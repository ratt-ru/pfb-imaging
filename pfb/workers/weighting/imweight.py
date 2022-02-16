# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('IMWEIGHT')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

@cli.command()
@clickify_parameters(schema.imweight)
def imweight(**kw):
    '''
    This worker has two modes depending on whether a measurement set of a xds
    is supplied. When a measurement set is supplied it will compute and write
    imaging weights to a measurement set-like storage format. When an xds is
    supplied it will update the imaging weights in the xds and also optionally
    recompute the dirty image and PSF contained in the xds. The latter mode
    is useful if eg. gain solutions remain unchanged but we want to change
    just the imaging weights.

    The type of weighting to apply is determined by the
    --robustness parameter. The default of None implies uniform weighting.
    You don't need this step for natural weighting.

    Counts (i.e. how many visibilities fall in a grid point) are determined
    independent of the natural visibility weights and then used to scale the
    natural weights to a specific robustness value. This implies visibility
    weights still affect the relative weighting within a cell. In other words,
    when --robustness==None, you will only have a truly uniformly weighted
    image if the visibilities all have the same weight to start with.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively.

    Disclaimer - Memory budgeting is still very crude!

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = nthreads/nworkers

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(f'{args.output_filename}_{args.product}.log')
    from glob import glob
    if args.ms is not None:
        args.ms = glob(args.ms)
        try:
            assert len(args.ms) > 0
        except:
            raise ValueError(f"No MS at {args.ms}")
        mode = 'ms'
    elif args.xds is not None:
        if mode == 'ms':
            raise ValueError("You cannot supply both an ms or an xds")
        mode = 'xds'
    else:
        raise ValueError("You must supply either an ms or an xds")

    if args.nworkers is None:
        if args.scheduler=='distributed':
            args.nworkers = args.nband
        else:
            args.nworkers = 1
    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log, scheduler=args.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        if mode == 'ms':
            return _imweight_ms(**args)
        else:
            return _imweight_xds(**args)

def _imweight_ms(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig):
        args.ms = [args.ms]

    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.distributed import performance_report
    import dask.array as da
    from daskms import xds_to_table
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.misc import stitch_images
    from pfb.utils.misc import restore_corrs, plan_row_chunk, set_image_size
    from pfb.utils.weighting import counts_to_weights, compute_counts
    from pfb.utils.fits import set_wcs, save_fits

    # TODO - optional grouping.
    # We need to construct an identifier between
    # dataset and field/spw/scan identifiers
    group_by = []
    if args.group_by_field:
        group_by.append('FIELD_ID')
    else:
        raise NotImplementedError("Grouping by field is currently mandatory")

    if args.group_by_ddid:
        group_by.append('DATA_DESC_ID')
    else:
        raise NotImplementedError("Grouping by DDID is currently mandatory")

    if args.group_by_scan:
        group_by.append('SCAN_NUMBER')
    else:
        raise NotImplementedError("Grouping by scan is currently mandatory")

    # chan <-> band mapping
    nband = args.nband
    freqs, fbin_idx, fbin_counts, freq_out, band_mapping, chan_chunks = \
        chan_to_band_mapping(args.ms, nband=args.nband, group_by=group_by)

    # gridder memory budget (TODO)
    max_chan_chunk = 0
    max_freq = 0
    for ms in args.ms:
        for spw in freqs[ms]:
            counts = fbin_counts[ms][spw].compute()
            freq = freqs[ms][spw].compute()
            max_chan_chunk = np.maximum(max_chan_chunk, counts.max())
            max_freq = np.maximum(max_freq, freq.max())

    # assumes measurement sets have the same columns
    xds = xds_from_ms(args.ms[0])
    columns = (args.data_column,
               args.flag_column,
               'UVW', 'ANTENNA1',
               'ANTENNA2', 'TIME', 'INTERVAL')
    schema = {}
    schema[args.data_column] = {'dims': ('chan', 'corr')}
    schema[args.flag_column] = {'dims': ('chan', 'corr')}

    # only WEIGHT column gets special treatment
    # any other column must have channel axis
    if args.weight_column is not None:
        columns += (args.weight_column,)
        if args.weight_column == 'WEIGHT':
            schema[args.weight_column] = {'dims': ('corr')}
        else:
            schema[args.weight_column] = {'dims': ('chan', 'corr')}

    # flag row
    if 'FLAG_ROW' in xds[0]:
        columns += ('FLAG_ROW',)

    # imaging weights
    if args.imaging_weight_column is not None:
        columns += (args.imaging_weight_column,)
        schema[args.imaging_weight_column] = {'dims': ('chan', 'corr')}

    # get max uv coords over all datasets
    uvw = []
    u_max = 0.0
    v_max = 0.0
    for ms in args.ms:
        xds = xds_from_ms(ms, columns=('UVW'), chunks={'row': -1},
                          group_cols=group_by)

        for ds in xds:
            uvw = ds.UVW.data
            u_max = da.maximum(u_max, abs(uvw[:, 0]).max())
            v_max = da.maximum(v_max, abs(uvw[:, 1]).max())
            uv_max = da.maximum(u_max, v_max)

    uv_max = uv_max.compute()
    del uvw

    # image size
    cell_N = 1.0 / (2 * uv_max * max_freq / lightspeed)

    if args.cell_size is not None:
        cell_size = args.cell_size
        cell_rad = cell_size * np.pi / 60 / 60 / 180
        if cell_N / cell_rad < 1:
            raise ValueError("Requested cell size too small. "
                             "Super resolution factor = ", cell_N / cell_rad)
        print(f"Super resolution factor = {cell_N/cell_rad}", file=log)
    else:
        cell_rad = cell_N / args.super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi
        print(f"Cell size set to {cell_size} arcseconds", file=log)

    if args.nx is None:
        fov = args.field_of_view * 3600
        npix = int(fov / cell_size)
        npix = good_size(npix)
        while npix % 2:
            npix += 1
            npix = good_size(npix)
        nx = npix
        ny = npix
    else:
        nx = args.nx
        ny = args.ny if args.ny is not None else nx

    print(f"Image size set to ({nband}, {nx}, {ny})", file=log)



    # compute counts
    counts = []
    radec = None  # assumes we are only imaging field 0 of first MS
    for ims in ms:
        xds = xds_from_ms(ims, chunks=chunks[ims], columns=columns)

        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD")
        spws = xds_from_table(ims + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ims + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]

        for ds in xds:
            field = fields[ds.FIELD_ID]

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            spw = ds.DATA_DESC_ID  # not optimal, need to use spw

            uvw = ds.UVW.data

            # MS may contain auto-correlations
            if 'FLAG_ROW' in xds[0]:
                frow = ds.FLAG_ROW.data | (ds.ANTENNA1.data == ds.ANTENNA2.data)
            else:
                frow = (ds.ANTENNA1.data == ds.ANTENNA2.data)

            # only keep data where both corrs are unflagged
            flag = getattr(ds, args.flag_column).data
            flagxx = flag[:, :, 0]
            flagyy = flag[:, :, -1]
            flag = da.logical_or((flagxx | flagyy), frow[:, None])

            count = compute_counts(
                uvw,
                freqs[ims][spw],
                freq_bin_idx[ims][spw],
                freq_bin_counts[ims][spw],
                flag,
                nx,
                ny,
                cell_rad,
                cell_rad,
                args.output_type if not args.double_accum else np.float64)

            counts.append(count)

    dask.visualize(counts, filename=args.output_filename + '_counts_graph.pdf', optimize_graph=False)
    with performance_report(filename=args.output_filename + '_counts_per.html'):
            counts = dask.compute(counts, optimize_graph=False)[0]

    counts = stitch_images(counts, nband, band_mapping)

    # save counts grid
    hdr = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec, freq_out)
    save_fits(args.output_filename + '_counts.fits', counts, hdr, dtype=args.output_type)

    counts = da.from_array(counts.astype(args.output_type),
                           chunks=(1, -1, -1), name=False)

    # convert counts to weights
    writes = []
    for ims in ms:
        xds = xds_from_ms(ims, chunks=chunks[ims], columns=('UVW'))

        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD")
        spws = xds_from_table(ims + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ims + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]

        out_data = []
        for ds in xds:
            field = fields[ds.FIELD_ID]

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            spw = ds.DATA_DESC_ID  # this is not correct, need to use spw

            uvw = ds.UVW.data

            # dask-ms.optimization inline-array
            weights = counts_to_weights(
                counts,
                uvw,
                freqs[ims][spw],
                freq_bin_idx[ims][spw],
                freq_bin_counts[ims][spw],
                nx,
                ny,
                cell_rad,
                cell_rad,
                np.dtype(args.output_type),
                args.robustness)

            # hack to get shape and chunking info
            weights = restore_corrs(weights, ncorr)
            if mstype == 'zarr':
                # flag = flag.rechunk(iw_chunks)
                weights = weights.rechunk(iw_chunks)
                uvw = uvw.rechunk((iw_chunks[0], 3))

            out_ds = ds.assign(**{args.imaging_weight_column: (("row", "chan", "corr"), weights),
                                  'UVW': (("row", "three"), uvw)})
            out_data.append(out_ds)
        writes.append(xds_to_table(out_data, ims,
                                   columns=[args.imaging_weight_column]))

    print("All done here.", file=log)


def _imweight_xds(**kw):
    args = OmegaConf.create(kw)
