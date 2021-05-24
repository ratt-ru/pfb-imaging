# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('IMWEIGHT')


@cli.command()
@click.argument('ms', nargs=-1)
@click.option('-iwc', '--imaging-weight-column',
              default='IMAGING_WEIGHT_SPECTRUM',
              help="Column to write imaging weights to.")
@click.option('-fc', '--flag-column', default='FLAG',
              help="Column containing data flags."
              "Must be the same across MSs")
@click.option('-rb', '--robustness', type=float,
              help="Robustness factor between -2, and 2. "
              "None implies uniform weighting.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-fov', '--field-of-view', type=float,
              help="Field of view in degrees")
@click.option('-srf', '--super-resolution-factor', type=float,
              help="Will over-sample Nyquist by this factor at max frequency")
@click.option('-cs', '--cell-size', type=float,
              help='Cell size in arcseconds')
@click.option('-nx', '--nx', type=int,
              help="Number of x pixels")
@click.option('-ny', '--ny', type=int,
              help="Number of x pixels")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
def imweight(ms, **kw):
    '''
    Routine to compute and write imaging weights to a measurement set-like
    storage format. The type of weighting to apply is determined by the
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
    if not len(ms):
        raise ValueError("You must specify at least one measurement set")
    with ExitStack() as stack:
        return _imweight(ms, stack, **kw)

def _imweight(ms, stack, **kw):
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
    pyscilog.enable_memory_logging(level=3)

    ms = list(ms)
    print('Input Options:', file=log)
    for key in kw.keys():
        print('     %25s = %s' % (key, kw[key]), file=log)

    if args.nthreads is None:
        if args.host_address is not None:
            raise ValueError("You have to specify nthreads when using a distributed scheduler")
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
    else:
        nthreads = args.nthreads

    if args.mem_limit is None:
        if args.host_address is not None:
            raise ValueError("You have to specify mem-limit when using a distributed scheduler")
        import psutil
        mem_limit = int(psutil.virtual_memory()[0]/1e9)  # 100% of memory by default
    else:
        mem_limit = args.mem_limit

    nband = args.nband
    if args.nworkers is None:
        nworkers = nband
    else:
        nworkers = args.nworkers

    if args.nthreads_per_worker is None:
        nthreads_per_worker = nthreads//nworkers
    else:
        nthreads_per_worker = args.nthreads_per_worker

    # numpy imports have to happen after this step
    from pfb import set_client
    set_client(nthreads, mem_limit, nworkers, nthreads_per_worker,
               args.host_address, stack, log)

    # the number of chunks being read in simultaneously is equal to
    # the number of dask threads
    nthreads_dask = nworkers * nthreads_per_worker

    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.distributed import performance_report
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.utils import dataset_type
    mstype = dataset_type(ms[0])
    if mstype == 'casa':
        from daskms import xds_to_table
    elif mstype == 'zarr':
        from daskms.experimental.zarr import xds_to_zarr as xds_to_table
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.misc import stitch_images
    from pfb.utils.misc import restore_corrs, plan_row_chunk, set_image_size
    from pfb.utils.weighting import counts_to_weights, compute_counts
    from pfb.utils.fits import set_wcs, save_fits

    # chan <-> band mapping
    freqs, freq_bin_idx, freq_bin_counts, freq_out, band_mapping, chan_chunks = chan_to_band_mapping(
        ms, nband=nband)

    # gridder memory budget
    max_chan_chunk = 0
    max_freq = 0
    for ims in ms:
        for spw in freqs[ims]:
            counts = freq_bin_counts[ims][spw].compute()
            freq = freqs[ims][spw].compute()
            max_chan_chunk = np.maximum(max_chan_chunk, counts.max())
            max_freq = np.maximum(max_freq, freq.max())

    # assumes measurement sets have the same columns,
    # number of correlations etc.
    xds = xds_from_ms(ms[0])
    ncorr = xds[0].dims['corr']
    nrow = xds[0].dims['row']

    # imaging weights
    memory_per_row = np.dtype(args.output_type).itemsize * max_chan_chunk * ncorr

    # flags (uint8 or bool)
    memory_per_row += np.dtype(np.uint8).itemsize * max_chan_chunk * ncorr

    # UVW
    memory_per_row += xds[0].UVW.data.itemsize * 3

    # ANTENNA1/2
    memory_per_row += xds[0].ANTENNA1.data.itemsize * 2

    columns = (args.flag_column,
               'UVW', 'ANTENNA1',
               'ANTENNA2')

    # flag row
    if 'FLAG_ROW' in xds[0]:
        columns += ('FLAG_ROW',)
        memory_per_row += xds[0].FLAG_ROW.data.itemsize

    if mstype == 'zarr':
        if args.imaging_weight_column in xds[0].keys():
            iw_chunks = getattr(xds[0], args.imaging_weight_column).data.chunks
        else:
            iw_chunks = xds[0].DATA.data.chunks
            print('Chunking imaging weights same as data')

    # get max uv coords over all fields
    uvw = []
    u_max = 0.0
    v_max = 0.0
    for ims in ms:
        xds = xds_from_ms(ims, columns=('UVW'), chunks={'row': -1})

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
        print("Super resolution factor = %f" % (cell_N / cell_rad), file=log)
    else:
        cell_rad = cell_N / args.super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi
        print("Cell size set to %5.5e arcseconds" % cell_size, file=log)

    if args.nx is None:
        fov = args.field_of_view * 3600
        npix = int(fov / cell_size)
        if npix % 2:
            npix += 1
        nx = good_size(npix)
        ny = good_size(npix)
    else:
        nx = args.nx
        ny = args.ny if args.ny is not None else nx

    print("Image size set to (%i, %i, %i)" % (nband, nx, ny), file=log)

    # get approx image size
    # this is not a conservative estimate when multiple SPW's map to a single
    # imaging band
    pixel_bytes = np.dtype(args.output_type).itemsize
    band_size = nx * ny * pixel_bytes


    if args.host_address is None:
        # full image on single node
        row_chunk = plan_row_chunk(mem_limit/nworkers, band_size, nrow,
                                   memory_per_row, nthreads_per_worker)
    else:
        # single band per node
        row_chunk = plan_row_chunk(mem_limit, band_size, nrow,
                                   memory_per_row, nthreads_per_worker)

    if args.row_chunks is not None:
        row_chunk = int(args.row_chunks)
        if row_chunk == -1:
            row_chunk = nrow

    print("nrows = %i, row chunks set to %i for a total of %i chunks per node" %
          (nrow, row_chunk, int(np.ceil(nrow / row_chunk))), file=log)

    chunks = {}
    for ims in ms:
        chunks[ims] = []  # xds_from_ms expects a list per ds
        for spw in freqs[ims]:
            chunks[ims].append({'row': row_chunk,
                                'chan': chan_chunks[ims][spw]['chan']})

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
        xds = xds_from_ms(ims, chunks=chunks[ims], columns=('UVW', args.imaging_weight_column))

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

    dask.visualize(*writes, filename=args.output_filename + 'weights_graph.pdf',
                   optimize_graph=False, collapse_outputs=True)

    with performance_report(filename=args.output_filename + 'weights_per.html'):
        dask.compute(writes, optimize_graph=False)

    print("All done here.", file=log)
