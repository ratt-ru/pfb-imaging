# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('DEGRID')


@cli.command()
@click.option('-ms', '--ms', required=True,
              help='Path to measurement set.')
@click.option('-m', '--model', required=True,
              help='Path to model.fits')
@click.option('-mc', '--model-column', default='MODEL_DATA',
              help="Model column to write visibilities to."
              "Must be the same across MSs")
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('--group-by-field/--no-group-by-field', default=True)
@click.option('--group-by-ddid/--no-group-by-ddid', default=True)
@click.option('--group-by-scan/--no-group-by-scan', default=True)
@click.option('--wstack/--no-wstack', default=True)
@click.option('--mock/--no-mock', default=False)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-nbo', '--nband-out', type=int,
              help="Number of imaging bands in output cube. "
              "Default is same as input but it is possible to increase it to "
              "improve degridding accuracy. If set to -1 the model will be "
              "interpolated to the resolution of the MS.")
@click.option('-spo', '--spectral-poly-order', type=int,
              help='Order of polynomial to fit to freq axis. '
              'Set to zero to turn off interpolation')
@click.option('-otype', '--output-type',
              help="Data type of output")
@click.option('-rtype', '--real-type', default='f4',
              help="Real data type")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use per worker")
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
def degrid(**kw):
    '''
    Predict model visibilities to measurement sets.
    Currently only predicts from .fits files which can optionally be interpolated
    along the frequency axis.

    By default the real-type argument specifies the type of the
    output unless --output-dtype is explicitly set.

    The row chunk size is determined automatically from mem_limit
    unless it is specified explicitly.

    The chan chunk size is determined automatically from chan <-> band
    mapping unless it is specified explicitly.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default the gridder will
    use all available resources.

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = 1

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    if LocalCluster:
        nvthreads = nthreads//(nworkers*nthreads_per_worker)
    else:
        nvthreads = nthreads//nthreads-per-worker

    where nvthreads refers to the number of threads used to scale vertically
    (eg. the number threads given to each gridder instance).

    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')

    from glob import glob
    ms = glob(args.ms)
    try:
        assert len(ms) > 0
        args.ms = ms
    except:
        raise ValueError(f"No MS at {args.ms}")

    if args.nworkers is None:
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _degrid(**args)

def _degrid(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig) :
        args.ms = [args.ms]
    OmegaConf.set_struct(args, True)

    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.distributed import performance_report
    from dask.graph_manipulation import clone
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.utils import dataset_type
    mstype = dataset_type(args.ms[0])
    if mstype == 'casa':
        from daskms import xds_to_table
    else:
        raise ValueError("predict currently only supports measurement sets")
        # from daskms.experimental.zarr import xds_to_zarr as xds_to_table
    import dask.array as da
    from africanus.constants import c as lightspeed
    from africanus.gridding.wgridder.dask import model as im2vis
    from pfb.utils.fits import load_fits, data_from_header
    from pfb.utils.misc import restore_corrs, model_from_comps
    from astropy.io import fits

    # always returns 4D
    # gridder expects freq axis
    model = np.atleast_3d(load_fits(args.model).squeeze())
    nband, nx, ny = model.shape
    hdr = fits.getheader(args.model)
    cell_d = np.abs(hdr['CDELT1'])
    cell_rad = np.deg2rad(cell_d)
    mfreqs, ref_freq = data_from_header(hdr, axis=3)

    if args.nband_out is None:
        nband_out = nband
    else:
        nband_out = args.nband_out

    # TODO - optional grouping. We need to construct an identifier between
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
    freqs, freq_bin_idx, freq_bin_counts, freq_out, band_mapping, chan_chunks = chan_to_band_mapping(
        args.ms, nband=nband_out, group_by=group_by)

    if args.output_type is not None:
        output_type = np.dtype(args.output_type)
    else:
        output_type = np.result_type(np.dtype(args.real_type), np.complex64)

    # if mstype == 'zarr':
    #     if args.model_column in xds[0].keys():
    #         model_chunks = getattr(xds[0], args.model_column).data.chunks
    #     else:
    #         model_chunks = xds[0].DATA.data.chunks
    #         print('Chunking model same as data', file=log)




    ms_chunks = {}
    ncorr = None
    for ms in args.ms:
        xds = xds_from_ms(ms, group_cols=group_by)
        ms_chunks[ms] = []  # daskms expects a list per ds

        for ds in xds:
            idt = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"

            if ncorr is None:
                ncorr = ds.dims['corr']
            else:
                try:
                    assert ncorr == ds.dims['corr']
                except Exception as e:
                    raise ValueError("All data sets must have the same number of correlations")

            if args.row_chunk in [0, -1, None]:
                rchunks = ds.dims['row']
            else:
                rchunks = args.row_chunk

            ms_chunks[ms].append({'row': rchunks,
                                  'chan': chan_chunks[ms][idt]})

    # interpolate model
    mask = np.any(model, axis=0)
    Ix, Iy = np.where(mask)
    ncomps = Ix.size

    # components excluding zeros
    beta = model[:, Ix, Iy]
    if args.spectral_poly_order:
        order = args.spectral_poly_order
        print(f"Fitting integrated polynomial of order {order}", file=log)
        if order > mfreqs.size:
            raise ValueError("spectral-poly-order can't be larger than nband")

        # we are given frequencies at bin centers, convert to bin edges
        delta_freq = mfreqs[1] - mfreqs[0]
        wlow = (mfreqs - delta_freq/2.0)/ref_freq
        whigh = (mfreqs + delta_freq/2.0)/ref_freq
        wdiff = whigh - wlow

        # set design matrix for each component
        Xfit = np.zeros([mfreqs.size, order])
        for i in range(1, order+1):
            Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

        # get wsum per band
        wsums = np.zeros((mfreqs.size,1))
        try:
            for i in range(mfreqs.size):
                wsums[i] = hdr[f'WSUM{i}']
        except Exception as e:
            print("Can't find WSUMS in header, using unity weights", file=log)
            wsums = np.ones((mfreqs.size,1))

        dirty_comps = Xfit.T.dot(wsums*beta)

        hess_comps = Xfit.T.dot(wsums*Xfit)

        comps = np.linalg.solve(hess_comps, dirty_comps)
        freq_fitted = True
    else:
        print("Not fitting frequency axis", file=log)
        comps = beta
        freq_fitted = False

    print("Computing model visibilities", file=log)
    mask = da.from_array(mask, chunks=(nx, ny), name=False)
    comps = da.from_array(comps.astype(args.real_type),
                          chunks=(-1, ncomps), name=False)
    freq_out = da.from_array(freq_out, chunks=-1, name=False)
    writes = []
    radec = None  # assumes we are only imaging field 0 of first MS
    for ms in args.ms:
        xds = xds_from_ms(ms, chunks=ms_chunks[ms], columns=('UVW'),
                          group_cols=group_by)

        # subtables
        ddids = xds_from_table(ms + "::DATA_DESCRIPTION")
        fields = xds_from_table(ms + "::FIELD")
        spws = xds_from_table(ms + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ms + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]

        out_data = []
        for ds in xds:
            field = fields[ds.FIELD_ID]
            radec = field.PHASE_DIR.data.squeeze()

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            # TODO - rephase if fields don't match, requires PB model
            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            idt = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"

            model = model_from_comps(comps, freq_out, mask,
                                     band_mapping[ms][idt],
                                     ref_freq, freq_fitted)

            uvw = ds.UVW.data
            vis_I = im2vis(uvw,
                           freqs[ms][idt],
                           model,
                           freq_bin_idx[ms][idt],
                           freq_bin_counts[ms][idt],
                           cell_rad,
                           nthreads=args.nvthreads,
                           epsilon=args.epsilon,
                           do_wstacking=args.wstack)

            # vis_Q = im2vis(uvw,
            #              freqs[ms][idt],
            #              model[1],
            #              freq_bin_idx[ms][idt],
            #              freq_bin_counts[ms][idt],
            #              cell_rad,
            #              nthreads=ngridder_threads,
            #              epsilon=args.epsilon,
            #              do_wstacking=args.wstack)

            model_vis = restore_corrs(vis_I, ncorr)

            model_vis = model_vis.rechunk({1: -1})


            # if mstype == 'zarr':
            #     model_vis = model_vis.rechunk(model_chunks)
            #     uvw = uvw.rechunk((model_chunks[0], 3))

            # out_ds = ds.assign(**{args.model_column: (("row", "chan", "corr"), model_vis),
            #                       'UVW': (("row", "three"), uvw)})
            out_ds = ds.assign(**{args.model_column: (("row", "chan", "corr"), model_vis)})
            out_data.append(out_ds)

        writes.append(xds_to_table(out_data, ms, columns=[args.model_column]))

    dask.visualize(*writes, filename=args.output_filename + '_predict_graph.pdf',
                   optimize_graph=False, collapse_outputs=True)

    # if not args.mock:
    #     with performance_report(filename=args.output_filename + '_predict_per.html'):
    #         dask.compute(writes, optimize_graph=False)

    from pfb.utils.misc import compute_context
    with compute_context(args.scheduler, args.output_filename):
        dask.compute(writes,
                     optimize_graph=False,
                     scheduler=args.scheduler)

    print("All done here.", file=log)

