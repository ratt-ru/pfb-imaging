# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('PREDICT')


@cli.command()
@click.argument('ms', nargs=-1)
@click.option('-m', '--model', required=True,
              help='Path to model.fits')
@click.option('-mc', '--model-column', default='MODEL_DATA',
              help="Model column to write visibilities to."
              "Must be the same across MSs")
@click.option('-rchunk', '--row-chunks',
              help="Number of rows in a chunk.")
@click.option('-cchunk', '--chan-chunks',
              help="Number of channels in a chunk.")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--mock/--no-mock', default=False)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
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
@click.option('-ngt', '--ngridder-threads', type=int,
              help="Total number of threads to use per worker")
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
def predict(ms, **kw):
    '''
    Routine to predict model visibilities to measurement sets.
    Currently only predicts from .fits files

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

    Disclaimer - Memory budgeting is still very crude!

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = 1

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    if LocalCluster:
        ngridder-threads = nthreads//(nworkers*nthreads_per_worker)
    else:
        ngridder-threads = nthreads//nthreads-per-worker

    '''
    if not len(ms):
        raise ValueError("You must specify at least one measurement set")
    with ExitStack() as stack:
        return _predict(ms, stack, **kw)

def _predict(ms, stack, **kw):
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
        nthreads_per_worker = 1
    else:
        nthreads_per_worker = args.nthreads_per_worker

    # numpy imports have to happen after this step
    from pfb import set_client
    set_client(nthreads, mem_limit, nworkers, nthreads_per_worker,
               args.host_address, stack, log)

    # the number of chunks being read in simultaneously is equal to
    # the number of dask threads
    nthreads_dask = nworkers * nthreads_per_worker

    if args.host_address is not None:
        gridder_nthreads = nthreads//nthreads_per_worker
    else:
        gridder_nthreads = nthreads//nthreads_dask

    print("Number of threads allocated to each gridding instance %i"%
          gridder_nthreads, file=log)

    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.distributed import performance_report
    from dask.graph_manipulation import clone
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
    from africanus.gridding.wgridder.dask import model as im2vis
    from pfb.utils.fits import load_fits
    from pfb.utils.misc import restore_corrs, plan_row_chunk
    from astropy.io import fits

    # always returns 4D
    # gridder expects freq axis
    model = np.atleast_3d(load_fits(args.model).squeeze())
    nband, nx, ny = model.shape
    hdr = fits.getheader(args.model)
    cell_d = np.abs(hdr['CDELT1'])
    cell_rad = np.deg2rad(cell_d)

    # chan <-> band mapping
    freqs, freq_bin_idx, freq_bin_counts, freq_out, band_mapping, chan_chunks = chan_to_band_mapping(
        ms, nband=nband)

    # degridder memory budget
    max_chan_chunk = 0
    for ims in ms:
        for spw in freqs[ims]:
            counts = freq_bin_counts[ims][spw].compute()
            max_chan_chunk = np.maximum(max_chan_chunk, counts.max())

    # assumes number of correlations are the same across MS/SPW
    xds = xds_from_ms(ms[0])
    ncorr = xds[0].dims['corr']
    nrow = xds[0].dims['row']
    if args.output_type is not None:
        output_type = np.dtype(args.output_type)
    else:
        output_type = np.result_type(np.dtype(args.real_type), np.complex64)
    data_bytes = output_type.itemsize
    bytes_per_row = max_chan_chunk * ncorr * data_bytes
    memory_per_row = bytes_per_row  # model
    memory_per_row += 3*8  # uvw

    if mstype == 'zarr':
        if args.model_column in xds[0].keys():
            model_chunks = getattr(xds[0], args.model_column).data.chunks
        else:
            model_chunks = xds[0].DATA.data.chunks
            print('Chunking model same as data')

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

    model = da.from_array(model.astype(args.real_type),
                          chunks=(1, nx, ny), name=False)
    writes = []
    radec = None  # assumes we are only imaging field 0 of first MS
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
            radec = field.PHASE_DIR.data.squeeze()

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            spw = ds.DATA_DESC_ID  # this is not correct, need to use spw

            uvw = clone(ds.UVW.data)

            bands = band_mapping[ims][spw]
            model = model[list(bands), :, :]
            vis = im2vis(uvw,
                         freqs[ims][spw],
                         model,
                         freq_bin_idx[ims][spw],
                         freq_bin_counts[ims][spw],
                         cell_rad,
                         nthreads=nthreads//nband,
                         epsilon=args.epsilon,
                         do_wstacking=args.wstack)

            model_vis = restore_corrs(vis, ncorr)
            if mstype == 'zarr':
                model_vis = model_vis.rechunk(model_chunks)
                uvw = uvw.rechunk((model_chunks[0], 3))

            out_ds = ds.assign(**{args.model_column: (("row", "chan", "corr"), model_vis),
                                  'UVW': (("row", "three"), uvw)})
            # out_ds = ds.assign(**{args.model_column: (("row", "chan", "corr"), model_vis)})
            out_data.append(out_ds)

        writes.append(xds_to_table(out_data, ims, columns=[args.model_column]))

    dask.visualize(*writes, filename=args.output_filename + '_graph.pdf',
                   optimize_graph=False, collapse_outputs=True)

    if not args.mock:
        with performance_report(filename=args.output_filename + '_per.html'):
            dask.compute(writes, optimize_graph=False)

    print("All done here.", file=log)

