# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('PSF')


@cli.command()
@click.argument('ms', nargs=-1)
@click.option('-dc', '--data-column',
              help="Data column to image."
              "Must be the same across MSs")
@click.option('-wc', '--weight-column',
              help="Column containing natural weights."
              "Must be the same across MSs")
@click.option('-iwc', '--imaging-weight-column',
              help="Column containing/to write imaging weights to."
              "Must be the same across MSs")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('--mock/--no-mock', default=False)
@click.option('-fc', '--flag-column', default='FLAG',
              help="Column containing data flags."
              "Must be the same across MSs")
@click.option('-muc', '--mueller-column',
              help="Column containing Mueller terms."
              "Must be the same across MSs")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-fov', '--field-of-view', type=float,
              help="Field of view in degrees")
@click.option('-srf', '--super-resolution-factor', type=float,
              help="Will over-sample Nyquist by this factor at max frequency")
@click.option('-psfo', '--psf-oversize', type=float, default=2.0,
              help='Size of the PSF relative to the dirty image.')
@click.option('-cs', '--cell-size', type=float,
              help='Cell size in arcseconds')
@click.option('-nx', '--nx', type=int,
              help="Number of x pixels")
@click.option('-ny', '--ny', type=int,
              help="Number of x pixels")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
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
@click.option('-roc', '--row-out-chunk', type=int, default=10000,
              help="Size of row chunks for output weights and uvw")
def psf(ms, **kw):
    '''
    Routine to create a psf image from a list of measurement sets.
    The psf image cube is not normalised by wsum as this destroyes
    information. The MFS image is written out in units of Jy/beam
    and should have a peak of one otherwise something has gone wrong.

    The --field-of-view and --super-resolution-factor options
    (equivalently --cell-size, --nx and --ny) pertain to the size of
    the image (eg. dirty and model). The size of the PSF output image
    is controlled by the --psf-oversize option.

    The Stokes I weights required to apply the Hessian are also written out
    to a zarr data set called output-filename.zarr. This data set does not
    adhere to the MSv2 specs and is only meant to be used to apply the
    Hessian. In particular, the weights written out are a combination of
    imaging weights and the "Mueller" weights.

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
        return _psf(ms, stack, **kw)

def _psf(ms, stack, **kw):
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
    pyscilog.enable_memory_logging(level=3)

    ms = list(ms)
    print('Input Options:', file=log)
    for key in kw.keys():
        print('     %25s = %s' % (key, kw[key]), file=log)

    # number of threads per worker
    if args.nthreads is None:
        if args.host_address is not None:
            raise ValueError("You have to specify nthreads when using a distributed scheduler")
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
    else:
        nthreads = args.nthreads

    # configure memory limit
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

    # numpy imports have to happen after this step
    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.graph_manipulation import clone
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms import Dataset
    from daskms.experimental.zarr import xds_to_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from africanus.gridding.wgridder.dask import dirty as vis2im
    from ducc0.fft import good_size
    from pfb.utils.misc import stitch_images
    from pfb.utils.fits import set_wcs, save_fits

    # chan <-> band mapping
    freqs, freq_bin_idx, freq_bin_counts, freq_out, band_mapping, chan_chunks = chan_to_band_mapping(
        ms, nband=args.nband)

    # gridder memory budget
    max_chan_chunk = 0
    max_freq = 0
    for ims in ms:
        for spw in freqs[ims]:
            counts = freq_bin_counts[ims][spw].compute()
            freq = freqs[ims][spw].compute()
            max_chan_chunk = np.maximum(max_chan_chunk, counts.max())
            max_freq = np.maximum(max_freq, freq.max())

    # assumes number of correlations are the same across MS/SPW
    xds = xds_from_ms(ms[0], columns=(args.data_column, args.weight_column))
    ncorr = xds[0].dims['corr']
    nrow = xds[0].dims['row']
    # we still have to cater for complex valued data because we cast
    # the weights to complex but we not longer need to factor the
    # weight column into our memory budget
    data_bytes = getattr(xds[0], args.data_column).data.itemsize
    bytes_per_row = max_chan_chunk * ncorr * data_bytes
    memory_per_row = bytes_per_row

    # flags (uint8 or bool)
    memory_per_row += bytes_per_row / 8

    # UVW (float64)
    memory_per_row += data_bytes * 3

    # data column is not actually read into memory just used to infer
    # dtype and chunking
    columns = (args.data_column,
               args.weight_column,
               args.flag_column,
               'UVW', 'TIME')

    # imaging weights
    if args.imaging_weight_column is not None:
        columns += (args.imaging_weight_column,)
        memory_per_row += bytes_per_row / 2

    # Mueller term (complex valued)
    if args.mueller_column is not None:
        columns += (args.mueller_column,)
        memory_per_row += bytes_per_row

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
        npix = int(args.psf_oversize * fov / cell_size)
        if npix % 2:
            npix += 1
        nx = good_size(npix)
        ny = good_size(npix)
    else:
        nx = args.nx
        ny = args.ny if args.ny is not None else nx

    print("PSF size set to (%i, %i, %i)" % (nband, nx, ny), file=log)

    # get approx image size
    # this is not a conservative estimate when multiple SPW's map to a single
    # imaging band
    pixel_bytes = np.dtype(args.output_type).itemsize

    # 0.8 assuming the gridder has about 20% memory overhead
    if args.host_address is None:  # full image on single node
        image_size = nband * nx * ny * pixel_bytes
        max_row_chunk = int(0.8*(mem_limit*1e9 - image_size)/(memory_per_row*nthreads_dask))
    else:  # max nthreads_per_workder bands per node
        image_size = nthreads_per_worker * nx * ny * pixel_bytes
        max_row_chunk = int(0.8*(mem_limit*1e9 - image_size)/(memory_per_row*nthreads_per_worker))
    print("Maximum row chunks set to %i for a total of %i chunks per node" %
          (max_row_chunk, np.ceil(nrow/max_row_chunk)), file=log)

    chunks = {}
    for ims in ms:
        chunks[ims] = []  # xds_from_ms expects a list per ds
        for spw in freqs[ims]:
            chunks[ims].append({'row': max_row_chunk,
                                'chan': chan_chunks[ims][spw]['chan']})

    psfs = []
    radec = None  # assumes we are only imaging field 0 of first MS
    out_datasets = []
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

            # this is not correct, need to use spw
            spw = ds.DATA_DESC_ID

            uvw = clone(ds.UVW.data)

            data_type = getattr(ds, args.data_column).data.dtype
            data_shape = getattr(ds, args.data_column).data.shape
            data_chunks = getattr(ds, args.data_column).data.chunks

            weights = getattr(ds, args.weight_column).data
            if len(weights.shape) < 3:
                weights = da.broadcast_to(
                    weights[:, None, :], data_shape, chunks=data_chunks)

            if args.imaging_weight_column is not None:
                imaging_weights = getattr(
                    ds, args.imaging_weight_column).data
                if len(imaging_weights.shape) < 3:
                    imaging_weights = da.broadcast_to(
                        imaging_weights[:, None, :], data_shape, chunks=data_chunks)

                weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
                weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
            else:
                weightsxx = weights[:, :, 0]
                weightsyy = weights[:, :, -1]

            # apply adjoint of mueller term.
            # Phases modify data amplitudes modify weights.
            if args.mueller_column is not None:
                mueller = getattr(ds, args.mueller_column).data
                weightsxx *= da.absolute(mueller[:, :, 0])**2
                weightsyy *= da.absolute(mueller[:, :, -1])**2

            # weighted sum corr to Stokes I
            weights = weightsxx + weightsyy

            # only keep data where both corrs are unflagged
            flag = getattr(ds, args.flag_column).data
            flagxx = flag[:, :, 0]
            flagyy = flag[:, :, -1]
            # ducc0 uses uint8 mask not flag
            flag = ~ (flagxx | flagyy)

            psf = vis2im(uvw,
                         freqs[ims][spw],
                         weights.astype(data_type),
                         freq_bin_idx[ims][spw],
                         freq_bin_counts[ims][spw],
                         nx,
                         ny,
                         cell_rad,
                         flag=flag.astype(np.uint8),
                         nthreads=gridder_nthreads,
                         epsilon=args.epsilon,
                         do_wstacking=args.wstack,
                         double_accum=args.double_accum)

            psfs.append(psf)

            data_vars = {
                'FIELD_ID':(('row',), da.full_like(ds.TIME.data, ds.FIELD_ID, chunks=args.row_out_chunk)),
                'DATA_DESC_ID':(('row',), da.full_like(ds.TIME.data, ds.DATA_DESC_ID, chunks=args.row_out_chunk)),
                'WEIGHT':(('row', 'chan'), weights.rechunk({0:args.row_out_chunk})),
                'UVW':(('row', 'uvw'), uvw.rechunk({0:args.row_out_chunk}))
            }

            coords = {
                'chan': (('chan',), freqs[ims][spw])
            }

            out_ds = Dataset(data_vars, coords)

            out_datasets.append(out_ds)

    writes = xds_to_zarr(out_datasets, args.output_filename + '.zarr', columns='ALL')

    dask.visualize(psfs, filename=args.output_filename + '_graph.pdf', optimize_graph=False)

    if not args.mock:
        psfs = dask.compute(psfs, writes, optimize_graph=False)[0]

        psf = stitch_images(psfs, nband, band_mapping)

        hdr = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec, freq_out)
        save_fits(args.output_filename + '_psf.fits', psf, hdr,
                dtype=args.output_type)

    print("All done here.", file=log)
