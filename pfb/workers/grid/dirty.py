# flake8: noqa

from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('DIRTY')


@cli.command()
@click.argument('ms', nargs=-1)
              #help="List of paths to measurement sets")
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
@click.option('-cs', '--cell-size', type=float,
              help='Cell size in arcseconds')
@click.option('-nx', '--nx', type=int,
              help="Number of x pixels")
@click.option('-ny', '--ny', type=int,
              help="Number of x pixels")
@click.option('-nthreads', '--nthreads', type=int, default=0,
              help="Total number of threads to use per worker")
@click.option('-mem', '--mem-limit', type=int, default=0,
              help="Memory limit in GB. Default of 0 means use all available memory")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
def dirty(ms, **kw):
    '''
    Routine to create a dirty image from a list of measurement sets.
    The dirty image cube is not normalised by wsum as this destroyes
    information. The MFS image is written out in units of Jy/beam.
    The normalisation factors can be obtained by making a psf image
    using the psf worker (see pfbworkers psf --help).

    If a host address is provided the computation will be distributed
    first over imaging band and then over rows in case a full imaging
    band does not fit into memory. When using a distributed scheduler
    mem_limit is per node, otherwise it should specify the maximum memory
    we can allocated to the gridder. In the latter case we step through
    the bands one at a time to minimise memory consumption.
    Disclaimer - Memory budgeting is still very crude!
    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
    pyscilog.enable_memory_logging(level=3)

    ms = list(ms)
    print('Input Options:', file=log)
    for key in kw.keys():
        print('     %25s = %s' % (key, kw[key]), file=log)

    # number of threads per worker
    if not args.nthreads:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
    else:
        nthreads = args.nthreads

    # configure memory limit
    if not args.mem_limit:
        import psutil
        mem_limit = int(psutil.virtual_memory()[0]/1e9)  # 100% of memory by default
    else:
        mem_limit = args.mem_limit

    # client has nband workers
    from pfb import set_client
    nband = args.nband
    gridder_threads = set_client(nthreads, nband, mem_limit, args.host_address)
    # numpy imports have to happen after this step
    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
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
    ncorr = xds[0].corr.size
    data_bytes = getattr(xds[0], args.data_column).data.itemsize
    bytes_per_row = max_chan_chunk * ncorr * data_bytes
    memory_per_row = bytes_per_row

    # real valued weights
    wdims = getattr(xds[0], args.weight_column).data.ndim
    if wdims == 2:  # WEIGHT
        memory_per_row += ncorr * data_bytes / 2
    else:           # WEIGHT_SPECTRUM
        memory_per_row += bytes_per_row / 2

    # flags (uint8 or bool)
    memory_per_row += bytes_per_row / 8

    # UVW (float64)
    memory_per_row += data_bytes * 3

    columns = (args.data_column,
               args.weight_column,
               args.flag_column,
               'UVW')

    # imaging weights
    if args.imaging_weight_column is not None:
        columns += (args.imaging_weight_column,)
        memory_per_row += bytes_per_row / 2

    # Mueller term (complex valued)
    if args.mueller_column is not None:
        columns += (args.mueller_column,)
        memory_per_row += bytes_per_row

    max_row_chunk = int(mem_limit*1e9/memory_per_row)
    print("Maximum row chunks set to %i"%max_row_chunk, file=log)

    chunks = {}
    for ims in ms:
        chunks[ims] = []  # xds_from_ms expects a list per ds
        for spw in freqs[ims]:
            chunks[ims].append({'row': max_row_chunk,
                                'chan': chan_chunks[ims][spw]['chan']})

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

    if args.nx is None or args.ny is None:
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

    dirties = []
    wsum = 0.0
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

            # this is not correct, need to use spw
            spw = ds.DATA_DESC_ID

            uvw = ds.UVW.data

            data = getattr(ds, args.data_column).data
            dataxx = data[:, :, 0]
            datayy = data[:, :, -1]

            weights = getattr(ds, args.weight_column).data
            if len(weights.shape) < 3:
                weights = da.broadcast_to(
                    weights[:, None, :], data.shape, chunks=data.chunks)

            if args.imaging_weight_column is not None:
                imaging_weights = getattr(
                    ds, args.imaging_weight_column).data
                if len(imaging_weights.shape) < 3:
                    imaging_weights = da.broadcast_to(
                        imaging_weights[:, None, :], data.shape, chunks=data.chunks)

                weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
                weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
            else:
                weightsxx = weights[:, :, 0]
                weightsyy = weights[:, :, -1]

            # apply adjoint of mueller term.
            # Phases modify data amplitudes modify weights.
            if args.mueller_column is not None:
                mueller = getattr(ds, args.mueller_column).data
                dataxx *= da.exp(-1j * da.angle(mueller[:, :, 0]))
                datayy *= da.exp(-1j * da.angle(mueller[:, :, -1]))
                weightsxx *= da.absolute(mueller[:, :, 0])
                weightsyy *= da.absolute(mueller[:, :, -1])

            # weighted sum corr to Stokes I
            weights = weightsxx + weightsyy
            data = (weightsxx * dataxx + weightsyy * datayy)
            # TODO - turn off this stupid warning
            data = da.where(weights, data / weights, 0.0j)

            # only keep data where both corrs are unflagged
            flag = getattr(ds, args.flag_column).data
            flagxx = flag[:, :, 0]
            flagyy = flag[:, :, -1]
            # ducc0 uses uint8 mask not flag
            flag = ~ (flagxx | flagyy)

            wsum += da.sum((weights * flag).ravel())

            dirty = vis2im(
                uvw,
                freqs[ims][spw],
                data,
                freq_bin_idx[ims][spw],
                freq_bin_counts[ims][spw],
                nx,
                ny,
                cell_rad,
                weights=weights,
                flag=flag.astype(np.uint8),
                nthreads=gridder_threads,
                epsilon=args.epsilon,
                do_wstacking=args.wstack,
                double_accum=args.double_accum)

            dirties.append(dirty)


    # dask.visualize(dirties, wsum, filename=args.output_filename + '_graph.pdf', optimize_graph=False)
    result = dask.compute(dirties, wsum)

    dirties = result[0]
    wsum = result[1]

    dirty = stitch_images(dirties, nband, band_mapping)

    hdr = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec, freq_out)
    save_fits(args.output_filename + '_dirty.fits', dirty, hdr, dtype=args.output_type)

    hdr_mfs = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec, np.mean(freq_out))
    dirty_mfs = np.sum(dirty, axis=0)/wsum
    save_fits(args.output_filename + '_dirty_mfs.fits', dirty_mfs, hdr_mfs, dtype=args.output_type)
