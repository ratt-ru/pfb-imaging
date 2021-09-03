# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('CLEAN')

@cli.command()
@click.option('-d', '--dirty', required=True,
              help="Path to dirty.")
@click.option('-p', '--psf', required=True,
              help="Path to PSF")
@click.option('-wt', '--weight-table',
              help="Path to weight table produced by psf worker")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('-nmiter', '--nmiter', type=int, default=5,
              help="Number of major cycles")
@click.option('-hbg', "--hb-gamma", type=float, default=0.1,
              help="Minor loop gain of Hogbom")
@click.option('-hbpf', "--hb-peak-factor", type=float, default=0.1,
              help="Peak factor of Hogbom")
@click.option('-hbmaxit', "--hb-maxit", type=int, default=5000,
              help="Maximum number of iterations for Hogbom")
@click.option('-hbverb', "--hb-verbose", type=int, default=0,
              help="Verbosity of Hogbom. Set to 2 for debugging or "
              "zero for silence.")
@click.option('-hbrf', "--hb-report-freq", type=int, default=10,
              help="Report freq for hogbom.")
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
def clean(**kw):
    '''
    Single-scale clean.

    If the optional weight-table argument points to a valid weight table
    (created by the psf worker) the algorithm will approximate gradients using
    the diagonal Mueller weights assumption (exact for Stokes I imaging) i.e.

    IR = ID - R.H W R x

    otherwise it is a pure image space algorithm i.e.

    IR = ID - PSF.convolve(x)

    The latter is exact in the absence of wide-field effects and is usually
    much faster.

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
        nvthreads = nthreads//(nworkers*nthreads_per_worker)
    else:
        nvthreads = nthreads//nthreads-per-worker
    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')

    if args.nworkers is None:
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _clean(**args)

def _clean(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)

    import numpy as np
    import numexpr as ne
    import dask
    import dask.array as da
    from dask.distributed import performance_report
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.opt.hogbom import hogbom
    from astropy.io import fits

    print("Loading dirty", file=log)
    dirty = load_fits(args.dirty, dtype=args.output_type).squeeze()
    if len(dirty.shape) == 2:
        dirty = dirty[None, :, :]
    nband, nx, ny = dirty.shape
    hdr = fits.getheader(args.dirty)

    print("Loading psf", file=log)
    psf = load_fits(args.psf, dtype=args.output_type).squeeze()
    if len(psf.shape) == 2:
        psf = psf[None, :, :]
    _, nx_psf, ny_psf = psf.shape
    hdr_psf = fits.getheader(args.psf)

    wsums = np.amax(psf.reshape(-1, nx_psf*ny_psf), axis=1)
    wsum = np.sum(wsums)

    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

    assert (psf_mfs.max() - 1.0) < 1e-4

    dirty /= wsum
    dirty_mfs = np.sum(dirty, axis=0)

    # get info required to set WCS
    ra = np.deg2rad(hdr['CRVAL1'])
    dec = np.deg2rad(hdr['CRVAL2'])
    radec = [ra, dec]

    cell_deg = np.abs(hdr['CDELT1'])
    if cell_deg != np.abs(hdr['CDELT2']):
        raise NotImplementedError('cell sizes have to be equal')
    cell_rad = np.deg2rad(cell_deg)

    freq_out, ref_freq = data_from_header(hdr, axis=3)

    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    save_fits(args.output_filename + '_dirty_mfs.fits', dirty_mfs, hdr_mfs,
              dtype=args.output_type)

    # set up Hessian approximation
    if args.weight_table is not None:
        normfact = wsum
        from africanus.gridding.wgridder.dask import hessian
        from pfb.utils.misc import plan_row_chunk
        from daskms.experimental.zarr import xds_from_zarr

        xds = xds_from_zarr(args.weight_table)[0]
        nrow = xds.row.size
        freqs = xds.chan.data
        nchan = freqs.size

        # bin edges
        fmin = freqs.min()
        fmax = freqs.max()
        fbins = np.linspace(fmin, fmax, nband + 1)

        # chan <-> band mapping
        band_mapping = {}
        chan_chunks = {}
        freq_bin_idx = {}
        freq_bin_counts = {}
        band_map = np.zeros(freqs.size, dtype=np.int32)
        for band in range(nband):
            indl = freqs >= fbins[band]
            indu = freqs < fbins[band + 1] + 1e-6
            band_map = np.where(indl & indu, band, band_map)

        # to dask arrays
        bands, bin_counts = np.unique(band_map, return_counts=True)
        band_mapping = tuple(bands)
        chan_chunks = {'chan': tuple(bin_counts)}
        freqs = da.from_array(freqs, chunks=tuple(bin_counts))
        bin_idx = np.append(np.array([0]), np.cumsum(bin_counts))[0:-1]
        freq_bin_idx = da.from_array(bin_idx, chunks=1)
        freq_bin_counts = da.from_array(bin_counts, chunks=1)

        max_chan_chunk = bin_counts.max()
        bin_counts = tuple(bin_counts)
        # the first factor of 3 accounts for the intermediate visibilities
        # produced in Hessian (i.e. complex data + real weights)
        memory_per_row = (3 * max_chan_chunk * xds.WEIGHT.data.itemsize +
                          3 * xds.UVW.data.itemsize)

        # get approx image size
        pixel_bytes = np.dtype(args.output_type).itemsize
        band_size = nx * ny * pixel_bytes

        if args.host_address is None:
            # nworker bands on single node
            row_chunk = plan_row_chunk(args.mem_limit/args.nworkers,
                                       band_size, nrow, memory_per_row,
                                       args.nthreads_per_worker)
        else:
            # single band per node
            row_chunk = plan_row_chunk(args.mem_limit, band_size, nrow,
                                       memory_per_row,
                                       args.nthreads_per_worker)

        print("nrows = %i, row chunks set to %i for a total of %i chunks per node" %
              (nrow, row_chunk, int(np.ceil(nrow / row_chunk))), file=log)


        def convolver(x):
            model = da.from_array(x,
                          chunks=(1, nx, ny), name=False)

            xds = xds_from_zarr(args.weight_table, chunks={'row': row_chunk,
                                'chan': bin_counts})[0]

            convolvedim = hessian(xds.UVW.data,
                                  freqs,
                                  model,
                                  freq_bin_idx,
                                  freq_bin_counts,
                                  cell_rad,
                                  weights=xds.WEIGHT.data.astype(args.output_type),
                                  nthreads=args.nvthreads,
                                  epsilon=args.epsilon,
                                  do_wstacking=args.wstack,
                                  double_accum=args.double_accum)
            return convolvedim
    else:
        normfact = 1.0
        from pfb.operators.psf import hessian
        from ducc0.fft import r2c
        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding[-1])
        psf_pad = iFs(psf, axes=(1, 2))
        psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                     nthreads=nthreads, inorm=0)

        psfhat = da.from_array(psfhat, chunks=(1, -1, -1))

        def convolver(x):
            model = da.from_array(x,
                          chunks=(1, nx, ny), name=False)


            convolvedim = hessian(model,
                                  psfhat,
                                  padding,
                                  nvthreads,
                                  unpad_x,
                                  unpad_y,
                                  lastsize)
            return convolvedim

        # psfo = PSF(psf, dirty.shape, nthreads=args.nthreads)
        # def convolver(x): return psfo.convolve(x)

    rms = np.std(dirty_mfs)
    rmax = np.abs(dirty_mfs).max()

    print("Iter %i: peak residual = %f, rms = %f" % (
                0, rmax, rms), file=log)

    residual = dirty.copy()
    residual_mfs = dirty_mfs.copy()
    model = np.zeros_like(residual)
    for k in range(args.nmiter):
        print("Running Hogbom", file=log)
        x = hogbom(residual, psf,
                   gamma=args.hb_gamma,
                   pf=args.hb_peak_factor,
                   maxit=args.hb_maxit,
                   verbosity=args.hb_verbose,
                   report_freq=args.hb_report_freq)

        model += x
        print("Getting residual", file=log)

        convimage = convolver(model).compute()
        # dask.visualize(convimage, filename=args.output_filename + '_hessian' + str(k) + '_graph.pdf', optimize_graph=False)
        # with performance_report(filename=args.output_filename + '_hessian' + str(k) + '_per.html'):
        #     convimage = dask.compute(convimage, optimize_graph=False)[0]
        ne.evaluate('dirty - convimage/normfact', out=residual, casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs, casting='same_kind')

        save_fits(args.output_filename + f'_residual_mfs{k}.fits', residual_mfs, hdr_mfs)
        save_fits(args.output_filename + f'_model_mfs{k}.fits', model, hdr)
        save_fits(args.output_filename + f'_convim_mfs{k}.fits', convimage/normfact, hdr)


        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()

        print("Iter %i: peak residual = %f, rms = %f" % (
                k+1, rmax, rms), file=log)


    print("Saving results", file=log)
    save_fits(args.output_filename + '_model.fits', model, hdr)
    model_mfs = np.mean(model, axis=0)
    save_fits(args.output_filename + '_model_mfs.fits', model_mfs, hdr_mfs)
    save_fits(args.output_filename + '_residual.fits', residual*wsums[:, None, None], hdr)
    save_fits(args.output_filename + '_residual.fits', residual_mfs, hdr_mfs)

    print("All done here.", file=log)