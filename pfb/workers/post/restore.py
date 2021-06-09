# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('RESTORE')


@cli.command()
@click.option('-m', '--model', required=True,
              help="Path to model image cube")
@click.option('-r', '--residual', required=True,
              help="Path to residual image cube")
@click.option('-p', '--psf', required=True,
              help="Path to PSF cube")
@click.option('-beam', '--beam', required=False,
              help="Path to power beam image cube")
@click.option('-o', '--output-filename', required=True,
              help="Basename of output.")
@click.option('-nthreads', '--nthreads', type=int, default=1,
              show_default=True, help='Number of threads to use')
@click.option('-cr', '--convolve-residuals', is_flag=True,
              help='Whether to convolve the residuals to a common resolution')
@click.option('-pf', '--padding-frac', type=float,
              default=0.5, show_default=True,
              help="Padding fraction for FFTs (half on either side)")
@click.option('-pb-min', '--pb-min', type=float, default=0.1,
              help="Set image to zero where pb falls below this value")
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
def restore(**kw):
    '''
    Create restored images.

    Can also be used to convolve images to a common resolution
    and/or perform a primary beam correction.
    '''
    with ExitStack() as stack:
        return _restore(stack, **kw)

def _restore(stack, **kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.output_filename + '.log')
    pyscilog.enable_memory_logging(level=3)

    # number of threads per worker
    if args.nthreads is None:
        if args.host_address is not None:
            raise ValueError("You have to specify nthreads when using a distributed scheduler")
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
        args.nthreads = nthreads
    else:
        nthreads = args.nthreads

    # configure memory limit
    if args.mem_limit is None:
        if args.host_address is not None:
            raise ValueError("You have to specify mem-limit when using a distributed scheduler")
        import psutil
        mem_limit = int(psutil.virtual_memory()[0]/1e9)  # 100% of memory by default
        args.mem_limit = mem_limit
    else:
        mem_limit = args.mem_limit

    nband = args.nband
    if args.nworkers is None:
        nworkers = nband
        args.nworkers = nworkers
    else:
        nworkers = args.nworkers

    if args.nthreads_per_worker is None:
        nthreads_per_worker = 1
        args.nthreads_per_worker = nthreads_per_worker
    else:
        nthreads_per_worker = args.nthreads_per_worker

    # the number of chunks being read in simultaneously is equal to
    # the number of dask threads
    nthreads_dask = nworkers * nthreads_per_worker

    if args.ngridder_threads is None:
        if args.host_address is not None:
            ngridder_threads = nthreads//nthreads_per_worker
        else:
            ngridder_threads = nthreads//nthreads_dask
        args.ngridder_threads = ngridder_threads
    else:
        ngridder_threads = args.ngridder_threads

    ms = list(ms)
    print('Input Options:', file=log)
    for key in kw.keys():
        print('     %25s = %s' % (key, args[key]), file=log)

    # numpy imports have to happen after this step
    from pfb import set_client
    set_client(nthreads, mem_limit, nworkers, nthreads_per_worker,
               args.host_address, stack, log)

    import numpy as np
    from astropy.io import fits
    mhdr = fits.getheader(args.model)

    from pfb.utils.fits import load_fits
    model = load_fits(args.model).squeeze()  # drop Stokes axis

    # check images compatible
    rhdr = fits.getheader(args.residual)

    from pfb.utils.fits import compare_headers
    compare_headers(mhdr, rhdr)
    residual = load_fits(args.residual).squeeze()

    # fit restoring psf
    from pfb.utils.misc import fitcleanbeam
    psf = load_fits(args.psf, dtype=args.real_type).squeeze()

    nband, nx_psf, ny_psf = psf.shape
    wsums = np.amax(psf.reshape(args.nband, nx_psf ny_psf), axis=1)
    wsum = np.sum(wsums)
    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

    # fit restoring psf
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)
    GaussPars = fitcleanbeam(psf, level=0.5, pixsize=1.0)

    cpsf_mfs = np.zeros(psf_mfs.shape, dtype=args.real_type)
    cpsf = np.zeros(psf.shape, dtype=args.real_type)

    lpsf = np.arange(-R.nx_psf / 2, R.nx_psf / 2)
    mpsf = np.arange(-R.ny_psf / 2, R.ny_psf / 2)
    xx, yy = np.meshgrid(lpsf, mpsf, indexing='ij')

    cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)

    for v in range(args.nband):
        cpsf[v] = Gaussian2D(xx, yy, GaussPars[v], normalise=False)

    from pfb.utils.fits import add_beampars
    GaussPar = list(GaussPar[0])
    GaussPar[0] *= args.cell_size / 3600
    GaussPar[1] *= args.cell_size / 3600
    GaussPar = tuple(GaussPar)
    hdr_psf_mfs = add_beampars(hdr_psf_mfs, GaussPar)

    save_fits(args.outfile + '_cpsf_mfs.fits', cpsf_mfs, hdr_psf_mfs)
    save_fits(args.outfile + '_psf_mfs.fits', psf_mfs, hdr_psf_mfs)

    if args.beam is not None:
        bhdr = fits.getheader(args.beam)
        compare_headers(mhdr, bhdr)
        beam = load_fits(args.beam).squeeze()
        model = np.where(beam > args.pb_min, model/beam, 0.0)

    nband, nx, ny = model.shape
    guassparf = ()
    if nband > 1:
        for b in range(nband):
            guassparf += (rhdr['BMAJ'+str(b)], rhdr['BMIN'+str(b)],
                          rhdr['BPA'+str(b)])
    else:
        guassparf += (rhdr['BMAJ'], rhdr['BMIN'], rhdr['BPA'])

    # if args.convolve_residuals:

    cellx = np.abs(mhdr['CDELT1'])
    celly = np.abs(mhdr['CDELT2'])

    from pfb.utils.restoration import restore_image
