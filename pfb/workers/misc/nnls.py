# flake8: noqa
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('NNLS')

@cli.command()
@click.option('-d', '--dirty',
              help="Path to dirty.")
@click.option('-p', '--psf',
              help="Path to PSF")
@click.option('-x0', '--x0',
              help="Initial model")
@click.option('-mv', '--min-value', type=float, default=0.0,
              help="Minimum value below which to threshold")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nthreads', '--nthreads', type=int, default=0,
              help="Total number of threads to use per worker")
@click.option('-ftol', '--fista-tol', type=float, default=1e-4,
              help="Tolerance for FISTA")
@click.option('-fmaxit', '--fista-maxit', type=int, default=100,
              help="Maximum iterations for FISTA")
@click.option('-fverb', '--fista-verbose', type=int, default=1,
              help="Verbosity of FISTA (>1 for debugging)")
@click.option('-frf', '--fista-report-freq', type=int, default=25,
              help="Report freq for FISTA")
@click.option('-pmtol', '--pm-tol', type=float, default=1e-4,
              help="Tolerance for FISTA")
@click.option('-pmmaxit', '--pm-maxit', type=int, default=100,
              help="Maximum iterations for FISTA")
@click.option('-pmverb', '--pm-verbose', type=int, default=1,
              help="Verbosity of FISTA (>1 for debugging)")
@click.option('-pmrf', '--pm-report-freq', type=int, default=25,
              help="Report freq for FISTA")
# @click.option('-x0')
def nnls(**kw):
    '''
    Minor cycle implementing non-negative least squares
    '''
    opts = OmegaConf.create(kw)
    pyscilog.log_to_file(opts.output_filename + '.log')
    pyscilog.enable_memory_logging(level=3)

    print('Input Options:', file=log)
    for key in kw.keys():
        print('     %25s = %s' % (key, kw[key]), file=log)

    from pfb.utils.fits import load_fits
    from astropy.io import fits
    import numpy as np

    def resid_func(x, dirty, psfo):
        """
        Returns the unattenuated residual
        """
        residual = dirty - psfo.convolve(x)
        residual_mfs = np.sum(residual, axis=0)
        return residual, residual_mfs

    def value_and_grad(x, dirty, psfo):
        model_conv = psfo.convolve(x)
        return np.vdot(x, model_conv - 2*dirty), 2*(model_conv - dirty)

    def prox(x):
        x[x<opts.min_value] = 0.0
        return x

    dirty = load_fits(opts.dirty).squeeze()
    nband, nx, ny = dirty.shape
    hdr = fits.getheader(opts.dirty)

    psf = load_fits(opts.psf).squeeze()
    _, nx_psf, ny_psf = psf.shape
    hdr_psf = fits.getheader(opts.psf)

    wsums = np.amax(psf.reshape(-1, nx_psf*ny_psf), axis=1)
    wsum = np.sum(wsums)

    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

    assert (psf_mfs.max() - 1.0) < 1e-4

    dirty /= wsum
    dirty_mfs = np.sum(dirty, axis=0)

    from pfb.operators.psf import PSF
    psfo = PSF(psf, dirty.shape, nthreads=opts.nthreads)

    from pfb.opt.power_method import power_method

    beta, betavec = power_method(psfo.convolve, dirty.shape,
                                 tol=opts.pm_tol,
                                 maxit=opts.pm_maxit,
                                 verbosity=opts.pm_verbose,
                                 report_freq=opts.pm_report_freq)

    fprime = partial(value_and_grad, dirty=dirty, psfo=psfo)

    from pfb.opt.fista import fista

    if opts.x0 is None:
        x0 = np.zeros_like(dirty)
    else:
        x0 = load_fits(opts.x0, dtype=dirty.dtype).squeeze()

    model = fista(x0, beta, fprime, prox,
                  tol=opts.fista_tol,
                  maxit=opts.fista_maxit,
                  verbosity=opts.fista_verbose,
                  report_freq=opts.fista_report_freq)

    residual, residual_mfs = resid_func(model, dirty, psfo)


    from pfb.utils.fits import save_fits

    save_fits(opts.output_filename + '_model.fits', model, hdr)
    save_fits(opts.output_filename + '_residual.fits', residual, hdr)

