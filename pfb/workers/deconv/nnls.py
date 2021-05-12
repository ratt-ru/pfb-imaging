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
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nthreads', '--nthreads', type=int, default=0,
              help="Total number of threads to use per worker")
def nnls(**kw):
    '''
    Minor cycle implementing non-negative least squares
    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
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
        x[x<1e-4] = 0.0
        return x

    dirty = load_fits(args.dirty).squeeze()
    nband, nx, ny = dirty.shape
    hdr = fits.getheader(args.dirty)

    psf = load_fits(args.psf).squeeze()
    _, nx_psf, ny_psf = psf.shape
    hdr_psf = fits.getheader(args.psf)

    wsums = np.amax(psf.reshape(-1, nx_psf*ny_psf), axis=1)
    wsum = np.sum(wsums)

    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

    dirty /= wsum
    dirty_mfs = np.sum(dirty, axis=0)

    from pfb.operators.psf import PSF
    psfo = PSF(psf, dirty.shape, nthreads=args.nthreads)

    from pfb.opt.power_method import power_method

    beta, betavec = power_method(psfo.convolve, dirty.shape)

    fprime = partial(value_and_grad, dirty=dirty, psfo=psfo)

    from pfb.opt.fista import fista

    model = fista(np.zeros_like(dirty), beta, fprime, prox)

    residual, residual_mfs = resid_func(model, dirty, psfo)


    from pfb.utils.fits import save_fits

    save_fits(args.output_filename + '_model.fits', model, hdr)
    save_fits(args.output_filename + '_residual.fits', residual, hdr)

