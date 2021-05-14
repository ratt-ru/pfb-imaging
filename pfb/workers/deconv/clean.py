# flake8: noqa
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('CLEAN')

@cli.command()
@click.option('-d', '--dirty',
              help="Path to dirty.")
@click.option('-p', '--psf',
              help="Path to PSF")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nthreads', '--nthreads', type=int, default=0,
              help="Total number of threads to use per worker")
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
def clean(**kw):
    '''
    Minor cycle implementing clean
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

    print("Loading dirty", file=log)
    dirty = load_fits(args.dirty).squeeze()
    nband, nx, ny = dirty.shape
    hdr = fits.getheader(args.dirty)

    print("Loading psf", file=log)
    psf = load_fits(args.psf).squeeze()
    _, nx_psf, ny_psf = psf.shape
    hdr_psf = fits.getheader(args.psf)

    wsums = np.amax(psf.reshape(-1, nx_psf*ny_psf), axis=1)
    wsum = np.sum(wsums)

    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

    assert (psf_mfs.max() - 1.0) < 1e-4

    dirty /= wsum
    dirty_mfs = np.sum(dirty, axis=0)

    from pfb.operators.psf import PSF
    psfo = PSF(psf, dirty.shape, nthreads=args.nthreads)

    from pfb.opt.hogbom import hogbom

    print("Running Hogbom", file=log)
    model = hogbom(dirty, psf,
                   gamma=args.hb_gamma,
                   pf=args.hb_peak_factor,
                   maxit=args.hb_maxit,
                   verbosity=args.hb_verbose,
                   report_freq=args.hb_report_freq)

    print("Getting residual", file=log)
    residual, residual_mfs = resid_func(model, dirty, psfo)


    from pfb.utils.fits import save_fits

    print("Saving results", file=log)
    save_fits(args.output_filename + '_model.fits', model, hdr)
    save_fits(args.output_filename + '_residual.fits', residual, hdr)

    print("All done here.", file=log)