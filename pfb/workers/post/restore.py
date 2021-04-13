'''
Utility to create restored images.
Can also be used to convolve images to a common resolution
and/or perform a primary beam correction.
'''

from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
log = pyscilog.get_logger('RESTORE')


@cli.command()
@click.option('-model', '--model', required=True,
              help="Path to model image cube")
@click.option('-resid', '--residual', required=False,
              help="Path to residual image cube")
@click.option('-beam', '--beam', required=False,
              help="Path to power beam image cube")
@click.option('-o', '--output-filename',
              help="Basename (next to input model if not provided).")
@click.option('-nthreads', '--nthreads', type=int, default=1,
              show_default=True, help='Number of threads to use')
@click.option('-cr', '--convolve-residuals', is_flag=True,
              help='Whether to convolve the residuals to a common resolution')
@click.option('-pf', '--padding-frac', type=float,
              default=0.5, show_default=True,
              help="Padding fraction for FFTs (half on either side)")
@click.option('-pb-min', '--pb-min', type=float, default=0.1,
              help="Set image to zero where pb falls below this value")
def restore(**kw):
    args = OmegaConf.create(kw)

    # use all threads if nthreads set to zero
    if not args.nthreads:
        import multiprocessing
        args.nthreads = multiprocessing.cpu_count()

    from pfb import set_threads
    set_threads(args.nthreads)

    if args.convolve_residuals and args.residual is None:
        raise ValueError("No residuals provided to convolve")

    # save next to model if no outfile is provided
    if args.output_filename is None:
        # strip .fits from model filename
        tmp = args.model[::-1]
        idx = tmp.find('.')
        outfile = args.model[0:-idx]
    else:
        outfile = args.output_filename

    from astropy.io import fits
    mhdr = fits.getheader(args.model)

    from pfb.utils.fits import load_fits
    model = load_fits(args.model).squeeze()  # drop Stokes axis

    if args.residual is not None:
        # check images compatible
        rhdr = fits.getheader(args.residual)

        from pfb.utils.fits import compare_headers
        compare_headers(mhdr, rhdr)
        residual = load_fits(args.residual).squeeze()

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
