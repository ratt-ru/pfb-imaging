import numpy as np
from functools import partial
from pfb.opt.fista import fista
from pfb.operators.psf import PSF
from pfb.utils.fits import save_fits
from pfb.opt.power_method import power_method
import pyscilog
log = pyscilog.get_logger('NNLS')


def resid_func(x, dirty, hessian, mask, beam, wsum):
    """
    Returns the unattenuated residual
    """
    residual = dirty - hessian(mask(beam(x)))/wsum
    residual_mfs = np.sum(residual, axis=0)
    residual = residual
    return residual, residual_mfs


def value_and_grad(x, dirty, psfo, mask, beam):
    convmod = mask(beam(psfo.convolve(mask(beam(x)))))
    attdirty = mask(beam(dirty))
    return np.vdot(x, convmod - 2*attdirty), 2*convmod - 2*attdirty

def prox(x):
    x[x<0] = 0.0
    return x

def nnls(psf, model, residual, mask=None, beam_image=None,
         hessian=None, wsum=None, gamma=0.95, nthreads=1,
         maxit=1, tol=1e-3,
         hdr=None, hdr_mfs=None, outfile=None,
         pmtol=1e-5, pmmaxit=50, pmverbose=1,
         ftol=1e-5, fmaxit=250, fverbose=3):

    if len(residual.shape) > 3:
        raise ValueError("Residual must have shape (nband, nx, ny)")

    nband, nx, ny = residual.shape

    if beam_image is None:
        def beam(x): return x
    else:
        try:
            assert beam.shape == (nband, nx, ny)
            def beam(x): return beam_image * x
        except BaseException:
            raise ValueError("Beam has incorrect shape")

    if mask is None:
        def mask(x): return x
    else:
        try:
            if mask.ndim == 2:
                assert mask.shape == (nx, ny)
                def mask(x): return mask[None] * x
            elif mask.ndim == 3:
                assert mask.shape == (1, nx, ny)
                def mask(x): return mask * x
            else:
                raise ValueError
        except BaseException:
            raise ValueError("Mask has incorrect shape")

    # PSF operator
    psfo = PSF(psf, residual.shape, nthreads=nthreads)

    residual_mfs = np.sum(residual, axis=0)
    residual = mask(beam(residual))
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)

    if hessian is None:
        hessian = psfo.convolve
        wsum = 1

    def hess(x):
        return mask(beam(psfo.convolve(mask(beam(x)))))

    beta, betavec = power_method(hess, residual.shape, tol=pmtol,
                                 maxit=pmmaxit, verbosity=pmverbose)

    if model.any():
        dirty = residual + hessian(mask(beam(model)))/wsum
    else:
        dirty = residual

    for i in range(maxit):
        fprime = partial(value_and_grad, dirty=residual,
                         psfo=psfo, mask=mask, beam=beam)

        x = fista(np.zeros_like(model), beta,
                  fprime, prox, tol=ftol, maxit=fmaxit,
                  verbosity=fverbose)

        modelp = model.copy()
        model += gamma * x

        residual, residual_mfs = resid_func(model, dirty, hessian, mask, beam, wsum)
        model_mfs = np.mean(model, axis=0)

        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        print("Iter %i: peak residual = %f, rms = %f, eps = %f" % (
              i+1, rmax, rms, eps), file=log)

        # save current iteration
        if outfile is not None:
            assert hdr is not None
            assert hdr_mfs is not None

            save_fits(outfile + str(i + 1) + '_NNLS_model_mfs.fits',
                        model_mfs, hdr_mfs)

            save_fits(outfile + str(i + 1) + '_NNLS_model.fits',
                        model, hdr)

            save_fits(outfile + str(i + 1) + '_NNLS_residual_mfs.fits',
                        residual_mfs, hdr_mfs)

            save_fits(outfile + str(i + 1) + '_NNLS_residual.fits',
                      residual*wsum, hdr)

        if eps < tol:
            print("Success, convergence after %i iterations" % (i+1),
                file=log)
            break

    return model