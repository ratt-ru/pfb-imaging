import numpy as np
from pfb.operators.psf import PSF
from pfb.opt.hogbom import hogbom
import pyscilog
log = pyscilog.get_logger('CLEAN')


def resid_func(x, dirty, psfo):
    residual = dirty - psfo.convolve(x)
    return residual


def clean(psf, model, residual, mask=None, beam=None,
          nthreads=0, maxit=10, gamma=1.0, peak_factor=0.01, threshold=0.0,
          hbgamma=0.1, hbpf=0.1, hbmaxit=5000, hbverbose=1):  # Hogbom options

    if len(residual.shape) > 3:
        raise ValueError("Residual must have shape (nband, nx, ny)")

    nband, nx, ny = residual.shape

    if beam is not None:
        raise NotImplementedError("Beam not supported in clean minor cycle")

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
    psfo = PSF(psf, nthreads, imsize=residual.shape)

    # init dirty
    if model.any():
        dirty = residual + psfo.convolve(model)
    else:
        dirty = residual

    # residual
    residual_mfs = np.sum(residual, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    rmax_orig = rmax
    threshold = np.maximum(peak_factor * rmax, threshold)
    # deconvolve
    for i in range(0, maxit):
        modelu = hogbom(mask(residual), psf,
                        gamma=hbgamma, pf=hbpf,
                        maxit=hbmaxit, verbosity=hbverbose)

        model += gamma * modelu

        residual = resid_func(model, dirty, psfo)

        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        if rmax < threshold * rmax_orig:
            print("Success, convergence after %i iterations."
                  "Peak of residual is %f, rms is %f" % (i + 1, rmax, rms),
                  file=log)
            break
        else:
            print("At iteration %i peak of residual is %f, rms is %f" %
                  (i + 1, rmax, rms), file=log)

    return model, residual_mfs
