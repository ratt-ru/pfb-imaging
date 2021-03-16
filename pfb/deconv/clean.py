import sys
import numpy as np
from pfb.operators import PSF
from pfb.opt import hogbom
import pyscilog
log = pyscilog.get_logger('CLEAN')

def grad_func(x, dirty, psfo):
    return psfo.convolve(x) - dirty

def clean(psf, model, residual, mask=None, beam=None, 
          nthreads=0, maxit=10, gamma=1.0, peak_factor=0.01, threshold=0.0,
          hbgamma=0.1, hbpf=0.1, hbmaxit=5000, hbverbose=1):  # Hogbom options

    if len(residual.shape) > 3:
        raise ValueError("Residual must have shape (nband, nx, ny)")
    
    nband, nx, ny = residual.shape

    if beam is None:
        beam = lambda x: x
    else:
        raise NotImplementedError("Beam not yet supported in clean minor cycle")

    if mask is None:
        mask = lambda x: x
    else:
        try:
            if mask.ndim == 2:
                assert mask.shape == (nx, ny)
                mask = lambda x: mask[None] * x
            elif mask.ndim == 3:
                assert mask.shape == (1, nx, ny)
                mask = lambda x: mask * x
            else:
                raise ValueError
        except:
            raise ValueError("Mask has incorrect shape")

    # PSF operator
    psfo = PSF(psf, nthreads, imsize=residual.shape, mask=mask, beam=beam)

    # init dirty
    residual = beam(mask(residual))
    if model.any():
        dirty = residual + psfo.convolve(model)
    else:
        dirty = residual

    # residual
    residual_mfs = np.sum(residual, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    rmax_orig = rmax
    threshold = np.maximum(peak_factor*rmax, threshold)
    # deconvolve
    for i in range(0, maxit):
        modelu = hogbom(residual, psf, gamma=hbgamma, pf=hbpf, maxit=hbmaxit, verbosity=hbverbose)

        model += gamma*modelu

        residual = -grad_func(model, dirty, psfo)

        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        if rmax < threshold*rmax_orig:
            print("Success, convergence after %i iterations. Peak of residual is %f, rms is %f"%(i+1, rmax, rms), file=log)
            break
        else:
            print("At iteration %i peak of residual is %f, rms is %f" % (i+1, rmax, rms), file=log)
    
    return model, residual_mfs