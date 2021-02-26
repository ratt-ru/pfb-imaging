import numpy as np
from pfb.operators import PSF
from pfb.opt import hogbom

def grad_func(x, dirty, psfo):
    return psfo.convolve(x) - dirty

def clean(psf, model, residual, mask=None, beam=None, nthreads=0, maxit=10, 
          gamma=0.1, peak_factor=0.85, threshold=0.1):

    nband, nx_psf, ny_psf = psf.shape
    
    if beam is None:
        beam = lambda x: x
    else:
        raise NotImplementedError("Beam not yet supported in clean minor cycle")

    if mask is None:
        mask = lambda x: x

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

    # deconvolve
    for i in range(0, maxit):
        modelu = hogbom(residual, psf, gamma=gamma, pf=peak_factor)

        model += modelu

        residual = -grad_func(model, dirty, psfo)

        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        if rmax < threshold:
            print("     CLEAN - Success, convergence after %i iterations. Peak of residual is %f, rms is %f"%(i+1, rmax, rms))
            break
        else:
            print("     CLEAN - At iteration %i peak of residual is %f, rms is %f" % (i+1, rmax, rms))
    
    return model, residual_mfs