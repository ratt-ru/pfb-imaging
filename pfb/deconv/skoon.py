import numpy as np
from pfb.operators import PSF, Dirac
from pfb.opt import hogbom, pcg, primal_dual

def grad_func(x, dirty, psfo):
    return psfo.convolve(x) - dirty

def skoon(psf, model, residual, mask, nthreads=0, maxit=10, 
          gamma=0.1, peak_factor=0.85, threshold=0.1):

    nband, nx, ny = model.shape
    
    # PSF operator
    psfo = PSF(psf, nthreads)

    # init dirty
    if model.any():
        dirty = residual + psfo.convolve(model)
    else:
        dirty = residual

    # residual
    residual_mfs = np.sum(residual, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()


    # set up point sources
    phi = Dirac(nband, nx, ny, mask=mask)
    dual = np.zeros((nband, nx, ny), dtype=np.float64)
    weights_21 = np.where(phi.mask, 1, np.inf)

    #  preconditioning operator
    def hess(x):  
        return psfo.convolve(x) + rms*x

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