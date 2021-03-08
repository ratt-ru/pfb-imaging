import numpy as np
from pfb.operators import PSF2, Dirac
from pfb.opt import pcg, primal_dual, power_method
from pfb.deconv import hogbom_mfs

def resid_func(x, dirty, psfo, mask, beam):
    residual = dirty - psfo.convolve(x)
    residual_mfs = np.sum(residual, axis=0)
    residual = mask(beam(residual))
    return residual, residual_mfs

def spotless(psf, model, residual, mask=None, beam=None, nthreads=1, 
             maxit=10, tol=1e-4, threshold=0.01, positivity=True, gamma=0.99,
             hbgamma=0.1, hbpf=0.1, hbmaxit=1000, hbverbose=1,  # Hogbom options
             pdtol=1e-6, pdmaxit=250, pdverbose=1,  # primal dual options
             cgtol=1e-6, cgminit=25, cgmaxit=150, cgverbose=1,  # conjugate gradient options
             pmtol=1e-5, pmmaxit=50, pmverbose=1):  # power method options
    """
    Modified clean algorithm:

    psf - PSF image i.e. R.H W where W contains the weights. Shape must be >= residual.shape
    model - current intrinsic model
    residual - apparent residual image i.e. R.H W (V - R A x)
    """


    if len(residual.shape) > 3:
        raise ValueError("Residual must have shape (nband, nx, ny)")
    
    nband, nx, ny = residual.shape

    if beam is None:
        beam = lambda x: x
    else:
        try:
            assert beam.shape == (nband, nx, ny)
            beam = lambda x: beam * x
        except:
            raise ValueError("Beam has incorrect shape")

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
    psf_mfs = np.sum(psf, axis=0)
    psfo = PSF2(psf, nthreads=nthreads, imsize=residual.shape, mask=mask, beam=beam)
    if model.any():
        dirty = residual + psfo.convolve(model)
    else:
        dirty = residual
    residual_mfs = np.sum(residual, axis=0)
    residual = mask(beam(residual))

    # set up point sources
    phi = Dirac(nband, nx, ny, mask=np.any(model, axis=0))
    dual = np.zeros((nband, nx, ny), dtype=np.float64)

    #  preconditioning operator
    def hess(x):  
        return phi.hdot(mask(beam(psfo.convolve(phi.dot(x))))) + 1e-6*x

    # deconvolve
    for i in range(0, maxit):
        # find point source candidates
        print('     hogbom')
        modelu = hogbom_mfs(residual_mfs, psf_mfs, gamma=hbgamma, pf=hbpf, maxit=hbmaxit, verbosity=hbverbose)
        
        print('     pm')
        phi.update_locs(modelu)
        beta, betavec = power_method(hess, model.shape, tol=pmtol, maxit=pmmaxit, verbosity=pmverbose)

        print('     pcg')
        # solve for beta updates
        x = pcg(hess, phi.hdot(residual), phi.hdot(np.tile(modelu[None], (nband, 1, 1))), 
                M=lambda x: x / 1e-6, tol=cgtol,
                maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)

        modelp = model.copy()
        model += gamma * x

        print('     pd')
        weights_21 = np.where(phi.mask, 1, 1e10)  # 1e10 for effective infinity
        model, dual = primal_dual(hess, model, modelp, dual, 1e-6, phi, weights_21, beta,
                                  tol=pdtol, maxit=pdmaxit, axis=0,
                                  positivity=positivity, report_freq=100, verbosity=pdverbose)

        # update Dirac dictionary (remove zero components)
        phi.trim_fat(model)
        print('     resid')
        residual, residual_mfs = resid_func(model, dirty, psfo, mask, beam)

        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)
        
        if rmax < threshold or eps < tol:
            print("     Spotless - Success, convergence after %i iterations"%(i+1))
            break
        else:
            print("     Spotless - At iteration %i peak of residual is %f, rms is %f" % (i+1, rmax, rms))
    
    return model, residual_mfs