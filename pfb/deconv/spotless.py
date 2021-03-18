import numpy as np
from pfb.operators import PSF2, Dirac
from pfb.opt import pcg, primal_dual, power_method
import numexpr as ne
import pyscilog
log = pyscilog.get_logger('SPOTLESS')

def hogbom(ID, PSF, x, gamma=0.1, pf=0.1, maxit=5000):
    nx, ny = ID.shape
    IR = ID.copy()
    IRsearch = IR*IR
    pq = IRsearch.argmax()
    p = pq//ny
    q = pq - p*ny
    IRmax = np.sqrt(IRsearch[p, q])
    tol = pf*IRmax
    k = 0
    while IRmax > tol and k < maxit:
        xhat = IR[p, q]
        x[p, q] += gamma * xhat
        # IR -= gamma * xhat * PSF[nx-p:2*nx - p, ny-q:2*ny - q]
        # IRsearch = IR*IR
        tmp = PSF[nx-p:2*nx - p, ny-q:2*ny - q]
        ne.evaluate('IR - gamma * xhat * tmp', out=IR, casting='same_kind')
        ne.evaluate('IR*IR', out=IRsearch, casting='same_kind')
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmax = np.sqrt(IRsearch[p, q])
        k += 1
    return x, np.std(IR)

def resid_func(x, dirty, psfo, mask, beam):
    """
    Returns the unattenuated but masked residual_mfs useful for peak finding
    and the masked residual cube attenuated by the beam pattern which is the
    data source in the pcg forward step. Masked regions are set to zero. 
    """
    residual = dirty - psfo.convolve(x)
    residual_mfs = np.sum(residual, axis=0)
    residual = mask(beam(residual))
    return residual, mask(residual_mfs)

def spotless(psf, model, residual, mask=None, beam=None, nthreads=1, sig_21=1e-3, sigma_frac=100,
             maxit=10, tol=1e-4, peak_factor=0.01, threshold=0.0, positivity=True, gamma=0.9999, tidy=True,
             hbgamma=0.1, hbpf=0.1, hbmaxit=5000, hbverbose=1,  # Hogbom options
             pdtol=1e-4, pdmaxit=250, pdverbose=1,  # primal dual options
             cgtol=1e-4, cgminit=15, cgmaxit=150, cgverbose=1,  # conjugate gradient options
             pmtol=1e-4, pmmaxit=50, pmverbose=1):  # power method options
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
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)

    # set up point sources
    phi = Dirac(nband, nx, ny, mask=np.any(model, axis=0))
    dual = np.zeros((nband, nx, ny), dtype=np.float64)

    #  preconditioning operator
    def hess(x):  
        return phi.hdot(mask(beam(psfo.convolve(phi.dot(x))))) + x / (sigma_frac * rmax)

    if tidy:
        # spectral norm
        posthess = hess
        beta, betavec = power_method(hess, residual.shape, tol=pmtol, maxit=pmmaxit, verbosity=pmverbose)
    else:
        posthess = lambda x: x
        beta = 1.0
        betavec = 1.0

    # deconvolve
    threshold = np.maximum(peak_factor*rmax, threshold)
    for i in range(0, maxit):
        # find point source candidates
        modelu, rms = hogbom_mfs(residual_mfs, psf_mfs, gamma=hbgamma, pf=hbpf, maxit=hbmaxit)
        
        phi.update_locs(modelu)

        # solve for beta updates
        x0 = np.tile(modelu[None], (nband, 1, 1))
        x = pcg(hess, phi.hdot(residual), phi.hdot(x0), 
                M=lambda x: x * (sigma_frac * rmax), tol=cgtol,
                maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)

        modelp = model.copy()
        model += gamma * x

        weights_21 = np.where(phi.mask, 1, 1e10)  # 1e10 for effective infinity

        if tidy:
            beta, betavec = power_method(hess, model.shape, b0=betavec, tol=pmtol, maxit=pmmaxit, verbosity=pmverbose)
            model, dual = primal_dual(posthess, model, modelp, dual, sig_21, phi, weights_21, beta,
                                      tol=pdtol, maxit=pdmaxit, axis=0,
                                      positivity=positivity, report_freq=100, verbosity=pdverbose)
            
        else:
            model, dual = primal_dual(posthess, model, modelp, dual, sig_21, phi, weights_21, beta,
                                      tol=pdtol, maxit=pdmaxit, axis=0,
                                      positivity=positivity, report_freq=100, verbosity=pdverbose)

        # update Dirac dictionary (remove zero components)
        phi.trim_fat(model)
        residual, residual_mfs = resid_func(model, dirty, psfo, mask, beam)

        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)
        
        if rmax < threshold or eps < tol:
            print("Success, convergence after %i iterations"%(i+1), file=log)
            break
        else:
            print("At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i+1, rmax, rms, eps), file=log)
    
    return model, residual_mfs