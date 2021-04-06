import numpy as np
from pfb.operators import PSF, Dirac
from pfb.opt import pcg, primal_dual, power_method, hogbom
from pfb.prox import prox_21m
import numexpr as ne
import pyscilog
log = pyscilog.get_logger('SPOTLESS')

def resid_func(x, dirty, psfo, mask, beam):
    """
    Returns the unattenuated residual
    """
    residual = dirty - psfo.convolve(mask(beam(x)))
    residual_mfs = np.sum(residual, axis=0)
    residual = residual
    return residual, residual_mfs


def spotless(psf, model, residual, mask=None, beam=None,
             nthreads=1, sig_21=1e-3, sigma_frac=100, maxit=10, tol=1e-4,
             peak_factor=0.01, threshold=0.0, positivity=True, gamma=0.9999,
             tidy=True, hbgamma=0.1, hbpf=0.1, hbmaxit=5000, hbverbose=1,
             pdtol=1e-4, pdmaxit=250, pdverbose=1,  # primal dual options
             cgtol=1e-4, cgminit=15, cgmaxit=150, cgverbose=1,  # pcg options
             pmtol=1e-4, pmmaxit=50, pmverbose=1):  # power method options
    """
    Modified clean algorithm:

    psf      - PSF image i.e. R.H W where W contains the weights.
               Shape must be >= residual.shape
    model    - current intrinsic model
    residual - apparent residual image i.e. R.H W (V - R A x)

    Note that peak finding happens in apparent residual because that
    is where it is easiest to accommodate convolution by the PSF.
    However, the beam and the mask have to be applied to the residual
    before we solve for the pre-conditioned updates.

    """

    if len(residual.shape) > 3:
        raise ValueError("Residual must have shape (nband, nx, ny)")

    nband, nx, ny = residual.shape

    if beam is None:
        def beam(x): return x
    else:
        try:
            assert beam.shape == (nband, nx, ny)
            def beam(x): return beam * x
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
    psf_mfs = np.sum(psf, axis=0)
    psfo = PSF(psf, nthreads=nthreads, imsize=residual.shape)
    if model.any():
        dirty = residual + psfo.convolve(mask(beam(model)))
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
        return phi.hdot(mask(beam(psfo.convolve(mask(beam(phi.dot(x))))))) +\
            x / (sigma_frac * rmax)

    # # test psf undersize for backward step
    # _, nx_psfo, ny_psfo = psf.shape
    # nx_psff = int(1.2*nx)
    # if nx_psff%2:
    #     nx_psff += 1

    # ny_psff = int(1.2*ny)
    # if ny_psff%2:
    #     ny_psff += 1

    # nx_trim = (nx_psfo - nx_psff)//2
    # ny_trim = (ny_psfo - ny_psff)//2
    # psf2 = psf[:, nx_trim:-nx_trim, ny_trim:-ny_trim]

    # psfo2 = PSF(psf2, nthreads=nthreads, imsize=residual.shape)

    # def posthess(x):
    #     return phi.hdot(mask(beam(psfo2.convolve(mask(beam(phi.dot(x))))))) +\
    #         x / (sigma_frac * rmax)

    if tidy:
        # spectral norm
        posthess = hess
        beta, betavec = power_method(posthess, residual.shape, tol=pmtol,
                                     maxit=pmmaxit, verbosity=pmverbose)
    else:
        def posthess(x): return x
        beta = 1.0
        betavec = 1.0

    # deconvolve
    threshold = np.maximum(peak_factor * rmax, threshold)
    for i in range(0, maxit):
        # find point source candidates
        modelu = hogbom(mask(residual), psf, gamma=hbgamma,
                        pf=hbpf, maxit=hbmaxit, verbosity=hbverbose)

        phi.update_locs(modelu)

        # solve for beta updates
        x = pcg(hess, phi.hdot(mask(beam(residual))), phi.hdot(modelu),
                M=lambda x: x * (sigma_frac * rmax), tol=cgtol,
                maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)

        modelp = model.copy()
        model += gamma * x

        weights_21 = np.where(phi.mask, 1, 1e10)  # 1e10 for effective infinity

        if tidy:
            beta, betavec = power_method(posthess, model.shape, b0=betavec,
                                         tol=pmtol, maxit=pmmaxit,
                                         verbosity=pmverbose)
            model, dual = primal_dual(posthess, model, modelp, dual, sig_21,
                                      phi, weights_21, beta, prox_21m,
                                      tol=pdtol, maxit=pdmaxit, axis=0,
                                      positivity=positivity, report_freq=100,
                                      verbosity=pdverbose)

        else:
            model, dual = primal_dual(posthess, model, modelp, dual, sig_21,
                                      phi, weights_21, beta, prox_21m,
                                      tol=pdtol, maxit=pdmaxit, axis=0,
                                      positivity=positivity, report_freq=100,
                                      verbosity=pdverbose)

        # update Dirac dictionary (remove zero components)
        phi.trim_fat(model)
        residual, residual_mfs = resid_func(model, dirty, psfo, mask, beam)

        # check stopping criteria
        rmax = np.abs(mask(residual_mfs)).max()
        rms = np.std(mask(residual_mfs))
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        if rmax < threshold or eps < tol:
            print("Success, convergence after %i iterations" %
                  (i + 1), file=log)
            break
        else:
            print("At iteration %i peak of residual is %f, rms is %f, current"
                  " eps is %f" % (i + 1, rmax, rms, eps), file=log)

    return model, residual_mfs
