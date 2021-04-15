import numpy as np
from pfb.operators.psf import PSF
from pfb.operators.dirac import Dirac
from pfb.opt.primal_dual import primal_dual
from pfb.opt.pcg import pcg
from pfb.opt.power_method import power_method
from pfb.opt.hogbom import hogbom
from pfb.prox.prox_21m import prox_21m
from skimage.filters import threshold_mean
import pyscilog
log = pyscilog.get_logger('SPOTLESS')


def resid_func(x, dirty, hessian, mask, beam, wsum):
    """
    Returns the unattenuated residual
    """
    residual = dirty - hessian(mask(beam(x)))/wsum
    residual_mfs = np.sum(residual, axis=0)
    residual = residual
    return residual, residual_mfs


def spotless(psf, model, residual, mask=None, beam_image=None, hessian=None,
             wsum=1, adapt_sig21=False,
             nthreads=1, sig_21=1e-3, sigma_frac=100, maxit=10, tol=1e-4,
             peak_factor=0.01, threshold=0.0, positivity=True, gamma=0.9999,
             hbgamma=0.1, hbpf=0.1, hbmaxit=5000, hbverbose=1,
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

    if beam_image is None:
        def beam(x): return x
        def beaminv(x): return x
    else:
        try:
            assert beam.shape == (nband, nx, ny)
            def beam(x): return beam_image * x
            def beaminv(x): return np.where(beam_image > 0.01,  x / beam_image, x)
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

    # set up point sources
    phi = Dirac(nband, nx, ny, mask=np.any(model, axis=0))
    dual = np.zeros((nband, nx, ny), dtype=np.float64)

    residual_mfs = np.sum(residual, axis=0)
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)

    #  preconditioning operator
    def hessb(x):
        return phi.hdot(mask(beam(psfo.convolveb(mask(beam(phi.dot(x))))))) +\
            x / (sigma_frac * rmax)

    def hessf(x):
        return phi.hdot(mask(beam(psfo.convolve(mask(beam(phi.dot(x))))))) +\
                    x / (sigma_frac * rmax)

    beta, betavec = power_method(hessb, residual.shape, tol=pmtol,
                                 maxit=pmmaxit, verbosity=pmverbose)

    if hessian is None:
        hessian = psf.convolve
        wsum = 1.0

    if model.any():
        dirty = residual + hessian(mask(beam(model)))/wsum
    else:
        dirty = residual

    # deconvolve
    threshold = np.maximum(peak_factor * rmax, threshold)
    for i in range(0, maxit):
        # find point source candidates
        modelu = hogbom(mask(residual), psf, gamma=hbgamma,
                        pf=hbpf, maxit=hbmaxit, verbosity=hbverbose)

        phi.update_locs(modelu)

        # solve for beta updates
        x = pcg(hessf,
                phi.hdot(mask(beam(residual))),
                phi.hdot(beaminv(modelu)),
                M=lambda x: x * (sigma_frac * rmax), tol=cgtol,
                maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)

        modelp = model.copy()
        model += gamma * x

        weights_21 = np.where(phi.mask,
                              sig_21/(sig_21 + np.mean(modelp, axis=0)),
                              1e10)  # 1e10 for effective infinity
        beta, betavec = power_method(hessb, model.shape, b0=betavec,
                                     tol=pmtol, maxit=pmmaxit,
                                     verbosity=pmverbose)

        model, dual = primal_dual(hessb, model, modelp, dual, sig_21,
                                  phi, weights_21, beta, prox_21m,
                                  tol=pdtol, maxit=pdmaxit, axis=0,
                                  positivity=positivity, report_freq=100,
                                  verbosity=pdverbose)

        # update Dirac dictionary (remove zero components)
        phi.trim_fat(model)
        residual, residual_mfs = resid_func(model, dirty, hessian, mask, beam, wsum)

        # check stopping criteria
        rmax = np.abs(mask(residual_mfs)).max()
        rms = np.std(mask(residual_mfs))
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        if rmax < threshold or eps < tol:
            print("Success, convergence after %i iterations, eps is %f" %
                  ((i + 1), eps), file=log)
            break
        else:
            print("At iteration %i peak of residual is %f, rms is %f, current"
                  " eps is %f" % (i + 1, rmax, rms, eps), file=log)

        if adapt_sig21:
            mean_th = threshold_mean(residual_mfs)
            sig_21 = np.minimum(rms, mean_th)

            print("Threshold set to ", sig_21, rms, mean_th)


            quit()

    return model, residual_mfs
