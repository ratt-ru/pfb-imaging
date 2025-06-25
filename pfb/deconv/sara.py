import numpy as np
from scipy.stats import expon
from pfb.opt.power_method import power_method
from pfb.opt.pcg import pcg
from pfb.opt.primal_dual import primal_dual
from pfb.operators.psi import DaskPSI
from pfb.operators.psf import PSF
from pfb.prox.prox_21 import prox_21
from pfb.utils.fits import save_fits
from pfb.utils.misc import Gaussian2D
import pyscilog
log = pyscilog.get_logger('SARA')


def resid_func(x, dirty, hessian, mask, beam, wsum):
    """
    Returns the unattenuated residual
    """
    residual = dirty - hessian(mask(beam(x)))/wsum
    residual_mfs = np.sum(residual, axis=0)
    residual = residual
    return residual, residual_mfs


def sara(psf, model, residual, mask=None, beam_image=None, hessian=None,
         wsum=1, adapt_sig21=True, hdr=None, hdr_mfs=None, outfile=None, cpsf=None,
         nthreads=1, sig_21=1e-6, sigma_frac=100, maxit=10, tol=1e-3,
         gamma=0.99,  psi_levels=2, psi_basis=None, alpha=None,
         pdtol=1e-6, pdmaxit=250, pdverbose=1, positivity=True,
         cgtol=1e-6, cgminit=25, cgmaxit=150, cgverbose=1,
         pmtol=1e-5, pmmaxit=50, pmverbose=1):

    if len(residual.shape) > 3:
        log.error_and_raise("Residual must have shape (nband, nx, ny)",
                            ValueError)

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
            log.error_and_raise("Beam has incorrect shape",
                                ValueError)

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
                raise
        except BaseException:
            log.error_and_raise("Mask has incorrect shape",
                                ValueError)

    # PSF operator
    psfo = PSF(psf, residual.shape, nthreads=nthreads)  #, backward_undersize=1.2)

    if cpsf is None:
        log.error_and_raise("Need to pass in cpsf", ValueError)
    else:
        cpsfo = PSF(cpsf, residual.shape, nthreads=nthreads)

    residual_mfs = np.sum(residual, axis=0)
    residual = mask(beam(residual))
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)

    # wavelet dictionary
    if psi_basis is None:
        psi = DaskPSI(imsize=residual.shape,
                      nlevels=psi_levels, nthreads=nthreads)
    else:
        if not isinstance(psi_basis, list):
            psi_basis = [psi_basis]
        psi = DaskPSI(imsize=residual.shape, nlevels=psi_levels,
                      nthreads=nthreads, bases=psi_basis)

    # set alpha's and sig21's
    # this assumes that the model has been initialised using NNLS
    alpha = np.zeros(psi.nbasis)
    sigmas = np.zeros(psi.nbasis)
    resid_comps = psi.hdot(residual/np.amax(residual.reshape(-1, nx*ny), axis=1)[:, None, None])
    l2_norm = np.linalg.norm(psi.hdot(cpsfo.convolve(model)), axis=1)
    for m in range(psi.nbasis):
        alpha[m] = np.std(resid_comps[m])
        _, sigmas[m] = expon.fit(l2_norm[m], floc=0.0)
        log.info("Basis %i, alpha %f, sigma %f"%(m, alpha[m], sigmas[m]))

    # l21 weights and dual
    weights21 = np.ones((psi.nbasis, psi.nmax), dtype=residual.dtype)
    for m in range(psi.nbasis):
        weights21[m] *= sigmas[m]/sig_21
    dual = np.zeros((psi.nbasis, nband, psi.nmax), dtype=residual.dtype)

    # use PSF to approximate Hessian if not passed in
    if hessian is None:
        hessian = psfo.convolve
        wsum = 1.0

    #  preconditioning operator
    if model.any():
        varmap = np.maximum(rms, sigma_frac * cpsfo.convolve(model))
    else:
        varmap = np.ones(model.shape) * sigma_frac * rms

    def hessf(x):
        # return mask(beam(hessian(mask(beam(x)))))/wsum + x / varmap
        return mask(beam(psfo.convolve(mask(beam(x))))) + x / varmap

    def hessb(x):
        return mask(beam(psfo.convolve(mask(beam(x))))) + x / varmap

    beta, betavec = power_method(hessb, residual.shape, tol=pmtol,
                                 maxit=pmmaxit, verbosity=pmverbose)

    if model.any():
        dirty = residual + hessian(mask(beam(model)))/wsum
    else:
        dirty = residual

    # deconvolve
    for i in range(0, maxit):
        x = pcg(hessf,
                mask(beam(residual)),
                np.zeros_like(residual),
                M=lambda x: x * varmap,
                tol=cgtol, maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)

        # update model
        modelp = model
        model = modelp + gamma * x

        model, dual = primal_dual(hessb, model, modelp, dual, sig_21, psi,
                                  weights21, beta, prox_21, tol=pdtol,
                                  maxit=pdmaxit, report_freq=50, mask=mask,
                                  verbosity=pdverbose, positivity=positivity)

        # get residual
        residual, residual_mfs = resid_func(model, dirty, hessian, mask, beam, wsum)
        model_mfs = np.mean(model, axis=0)
        x_mfs = np.mean(x, axis=0)


        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        # update variance map (positivity constraint optional)
        varmap = np.maximum(rms, sigma_frac * cpsfo.convolve(model))

        # update spectral norm
        beta, betavec = power_method(hessb, residual.shape, b0=betavec,
                                     tol=pmtol, maxit=pmmaxit,
                                     verbosity=pmverbose)

        log.info("Iter %i: peak residual = %f, rms = %f, eps = %f" % (
              i+1, rmax, rms, eps))

        # reweight
        l2_norm = np.linalg.norm(psi.hdot(model), axis=1)
        for m in range(psi.nbasis):
            if adapt_sig21:
                _, sigmas[m] = expon.fit(l2_norm[m], floc=0.0)
                log.info('basis %i, sigma %f'%sigmas[m])

            weights21[m] = alpha[m]/(alpha[m] + l2_norm[m]) * sigmas[m]/sig_21

        # save current iteration
        if outfile is not None:
            assert hdr is not None
            assert hdr_mfs is not None

            save_fits(outfile + str(i + 1) + '_model_mfs.fits',
                      model_mfs, hdr_mfs)

            save_fits(outfile + str(i + 1) + '_model.fits',
                      model, hdr)

            save_fits(outfile + str(i + 1) + '_update.fits',
                      x, hdr)

            save_fits(outfile + str(i + 1) + '_update_mfs.fits',
                      x_mfs, hdr)

            save_fits(outfile + str(i + 1) + '_residual_mfs.fits',
                      residual_mfs, hdr_mfs)

            save_fits(outfile + str(i + 1) + '_residual.fits',
                      residual*wsum, hdr)


        if eps < tol:
            log.info("Success, convergence after %i iterations" % (i+1))
            break

    return model
