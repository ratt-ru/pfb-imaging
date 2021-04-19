import numpy as np
from pfb.opt.power_method import power_method
from pfb.opt.pcg import pcg
from pfb.opt.primal_dual import primal_dual
from pfb.operators.psi import DaskPSI
from pfb.operators.psf import PSF
from pfb.prox.prox_21 import prox_21
from pfb.utils.fits import save_fits
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
         wsum=1, adapt_sig21=False, hdr=None, hdr_mfs=None, outfile=None,
         nthreads=1, sig_21=1e-6, sigma_frac=100, maxit=10, tol=1e-3,
         gamma=0.99,  psi_levels=2, psi_basis=None, alpha=None,
         pdtol=1e-6, pdmaxit=250, pdverbose=1, positivity=True,
         cgtol=1e-6, cgminit=25, cgmaxit=150, cgverbose=1,
         pmtol=1e-5, pmmaxit=50, pmverbose=1):

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
    psfo = PSF(psf, residual.shape, nthreads=nthreads)  #, backward_undersize=1.2)

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

    # l21 weights and dual
    weights21 = np.ones((psi.nbasis, psi.nmax), dtype=residual.dtype)
    dual = np.zeros((psi.nbasis, nband, psi.nmax), dtype=residual.dtype)

    #  preconditioning operator
    def hessf(x):
        return mask(beam(hessian(mask(beam(x)))))/wsum + x / (sigma_frac*rmax)

    def hessb(x):
        return mask(beam(psfo.convolve(mask(beam(x))))) + x / (sigma_frac*rmax)

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
    if alpha is None:
        alpha = 0.00005
    for i in range(0, maxit):
        x = pcg(hessf,
                mask(beam(residual)),
                np.zeros_like(residual),
                M=lambda x: x * (sigma_frac * rmax),
                tol=cgtol, maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)

        # update model
        modelp = model
        model = modelp + gamma * x

        # reweighting
        l2_norm = np.linalg.norm(psi.hdot(model), axis=1)
        for m in range(psi.nbasis):
            weights21[m] *= alpha/(alpha + l2_norm[m])

        model, dual = primal_dual(hessb, model, modelp, dual, sig_21, psi,
                                  weights21, beta, prox_21, tol=pdtol,
                                  maxit=pdmaxit, report_freq=50, mask=mask,
                                  verbosity=pdverbose, positivity=positivity)

        # reset weights
        for m in range(psi.nbasis):
            weights21[m] = np.where(np.any(dual[m], axis=0), weights21[m], 1.0)

        # get residual
        residual, residual_mfs = resid_func(model, dirty, hessian, mask, beam, wsum)
        model_mfs = np.mean(model, axis=0)

        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        print("Iter %i: peak residual = %f, rms = %f, eps = %f" % (
              i+1, rmax, rms, eps), file=log)

        # rmax can change so we need to redo spectral norm
        beta, betavec = power_method(hessb, residual.shape, b0=betavec,
                                     tol=pmtol, maxit=pmmaxit,
                                     verbosity=pmverbose)

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

            save_fits(outfile + str(i + 1) + '_residual_mfs.fits',
                      residual_mfs, hdr_mfs)

            save_fits(outfile + str(i + 1) + '_residual.fits',
                      residual*wsum, hdr)


        if eps < tol:
            print("Success, convergence after %i iterations" % (i+1),
                  file=log)
            break

        if adapt_sig21:
            # sig_21 should be set to the std of the image noise
            from scipy.stats import skew, kurtosis
            alpha = rms
            tmp = residual_mfs
            z = tmp/alpha
            k = 0
            while (np.abs(skew(z.ravel(), nan_policy='omit')) > 0.05 or
                   np.abs(kurtosis(z.ravel(), fisher=True, nan_policy='omit')) > 0.5) and k < 10:
                # eliminate outliers
                tmp = np.where(np.abs(z) < 3, residual_mfs, np.nan)
                alpha = np.nanstd(tmp)
                z = tmp/alpha
                print(alpha, skew(z.ravel(), nan_policy='omit'), kurtosis(z.ravel(), fisher=True, nan_policy='omit'))
                k += 1

            sig_21 = alpha
            print("alpha set to %f"%(alpha), file=log)

    return model
