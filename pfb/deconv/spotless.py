import numpy as np
import scipy
from pfb.operators.psf import PSF
from pfb.operators.dirac import Dirac
from pfb.opt.primal_dual import primal_dual
from pfb.opt.pcg import pcg
from pfb.opt.power_method import power_method
from pfb.opt.hogbom import hogbom
from pfb.prox.prox_21m import prox_21m
from pfb.utils.fits import save_fits
from skimage.filters import threshold_mean
import pyscilog
log = pyscilog.get_logger('SPOTLESS')

def make_noise_map(restored_image, boxsize):
    # Modified version of Cyril's magic minimum filter
    # Plundered from the depths of
    # https://github.com/cyriltasse/DDFacet/blob/master/SkyModel/MakeMask.py
    box = (boxsize, boxsize)
    n = boxsize**2.0
    x = np.linspace(-10, 10, 1000)
    f = 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))
    F = 1.0 - (1.0 - f)**n
    ratio = np.abs(np.interp(0.5, F, x))
    noise = -scipy.ndimage.filters.minimum_filter(restored_image, box) / ratio
    negative_mask = noise < 0.0
    noise[negative_mask] = 1.0e-10
    median_noise = np.median(noise)
    median_mask = noise < median_noise
    noise[median_mask] = median_noise
    return noise


def resid_func(x, dirty, hessian, mask, beam, wsum):
    """
    Returns the unattenuated residual
    """
    residual = dirty - hessian(mask(beam(x)))/wsum
    residual_mfs = np.sum(residual, axis=0)
    residual = residual
    return residual, residual_mfs


def spotless(psf, model, residual, mask=None, beam_image=None, hessian=None,
             wsum=1, adapt_sig21=False, cpsf=None, hdr=None, hdr_mfs=None, outfile=None,
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
    psfo = PSF(psf, residual.shape, nthreads=nthreads, backward_undersize=1.2)

    # set up point sources
    phi = Dirac(nband, nx, ny, mask=np.any(model, axis=0))
    dual = np.zeros((nband, nx, ny), dtype=np.float64)

    # clean beam
    if cpsf is not None:
        try:
            assert cpsf.shape == (1,) + psf.shape[1::]
        except Exception as e:
            cpsf = cpsf[None, :, :]
        cpsfo = PSF(cpsf, residual.shape, nthreads=nthreads)

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
    alpha = sig_21
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
                              alpha/(alpha + np.abs(np.mean(modelp, axis=0))),
                              1e10)  # 1e10 for effective infinity
        beta, betavec = power_method(hessb, model.shape, b0=betavec,
                                     tol=pmtol, maxit=pmmaxit,
                                     verbosity=pmverbose)

        model, dual = primal_dual(hessb, model, modelp, dual, sig_21,
                                  phi, weights_21, beta, prox_21m,
                                  tol=pdtol, maxit=pdmaxit, axis=0,
                                  positivity=positivity, report_freq=50,
                                  verbosity=pdverbose)

        # update Dirac dictionary (remove zero components)
        phi.trim_fat(model)
        residual, residual_mfs = resid_func(model, dirty, hessian, mask, beam, wsum)

        model_mfs = np.mean(model, axis=0)

        # check stopping criteria
        rmax = np.abs(mask(residual_mfs)).max()
        rms = np.std(mask(residual_mfs))
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)


        print("Iter %i: peak residual = %f, rms = %f, eps = %f" % (
                i+1, rmax, rms, eps), file=log)


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


        if rmax < threshold or eps < tol:
            print("Success, convergence after %i iterations", file=log)
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
