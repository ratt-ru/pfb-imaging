
import numpy as np
import dask.array as da
from ducc0.wgridder import ms2dirty, dirty2ms
from ducc0.fft import r2c, c2r, c2c, good_size
from pfb.operators.psi import im2coef, coef2im
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def hessian_xds(alpha, xdss,
                cell=None,
                wstack=None,
                epsilon=None,
                double_accum=None,
                nthreads=None,
                sigmainv=None,
                wsum=None,
                pmask=None,
                padding=None,
                bases=None,
                iy=None,
                sy=None,
                ntot=None,
                nmax=None,
                nlevels=None,
                nx=None,
                ny=None):

    hesses = []

    for xds in xdss:
        wgt = xds.WEIGHT.data
        uvw = xds.UVW.data
        freq = xds.FREQ.data
        beam = xds.BEAM.data

        hesses.append(hessian_reg(alpha, beam, uvw, weight, freq))

    return da.stack(hesses).sum(axis=0) + alpha * sigmainv**2


def _hessian_reg_wgt(alpha, beam, uvw, weight, freq,
                     cell=None,
                     wstack=None,
                     epsilon=None,
                     double_accum=None,
                     nthreads=None,
                     sigmainv=None,
                     wsum=None,
                     pmask=None,
                     padding=None,
                     bases=None,
                     iy=None,
                     sy=None,
                     ntot=None,
                     nmax=None,
                     nlevels=None,
                     nx=None,
                     ny=None):
    """
    Tikhonov regularised Hessian of coeffs
    """

    x = coef2im(alpha, pmask, bases, padding, iy, sy, nx, ny)

    mvis = dirty2ms(uvw=uvw,
                    freq=freq,
                    dirty=beam * x,
                    wgt=None,
                    pixsize_x=cell,
                    pixsize_y=cell,
                    epsilon=epsilon,
                    nthreads=nthreads,
                    do_wstacking=wstack)

    im = ms2dirty(uvw=uvw,
                  freq=freq,
                  ms=mvis,
                  wgt=weight,
                  npix_x=nx,
                  npix_y=ny,
                  pixsize_x=cell,
                  pixsize_y=cell,
                  epsilon=epsilon,
                  nthreads=nthreads,
                  do_wstacking=wstack,
                  double_precision_accumulation=double_accum)/wsum

    alpha_rec = im2coef(beam*im, pmask, bases, ntot, nmax, nlevels)

    return alpha_rec + alpha * sigmainv**2


def _hessian_reg_psf(alpha, beam, psfhat,
                     nthreads=None,
                     sigmainv=None,
                     pmask=None,
                     padding_psf=None,
                     unpad_x=None,
                     unpad_y=None,
                     lastsize=None,
                     padding=None,
                     bases=None,
                     iy=None,
                     sy=None,
                     ntot=None,
                     nmax=None,
                     nlevels=None,
                     nx=None,
                     ny=None):
    """
    Tikhonov regularised Hessian of coeffs
    """

    x = coef2im(alpha, pmask, bases, padding, iy, sy, nx, ny)

    xhat = iFs(np.pad(beam*x, padding_psf, mode='constant'), axes=(0, 1))
    xhat = r2c(xhat, axes=(0, 1), nthreads=nthreads,
                forward=True, inorm=0)
    xhat = c2r(xhat * psfhat, axes=(0, 1), forward=False,
                lastsize=lastsize, inorm=2, nthreads=nthreads)
    im = Fs(xhat, axes=(0, 1))[unpad_x, unpad_y]

    alpha_rec = im2coef(beam*im, pmask, bases, ntot, nmax, nlevels)

    return alpha_rec + alpha * sigmainv**2
