
import numpy as np
import dask.array as da
from ducc0.wgridder import ms2dirty, dirty2ms
from pfb.operators.psi import im2coef, coef2im


def _hessian_reg(alpha, beam, uvw, weight, freq,
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
    Tikhonov regularised Hessian of wavelet coeffs
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
