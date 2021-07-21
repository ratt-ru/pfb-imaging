
import numpy as np
import dask.array as da
from ducc0.wgridder import ms2dirty, dirty2ms


# Tikhonov regularised Hessian
def _hessian_reg(x, beam, uvw, weight, freq, cell, wstack, epsilon,
                 double_accum, nthreads, sigmainvsq, wsum):
    """
    beam * x is equivalent to mask(beam(x))
    """

    nx, ny = x.shape
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
                  double_precision_accumulation=double_accum)

    return beam * im / wsum + x * sigmainvsq