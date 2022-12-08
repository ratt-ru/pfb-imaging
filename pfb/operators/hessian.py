
import numpy as np
import dask
import dask.array as da
from daskms.optimisation import inlined_array
from ducc0.wgridder import ms2dirty, dirty2ms
from uuid import uuid4
from pfb.operators.psf import (psf_convolve_slice,
                               psf_convolve_cube)


def hessian_xds(x, xds, hessopts, wsum, sigmainv, mask,
                compute=True, use_beam=True):
    '''
    Vis space Hessian reduction over dataset.
    Hessian will be applied to x
    '''
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1),
                          name="x-" + uuid4().hex)

    if not isinstance(mask, da.Array):
        mask = da.from_array(mask, chunks=(-1, -1),
                             name="mask-" + uuid4().hex)

    assert mask.ndim == 2

    nband, nx, ny = x.shape

    # LB - what is the point of specifying name?
    convims = [da.zeros((nx, ny),
               chunks=(-1, -1), name="zeros-" + uuid4().hex)
               for _ in range(nband)]

    for ds in xds:
        wgt = ds.WEIGHT.data
        vis_mask = ds.MASK.data
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        b = ds.bandid
        if use_beam:
            beam = ds.BEAM.data * mask
        else:
            # TODO - separate implementation without
            # unnecessary beam application
            beam = mask

        convim = hessian(x[b], uvw, wgt, vis_mask, freq, beam, hessopts)

        # convim = inlined_array(convim, uvw)

        convims[b] += convim

    convim = da.stack(convims)/wsum

    if sigmainv:
        convim += x * sigmainv**2

    if compute:
        return convim.compute()
    else:
        return convim


def _hessian_impl(x, uvw, weight, vis_mask, freq, beam,
                  cell=None,
                  wstack=None,
                  epsilon=None,
                  double_accum=None,
                  nthreads=None):
    if not x.any():
        return np.zeros_like(x)
    nx, ny = x.shape
    mvis = dirty2ms(uvw=uvw,
                    freq=freq,
                    mask=vis_mask,
                    dirty=x if beam is None else x * beam,
                    pixsize_x=cell,
                    pixsize_y=cell,
                    epsilon=epsilon,
                    nthreads=nthreads,
                    do_wstacking=wstack)

    convim = ms2dirty(uvw=uvw,
                      freq=freq,
                      ms=mvis,
                      wgt=weight,
                      mask=vis_mask,
                      npix_x=nx,
                      npix_y=ny,
                      pixsize_x=cell,
                      pixsize_y=cell,
                      epsilon=epsilon,
                      nthreads=nthreads,
                      do_wstacking=wstack,
                      double_precision_accumulation=double_accum)

    if beam is not None:
        convim *= beam

    return convim


def _hessian(x, uvw, weight, vis_mask, freq, beam, hessopts):
    return _hessian_impl(x, uvw[0][0], weight[0][0], vis_mask[0][0], freq[0],
                         beam, **hessopts)

def hessian(x, uvw, weight, vis_mask, freq, beam, hessopts):
    if beam is None:
        bout = None
    else:
        bout = ('nx', 'ny')
    return da.blockwise(_hessian, ('nx', 'ny'),
                        x, ('nx', 'ny'),
                        uvw, ('row', 'three'),
                        weight, ('row', 'chan'),
                        vis_mask, ('row', 'chan'),
                        freq, ('chan',),
                        beam, bout,
                        hessopts, None,
                        dtype=x.dtype)


def hessian_psf_slice_dask(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    psfhat,
                    beam,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    sigmainv=1,
                    wsum=1):
    return da.blockwise(hessian_psf_slice,
                        xpad, 'xy',
                        xhat, 'xy',
                        xout, 'xy',
                        psfhat, 'xy',
                        beam, 'xy',
                        lastsize, None,
                        x, 'xy',
                        nthreads, None,
                        sigmainv, None,
                        wsum, None,
                        allign_arrays=False,
                        dtype=xpad.dtype)


def hessian_psf_slice(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    psfhat,
                    beam,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    sigmainv=1,
                    wsum=1):
    """
    Tikhonov regularised Hessian approx
    """

    if beam is not None:
        psf_convolve_slice(xpad, xhat, xout,
                           psfhat/wsum, lastsize, x*beam)
    else:
        psf_convolve_slice(xpad, xhat, xout,
                           psfhat/wsum, lastsize, x)

    if beam is not None:
        xout *= beam

    return xout + x * sigmainv


def hessian_psf_cube(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    beam,
                    psfhat,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    sigmainv=1):
    """
    Tikhonov regularised Hessian approx
    """

    if beam is not None:
        psf_convolve_cube(x*beam, xpad, xhat, xout, psfhat, lastsize)
    else:
        psf_convolve_cube(x, xpad, xhat, xout, psfhat, lastsize)

    if beam is not None:
        xout *= beam

    return xout + x * sigmainv
