import numpy as np
import dask
import dask.array as da
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from ducc0.fft import r2c, c2r
from ducc0.misc import make_noncritical
from uuid import uuid4
from pfb.operators.psf import (psf_convolve_slice,
                               psf_convolve_cube)


def _hessian_impl(x,
                  uvw=None,
                  weight=None,
                  vis_mask=None,
                  freq=None,
                  beam=None,
                  x0=0.0,
                  y0=0.0,
                  flip_u=False,
                  flip_v=True,
                  flip_w=False,
                  cell=None,
                  do_wgridding=None,
                  epsilon=None,
                  double_accum=None,
                  nthreads=None,
                  sigmainvsq=None,
                  wsum=1.0):
    '''
    Apply vis space Hessian approximation on a slice of an image.

    Important!
    x0, y0, flip_u, flip_v and flip_w must be consistent with the
    conventions defined in pfb.operators.gridder.wgridder_conventions

    These are inputs here to allow for testing but should generally be taken
    from the attrs of the datasets produced by
    pfb.operators.gridder.image_data_products
    '''
    if not x.any():
        return np.zeros_like(x)
    nx, ny = x.shape
    mvis = dirty2vis(uvw=uvw,
                    freq=freq,
                    mask=vis_mask,
                    dirty=x if beam is None else x * beam,
                    pixsize_x=cell,
                    pixsize_y=cell,
                    center_x=x0,
                    center_y=y0,
                    flip_u=flip_u,
                    flip_v=flip_v,
                    flip_w=flip_w,
                    epsilon=epsilon,
                    nthreads=nthreads,
                    do_wgridding=do_wgridding,
                    divide_by_n=False)

    convim = vis2dirty(
                    uvw=uvw,
                    freq=freq,
                    vis=mvis,
                    wgt=weight,
                    mask=vis_mask,
                    npix_x=nx,
                    npix_y=ny,
                    pixsize_x=cell,
                    pixsize_y=cell,
                    center_x=x0,
                    center_y=y0,
                    flip_u=flip_u,
                    flip_v=flip_v,
                    flip_w=flip_w,
                    epsilon=epsilon,
                    nthreads=nthreads,
                    do_wgridding=do_wgridding,
                    double_precision_accumulation=double_accum,
                    divide_by_n=False)
    convim /= wsum

    if beam is not None:
        convim *= beam

    if sigmainvsq is not None:
        convim += sigmainvsq * x

    return convim


# Kept in case we need them in the future
# def _hessian(x, uvw, weight, vis_mask, freq, beam, hessopts):
#     return _hessian_impl(x, uvw[0][0], weight[0][0], vis_mask[0][0], freq[0],
#                          beam, **hessopts)

# def hessian(x, uvw, weight, vis_mask, freq, beam, hessopts):
#     if beam is None:
#         bout = None
#     else:
#         bout = ('nx', 'ny')
#     return da.blockwise(_hessian, ('nx', 'ny'),
#                         x, ('nx', 'ny'),
#                         uvw, ('row', 'three'),
#                         weight, ('row', 'chan'),
#                         vis_mask, ('row', 'chan'),
#                         freq, ('chan',),
#                         beam, bout,
#                         hessopts, None,
#                         dtype=x.dtype)


def _hessian_psf_slice(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    psfhat,
                    beam,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    sigmainv=1,
                    wsum=None):
    """
    Tikhonov regularised Hessian approx
    """
    if beam is not None:
        psf_convolve_slice(xpad, xhat, xout,
                           psfhat, lastsize, x*beam,
                           nthreads=nthreads)
    else:
        psf_convolve_slice(xpad, xhat, xout,
                           psfhat, lastsize, x,
                           nthreads=nthreads)

    if beam is not None:
        xout *= beam

    if wsum is not None:
        xout /= wsum

    if sigmainv:
        xout += x * sigmainv

    return xout


def hessian_psf_cube(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    beam,
                    psfhat,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    sigmainv=1,
                    wsum=None,
                    mode='forward'):
    """
    Tikhonov regularised Hessian approx
    """
    if mode=='forward':
        if beam is not None:
            psf_convolve_cube(xpad, xhat, xout, psfhat, lastsize, x*beam,
                            nthreads=nthreads)
        else:
            psf_convolve_cube(xpad, xhat, xout, psfhat, lastsize, x,
                            nthreads=nthreads)

        if beam is not None:
            xout *= beam

        if wsum is not None:
            xout /= wsum

        if sigmainv:
            xout += x * sigmainv

        return xout
    else:
        raise NotImplementedError


def hess_direct(x,     # input image, not overwritten
                xpad=None,  # preallocated array to store padded image
                xhat=None,  # preallocated array to store FTd image
                xout=None,  # preallocated array to store output image
                psfhat=None,
                taperxy=None,
                lastsize=None,
                nthreads=1,
                sigmainvsq=1,
                wsum=None,
                mode='forward'):
    nband, nx, ny = x.shape
    xpad[...] = 0.0
    if mode == 'forward':
        xpad[:, 0:nx, 0:ny] = x / taperxy[None]
    else:
        xpad[:, 0:nx, 0:ny] = x * taperxy[None]
    r2c(xpad, out=xhat, axes=(1,2),
        forward=True, inorm=0, nthreads=nthreads)
    if mode=='forward':
        xhat *= (psfhat + sigmainvsq)
    else:
        xhat /= (psfhat + sigmainvsq)
    c2r(xhat, axes=(1, 2), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    xout[...] = xpad[:, 0:nx, 0:ny]
    if mode=='forward':
        xout /= taperxy[None]
    else:
        xout *= taperxy[None]
    return xout


def hess_direct_slice(x,     # input image, not overwritten
                xpad=None,  # preallocated array to store padded image
                xhat=None,  # preallocated array to store FTd image
                xout=None,  # preallocated array to store output image
                psfhat=None,
                taperxy=None,
                lastsize=None,
                nthreads=1,
                sigmainvsq=1,
                wsum=None,
                mode='forward'):
    nx, ny = x.shape
    xpad[...] = 0.0
    if mode == 'forward':
        xpad[0:nx, 0:ny] = x / taperxy
    else:
        xpad[0:nx, 0:ny] = x * taperxy
    r2c(xpad, out=xhat, axes=(0,1),
        forward=True, inorm=0, nthreads=nthreads)
    if mode=='forward':
        xhat *= (psfhat + sigmainvsq)
    else:
        xhat /= (psfhat + sigmainvsq)
    c2r(xhat, axes=(0, 1), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    xout[...] = xpad[0:nx, 0:ny]
    if mode=='forward':
        xout /= taperxy
    else:
        xout *= taperxy
    return xout
