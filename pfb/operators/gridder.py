"""
Dask wrappers around the wgridder. These operators are per band
because we are not guaranteed that each imaging band has the same
number of rows after BDA.
"""

import numpy as np
import dask
import dask.array as da
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from africanus.constants import c as lightspeed


def vis2im(uvw=None,
           freq=None,
           vis=None,
           wgt=None,
           nx=None, ny=None,
           cellx=None, celly=None,
           nthreads=None,
           epsilon=None,
           mask=None,
           do_wgridding=True,
           flip_v=False,
           x0=0, y0=0,
           precision='single',
           divide_by_n=False,
           sigma_min=1.1, sigma_max=2.6,
           double_precision_accumulation=True):

    if precision.lower() == 'single':
        real_type = np.float32
    elif precision.lower() == 'double':
        real_type = np.float64

    if wgt is not None:
        wgt_out = 'rf'
    else:
        wgt_out = None

    if mask is not None:
        mask_out = 'rf'
    else:
        mask_out = None

    return da.blockwise(_vis2im, 'xy',
                        uvw, 'r3',
                        freq, 'f',
                        vis, 'rf',
                        wgt, wgt_out,
                        mask, mask_out,
                        nx, None,
                        ny, None,
                        cellx, None,
                        celly, None,
                        x0, None,
                        y0, None,
                        epsilon, None,
                        precision, None,
                        flip_v, None,
                        do_wgridding, None,
                        divide_by_n, None,
                        nthreads, None,
                        sigma_min, None,
                        sigma_max, None,
                        double_precision_accumulation, None,
                        new_axes={'x':nx, 'y': ny},
                        dtype=real_type)

def _vis2im(uvw,
            freq,
            vis,
            wgt,
            mask,
            nx, ny,
            cellx, celly,
            x0, y0,
            epsilon,
            precision,
            flip_v,
            do_wgridding,
            divide_by_n,
            nthreads,
            sigma_min,
            sigma_max,
            double_precision_accumulation):

    if wgt is not None:
        wgt_out = wgt[0][0]
    else:
        wgt_out = None

    if mask is not None:
        mask_out = mask[0][0]
    else:
        mask_out = None

    return _vis2im_impl(uvw[0][0],
                        freq[0],
                        vis[0][0],
                        wgt_out,
                        mask_out,
                        nx, ny,
                        cellx, celly,
                        x0, y0,
                        epsilon,
                        precision,
                        flip_v,
                        do_wgridding,
                        divide_by_n,
                        nthreads,
                        sigma_min, sigma_max,
                        double_precision_accumulation)

def _vis2im_impl(uvw,
                 freq,
                 vis,
                 wgt,
                 mask,
                 nx, ny,
                 cellx, celly,
                 x0, y0,
                 epsilon,
                 precision,
                 flip_v,
                 do_wgridding,
                 divide_by_n,
                 nthreads,
                 sigma_min, sigma_max,
                 double_precision_accumulation):
    uvw = np.require(uvw, dtype=np.float64)
    freq = np.require(freq, np.float64)

    if precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    vis = np.require(vis, dtype=complex_type)

    if wgt is not None:
        wgt = np.require(wgt, dtype=real_type)

    if mask is not None:
        mask = np.require(mask, dtype=np.uint8)

    return vis2dirty(uvw=uvw,
                     freq=freq,
                     vis=vis,
                     wgt=wgt,
                     mask=mask,
                     npix_x=nx, npix_y=ny,
                     pixsize_x=cellx, pixsize_y=celly,
                     center_x=x0, center_y=y0,
                     epsilon=epsilon,
                     flip_v=flip_v,
                     do_wgridding=do_wgridding,
                     divide_by_n=divide_by_n,
                     nthreads=nthreads,
                     sigma_min=sigma_min, sigma_max=sigma_max,
                     double_precision_accumulation=double_precision_accumulation)



def im2vis(uvw,
           freq,
           image,
           wgt,
           mask,
           cellx,
           celly,
           nthreads,
           epsilon,
           do_wgridding=True,
           flip_v=False,
           x0=0, y0=0,
           precision='single',
           divide_by_n=False,
           sigma_min=1.1,
           sigma_max=2.6):

    if precision.lower() == 'single':
        complex_type = np.float32
    elif precision.lower() == 'double':
        complex_type = np.float64

    if wgt is not None:
        wgt_out = 'rf'
    else:
        wgt_out = None

    if mask is not None:
        mask_out = 'rf'
    else:
        mask_out = None

    return da.blockwise(_im2vis, 'rf',
                        uvw, 'r3',
                        freq, 'f',
                        image, 'xy',
                        wgt, 'rf',
                        mask, 'rf',
                        npix_x, None,
                        npix_y, None,
                        cellx, None,
                        celly, None,
                        x0, None,
                        y0, None,
                        epsilon, None,
                        flip_v, None,
                        do_wgridding, None,
                        divide_by_n, None,
                        nthreads, None,
                        sigma_min, None,
                        sigma_max, None,
                        dtype=complex_type)

def _im2vis(uvw,
           freq,
           image,
           wgt,
           mask,
           cellx,
           celly,
           nthreads,
           epsilon,
           do_wgridding=True,
           flip_v=False,
           x0=0, y0=0,
           precision='single',
           divide_by_n=False,
           sigma_min=1.1,
           sigma_max=2.6):

    return _im2vis_impl(uvw[0],
                        freq,
                        image[0][0],
                        wgt,
                        cellx, celly,
                        nthreads,
                        epsilon,
                        do_wgridding,
                        flip_v,
                        x0, y0,
                        precision,
                        divide_by_n,
                        sigma_min,
                        sigma_max)



def _im2vis_impl(uvw,
                 freq,
                 image,
                 wgt,
                 cellx,
                 celly,
                 nthreads,
                 epsilon,
                 do_wgridding=True,
                 flip_v=False,
                 x0=0,
                 y0=0,
                 precision='single',
                 divide_by_n=False,
                 sigma_min=1.1,
                 sigma_max=2.6):
    uvw = np.require(uvw, dtype=np.float64)
    freq = np.require(freq, np.float64)
    if precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128
    image = np.require(image, dtype=real_type)
    wgt = np.require(wgt, dtype=real_type)
    mask = np.require(mask, dtype=np.uint8)
    return dirty2vis(uvw=uvw, freq=freq, dirty=image, wgt=wgt, mask=mask,
                     npix_x=nx, npix_y=ny, pixsize_x=cellx, pixsize_y=celly,
                     center_x=x0, center_y=y0, epsilon=epsilon, flip_v=flip_v,
                     do_wgridding=do_wgridding, divide_by_n=divide_by_n,
                     nthreads=nthreads, sigma_min=sigma_min,
                     sigma_max=sigma_max)



def _loc2psf_vis_impl(uvw,
                      freq,
                      cell,
                      x0=0,
                      y0=0,
                      wstack=True,
                      epsilon=1e-7,
                      nthreads=1,
                      precision='single',
                      divide_by_n=False,
                      convention='CASA'):
    if x0 or y0:
        # LB - what is wrong with this?
        # n = np.sqrt(1 - x0**2 - y0**2)
        # if convention.upper() == 'CASA':
        #     freqfactor = -2j*np.pi*freq[None, :]/lightspeed
        # else:
        #     freqfactor = 2j*np.pi*freq[None, :]/lightspeed
        # psf_vis = np.exp(freqfactor*(uvw[:, 0:1]*x0 +
        #                              uvw[:, 1:2]*y0 +
        #                              uvw[:, 2:]*(n-1)))
        # if divide_by_n:
        #     psf_vis /= n
        real_type = np.float32 if precision == 'single' else np.float64
        x = np.zeros((128,128), dtype=real_type)
        x[64,64] = 1.0
        psf_vis = dirty2vis(uvw=uvw,
                            freq=freq,
                            dirty=x,
                            pixsize_x=cell,
                            pixsize_y=cell,
                            center_x=x0,
                            center_y=y0,
                            epsilon=1e-7,
                            do_wgridding=wstack,
                            nthreads=nthreads,
                            divide_by_n=divide_by_n)

    else:
        nrow, _ = uvw.shape
        nchan = freq.size
        dtype = np.complex64 if precision == 'single' else np.complex128
        psf_vis = np.ones((nrow, nchan), dtype=dtype)
    return psf_vis


def _loc2psf_vis(uvw,
                 freq,
                 cell,
                 x0=0,
                 y0=0,
                 wstack=True,
                 epsilon=1e-7,
                 nthreads=1,
                 precision='single',
                 divide_by_n=False,
                 convention='CASA'):
    return _loc2psf_vis_impl(uvw[0],
                             freq,
                             cell,
                             x0,
                             y0,
                             wstack,
                             epsilon,
                             nthreads,
                             precision,
                             divide_by_n,
                             convention)


def loc2psf_vis(uvw,
                freq,
                cell,
                x0=0,
                y0=0,
                wstack=True,
                epsilon=1e-7,
                nthreads=1,
                precision='single',
                divide_by_n=False,
                convention='CASA'):
    return da.blockwise(_loc2psf_vis, 'rf',
                        uvw, 'r3',
                        freq, 'f',
                        cell, None,
                        x0, None,
                        y0, None,
                        wstack, None,
                        epsilon, None,
                        nthreads, None,
                        precision, None,
                        divide_by_n, None,
                        convention, None,
                        dtype=np.complex64 if precision == 'single' else np.complex128)


