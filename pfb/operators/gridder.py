"""
Dask wrappers around the wgridder. These operators are per band
because we are not guaranteed that each imaging band has the same
number of rows after BDA.
"""

import numpy as np
import dask
import dask.array as da
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from ducc0.fft import c2r, r2c
from africanus.constants import c as lightspeed
from quartical.utils.dask import Blocker
from pfb.utils.weighting import _counts_to_weights
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

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



def im2vis(uvw=None,
           freq=None,
           image=None,
           wgt=None,
           mask=None,
           cellx=None,
           celly=None,
           nthreads=1,
           epsilon=1e-7,
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
                        wgt, wgt_out,
                        mask, mask_out,
                        cellx, None,
                        celly, None,
                        x0, None,
                        y0, None,
                        epsilon, None,
                        flip_v, None,
                        do_wgridding, None,
                        precision, None,
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
           x0=0, y0=0,
           epsilon=1e-7,
           flip_v=False,
           do_wgridding=True,
           precision='single',
           divide_by_n=False,
           nthreads=1,
           sigma_min=1.1,
           sigma_max=2.6):

    return _im2vis_impl(uvw[0],
                        freq,
                        image[0][0],
                        wgt,
                        mask,
                        cellx, celly,
                        x0, y0,
                        epsilon,
                        flip_v,
                        do_wgridding,
                        precision,
                        divide_by_n,
                        nthreads,
                        sigma_min,
                        sigma_max)



def _im2vis_impl(uvw,
                 freq,
                 image,
                 wgt,
                 mask,
                 cellx,
                 celly,
                 x0=0, y0=0,
                 epsilon=1e-7,
                 flip_v=False,
                 do_wgridding=True,
                 precision='single',
                 divide_by_n=False,
                 nthreads=1,
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
    if wgt is not None:
        wgt = np.require(wgt, dtype=real_type)
    if mask is not None:
        mask = np.require(mask, dtype=np.uint8)
    return dirty2vis(uvw=uvw, freq=freq, dirty=image, wgt=wgt, mask=mask,
                     pixsize_x=cellx, pixsize_y=celly,
                     center_x=x0, center_y=y0, epsilon=epsilon, flip_v=flip_v,
                     do_wgridding=do_wgridding, divide_by_n=divide_by_n,
                     nthreads=nthreads, sigma_min=sigma_min,
                     sigma_max=sigma_max)



def _loc2psf_vis_impl(uvw,
                      freq,
                      cell,
                      x0=0,
                      y0=0,
                      do_wgridding=True,
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
                            do_wgridding=do_wgridding,
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
                 do_wgridding=True,
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
                             do_wgridding,
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
                do_wgridding=True,
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
                        do_wgridding, None,
                        epsilon, None,
                        nthreads, None,
                        precision, None,
                        divide_by_n, None,
                        convention, None,
                        dtype=np.complex64 if precision == 'single' else np.complex128)


def comps2vis(uvw,
              utime,
              freq,
              rbin_idx, rbin_cnts,
              tbin_idx, tbin_cnts,
              fbin_idx, fbin_cnts,
              comps,
              Ix, Iy,
              modelf,
              tfunc,
              ffunc,
              nx, ny,
              cellx, celly,
              x0=0, y0=0,
              epsilon=1e-7,
              nthreads=1,
              do_wgridding=True,
              divide_by_n=False,
              ncorr_out=4):

    # determine output type
    complex_type = da.result_type(comps, np.complex64)

    return da.blockwise(_comps2vis, 'rfc',
                        uvw, 'r3',
                        utime, 'r',
                        freq, 'f',
                        rbin_idx, 'r',
                        rbin_cnts, 'r',
                        tbin_idx, 'r',
                        tbin_cnts, 'r',
                        fbin_idx, 'f',
                        fbin_cnts, 'f',
                        comps, None,
                        Ix, None,
                        Iy, None,
                        modelf, None,
                        tfunc, None,
                        ffunc, None,
                        nx, None,
                        ny, None,
                        cellx, None,
                        celly, None,
                        x0, None,
                        y0, None,
                        epsilon, None,
                        nthreads, None,
                        do_wgridding, None,
                        divide_by_n, None,
                        ncorr_out, None,
                        new_axes={'c': ncorr_out},
                        # it should be pulling these from uvw and freq so shouldn't need this?
                        adjust_chunks={'r': uvw.chunks[0]},
                        dtype=complex_type,
                        align_arrays=False)


def _comps2vis(uvw,
                utime,
                freq,
                rbin_idx, rbin_cnts,
                tbin_idx, tbin_cnts,
                fbin_idx, fbin_cnts,
                comps,
                Ix, Iy,
                modelf,
                tfunc,
                ffunc,
                nx, ny,
                cellx, celly,
                x0=0, y0=0,
                epsilon=1e-7,
                nthreads=1,
                do_wgridding=True,
                divide_by_n=False,
                ncorr_out=4):
    return _comps2vis_impl(uvw[0],
                           utime,
                           freq,
                           rbin_idx, rbin_cnts,
                           tbin_idx, tbin_cnts,
                           fbin_idx, fbin_cnts,
                           comps,
                           Ix, Iy,
                           modelf,
                           tfunc,
                           ffunc,
                           nx, ny,
                           cellx, celly,
                           x0=x0, y0=y0,
                           epsilon=epsilon,
                           nthreads=nthreads,
                           do_wgridding=do_wgridding,
                           divide_by_n=divide_by_n,
                           ncorr_out=ncorr_out)



def _comps2vis_impl(uvw,
                    utime,
                    freq,
                    rbin_idx, rbin_cnts,
                    tbin_idx, tbin_cnts,
                    fbin_idx, fbin_cnts,
                    comps,
                    Ix, Iy,
                    modelf,
                    tfunc,
                    ffunc,
                    nx, ny,
                    cellx, celly,
                    x0=0, y0=0,
                    epsilon=1e-7,
                    nthreads=1,
                    do_wgridding=True,
                    divide_by_n=False,
                    ncorr_out=4):
    # adjust for chunking
    # need a copy here if using multiple row chunks
    rbin_idx2 = rbin_idx - rbin_idx.min()
    tbin_idx2 = tbin_idx - tbin_idx.min()
    fbin_idx2 = fbin_idx - fbin_idx.min()

    # currently not interpolating in time
    ntime = tbin_idx.size
    nband = fbin_idx.size

    nrow = uvw.shape[0]
    nchan = freq.size
    vis = np.zeros((nrow, nchan, ncorr_out), dtype=np.result_type(comps, np.complex64))
    for t in range(ntime):
        indt = slice(tbin_idx2[t], tbin_idx2[t] + tbin_cnts[t])
        # TODO - clean up this logic. row_mapping holds the number of rows per
        # utime and there are multiple utimes per row chunk
        indr = slice(rbin_idx2[indt][0], rbin_idx2[indt][-1] + rbin_cnts[indt][-1])
        for b in range(nband):
            indf = slice(fbin_idx2[b], fbin_idx2[b] + fbin_cnts[b])
            # render components to image
            # we want to do this on each worker
            tout = tfunc(np.mean(utime[indt]))
            fout = ffunc(np.mean(freq[indf]))
            image = np.zeros((nx, ny), dtype=comps.dtype)
            image[Ix, Iy] = modelf(tout, fout, *comps[:, :])  # too magical?
            vis[indr, indf, 0] = dirty2vis(uvw=uvw,
                                           freq=freq[indf],
                                           dirty=image,
                                           pixsize_x=cellx, pixsize_y=celly,
                                           center_x=x0, center_y=y0,
                                           epsilon=epsilon,
                                           do_wgridding=do_wgridding,
                                           divide_by_n=divide_by_n,
                                           nthreads=nthreads)
            if ncorr_out > 1:
                vis[indr, indf, -1] = vis[indr, indf, 0]
    return vis


def image_data_products(uvw,
                        freq,
                        vis,
                        wgt,
                        mask,
                        counts,
                        nx, ny,
                        nx_psf, ny_psf,
                        cellx, celly,
                        model=None,
                        robustness=None,
                        x0=0.0, y0=0.0,
                        nthreads=1,
                        epsilon=1e-7,
                        do_wgridding=True,
                        double_accum=True,
                        # divide_by_n=False,
                        l2reweight_dof=None,
                        do_dirty=True,
                        do_psf=True,
                        do_weight=True,
                        do_residual=False):
    '''
    Function to compute image space data products in one go
        dirty
        psf
        psfhat
        imweight
        residual
        wsum
    '''
    out_dict = {}

    if model is None:
        if l2reweight_dof:
            raise ValueError('Requested l2 reweight but no model passed in. '
                             'Perhaps transfer model from somewhere?')
    else:
        # do not apply weights in this direction
        model_vis = dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=model,
            pixsize_x=cellx,
            pixsize_y=celly,
            center_x=x0,
            center_y=y0,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            flip_v=False,
            nthreads=nthreads,
            divide_by_n=False,
            sigma_min=1.1, sigma_max=3.0)

        residual_vis = vis - model_vis
        # apply mask to both
        residual_vis *= mask

    if l2reweight_dof:
        ressq = (residual_vis*residual_vis.conj()).real
        wcount = mask.sum()
        if wcount:
            ovar = ressq.sum()/wcount
            l2wgt = (l2reweight_dof + 1)/(l2reweight_dof + ressq/ovar)/ovar
        else:
            l2wgt = None
    else:
        l2wgt = None


    # we usually want to re-evaluate this since the robustness may change
    if robustness is not None:
        if counts is None:
            raise ValueError('counts are None but robustness specified. '
                             'This is probably a bug!')
        imwgt = _counts_to_weights(
            counts,
            uvw,
            freq,
            nx, ny,
            cellx, celly,
            robustness)

        # this is necessitated by the way the weighting is done i.e.
        # wgt applied to vis@init so only imwgt*l2wgt needs to be applied
        if l2wgt is not None:
            imwgt *= l2wgt

        # wgt*imwgt*l2wgt required for PSF
        wgt *= imwgt

    if do_weight:
        out_dict['WEIGHT'] = wgt

    wsum = wgt[mask.astype(bool)].sum()
    out_dict['WSUM'] = np.atleast_1d(wsum)

    dirty = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=imwgt,  # data already naturally weighted
        mask=mask,
        npix_x=nx, npix_y=ny,
        pixsize_x=cellx, pixsize_y=celly,
        center_x=x0, center_y=y0,
        epsilon=epsilon,
        flip_v=False,  # hardcoded for now
        do_wgridding=do_wgridding,
        divide_by_n=False,  # hardcoded for now
        nthreads=nthreads,
        sigma_min=1.1, sigma_max=3.0,
        double_precision_accumulation=double_accum)
    out_dict['DIRTY'] = dirty

    if do_psf:
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
            x = np.zeros((128,128), dtype=wgt.dtype)
            x[64,64] = 1.0
            psf_vis = dirty2vis(
                uvw=uvw,
                freq=freq,
                dirty=x,
                pixsize_x=cellx,
                pixsize_y=celly,
                center_x=x0,
                center_y=y0,
                epsilon=1e-7,
                do_wgridding=do_wgridding,
                nthreads=nthreads,
                divide_by_n=False,
                flip_v=False,  # hardcoded for now
                sigma_min=1.1, sigma_max=3.0)

        else:
            nrow, _ = uvw.shape
            nchan = freq.size
            psf_vis = np.ones((nrow, nchan), dtype=vis.dtype)

        psf = vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=psf_vis,
            wgt=wgt,
            mask=mask,
            npix_x=nx_psf, npix_y=ny_psf,
            pixsize_x=cellx, pixsize_y=celly,
            center_x=x0, center_y=y0,
            epsilon=epsilon,
            flip_v=False,  # hardcoded for now
            do_wgridding=do_wgridding,
            divide_by_n=False,  # hardcoded for now
            nthreads=nthreads,
            sigma_min=1.1, sigma_max=3.0,
            double_precision_accumulation=double_accum)

        # get FT of psf
        psfhat = r2c(iFs(psf, axes=(0, 1)), axes=(0, 1),
                     nthreads=nthreads,
                     forward=True, inorm=0)

        out_dict["PSF"] = psf
        out_dict["PSFHAT"] = psfhat


    if do_residual and model is not None:
        residual = vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=residual_vis,
            wgt=wgt,
            mask=mask,
            npix_x=nx, npix_y=ny,
            pixsize_x=cellx, pixsize_y=celly,
            center_x=x0, center_y=y0,
            epsilon=epsilon,
            flip_v=False,  # hardcoded for now
            do_wgridding=do_wgridding,
            divide_by_n=False,  # hardcoded for now
            nthreads=nthreads,
            sigma_min=1.1, sigma_max=3.0,
            double_precision_accumulation=double_accum)

        out_dict['RESIDUAL'] = residual

    return out_dict
