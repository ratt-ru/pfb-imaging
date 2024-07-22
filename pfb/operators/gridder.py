"""
Dask wrappers around the wgridder. These operators are per band
because we are not guaranteed that each imaging band has the same
number of rows after BDA.
"""

import numpy as np
import numba
import concurrent.futures as cf
import xarray as xr
import dask
import dask.array as da
from ducc0.wgridder import vis2dirty, dirty2vis
from ducc0.fft import c2r, r2c, c2c
from africanus.constants import c as lightspeed
from quartical.utils.dask import Blocker
from pfb.utils.weighting import counts_to_weights, _compute_counts
from pfb.utils.beam import eval_beam
from pfb.utils.naming import xds_from_list
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def vis2im(uvw,
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
           cellx,
           celly,
           freq_bin_idx,
           freq_bin_counts,
           x0=0, y0=0,
           epsilon=1e-7,
           flip_v=False,
           do_wgridding=True,
           divide_by_n=False,
           nthreads=1):
    # adjust for chunking
    # need a copy here if using multiple row chunks
    freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()
    nband, nx, ny = image.shape
    nrow = uvw.shape[0]
    nchan = freq.size
    vis = np.zeros((nrow, nchan), dtype=np.result_type(image, np.complex64))
    for i in range(nband):
        ind = slice(freq_bin_idx2[i], freq_bin_idx2[i] + freq_bin_counts[i])
        vis[:, ind] = dirty2vis(
            uvw=uvw,
            freq=freq[ind],
            dirty=image[i],
            pixsize_x=cellx,
            pixsize_y=celly,
            center_x=x0,
            center_y=y0,
            epsilon=epsilon,
            nthreads=nthreads,
            do_wgridding=do_wgridding,
            divide_by_n=divide_by_n,
            flip_v=flip_v
        )
    return vis


# we still need the collections interface here for the degridder
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
            # negate w for wgridder bug
            # uvw[:, 2] = -uvw[:, 2]
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


def image_data_products(dsl,
                        counts,
                        nx, ny,
                        nx_psf, ny_psf,
                        cellx, celly,
                        output_name,
                        attrs,
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
                        do_residual=True,
                        do_weight=True,
                        do_noise=False):
    '''
    Function to compute image space data products in one go

        dirty
        psf
        residual
        noise
        beam
        imweight
        wsum

    Assumes all datasets are concatenatable and will compute weighted
    sum of beam
    '''

    # TODO - assign ug,vg-coordinates
    x = (-nx/2 + np.arange(nx)) * cellx + x0
    y = (-ny/2 + np.arange(ny)) * celly + y0
    coords = {
        'x': x,
        'y': y
    }

    # expects a list
    if isinstance(dsl, str):
        dsl = [dsl]

    dsl = xds_from_list(dsl, nthreads=nthreads)

    # need to take weighted sum of beam before concat
    beam = np.zeros((nx, ny), dtype=float)
    wsumb = 0.0
    freq = dsl[0].FREQ.values  # must all be the same
    xx, yy = np.meshgrid(np.rad2deg(x), np.rad2deg(y), indexing='ij')
    for i, ds in enumerate(dsl):
        wgt = ds.WEIGHT.values
        mask = ds.MASK.values
        wsumt = (wgt*mask).sum()
        wsumb += wsumt
        l_beam = ds.l_beam.values
        m_beam = ds.m_beam.values
        beamt = eval_beam(ds.BEAM.values, l_beam, m_beam,
                          xx, yy)
        beam += beamt * wsumt
        assert (ds.FREQ.values == freq).all()
        ds = ds.drop_vars(('BEAM','FREQ'))
        dsl[i] = ds.drop_dims(('l_beam', 'm_beam'))

    # weighted sum of beam computed using natural weights
    beam /= wsumb

    ds = xr.concat(dsl, dim='row')
    uvw = ds.UVW.values
    vis = ds.VIS.values
    wgt = ds.WEIGHT.values
    mask = ds.MASK.values
    bandid = int(ds.bandid)

    # output ds
    dso = xr.Dataset(attrs=attrs, coords=coords)
    dso['FREQ'] = (('chan',), freq)
    if counts is not None:
        dso['COUNTS'] = (('x', 'y'), counts)

    if model is None:
        if l2reweight_dof:
            raise ValueError('Requested l2 reweight but no model passed in. '
                             'Perhaps transfer model from somewhere?')
    else:
        # do not apply weights in this direction
        residual_vis = dirty2vis(
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

        residual_vis *= -1  # negate model
        residual_vis += vis
        # apply mask for reweighting
        # residual_vis *= mask

    if l2reweight_dof:
        # careful mask needs to be bool here
        ressq = (residual_vis*residual_vis.conj()).real
        ssq = ressq[mask>0].sum()
        ovar = ssq/mask.sum()
        # ovar = np.var(residual_vis[mask])
        chi2_dofp = np.mean(ressq[mask>0]*wgt[mask>0])
        mean_dev = np.mean(ressq[mask>0]/ovar)
        if ovar:
            wgt = (l2reweight_dof + 1)/(l2reweight_dof + ressq/ovar)
            # now divide by ovar to scale to absolute units
            # the chi2_dof after reweighting should be close to one
            wgt /= ovar
            chi2_dof = np.mean(ressq[mask>0]*wgt[mask>0])
            print(f'Band {bandid} chi2-dof changed from {chi2_dofp} to {chi2_dof} with mean deviation of {mean_dev}')
        else:
            wgt = None




    # we usually want to re-evaluate this since the robustness may change
    if robustness is not None:
        if counts is None:
            raise ValueError('counts are None but robustness specified. '
                             'This is probably a bug!')
        counts = _compute_counts(uvw,
                                 freq,
                                 mask,
                                 wgt,
                                 nx, ny,
                                 cellx, celly,
                                 uvw.dtype,
                                 ngrid=np.minimum(nthreads, 8))  # limit number of grids
        imwgt = counts_to_weights(
            counts,
            uvw,
            freq,
            nx, ny,
            cellx, celly,
            robustness)
        if wgt is not None:
            wgt *= imwgt
        else:
            wgt = imwgt

    # these are always used together
    if do_weight:
        dso['WEIGHT'] = (('row','chan'), wgt)
        dso['UVW'] = (('row', 'three'), uvw)
        dso['MASK'] = (('row','chan'), mask)

    wsum = wgt[mask.astype(bool)].sum()
    dso['WSUM'] = (('scalar',), np.atleast_1d(wsum))

    if do_dirty:
        dirty = vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis,
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
        dso['DIRTY'] = (('x','y'), dirty)

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
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                nthreads=nthreads,
                divide_by_n=False,
                flip_v=False,  # hardcoded for now
                sigma_min=1.1, sigma_max=3.0)

        else:
            nrow, _ = uvw.shape
            nchan = freq.size
            tmp = np.ones((1,), dtype=vis.dtype)
            # should be tiny
            psf_vis = np.broadcast_to(tmp, vis.shape)

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

        dso["PSF"] = (('x_psf', 'y_psf'), psf)
        dso["PSFHAT"] = (('x_psf', 'yo2'), psfhat)


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

        dso['MODEL'] = (('x','y'), model)
        dso['RESIDUAL'] = (('x','y'), residual)

    if do_noise:
        # sample noise and project into image space
        nrow, nchan = vis.shape
        from pfb.utils.misc import parallel_standard_normal
        vis = (parallel_standard_normal((nrow, nchan)) +
               1j*parallel_standard_normal((nrow, nchan)))
        wmask = wgt > 0.0
        vis[wmask] /= np.sqrt(wgt[wmask])
        vis[~wmask] = 0j
        noise = vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis,
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

        dso['NOISE'] = (('x','y'), noise)

    if beam is not None:
        dso['BEAM'] = (('x', 'y'), beam)
    else:
        dso['BEAM'] = (('x', 'y'), np.ones((nx, ny), dtype=wgt.dtype))

    # save
    dso = dso.assign_attrs(wsum=wsum)
    dso.to_zarr(output_name, mode='a')

    # return residual to report stats
    # one of these should always succeed
    try:
        return residual, wsum
    except:
        return dirty, wsum


def compute_residual(dsl,
                     nx, ny,
                     cellx, celly,
                     output_name,
                     model,
                     x0=0.0, y0=0.0,
                     nthreads=1,
                     epsilon=1e-7,
                     do_wgridding=True,
                     double_accum=True):
    '''
    Function to compute residual and write it to disk
    '''

    # expects a list
    if isinstance(dsl, str):
        dsl = [dsl]

    # currently only a single dds
    ds = xds_from_list(dsl, nthreads=nthreads)[0]

    uvw = ds.UVW.values
    wgt = ds.WEIGHT.values
    mask = ds.MASK.values
    beam = ds.BEAM.values
    dirty = ds.DIRTY.values
    freq = ds.FREQ.values

    # do not apply weights in this direction
    model_vis = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=beam*model,
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


    convim = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=model_vis,
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

    # this is the once attenuated residual since
    # dirty is only attenuated once
    residual = dirty - convim

    ds['MODEL'] = (('x','y'), model)
    ds['RESIDUAL'] = (('x','y'), residual)

    # save
    ds.to_zarr(output_name, mode='a')

    return residual
