"""
Dask wrappers around the wgridder. These operators are per band
because we are not guaranteed that each imaging band has the same
number of rows after BDA.
"""
import concurrent.futures as cf
from time import time
import numpy as np
import numba
from numba import njit, prange, literally, types
from numba.extending import overload
import concurrent.futures as cf
import xarray as xr
import dask.array as da
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from ducc0.fft import r2c
from ducc0.misc import resize_thread_pool
from pfb.utils.weighting import counts_to_weights, _compute_counts
from pfb.utils.beam import eval_beam
from pfb.utils.naming import xds_from_list
from pfb.utils.misc import fitcleanbeam
from pfb.utils.stokes import stokes_funcs
from pfb.utils.misc import JIT_OPTIONS, _es_kernel
from scipy.constants import c as lightspeed
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def wgridder_conventions(l0, m0):
    '''
    Returns

    flip_u, flip_v, flip_w, x0, y0

    according to the conventions documented here https://github.com/mreineck/ducc/issues/34

    Note that these conventions are stored as dataset attributes in order
    to call the operators acting on datasets with a consistent convention.
    '''
    return False, True, False, -l0, -m0


def vis2im(uvw,
           freq,
           vis,
           wgt,
           mask,
           nx, ny,
           cellx, celly,
           l0, m0,
           epsilon,
           precision,
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

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(l0, m0)

    return vis2dirty(uvw=uvw,
                     freq=freq,
                     vis=vis,
                     wgt=wgt,
                     mask=mask,
                     npix_x=nx, npix_y=ny,
                     pixsize_x=cellx, pixsize_y=celly,
                     center_x=x0, center_y=y0,
                     epsilon=epsilon,
                     flip_u=flip_u,
                     flip_v=flip_v,
                     flip_w=flip_w,
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
           l0=0, m0=0,
           epsilon=1e-7,
           do_wgridding=True,
           divide_by_n=False,
           nthreads=1):
    # adjust for chunking
    # need a copy here if using multiple row chunks
    freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(l0, m0)
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
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=epsilon,
            nthreads=nthreads,
            do_wgridding=do_wgridding,
            divide_by_n=divide_by_n)
    return vis


# we still need the collections interface here for the degridder
def comps2vis(
            uvw,
            utime,
            freq,
            rbin_idx, rbin_cnts,
            tbin_idx, tbin_cnts,
            fbin_idx, fbin_cnts,
            region_mask,
            mds,
            modelf,
            tfunc,
            ffunc,
            epsilon=1e-7,
            nthreads=1,
            do_wgridding=True,
            divide_by_n=False,
            freq_min=-np.inf,
            freq_max=np.inf,
            ncorr_out=4,
            product='I',
            poltype='linear'):

    # determine output type
    complex_type = da.result_type(mds.coefficients.dtype, np.complex64)

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
                        region_mask, None,
                        mds, None,
                        modelf, None,
                        tfunc, None,
                        ffunc, None,
                        epsilon, None,
                        nthreads, None,
                        do_wgridding, None,
                        divide_by_n, None,
                        freq_min, None,
                        freq_max, None,
                        ncorr_out, None,
                        product, None,
                        poltype, None,
                        new_axes={'c': ncorr_out},
                        # it should be getting these from uvw and freq?
                        adjust_chunks={'r': uvw.chunks[0]},
                        dtype=complex_type,
                        align_arrays=False)


def _comps2vis(
            uvw,
            utime,
            freq,
            rbin_idx, rbin_cnts,
            tbin_idx, tbin_cnts,
            fbin_idx, fbin_cnts,
            region_mask,
            mds,
            modelf,
            tfunc,
            ffunc,
            epsilon=1e-7,
            nthreads=1,
            do_wgridding=True,
            divide_by_n=False,
            freq_min=-np.inf,
            freq_max=np.inf,
            ncorr_out=4,
            product='I',
            poltype='linear'):
    return _comps2vis_impl(
                        uvw[0],
                        utime,
                        freq,
                        rbin_idx, rbin_cnts,
                        tbin_idx, tbin_cnts,
                        fbin_idx, fbin_cnts,
                        region_mask,
                        mds,
                        modelf,
                        tfunc,
                        ffunc,
                        epsilon=epsilon,
                        nthreads=nthreads,
                        do_wgridding=do_wgridding,
                        divide_by_n=divide_by_n,
                        freq_min=freq_min,
                        freq_max=freq_max,
                        ncorr_out=ncorr_out,
                        product=product,
                        poltype=poltype)



def _comps2vis_impl(uvw,
                    utime,
                    freq,
                    rbin_idx, rbin_cnts,
                    tbin_idx, tbin_cnts,
                    fbin_idx, fbin_cnts,
                    region_mask,
                    mds,
                    modelf,
                    tfunc,
                    ffunc,
                    epsilon=1e-7,
                    nthreads=1,
                    do_wgridding=True,
                    divide_by_n=False,
                    freq_min=-np.inf,
                    freq_max=np.inf,
                    ncorr_out=4,
                    product='I',
                    poltype='linear'):
    # why is this necessary?
    resize_thread_pool(nthreads)
    msg = f"Polarisation product {product} is not compatible with the "\
          f"number of correlations {ncorr_out}"

    # adjust for chunking
    # need a copy here if using multiple row chunks
    rbin_idx2 = rbin_idx - rbin_idx.min()
    tbin_idx2 = tbin_idx - tbin_idx.min()
    fbin_idx2 = fbin_idx - fbin_idx.min()

    ntime = tbin_idx.size
    nband = fbin_idx.size

    nrow = uvw.shape[0]
    nchan = freq.size
    vis = np.zeros((nrow, nchan, ncorr_out),
                   dtype=np.result_type(mds.coefficients.dtype, np.complex64))
    if not ((freq>=freq_min) & (freq<=freq_max)).any():
        return vis

    comps = mds.coefficients.values
    Ix = mds.location_x.values
    Iy = mds.location_y.values
    cellx = mds.cell_rad_x
    celly = mds.cell_rad_x
    nx = mds.npix_x
    ny = mds.npix_y
    # these are taken from dataset attrs to make sure they remain consistent
    x0 = mds.center_x
    y0 = mds.center_y
    flip_u = mds.flip_u
    flip_v = mds.flip_v
    flip_w = mds.flip_w
    for t in range(ntime):
        indt = slice(tbin_idx2[t], tbin_idx2[t] + tbin_cnts[t])
        # TODO - clean up this logic. row_mapping holds the number of rows per
        # utime and there are multiple utimes per row chunk
        indr = slice(rbin_idx2[indt][0], rbin_idx2[indt][-1] + rbin_cnts[indt][-1])
        for b in range(nband):
            indf = slice(fbin_idx2[b], fbin_idx2[b] + fbin_cnts[b])
            f = freq[indf]
            # don't degrid outside requested frequency range
            if not ((f>=freq_min) & (f<=freq_max)).any():
                continue
            # render components to image
            tout = tfunc(np.mean(utime[indt]))
            fout = ffunc(np.mean(freq[indf]))
            image = np.zeros((nx, ny), dtype=comps.dtype)
            image[Ix, Iy] = modelf(tout, fout, *comps[:, :])  # too magical?
            if np.any(region_mask):
                image = np.where(region_mask, image, 0.0)
                vis_stokes = dirty2vis(uvw=uvw,
                                       freq=f,
                                       dirty=image,
                                       pixsize_x=cellx, pixsize_y=celly,
                                       center_x=x0, center_y=y0,
                                       flip_u=flip_u,
                                       flip_v=flip_v,
                                       flip_w=flip_w,
                                       epsilon=epsilon,
                                       do_wgridding=do_wgridding,
                                       divide_by_n=divide_by_n,
                                       nthreads=nthreads)
                if ncorr_out == 1:
                    vis[indr, indf, 0] = vis_stokes
                elif ncorr_out == 2:
                    if product.upper() == 'I':
                        vis[indr, indf, 0] = vis_stokes
                        vis[indr, indf, -1] = vis_stokes
                    elif product.upper() == 'Q':
                        if poltype.lower() == 'linear':
                            vis[indr, indf, 0] = vis_stokes
                            vis[indr, indf, -1] = vis_stokes
                        else:
                            raise ValueError(msg)
                    elif product.upper() == 'V':
                        if poltype.lower() == 'linear':
                            raise ValueError(msg)
                        else:
                            vis[indr, indf, 0] = vis_stokes
                            vis[indr, indf, -1] = -vis_stokes
                    else:
                        raise ValueError(msg)
                elif ncorr_out == 4:
                    if product.upper() == 'I':
                        vis[indr, indf, 0] = vis_stokes
                        vis[indr, indf, -1] = vis_stokes
                    elif product.upper() == 'Q':
                        if poltype.lower() == 'linear':
                            vis[indr, indf, 0] = vis_stokes
                            vis[indr, indf, -1] = vis_stokes
                        else:
                            vis[indr, indf, 1] = vis_stokes
                            vis[indr, indf, 2] = vis_stokes
                    elif product.upper() == 'U':
                        if poltype.lower() == 'linear':
                            vis[indr, indf, 1] = vis_stokes
                            vis[indr, indf, 2] = vis_stokes
                        else:
                            vis[indr, indf, 1] = 1.0j*vis_stokes
                            vis[indr, indf, 2] = -1.0j*vis_stokes
                    elif product.upper() == 'V':
                        if poltype.lower() == 'linear':
                            vis[indr, indf, 1] = 1.0j*vis_stokes
                            vis[indr, indf, 2] = -1.0j*vis_stokes
                        else:
                            vis[indr, indf, 0] = vis_stokes
                            vis[indr, indf, 1] = vis_stokes
                    else:
                        raise ValueError(f"Unknown product {product}")

    return vis


def image_data_products(dsl,
                        dsp,
                        nx, ny,
                        nx_psf, ny_psf,
                        cellx, celly,
                        output_name,
                        attrs,
                        model=None,
                        robustness=None,
                        l0=0.0, m0=0.0,
                        nthreads=1,
                        epsilon=1e-7,
                        do_wgridding=True,
                        double_accum=True,
                        l2_reweight_dof=None,
                        do_dirty=True,
                        do_psf=True,
                        do_residual=True,
                        do_weight=True,
                        do_noise=False,
                        do_beam=False):
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
    resize_thread_pool(nthreads)
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(l0, m0)
    signu = -1.0 if flip_u else 1.0
    signv = -1.0 if flip_v else 1.0
    # we need these in the test because of flipped wgridder convention
    # https://github.com/mreineck/ducc/issues/34
    signx = -1.0 if flip_u else 1.0
    signy = -1.0 if flip_v else 1.0

    # TODO - assign ug,vg-coordinates
    x = (-nx/2 + np.arange(nx)) * cellx + x0
    y = (-ny/2 + np.arange(ny)) * celly + y0
    n = np.sqrt(1 - x0**2 - y0**2)

    # expects a list
    if isinstance(dsl, str):
        dsl = [dsl]

    dsl = xds_from_list(dsl, nthreads=nthreads)

    ncorr = dsl[0].corr.size

    # need to take weighted sum of beam before concat
    beam = np.zeros((ncorr, nx, ny), dtype=float)
    wsumb = np.zeros(ncorr, dtype=float)
    freq = dsl[0].FREQ.values  # must all be the same
    xx, yy = np.meshgrid(np.rad2deg(x), np.rad2deg(y), indexing='ij')
    for i, ds in enumerate(dsl):
        wgt = ds.WEIGHT.values
        mask = ds.MASK.values
        for c in range(ncorr):
            wsumt = (wgt[c]*mask).sum()
            wsumb[c] += wsumt
            l_beam = ds.l_beam.values
            m_beam = ds.m_beam.values
            beamt = eval_beam(ds.BEAM.values[c], l_beam, m_beam,
                              xx, yy)
            beam[c] += beamt * wsumt
            assert (ds.FREQ.values == freq).all()
        ds = ds.drop_vars(('BEAM','FREQ'))
        dsl[i] = ds.drop_dims(('l_beam', 'm_beam'))

    # TODO - weighted sum of beam computed using natural weights?
    beam /= wsumb[:, None, None]

    ds = xr.concat(dsl, dim='row')
    uvw = ds.UVW.values
    vis = ds.VIS.values
    wgt = ds.WEIGHT.values
    mask = ds.MASK.values
    bandid = int(ds.bandid)

    _, nrow, nchan = vis.shape

    coords = {
        'x': x,
        'y': y,
        'corr': ds.corr.values,
        'bpar': ['BMAJ', 'BMIN', 'BPA']
    }

    # output ds
    dso = xr.Dataset(attrs=attrs, coords=coords)
    dso['FREQ'] = (('chan',), freq)

    if model is None:
        if l2_reweight_dof:
            raise ValueError('Requested l2 reweight but no model passed in. '
                             'Perhaps transfer model from somewhere?')
    else:
        # do not apply weights in this direction
        # actually model vis, this saves memory
        residual_vis = np.zeros_like(vis)
        for c in range(ncorr):
            dirty2vis(
                uvw=uvw,
                freq=freq,
                dirty=model[c],
                pixsize_x=cellx,
                pixsize_y=celly,
                center_x=x0,
                center_y=y0,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                nthreads=nthreads,
                divide_by_n=False,  # incorporate in smooth beam
                sigma_min=1.1, sigma_max=3.0,
                vis=residual_vis[c])

        # this is an attempt to avoid the tmp model_vis array
        residual_vis *= -1  # negate model
        residual_vis += vis

    if l2_reweight_dof:
        if dsp:
            dsp = xds_from_list([dsp], drop_all_but='WEIGHT')
            wgtp = dsp[0].WEIGHT.values
        else:
            wgtp = 1.0
        # mask needs to be bool here
        ressq = (residual_vis*wgtp*residual_vis.conj()).real
        # we are currently doing this per correlation
        # should it be done jointly?
        ssq = ressq[:, mask>0].sum(axis=-1)
        ovar = ssq/mask.sum()  # same mask for all corrs
        if ovar:
            # scale the natural weights
            # RHS is weight relative to unity since wgtp included in ressq
            meani = np.mean(ressq[:, mask>0]/ovar)
            stdi = np.std(ressq[:, mask>0]/ovar)
            print(f"Band {bandid} before: mean = {meani:.3e}, std = {stdi:.3e}")
            denom = (l2_reweight_dof + ressq/ovar[:, None, None])
            # the expectation value of the complex ressq above is 2
            # if we do this jointly over correlations this should be 2*ncorr
            wgt *= (l2_reweight_dof + 2)/denom
        else:
            wgt = None
    # re-evaluate since robustness and or wgt after reweight may change
    if robustness is not None:
        numba_threads = np.maximum(nthreads, 1)
        numba.set_num_threads(numba_threads)
        # import ipdb; ipdb.set_trace()
        counts = _compute_counts(uvw,
                                 freq,
                                 mask,
                                 wgt,
                                 nx, ny,
                                 cellx, celly,
                                 uvw.dtype,
                                 # limit number of grids
                                 ngrid=np.minimum(numba_threads, 8),
                                 usign=1.0 if flip_u else -1.0,
                                 vsign=1.0 if flip_v else -1.0)
        wgt = counts_to_weights(
            counts,
            uvw,
            freq,
            wgt,
            mask,
            nx, ny,
            cellx, celly,
            robustness,
            usign=1.0 if flip_u else -1.0,
            vsign=1.0 if flip_v else -1.0)

    if l2_reweight_dof:
        # normalise to absolute units
        ressq = (residual_vis*wgt*residual_vis.conj()).real
        ssq = ressq[:, mask>0].sum()
        ovar = ssq/mask.sum()/2  # complex
        wgt /= ovar[:, None, None]
        ressq = (residual_vis*wgt*residual_vis.conj()).real
        meanf = np.mean(ressq[mask>0])
        stdf = np.std(ressq[mask>0])
        print(f"Band {bandid} after: mean = {meanf:.3e}, std = {stdf:.3e}")

        # import matplotlib.pyplot as plt
        # from scipy.stats import norm
        # x = np.linspace(-5, 5, 150)
        # y = norm.pdf(x, 0, 1)
        # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 12))
        # ax[0,0].hist((residual_vis.real*np.sqrt(wgtp/2)).ravel(), bins=15, density=True)
        # ax[0,0].plot(x, y, 'k')
        # ax[0,1].hist((residual_vis.real*np.sqrt(wgt/2)).ravel(), bins=15, density=True)
        # ax[0,1].plot(x, y, 'k')
        # ax[1,0].hist((residual_vis.imag*np.sqrt(wgtp/2)).ravel(), bins=15, density=True)
        # ax[1,0].plot(x, y, 'k')
        # ax[1,1].hist((residual_vis.imag*np.sqrt(wgt/2)).ravel(), bins=15, density=True)
        # ax[1,1].plot(x, y, 'k')
        # import os
        # cwd = os.getcwd()
        # bid = dso.attrs['bandid']
        # fig.savefig(f'{cwd}/resid_hist_{bid}.png')

    # these are always used together
    if do_weight:
        dso['WEIGHT'] = (('corr', 'row','chan'), wgt)
        dso['UVW'] = (('row', 'three'), uvw)
        dso['MASK'] = (('row','chan'), mask)

    wsum = wgt[:, mask.astype(bool)].sum(axis=-1)
    dso['WSUM'] = (('corr',), wsum)

    if do_dirty:
        dirty = np.zeros((ncorr, nx, ny), dtype=float)
        for c in range(ncorr):
            vis2dirty(
                uvw=uvw,
                freq=freq,
                vis=vis[c],
                wgt=wgt[c],
                mask=mask,
                npix_x=nx, npix_y=ny,
                pixsize_x=cellx, pixsize_y=celly,
                center_x=x0, center_y=y0,
                epsilon=epsilon,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                do_wgridding=do_wgridding,
                divide_by_n=False,  # incorporte in smooth beam
                nthreads=nthreads,
                sigma_min=1.1, sigma_max=3.0,
                double_precision_accumulation=double_accum,
                dirty=dirty[c])
        dso['DIRTY'] = (('corr', 'x','y'), dirty)

    if do_psf:
        if x0 or y0:
            freqfactor = 2j*np.pi*freq[None, :]/lightspeed
            psf_vis = np.exp(freqfactor*(signu*uvw[:, 0:1]*x0*signx +
                             signv*uvw[:, 1:2]*y0*signy -
                             uvw[:, 2:]*(n-1)))

        else:
            nrow, _ = uvw.shape
            nchan = freq.size
            tmp = np.ones((1,), dtype=vis.dtype)
            # should be tiny
            psf_vis = np.broadcast_to(tmp, (nrow, nchan))

        psf = np.zeros((ncorr, nx_psf, ny_psf), dtype=float)
        for c in range(ncorr):
            vis2dirty(
                uvw=uvw,
                freq=freq,
                vis=psf_vis,
                wgt=wgt[c],
                mask=mask,
                npix_x=nx_psf, npix_y=ny_psf,
                pixsize_x=cellx, pixsize_y=celly,
                center_x=x0, center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                divide_by_n=False,  # incorporte in smooth beam
                nthreads=nthreads,
                sigma_min=1.1, sigma_max=3.0,
                double_precision_accumulation=double_accum,
                dirty=psf[c])

        # get FT of psf
        psfhat = r2c(iFs(psf, axes=(1, 2)), axes=(1, 2),
                     nthreads=nthreads,
                     forward=True, inorm=0)

        dso["PSF"] = (('corr', 'x_psf', 'y_psf'), psf)
        dso["PSFHAT"] = (('corr', 'x_psf', 'yo2'), psfhat)

        # add natural resolution info
        # normalised internally
        gausspar = fitcleanbeam(psf, level=0.5, pixsize=1.0)
        dso["PSFPARSN"] = (('corr', 'bpar'), np.array(gausspar))


    if do_residual and model is not None:
        residual = np.zeros((ncorr, nx, ny), dtype=float)
        for c in range(ncorr):
            vis2dirty(
                uvw=uvw,
                freq=freq,
                vis=residual_vis[c],
                wgt=wgt,
                mask=mask,
                npix_x=nx, npix_y=ny,
                pixsize_x=cellx, pixsize_y=celly,
                center_x=x0, center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                divide_by_n=False,  # incorporte in smooth beam
                nthreads=nthreads,
                sigma_min=1.1, sigma_max=3.0,
                double_precision_accumulation=double_accum,
                dirty=dirty[c])

        dso['MODEL'] = (('corr','x','y'), model)
        dso['RESIDUAL'] = (('corr','x','y'), residual)

    if do_noise:
        # sample noise and project into image space
        _, nrow, nchan = vis.shape
        from pfb.utils.misc import parallel_standard_normal
        noise = np.zeros((ncorr, nx, ny), dtype=float)
        for c in range(ncorr):
            vis = (parallel_standard_normal((nrow, nchan)) +
                1j*parallel_standard_normal((nrow, nchan)))
            wmask = wgt[c] > 0.0
            vis[wmask] /= np.sqrt(wgt[c, wmask])
            vis[~wmask] = 0j
            vis2dirty(
                uvw=uvw,
                freq=freq,
                vis=vis,
                wgt=wgt[c],
                mask=mask,
                npix_x=nx, npix_y=ny,
                pixsize_x=cellx, pixsize_y=celly,
                center_x=x0, center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                divide_by_n=False,  # incorporte in smooth beam
                nthreads=nthreads,
                sigma_min=1.1, sigma_max=3.0,
                double_precision_accumulation=double_accum,
                dirty=noise[c])

        dso['NOISE'] = (('corr', 'x','y'), noise)

    if do_beam:
        if beam is not None:
            dso['BEAM'] = (('corr', 'x', 'y'), beam)
        else:
            dso['BEAM'] = (('corr', 'x', 'y'), np.ones((ncorr, nx, ny),
                                                       dtype=wgt.dtype))

    # save
    dso = dso.assign_attrs(wsum=wsum, x0=x0, y0=y0, l0=l0, m0=m0,
                           flip_u=flip_u,
                           flip_v=flip_v,
                           flip_w=flip_w)
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
                     nthreads=1,
                     epsilon=1e-7,
                     do_wgridding=True,
                     double_accum=True,
                     verbosity=1):
    '''
    Function to compute residual and write it to disk
    '''
    resize_thread_pool(nthreads)
    # expects a list
    if isinstance(dsl, str):
        dsl = [dsl]

    tii = time()

    # currently only a single dds
    ti = time()
    ds = xds_from_list(dsl, nthreads=nthreads)[0]

    uvw = ds.UVW.values
    wgt = ds.WEIGHT.values
    mask = ds.MASK.values
    beam = ds.BEAM.values
    dirty = ds.DIRTY.values
    freq = ds.FREQ.values
    flip_u = ds.flip_u
    flip_v = ds.flip_v
    flip_w = ds.flip_w
    x0 = ds.x0
    y0 = ds.y0

    tread = time() - ti

    ti = time()
    # do not apply weights in this direction
    model_vis = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=beam*model,
        pixsize_x=cellx,
        pixsize_y=celly,
        center_x=x0,
        center_y=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
        divide_by_n=False,  # incorporate in smooth beam
        sigma_min=1.1, sigma_max=3.0,
        verbosity=0)
    tdegrid = time() - ti

    ti = time()
    convim = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=model_vis,
        wgt=wgt,
        mask=mask,
        npix_x=nx, npix_y=ny,
        pixsize_x=cellx, pixsize_y=celly,
        center_x=x0, center_y=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        divide_by_n=False,  # incorporate in smooth beam
        nthreads=nthreads,
        sigma_min=1.1, sigma_max=3.0,
        double_precision_accumulation=double_accum,
        verbosity=0)
    tgrid = time() - ti

    # this is the once attenuated residual since
    # dirty is only attenuated once
    ti = time()
    residual = dirty - convim
    tdiff = time() - ti

    ti = time()
    ds['MODEL'] = (('x','y'), model)
    ds['RESIDUAL'] = (('x','y'), residual)
    tassign = time() - ti

    # we only need to write MODEL and RESIDUAL
    ds = ds[['RESIDUAL','MODEL']]

    # save
    ti = time()
    with cf.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ds.to_zarr, output_name, mode='a')
    twrite = time() - ti

    ttot = time() - tii
    ttally = tread + tdegrid + tgrid + tdiff + tassign + twrite
    if verbosity > 1:
        print(f'tread = {tread/ttot}')
        print(f'tdegrid = {tdegrid/ttot}')
        print(f'tgrid = {tgrid/ttot}')
        print(f'tdiff = {tdiff/ttot}')
        print(f'tassign = {tassign/ttot}')
        print(f'twrite = {twrite/ttot}')
        print(f'ttally = {ttally/ttot}')
    return residual, future

def dataset_to_zarr(ds, output_name):
    ds.to_zarr(output_name, mode='a')
