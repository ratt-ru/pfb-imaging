import concurrent.futures as cf

import numba
import numpy as np
from numba import literally, njit, prange, types
from numba.extending import overload
from scipy.constants import c as lightspeed

from pfb_imaging.utils.naming import xds_from_list
from pfb_imaging.utils.stokes import stokes_funcs

ifftshift = np.fft.ifftshift
fftshift = np.fft.fftshift

JIT_OPTIONS = {
    "nogil": True,
    "cache": True,
    "error_model": "numpy",
    # "fastmath": True  # causes weighting tests to fail
}


@njit(**JIT_OPTIONS, inline="always")
def _es_kernel(x, y, xkern, ykern, betak):
    for i in range(x.size):
        if x[i] * x[i] < 1:
            xkern[i] = np.exp(betak * (np.sqrt(1 - x[i] * x[i]) - 1))
        else:
            xkern[i] = 0.0
        if y[i] * y[i] < 1:
            ykern[i] = np.exp(betak * (np.sqrt(1 - y[i] * y[i]) - 1))
        else:
            ykern[i] = 0.0


def compute_counts(dsl, nx, ny, cell_size_x, cell_size_y, tbid=0, nthreads=1):
    """
    Sum the weights on the grid over all datasets
    """

    if isinstance(dsl, str):
        dsl = [dsl]

    dsl = xds_from_list(dsl, nthreads=nthreads, drop_vars=("VIS", "BEAM"))

    nds = len(dsl)
    maxw = np.minimum(nthreads, nds)
    if nthreads > nds:  # this is not perfect
        ngrid = np.maximum(nthreads // nds, 1)
    else:
        ngrid = 1

    counts = np.zeros((nx, ny), dtype=dsl[0].WEIGHT.dtype)
    with cf.ThreadPoolExecutor(max_workers=maxw) as executor:
        futures = []
        for ds in dsl:
            fut = executor.submit(
                _compute_counts,
                ds.UVW.values,
                ds.FREQ.values,
                ds.MASK.values,
                ds.WEIGHT.values,
                nx,
                ny,
                cell_size_x,
                cell_size_y,
                ds.WEIGHT.dtype,
                ngrid=ngrid,
            )
            futures.append(fut)

        for fut in cf.as_completed(futures):
            # sum over number of datasets
            counts += fut.result().sum(axis=0)

    return counts, tbid


@njit(nogil=True, cache=True, parallel=True)
def _compute_counts(
    uvw, freq, mask, wgt, nx, ny, cell_size_x, cell_size_y, dtype, ngrid=1, usign=1.0, vsign=-1.0
):  # support hardcoded for now
    # ufreq
    u_cell = 1 / (nx * cell_size_x)
    # factor of 2 because -umax/2 <= u < umax
    umax = np.abs(1 / cell_size_x / 2)

    # vfreq
    v_cell = 1 / (ny * cell_size_y)
    vmax = np.abs(1 / cell_size_y / 2)

    ncorr, nrow, nchan = wgt.shape

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    counts = np.zeros((ngrid, ncorr, nx, ny), dtype=dtype)

    # accumulate counts
    bin_counts = [nrow // ngrid + (1 if x < nrow % ngrid else 0) for x in range(ngrid)]
    bin_idx = np.zeros(ngrid, dtype=np.int64)
    bin_counts = np.asarray(bin_counts).astype(bin_idx.dtype)
    bin_idx[1:] = np.cumsum(bin_counts)[0:-1]

    for g in prange(ngrid):
        for r in range(bin_idx[g], bin_idx[g] + bin_counts[g]):
            uvw_row = uvw[r]
            wgt_row = wgt[:, r]
            mask_row = mask[r]
            for f in range(nchan):
                if not mask_row[f]:
                    continue
                # current uv coords
                chan_normfreq = freq[f] / lightspeed
                u_tmp = uvw_row[0] * chan_normfreq * usign
                v_tmp = uvw_row[1] * chan_normfreq * vsign
                if v_tmp < 0:
                    u_tmp = -u_tmp
                    v_tmp = -v_tmp
                # pixel coordinates
                ug = (u_tmp + umax) / u_cell
                vg = (v_tmp + vmax) / v_cell
                # indices
                u_idx = np.int32(np.floor(ug))
                v_idx = np.int32(np.floor(vg))

                # corr weights
                wrf = wgt_row[:, f]

                # LB - is there an easier check for this?
                if (u_idx < 0) or (u_idx >= nx) or (v_idx < 0) or (v_idx >= ny):
                    # out of bounds so continue.
                    # raising an error means we can't grid at sub-Nyquist
                    continue

                # nearest neighbour
                counts[g, :, u_idx, v_idx] += wrf

    return counts.sum(axis=0)


@njit(**JIT_OPTIONS)
def counts_to_weights(counts, uvw, freq, weight, mask, nx, ny, cell_size_x, cell_size_y, robust, usign=1.0, vsign=-1.0):
    # when does this happen?
    if not counts.any():
        return weight

    real_type = weight.dtype

    # ufreq
    u_cell = 1 / (nx * cell_size_x)
    umax = np.abs(1 / cell_size_x / 2)

    # vfreq
    v_cell = 1 / (ny * cell_size_y)
    vmax = np.abs(1 / cell_size_y / 2)

    ncorr, nrow, nchan = weight.shape

    # Briggs weighting factor
    if robust > -2:
        numsqrt = 5 * 10 ** (-robust)
        avgwnum = np.zeros(ncorr, dtype=real_type)
        avgwden = np.zeros(ncorr, dtype=real_type)
        for c in range(ncorr):
            for i in range(nx):
                for j in range(ny):
                    cval = counts[c, i, j]
                    avgwnum[c] += cval * cval
                    avgwden[c] += cval
        ssq = numsqrt * numsqrt * avgwden / avgwnum
        counts *= ssq[:, None, None]
        counts += 1

    for r in prange(nrow):
        uvw_row = uvw[r]
        wgt_row = weight[:, r]
        mask_row = mask[r]
        for f in range(nchan):
            if mask_row[f] == 0:
                continue
            # current uv coords
            chan_normfreq = freq[f] / lightspeed
            u_tmp = uvw_row[0] * chan_normfreq * usign
            v_tmp = uvw_row[1] * chan_normfreq * vsign
            if v_tmp < 0:
                u_tmp = -u_tmp
                v_tmp = -v_tmp
            # pixel coordinates
            ug = (u_tmp + umax) / u_cell
            vg = (v_tmp + vmax) / v_cell

            # indices
            u_idx = np.int32(np.floor(ug))
            v_idx = np.int32(np.floor(vg))

            if (u_idx < 0) or (u_idx >= nx) or (v_idx < 0) or (v_idx >= ny):
                # out of bounds so continue.
                # raising an error means we can't grid at sub-Nyquist
                continue

            # counts can be zero if there are zero weights
            for c in range(ncorr):
                if counts[c, u_idx, v_idx] > 0:
                    wgt_row[c, f] /= counts[c, u_idx, v_idx]

    return weight


# @njit(nogil=True, cache=True)
def filter_extreme_counts(counts, level=10.0):
    """
    Replaces extremely small counts by median to prevent
    upweighting nearly empty cells
    """
    # get the median counts value
    if not level:
        return counts
    ic, ix, iy = np.where(counts > 0)
    cnts = counts[ic, ix, iy]
    med = np.median(cnts)
    lowval = med / level
    cnts = np.maximum(cnts, lowval)
    counts[ic, ix, iy] = cnts
    return counts


def box_sum_counts(counts, npix_super):
    """Box-sum the counts grid for super-uniform weighting.

    Replaces each cell of ``counts`` with the sum over a
    ``(2*npix_super+1)^2`` window centred on that cell, with zero-padding at
    image edges. Returns ``counts`` unchanged when ``npix_super`` is ``None``
    or non-positive, which recovers standard uniform weighting.

    Args:
        counts: Array of shape ``(ncorr, nx, ny)``.
        npix_super: Half-size of the box in pixels. ``0`` => standard uniform.

    Returns:
        Array of shape ``(ncorr, nx, ny)`` with box-summed counts.
    """
    if npix_super is None or npix_super <= 0:
        return counts
    from scipy.ndimage import uniform_filter

    assert np.issubdtype(counts.dtype, np.floating), (
        f"box_sum_counts requires a floating-point counts array; got dtype={counts.dtype}"
    )
    size = 2 * npix_super + 1
    out = np.empty_like(counts)
    for c in range(counts.shape[0]):
        out[c] = uniform_filter(counts[c], size=size, mode="constant", cval=0.0) * (size * size)
    return out


@njit(nogil=True, cache=False, parallel=False)
def weight_data(
    data,
    weight,
    flag,
    jones,
    tbin_idx,
    tbin_counts,
    ant1,
    ant2,
    pol,
    product,
    nc,
    wgt_mode,
):
    vis, wgt = _weight_data_impl(
        data,
        weight,
        flag,
        jones,
        tbin_idx,
        tbin_counts,
        ant1,
        ant2,
        literally(pol),
        literally(product),
        literally(nc),
        literally(wgt_mode),
    )

    return vis, wgt


def _weight_data_impl(
    data,
    weight,
    flag,
    jones,
    tbin_idx,
    tbin_counts,
    ant1,
    ant2,
    pol,
    product,
    nc,
    wgt_mode,
):
    raise NotImplementedError


@overload(_weight_data_impl, prefer_literal=True, jit_options={**JIT_OPTIONS, "parallel": True})
def nb_weight_data_impl(
    data,
    weight,
    flag,
    jones,
    tbin_idx,
    tbin_counts,
    ant1,
    ant2,
    pol,
    product,
    nc,
    wgt_mode,
):
    try:
        vis_func, wgt_func = stokes_funcs(data, jones, product, pol, nc, wgt_mode)
    except Exception as e:
        raise numba.core.errors.TypingError(f"Failed in overload resolution: {e}") from e
    ns = len(product.literal_value)

    def _impl(
        data,
        weight,
        flag,
        jones,
        tbin_idx,
        tbin_counts,
        ant1,
        ant2,
        pol,
        product,
        nc,
        wgt_mode,
    ):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        tbin_idx = tbin_idx - tbin_idx.min()
        nt = np.shape(tbin_idx)[0]
        nrow, nchan, ncorr = data.shape
        vis = np.zeros((nrow, nchan, ns), dtype=data.dtype)
        wgt = np.zeros((nrow, nchan, ns), dtype=data.real.dtype)

        for t in prange(nt):
            for row in range(tbin_idx[t], tbin_idx[t] + tbin_counts[t]):
                p = int(ant1[row])
                q = int(ant2[row])
                gp = jones[t, p, :, 0]
                gq = jones[t, q, :, 0]
                for chan in range(nchan):
                    if flag[row, chan].any():
                        continue
                    wgt[row, chan] = wgt_func(gp[chan], gq[chan], weight[row, chan])
                    vis[row, chan] = vis_func(gp[chan], gq[chan], weight[row, chan], data[row, chan])

        return (vis, wgt)

    _impl.returns = types.Tuple([types.Array(types.complex128, 3, "C"), types.Array(types.float64, 3, "C")])

    return _impl


@njit
def weight_data_np(data, weight, flag, jones, tbin_idx, tbin_counts, ant1, ant2, ns, vis_func, wgt_func):
    # for dask arrays we need to adjust the chunks to
    # start counting from zero
    tbin_idx = tbin_idx - tbin_idx.min()
    nt = np.shape(tbin_idx)[0]
    nrow, nchan, ncorr = data.shape
    vis = np.zeros((nrow, nchan, ns), dtype=data.dtype)
    wgt = np.zeros((nrow, nchan, ns), dtype=data.real.dtype)

    for t in prange(nt):
        for row in range(tbin_idx[t], tbin_idx[t] + tbin_counts[t]):
            p = int(ant1[row])
            q = int(ant2[row])
            gp = jones[t, p, :, 0]
            gq = jones[t, q, :, 0]
            for chan in range(nchan):
                if flag[row, chan].any():
                    continue
                wgt[row, chan] = wgt_func(gp[chan], gq[chan], weight[row, chan])
                vis[row, chan] = vis_func(gp[chan], gq[chan], weight[row, chan], data[row, chan])

    return (vis, wgt)
