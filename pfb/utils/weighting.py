import concurrent.futures as cf
import numpy as np
from numba import njit, prange, literally
from numba.extending import overload
import dask.array as da
from ducc0.fft import c2c
from africanus.constants import c as lightspeed
from quartical.utils.dask import Blocker
from pfb.utils.misc import JIT_OPTIONS
from pfb.utils.stokes import stokes_funcs
from pfb.utils.naming import xds_from_list
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def compute_counts(dsl,
                   nx, ny,
                   cell_size_x, cell_size_y,
                   tbid=0,
                   k=6,
                   nthreads=1):
    '''
    Sum the weights on the grid over all datasets
    '''

    if isinstance(dsl, str):
        dsl = [dsl]

    dsl = xds_from_list(dsl, nthreads=nthreads,
                        drop_vars=('VIS','BEAM'))

    nds = len(dsl)
    maxw = np.minimum(nthreads, nds)
    if nthreads > nds:  # this is not perfect
        ngrid = np.maximum(nthreads//nds, 1)
    else:
        ngrid = 1

    counts = np.zeros((nx, ny), dtype=dsl[0].WEIGHT.dtype)
    with cf.ThreadPoolExecutor(max_workers=maxw) as executor:
        futures = []
        for ds in dsl:
            fut = executor.submit(_compute_counts,
                                  ds.UVW.values,
                                  ds.FREQ.values,
                                  ds.MASK.values,
                                  ds.WEIGHT.values,
                                  nx, ny,
                                  cell_size_x,
                                  cell_size_y,
                                  ds.WEIGHT.dtype,
                                  k=6,
                                  ngrid=ngrid)
            futures.append(fut)

        for fut in cf.as_completed(futures):
            # sum over number of datasets
            counts += fut.result().sum(axis=0)

    return counts, tbid



@njit(nogil=True, cache=True, parallel=True)
def _compute_counts(uvw, freq, mask, wgt, nx, ny,
                    cell_size_x, cell_size_y, dtype,
                    k=6, ngrid=1):  # support hardcoded for now
    # ufreq
    u_cell = 1/(nx*cell_size_x)
    # shifts fftfreq such that they start at zero
    # convenient to look up the pixel value
    umax = np.abs(-1/cell_size_x/2 - u_cell/2)

    # vfreq
    v_cell = 1/(ny*cell_size_y)
    vmax = np.abs(-1/cell_size_y/2 - v_cell/2)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    counts = np.zeros((ngrid, nx, ny), dtype=dtype)

    # accumulate counts
    nrow = uvw.shape[0]
    nchan = freq.size
    bin_counts = [nrow // ngrid + (1 if x < nrow % ngrid else 0)  for x in range (ngrid)]
    bin_idx = np.zeros(ngrid, dtype=np.int64)
    bin_counts = np.asarray(bin_counts).astype(bin_idx.dtype)
    bin_idx[1:] = np.cumsum(bin_counts)[0:-1]

    normfreq = freq / lightspeed
    ko2 = k//2
    ko2sq = ko2**2

    if wgt is None:
        # this should be a small array
        # wgt = np.broadcast_to(np.ones((1,), dtype=dtype), (nrow, nchan))
        wgt = np.ones((nrow, nchan), dtype=uvw.dtype)

    for g in prange(ngrid):
        for r in range(bin_idx[g], bin_idx[g] + bin_counts[g]):
            uvw_row = uvw[r]
            wgt_row = wgt[r]
            for c in range(nchan):
                if not mask[r, c]:
                    continue
                # current uv coords
                chan_normfreq = normfreq[c]
                u_tmp = uvw_row[0] * chan_normfreq
                v_tmp = uvw_row[1] * chan_normfreq
                # pixel coordinates
                ug = (u_tmp + umax)/u_cell
                vg = (v_tmp + vmax)/v_cell
                wrc = wgt_row[c]
                if k:
                    # indices
                    u_idx = int(np.round(ug))
                    v_idx = int(np.round(vg))
                    for i in range(-ko2, ko2):
                        x_idx = i + u_idx
                        x = x_idx - ug + 0.5
                        val = _es_kernel(x/ko2, 2.3, k) * wrc
                        for j in range(-ko2, ko2):
                            y_idx = j + v_idx
                            y = y_idx - vg + 0.5
                            counts[g, x_idx, y_idx] += val * _es_kernel(y/ko2, 2.3, k)
                else:  # nearest neighbour
                    # indices
                    u_idx = int(np.floor(ug))
                    v_idx = int(np.floor(vg))
                    counts[g, u_idx, v_idx] += 1.0
    return counts.sum(axis=0)


@njit(nogil=True, cache=True, inline='always')
def _es_kernel(x, beta, k):
    return np.exp(beta*k*(np.sqrt((1-x)*(1+x)) - 1))


@njit(nogil=True, cache=True, parallel=True)
def counts_to_weights(counts, uvw, freq, nx, ny,
                       cell_size_x, cell_size_y, robust):
    # ufreq
    u_cell = 1/(nx*cell_size_x)
    umax = np.abs(-1/cell_size_x/2 - u_cell/2)

    # vfreq
    v_cell = 1/(ny*cell_size_y)
    vmax = np.abs(-1/cell_size_y/2 - v_cell/2)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    nchan = freq.size
    nrow = uvw.shape[0]

    weights = np.zeros((nrow, nchan), dtype=counts.dtype)
    if not counts.any():
        return weights

    # Briggs weighting factor
    if robust > -2:
        numsqrt = 5*10**(-robust)
        avgW = (counts ** 2).sum() / counts.sum()
        ssq = numsqrt * numsqrt/avgW
        counts = 1 + counts * ssq

    normfreq = freq / lightspeed
    for r in prange(nrow):
        uvw_row = uvw[r]
        for c in range(nchan):
            # get current uv
            chan_normfreq = normfreq[c]
            u_tmp = uvw_row[0] * chan_normfreq
            v_tmp = uvw_row[1] * chan_normfreq
            # get u index
            u_idx = int(np.floor((u_tmp + umax)/u_cell))
            # get v index
            v_idx = int(np.floor((v_tmp + vmax)/v_cell))
            if counts[u_idx, v_idx]:
                weights[r, c] = 1.0/counts[u_idx, v_idx]
    return weights



# @njit(nogil=True, cache=True)
def filter_extreme_counts(counts, level=10.0):
    '''
    Replaces extremely small counts by median to prevent
    upweighting nearly empty cells
    '''
    # get the median counts value
    ix, iy = np.where(counts > 0)
    cnts = counts[ix,iy]
    med = np.median(cnts)
    lowval = med/level
    cnts = np.maximum(cnts, lowval)
    counts[ix,iy] = cnts
    return counts


@njit(**JIT_OPTIONS, parallel=True)
def weight_data(data, weight, flag, jones, tbin_idx, tbin_counts,
                ant1, ant2, pol, product, nc):

    vis, wgt = _weight_data_impl(data, weight, flag, jones,
                                 tbin_idx, tbin_counts,
                                 ant1, ant2,
                                 literally(pol),
                                 literally(product),
                                 literally(nc))

    return vis, wgt


def _weight_data_impl(data, weight, flag, jones, tbin_idx, tbin_counts,
                 ant1, ant2, pol, product, nc):
    raise NotImplementedError


@overload(_weight_data_impl, **JIT_OPTIONS, parallel=True, prefer_literal=True)
def nb_weight_data_impl(data, weight, flag, jones, tbin_idx, tbin_counts,
                      ant1, ant2, pol, product, nc):

    vis_func, wgt_func = stokes_funcs(data, jones, product, pol, nc)

    def _impl(data, weight, flag, jones, tbin_idx, tbin_counts,
              ant1, ant2, pol, product, nc):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        tbin_idx -= tbin_idx.min()
        nt = np.shape(tbin_idx)[0]
        nrow, nchan, ncorr = data.shape
        vis = np.zeros((nrow, nchan), dtype=data.dtype)
        wgt = np.zeros((nrow, nchan), dtype=data.real.dtype)

        for t in prange(nt):
            for row in range(tbin_idx[t],
                             tbin_idx[t] + tbin_counts[t]):
                p = int(ant1[row])
                q = int(ant2[row])
                gp = jones[t, p, :, 0]
                gq = jones[t, q, :, 0]
                for chan in range(nchan):
                    if flag[row, chan]:
                        continue
                    wgt[row, chan] = wgt_func(gp[chan], gq[chan],
                                              weight[row, chan])
                    vis[row, chan] = vis_func(gp[chan], gq[chan],
                                              weight[row, chan],
                                              data[row, chan])

        return vis, wgt
    return _impl
