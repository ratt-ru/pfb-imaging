import concurrent.futures as cf
import numpy as np
from numba import njit, prange, literally, types
from numba.extending import overload
from africanus.constants import c as lightspeed
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
                    k=6, ngrid=1, usign=1.0, vsign=-1.0):  # support hardcoded for now
    # ufreq
    u_cell = 1/(nx*cell_size_x)
    # shifts fftfreq such that they start at zero
    # convenient to look up the pixel value
    umax = np.abs(-1/cell_size_x/2 - u_cell/2)

    # vfreq
    v_cell = 1/(ny*cell_size_y)
    vmax = np.abs(-1/cell_size_y/2 - v_cell/2)

    # are we always passing in wgt? 
    nrow, nchan, ncorr = wgt.shape

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    counts = np.zeros((ngrid, nx, ny, ncorr), dtype=dtype)

    # accumulate counts
    bin_counts = [nrow // ngrid + (1 if x < nrow % ngrid else 0)  for x in range (ngrid)]
    bin_idx = np.zeros(ngrid, dtype=np.int64)
    bin_counts = np.asarray(bin_counts).astype(bin_idx.dtype)
    bin_idx[1:] = np.cumsum(bin_counts)[0:-1]

    normfreq = freq / lightspeed
    ko2 = k//2

    for g in prange(ngrid):
        for r in range(bin_idx[g], bin_idx[g] + bin_counts[g]):
            uvw_row = uvw[r]
            wgt_row = wgt[r]
            mask_row = mask[r]
            for f in range(nchan):
                if not mask_row[f]:
                    continue
                # current uv coords
                chan_normfreq = normfreq[f]
                u_tmp = uvw_row[0] * chan_normfreq * usign
                v_tmp = uvw_row[1] * chan_normfreq * vsign
                # pixel coordinates
                ug = (u_tmp + umax)/u_cell
                vg = (v_tmp + vmax)/v_cell
                # indices
                u_idx = int(np.round(ug))
                v_idx = int(np.round(vg))
                # per correlation weights
                wrf = wgt_row[f]
                for i in range(k):
                    idx = i - ko2
                    x_idx = idx + u_idx
                    x = x_idx - ug + 0.5
                    xkern = _es_kernel(x/ko2, 2.3, k)
                    for j in range(k):
                        jdx = j - ko2
                        y_idx = jdx + v_idx
                        y = y_idx - vg + 0.5
                        ykern = _es_kernel(y/ko2, 2.3, k)
                        for c in range(ncorr):
                            counts[g, x_idx, y_idx, c] += xkern * ykern * wrf[c]
    return counts.sum(axis=0)


@njit(nogil=True, cache=True, inline='always')
def _es_kernel(x, beta, k):
    return np.exp(beta*k*(np.sqrt((1-x)*(1+x)) - 1))


@njit(nogil=True, cache=True, parallel=True)
def counts_to_weights(counts, uvw, freq, weight, nx, ny,
                      cell_size_x, cell_size_y, robust,
                      usign=1.0, vsign=-1.0):
    # when does this happen?
    if not counts.any():
        return weight
    
    # ufreq
    u_cell = 1/(nx*cell_size_x)
    umax = np.abs(-1/cell_size_x/2 - u_cell/2)

    # vfreq
    v_cell = 1/(ny*cell_size_y)
    vmax = np.abs(-1/cell_size_y/2 - v_cell/2)

    nrow, nchan, ncorr = weight.shape
    
    # Briggs weighting factor
    if robust > -2:
        numsqrt = 5*10**(-robust)
        avgW = (counts ** 2).sum() / counts.sum()
        ssq = numsqrt * numsqrt/avgW
        counts = 1 + counts * ssq

    normfreq = freq / lightspeed
    for r in prange(nrow):
        uvw_row = uvw[r]
        weight_row = weight[r]
        for f in range(nchan):
            # get current uv
            chan_normfreq = normfreq[f]
            u_tmp = uvw_row[0] * chan_normfreq * usign
            v_tmp = uvw_row[1] * chan_normfreq * vsign
            # get u index
            u_idx = int(np.floor((u_tmp + umax)/u_cell))
            # get v index
            v_idx = int(np.floor((v_tmp + vmax)/v_cell))
            for c in range(ncorr):
                if counts[u_idx, v_idx, c]:
                    weight_row[f, c] = weight_row[f, c]/counts[u_idx, v_idx, c]
    return weight



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


@njit(nogil=True, cache=False, parallel=False)
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


@overload(_weight_data_impl, prefer_literal=True,
          jit_options={**JIT_OPTIONS, "parallel":True})
def nb_weight_data_impl(data, weight, flag, jones, tbin_idx, tbin_counts,
                      ant1, ant2, pol, product, nc):

    vis_func, wgt_func = stokes_funcs(data, jones, product, pol, nc)

    if product.literal_value in ['I','Q','U','V']:
        ns = 1
    elif product.literal_value == 'DS':
        ns = 2
    elif product.literal_value == 'FS':
        ns = int(nc.literal_value)

    def _impl(data, weight, flag, jones, tbin_idx, tbin_counts,
              ant1, ant2, pol, product, nc):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        tbin_idx -= tbin_idx.min()
        nt = np.shape(tbin_idx)[0]
        nrow, nchan, ncorr = data.shape
        vis = np.zeros((nrow, nchan, ns), dtype=data.dtype)
        wgt = np.zeros((nrow, nchan, ns), dtype=data.real.dtype)

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

        return (vis, wgt)
    
    _impl.returns = types.Tuple([types.Array(types.complex128, 3, 'C'),
                                 types.Array(types.float64, 3, 'C')])
    return _impl
