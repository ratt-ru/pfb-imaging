import numpy as np
from numba import njit, prange, literally
from numba.extending import overload
import dask.array as da
from ducc0.fft import c2c
from africanus.constants import c as lightspeed
from quartical.utils.dask import Blocker
from pfb.utils.misc import JIT_OPTIONS
from pfb.utils.stokes import stokes_funcs
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def compute_counts(uvw, freq, mask, nx, ny,
                   cell_size_x, cell_size_y, dtype, k=6, ngrid=1):

    counts = da.blockwise(compute_counts_wrapper, ('row', 'nx', 'ny'),
                          uvw, ('row', 'three'),
                          freq, ('chan',),
                          mask, ('row', 'chan'),
                          nx, None,
                          ny, None,
                          cell_size_x, None,
                          cell_size_y, None,
                          dtype, None,
                          k, None,
                          ngrid, None,
                          new_axes={"nx": nx, "ny": ny},
                          adjust_chunks={'row': ngrid},
                          align_arrays=False,
                          dtype=dtype)

    return counts.sum(axis=0)


def compute_counts_wrapper(uvw, freq, mask, nx, ny,
                           cell_size_x, cell_size_y, dtype, k, ngrid):
    return _compute_counts(uvw[0], freq[0], mask[0], nx, ny,
                        cell_size_x, cell_size_y, dtype, k, ngrid)



@njit(nogil=True, cache=True, parallel=True)
def _compute_counts(uvw, freq, mask, nx, ny,
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

    for g in prange(ngrid):
        for r in range(bin_idx[g], bin_idx[g] + bin_counts[g]):
            uvw_row = uvw[r]
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
                if k:
                    # indices
                    u_idx = int(np.round(ug))
                    v_idx = int(np.round(vg))
                    for i in range(-ko2, ko2):
                        x_idx = i + u_idx
                        x = x_idx - ug + 0.5
                        val = _es_kernel(x/ko2, 2.3, k)
                        for j in range(-ko2, ko2):
                            y_idx = j + v_idx
                            y = y_idx - vg + 0.5
                            counts[g, x_idx, y_idx] += val * _es_kernel(y/ko2, 2.3, k)
                else:  # nearest neighbour
                    # indices
                    u_idx = int(np.floor(ug))
                    v_idx = int(np.floor(vg))
                    counts[g, u_idx, v_idx] += 1.0
    return counts  #.sum(axis=0, keepdims=True)

@njit(nogil=True, cache=True, inline='always')
def _es_kernel(x, beta, k):
    return np.exp(beta*k*(np.sqrt((1-x)*(1+x)) - 1))

def counts_to_weights(counts, uvw, freq, nx, ny,
                      cell_size_x, cell_size_y, robust):

    weights = da.blockwise(counts_to_weights_wrapper, ('row', 'chan'),
                           counts, ('nx', 'ny'),
                           uvw, ('row', 'three'),
                           freq, ('chan',),
                           nx, None,
                           ny, None,
                           cell_size_x, None,
                           cell_size_y, None,
                           robust, None,
                           dtype=counts.dtype)
    return weights

def counts_to_weights_wrapper(counts, uvw, freq, nx, ny,
                              cell_size_x, cell_size_y, robust):
    return _counts_to_weights(counts[0][0], uvw[0], freq, nx, ny,
                              cell_size_x, cell_size_y, robust)


@njit(nogil=True, cache=True)
def _counts_to_weights(counts, uvw, freq, nx, ny,
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


def filter_extreme_counts(counts, nbox=16, nlevel=10):

    return da.blockwise(_filter_extreme_counts, 'xy',
                        counts, 'xy',
                        nbox, None,
                        nlevel, None,
                        dtype=counts.dtype,
                        meta=np.empty((0,0), dtype=float))



# @njit(nogil=True, cache=True)
def _filter_extreme_counts(counts, nbox=16, level=10.0):
    '''
    Replaces extremely small counts by median to prevent
    upweighting nearly empty cells
    '''
    # nx, ny = counts.shape
    # I, J = np.where(counts>0)
    # for i, j in zip(I, J):
    #     ilow = np.maximum(0, i-nbox//2)
    #     ihigh = np.minimum(nx, i+nbox//2)
    #     jlow = np.maximum(0, j-nbox//2)
    #     jhigh = np.minimum(ny, j+nbox//2)
    #     tmp = counts[ilow:ihigh, jlow:jhigh]
    #     ix, iy = np.where(tmp)
    #     # check if there are too few values to compare to
    #     if ix.size < nbox:
    #         counts[i, j] = 0
    #         continue
    #     print(ix, iy)
    #     local_mean = np.mean(tmp[ix, iy])
    #     if counts[i,j] < local_mean/level:
    #         counts[i, j] = local_mean
    # get the median counts value
    ix, iy = np.where(counts > 0)
    cnts = counts[ix,iy]
    med = np.median(cnts)
    lowval = med/level
    cnts = np.maximum(cnts, lowval)
    counts[ix,iy] = cnts
    return counts


# from scipy.special import polygamma
# import dask
# @dask.delayed()
# def etaf(ressq, ovar, dof):
#     eta = (dof + 1)/(dof + ressq/ovar)
#     logeta = polygamma(0, dof+1) - np.log(dof + ressq/ovar)
#     return eta, logeta


# from pfb.operators.gridder import im2vis
# def l2reweight(dsv, dsi, epsilon, nthreads, do_wgridding, precision, dof=2):
#     # vis data products
#     uvw = dsv.UVW.data
#     freq = dsv.FREQ.data
#     vis = dsv.VIS.data
#     vis_mask = dsv.MASK.data

#     # image data products
#     model = dsi.MODEL.data
#     cell_rad = dsi.cell_rad
#     x0 = dsi.x0
#     y0 = dsi.y0
#     beam = dsi.BEAM.data

#     # residual from model visibilities
#     mvis = im2vis(uvw=uvw,
#                     freq=freq,
#                     image=model*beam,
#                     cellx=cell_rad,
#                     celly=cell_rad,
#                     nthreads=nthreads,
#                     epsilon=epsilon,
#                     do_wgridding=do_wgridding,
#                     x0=x0,
#                     y0=y0,
#                     precision=precision)
#     res = vis - mvis
#     res *= vis_mask

#     # Mahalanobis distance
#     ressq = (res*res.conj()).real

#     # overall variance factor
#     eta = da.map_blocks(update_eta,
#                         ressq,
#                         vis_mask,
#                         dof,
#                         chunks=ressq.chunks)

#     return eta, res


# def update_eta(ressq, vis_mask, dof):
#     wcount = vis_mask.sum()
#     if wcount:
#         ovar = ressq.sum()/wcount
#         eta = (dof + 1)/(dof + ressq/ovar)
#         return eta/ovar
#     else:
#         return np.zeros_like(ressq)



def weight_data(data, weight, flag, jones, tbin_idx, tbin_counts,
                ant1, ant2, pol, product, nc):

    vis, wgt = _weight_data(data, weight, flag, jones,
                                 tbin_idx, tbin_counts,
                                 ant1, ant2,
                                 literally(pol),
                                 literally(product),
                                 literally(nc))

    out_dict = {}
    out_dict['vis'] = vis
    out_dict['wgt'] = wgt

    return out_dict


@njit(**JIT_OPTIONS, parallel=True)
def _weight_data(data, weight, flag, jones, tbin_idx, tbin_counts,
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


@overload(_weight_data_impl, **JIT_OPTIONS, parallel=True)
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
