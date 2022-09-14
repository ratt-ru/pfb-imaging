import numpy as np
from numba import njit, prange
import dask.array as da
from ducc0.wgridder.experimental import vis2dirty
from ducc0.fft import c2c, genuine_hartley
from africanus.constants import c as lightspeed
from skimage.measure import block_reduce
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def compute_counts(uvw, freq, mask, nx, ny,
                   cell_size_x, cell_size_y, dtype, wgt=None):

    if wgt is not None:
        wgt_out = ('row', 'chan')
    else:
        wgt_out = None

    counts = da.blockwise(compute_counts_wrapper, ('row', 'nx', 'ny'),
                          uvw, ('row', 'three'),
                          freq, ('chan',),
                          mask, ('row', 'chan'),
                          nx, None,
                          ny, None,
                          cell_size_x, None,
                          cell_size_y, None,
                          dtype, None,
                          wgt, wgt_out,
                          new_axes={"nx": nx, "ny": ny},
                          adjust_chunks={'row': 1},
                          align_arrays=False,
                          dtype=dtype)

    return counts.sum(axis=0)


def compute_counts_wrapper(uvw, freq, mask, nx, ny,
                           cell_size_x, cell_size_y, dtype, wgt):
    if wgt is not None:
        wgt = wgt[0]
    return _compute_counts(uvw[0], freq[0], mask[0], nx, ny,
                        cell_size_x, cell_size_y, dtype, wgt)



@njit(nogil=True, fastmath=True, cache=True)
def _compute_counts(uvw, freq, mask, nx, ny,
                    cell_size_x, cell_size_y, dtype,
                    wgt=None, k=6):  # support hardcoded for now
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
    counts = np.zeros((1, nx, ny), dtype=dtype)

    # accumulate counts
    nrow = uvw.shape[0]
    nchan = freq.size
    if wgt is None:
        wgt = np.ones((nrow, nchan))

    normfreq = freq / lightspeed
    ko2 = k//2
    ko2sq = ko2**2
    for r in range(nrow):
        uvw_row = uvw[r]
        wgt_row = wgt[r]
        for c in range(nchan):
            if not mask[r, c]:
                continue
            wgt_row_chan = wgt_row[c]
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
                    val = _es_kernel(x/ko2, 2.3, k) * wgt_row_chan
                    for j in range(-ko2, ko2):
                        y_idx = j + v_idx
                        y = y_idx - vg + 0.5
                        counts[0, x_idx, y_idx] += val * _es_kernel(y/ko2, 2.3, k)
            else:  # nearest neighbour
                # indices
                u_idx = int(np.floor(ug))
                v_idx = int(np.floor(vg))
                counts[0, u_idx, v_idx] += wgt_row_chan
    return counts

@njit(nogil=True, fastmath=True, cache=True, inline='always')
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


@njit(nogil=True, fastmath=True, cache=True)
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

    # Briggs weighting factor
    if robust > -2:
        numsqrt = 5*10**(-robust)
        avgW = (counts ** 2).sum() / counts.sum()
        ssq = numsqrt * numsqrt/avgW
        counts = 1 + counts * ssq

    normfreq = freq / lightspeed
    weights = np.zeros((nrow, nchan), dtype=counts.dtype)
    for r in range(nrow):
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

    return da.blockwise(_filter_extreme_counts, 'bxy',
                        counts, 'bxy',
                        nbox, None,
                        nlevel, None,
                        dtype=counts.dtype)



@njit(nogil=True, fastmath=True, cache=True)
def _filter_extreme_counts(counts, nbox=16, level=10):
    '''
    Replaces extreme counts by local mean computed i
    '''
    nband, nx, ny = counts.shape
    for b in range(nband):
        cb = counts[b]
        I, J = np.where(cb>0)
        for i, j in zip(I, J):
            ilow = np.maximum(0, i-nbox//2)
            ihigh = np.minimum(nx, i+nbox//2)
            jlow = np.maximum(0, j-nbox//2)
            jhigh = np.minimum(ny, j+nbox//2)
            tmp = cb[ilow:ihigh, jlow:jhigh]
            ix, iy = np.where(tmp)
            # check if there are too few values to compare to
            if ix.size < nbox:
                counts[b, i, j] = 0
                continue
            local_mean = np.mean(tmp[ix, iy])
            if cb[i,j] < local_mean/level:
                counts[b, i, j] = local_mean
    return counts


