import numpy as np
from numba import njit, prange
import dask.array as da
from africanus.constants import c as lightspeed


def compute_counts(uvw, freqs, mask, nx, ny,
                   cell_size_x, cell_size_y, dtype, wgt=None):

    if wgt is not None:
        wgt_out = ('row', 'chan')
    else:
        wgt_out = None

    counts = da.blockwise(compute_counts_wrapper, ('row', 'nx', 'ny'),
                          uvw, ('row', 'three'),
                          freqs, ('chan',),
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


def compute_counts_wrapper(uvw, freqs, mask, nx, ny,
                           cell_size_x, cell_size_y, dtype, wgt):
    if wgt is not None:
        wgt = wgt[0]
    return _compute_counts(uvw[0], freqs[0], mask[0], nx, ny,
                           cell_size_x, cell_size_y, dtype, wgt)


@njit(nogil=True, fastmath=True, cache=True)
def _compute_counts(uvw, freqs, mask, nx, ny,
                    cell_size_x, cell_size_y, dtype,
                    wgt=None):
    # ufreqs
    u_cell = 1/(nx*cell_size_x)
    # shifts fftfreqs such that they start at zero
    # convenient to look up the pixel value
    umax = np.abs(-1/cell_size_x/2 - u_cell/2)

    # vfreqs
    v_cell = 1/(ny*cell_size_y)
    vmax = np.abs(-1/cell_size_y/2 - v_cell/2)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    counts = np.zeros((1, nx, ny), dtype=dtype)

    # accumulate counts
    nrow = uvw.shape[0]
    nchan = freqs.size
    if wgt is None:
        wgt = np.ones((nrow, nchan))

    normfreqs = freqs / lightspeed
    for r in range(nrow):
        uvw_row = uvw[r]
        for c in range(nchan):
            if not mask[r, c]:
                continue
            # get current uv coords
            chan_normfreq = normfreqs[c]
            u_tmp = uvw_row[0] * chan_normfreq
            v_tmp = uvw_row[1] * chan_normfreq
            # get u index
            u_idx = int(np.floor((u_tmp + umax)/u_cell))
            # get v index
            v_idx = int(np.floor((v_tmp + vmax)/v_cell))
            counts[0, u_idx, v_idx] += wgt[r, c]
    return counts


def counts_to_weights(counts, uvw, freqs, nx, ny,
                      cell_size_x, cell_size_y, robust):

    weights = da.blockwise(counts_to_weights_wrapper, ('row', 'chan'),
                           counts, ('nx', 'ny'),
                           uvw, ('row', 'three'),
                           freqs, ('chan',),
                           nx, None,
                           ny, None,
                           cell_size_x, None,
                           cell_size_y, None,
                           robust, None,
                           dtype=counts.dtype)
    return weights

def counts_to_weights_wrapper(counts, uvw, freqs, nx, ny,
                              cell_size_x, cell_size_y, robust):
    return _counts_to_weights(counts[0][0], uvw[0], freqs, nx, ny,
                              cell_size_x, cell_size_y, robust)


@njit(nogil=True, fastmath=True, cache=True)
def _counts_to_weights(counts, uvw, freqs, nx, ny,
                       cell_size_x, cell_size_y, robust):
    # ufreqs
    u_cell = 1/(nx*cell_size_x)
    umax = np.abs(-1/cell_size_x/2 - u_cell/2)

    # vfreqs
    v_cell = 1/(ny*cell_size_y)
    vmax = np.abs(-1/cell_size_y/2 - v_cell/2)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    nchan = freqs.size
    nrow = uvw.shape[0]

    # Briggs weighting factor
    if robust > -2:
        numsqrt = 5*10**(-robust)
        avgW = (counts ** 2).sum() / counts.sum()
        ssq = numsqrt * numsqrt/avgW
        counts = 1 + counts * ssq

    normfreqs = freqs / lightspeed
    weights = np.zeros((nrow, nchan), dtype=counts.dtype)
    for r in range(nrow):
        uvw_row = uvw[r]
        for c in range(nchan):
            # get current uv
            chan_normfreq = normfreqs[c]
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
            local_mean = np.mean(cb[ilow:ihigh, jlow:jhigh])
            if 1/cb[i,j] > level/local_mean:
                counts[b, i, j] = local_mean
    return counts

