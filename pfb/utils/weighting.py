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
    umax = 1/cell_size_x/2
    u_diff = 1/((nx+1)*cell_size_x)  # +1 for bin edges

    # vfreqs
    vmax = 1/cell_size_y/2
    v_diff = 1/((ny+1)*cell_size_y)

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
            # get u index (-1 because we start counting at zero)
            u_idx = int(np.round((u_tmp + umax)/u_diff))-1
            # get v index
            v_idx = int(np.round((v_tmp + vmax)/v_diff))-1
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
    umax = 1/cell_size_x/2
    u_diff = 1/((nx+1)*cell_size_x)  # +1 for bin edges

    # vfreqs
    vmax = 1/cell_size_y/2
    v_diff = 1/((ny+1)*cell_size_y)

    # initialise array to store counts
    # the additional axis is to allow chunking over ro
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
            # get u index (-1 because we start counting at zero)
            u_idx = int(np.round((u_tmp + umax)/u_diff)) - 1
            # get v index
            v_idx = int(np.round((v_tmp + vmax)/v_diff)) - 1
            if counts[u_idx, v_idx]:
                weights[r, c] = 1.0/counts[u_idx, v_idx]
    return weights
