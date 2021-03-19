import numpy as np
from numba import njit, prange
import dask.array as da
from africanus.constants import c as lightspeed


def compute_counts(uvw, freqs, fbin_idx, fbin_counts, nx, ny,
                   cell_size_x, cell_size_y, dtype):
    counts = da.blockwise(compute_counts_wrapper, ('one', 'chan', 'nx', 'ny'),
                          uvw, ('row', 'three'),
                          freqs, ('chan',),
                          fbin_idx, ('chan',),
                          fbin_counts, ('chan',),
                          nx, None,
                          ny, None,
                          cell_size_x, None,
                          cell_size_y, None,
                          dtype, None,
                          new_axes={"one": 1, "nx": nx, "ny": ny},
                          adjust_chunks={'chan': fbin_idx.chunks[0],
                                         'row': (1,)*len(uvw.chunks[0])},
                          align_arrays=False,
                          dtype=dtype)

    return counts.sum(axis=0)


def compute_counts_wrapper(uvw, freqs, fbin_idx, fbin_counts, nx, ny,
                           cell_size_x, cell_size_y, dtype):
    return _compute_counts(uvw[0][0], freqs, fbin_idx, fbin_counts,
                           nx, ny, cell_size_x, cell_size_y, dtype)


@njit(nogil=True, fastmath=True, cache=True)
def _compute_counts(uvw, freqs, fbin_idx, fbin_counts, nx, ny,
                    cell_size_x, cell_size_y, dtype):
    # ufreqs
    umax = 1/cell_size_x/2
    u_diff = 1/(nx*cell_size_x)

    # vfreqs
    vmax = 1/cell_size_y/2
    v_diff = 1/(ny*cell_size_y)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    nband = fbin_idx.size
    counts = np.zeros((1, nband, nx, ny), dtype=dtype)

    # accumulate counts
    nrow = uvw.shape[0]
    normfreqs = freqs / lightspeed
    # adjust for chunking (need a copy here if using multiple row chunks)
    fbin_idx2 = fbin_idx - fbin_idx.min()
    for r in range(nrow):
        uvw_row = uvw[r]
        for b in range(nband):
            for c in range(fbin_idx2[b], fbin_idx2[b] + fbin_counts[b]):
                # get current uv coords
                chan_normfreq = normfreqs[c]
                u_tmp = uvw_row[0] * chan_normfreq
                v_tmp = uvw_row[1] * chan_normfreq
               # get u index
                u_idx = int(np.round((u_tmp + umax)/u_diff))
                # get v index
                v_idx = int(np.round((v_tmp + vmax)/v_diff))
                counts[0, b, u_idx, v_idx] += 1
    return counts


def counts_to_weights(counts, uvw, freqs, fbin_idx, fbin_counts, nx, ny,
                      cell_size_x, cell_size_y, dtype, robust):

    weights = da.blockwise(counts_to_weights_wrapper, ('row', 'chan'),
                           counts, ('chan', 'nx', 'ny'),
                           uvw, ('row', 'three'),
                           freqs, ('chan',),
                           fbin_idx, ('chan',),
                           fbin_counts, ('chan',),
                           nx, None,
                           ny, None,
                           cell_size_x, None,
                           cell_size_y, None,
                           dtype, None,
                           robust, None,
                           adjust_chunks={'chan': freqs.chunks[0]},
                           align_arrays=False,
                           dtype=dtype)
    return weights


def counts_to_weights_wrapper(counts, uvw, freqs, fbin_idx, fbin_counts,
                              nx, ny, cell_size_x, cell_size_y, dtype,
                              robust):
    return _counts_to_weights(counts[0][0], uvw[0], freqs, fbin_idx,
                              fbin_counts, nx, ny, cell_size_x, cell_size_y,
                              dtype, robust)


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _counts_to_weights(counts, uvw, freqs, fbin_idx, fbin_counts, nx, ny,
                       cell_size_x, cell_size_y, dtype, robust):
    # ufreqs
    # ug = np.fft.fftshift(np.fft.fftfreq(nx, d=cell_size_x))
    umax = 1/cell_size_x/2
    u_diff = 1/(nx*cell_size_x)

    # vfreqs
    # vg = np.fft.fftshift(np.fft.fftfreq(ny, d=cell_size_y))
    vmax = 1/cell_size_y/2
    v_diff = 1/(ny*cell_size_y)

    # initialise array to store counts
    # the additional axis is to allow chunking over row
    nband = fbin_idx.size
    # accumulate counts
    nchan = freqs.size
    nrow = uvw.shape[0]

    if robust is not None:
        numsqrt = 5*10**(-robust)
        for b in range(nband):
            counts_band = counts[b]
            avgW = (counts_band ** 2).sum() / counts_band.sum()
            ssq = numsqrt**2/avgW
            counts_band[...] = 1 + counts_band * ssq

    normfreqs = freqs / lightspeed

    # adjust for chunking
    # need a copy here if using multiple row chunks
    fbin_idx2 = fbin_idx - fbin_idx.min()

    weights = np.zeros((nrow, nchan), dtype=dtype)
    for r in prange(nrow):
        uvw_row = uvw[r]
        for b in range(nband):
            for c in range(fbin_idx2[b], fbin_idx2[b] + fbin_counts[b]):
                # get current uv
                chan_normfreq = normfreqs[c]
                u_tmp = uvw_row[0] * chan_normfreq
                v_tmp = uvw_row[1] * chan_normfreq
                # get u index
                u_idx = int(np.round((u_tmp + umax)/u_diff))
                # get v index
                v_idx = int(np.round((v_tmp + vmax)/v_diff))
                if counts[b, u_idx, v_idx]:
                    weights[r, c] = 1.0/counts[b, u_idx, v_idx]
    return weights

# def robust_reweight(residuals, weights, v=None):
#     """
#     Find the robust weights corresponding to a soln that generated residuals

#     residuals   - residuals i.e. (data - model) (nrow, nchan, ncorr)
#     weights     - inverse of data covariance (nrow, nchan, ncorr)
#     v           - initial guess for degrees for freedom parameter (float)
#     corrs       - which correlation axes to compute new weights for
#                   Default is for LL and RR (or XX and YY)

#     Correlation axis not currently supported
#     """
#     # elements of Mahalanobis distance (Delta^2_i's)
#     nrow, nchan = residuals.shape
#     ressq = (residuals.conj()*residuals).real

#     # func to solve for degrees of freedom parameter
#     def func(v, N, ressq):
#         # expectation values
#         tmp = v+ressq
#         Etau = (v + 1)/tmp
#         Elogtau = digamma(v + 1) - np.log(tmp)

#         # derivatives of expectation value terms
#         dEtau = 1.0/tmp - (v+1)/tmp**2
#         dElogtau = polygamma(1, v + 1) - 1.0/(v + ressq)
#         return (N*np.log(v) - N*digamma(v) + np.sum(Elogtau) - np.sum(Etau),
#                 N/v - N*polygamma(1, v) + np.sum(dElogtau) - np.sum(dEtau)


#     v, f, d = fmin_l_bfgs_b(func, v, args=(nrow, ressq), pgtol=1e-2,
#                             approx_grad=False, bounds=[(1e-3, 30)])
#     Etau = (v + 1.0)/(v + ressq)  # used as new weights
#     Lambda = np.mean(ressq*Etau, axis=0)
#     return v, np.sqrt(Etau / Lambda[None, :])
