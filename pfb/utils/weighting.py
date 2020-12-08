import numpy as np
from numba import jit, prange


@njit(nogil=True, fastmath=True, cache=True)
def compute_wsums_band(uvw, weights, freqs, nx, ny, cell_size_x, cell_size_y, dtype):
    # get u coordinates of the grid 
    umax = 1.0/cell_size_x
    # get end points
    ug = np.linspace(-umax, umax, nx + 1)[1::]
    # get v coordinates of the grid
    vmax = 1.0/cell_size_y
    # get end points
    vg = np.linspace(-vmax, vmax, ny + 1)[1::]
    # initialise array to store counts
    counts = np.zeros((nx, ny), dtype=dtype)
    # accumulate counts
    nchan = freqs.size
    nrow = uvw.shape[0]
    for r in range(nrow):
        for c in range(nchan):
            # get current uv coords
            u_tmp, v_tmp = uvw[r, 0:2] * freqs[c] / lightspeed
            # get u index
            u_idx = (ug < u_tmp).nonzero()[0][-1]
            # get v index
            v_idx = (vg < v_tmp).nonzero()[0][-1]
            counts[u_idx, v_idx] += weights[r, c]
    return counts

@njit(nogil=True, fastmath=True, cache=True)
def wsums_to_weights_band(wsums, uvw, freqs, nx, ny, cell_size_x, cell_size_y, dtype):
    # get u coordinates of the grid 
    umax = 1.0/cell_size_x
    # get end points
    ug = np.linspace(-umax, umax, nx + 1)[1::]
    # get v coordinates of the grid
    vmax = 1.0/cell_size_y
    # get end points
    vg = np.linspace(-vmax, vmax, ny + 1)[1::]
    nchan = freqs.size
    nrow = uvw.shape[0]
    # initialise array to store weights
    weights = np.zeros((nrow, nchan), dtype=dtype)
    for r in range(nrow):
        for c in range(nchan):
            # get current uv
            u_tmp, v_tmp = uvw[r, 0:2] * freqs[c] / lightspeed
            # get u index
            u_idx = (ug < u_tmp).nonzero()[0][-1]
            # get v index
            v_idx = (vg < v_tmp).nonzero()[0][-1]
            if wsums[u_idx, v_idx]:
                weights[r, c] = 1.0/wsums[u_idx, v_idx]
    return weights

def robust_reweight(residuals, weights, v=None):
    """
    Find the robust weights corresponding to a soln that generated residuals

    residuals - residuals i.e. (data - model) (nrow, nchan, ncorr)
    weights - inverse of data covariance (nrow, nchan, ncorr)
    v - initial guess for degrees for freedom parameter (float)
    corrs - which correlation axes to compute new weights for (default is for LL and RR (or XX and)) 

    Correlation axis not currently supported
    """
    # elements of Mahalanobis distance (Delta^2_i's)
    nrow, nchan = residuals.shape
    ressq = (residuals.conj()*residuals).real

    # func to solve for degrees of freedom parameter
    def func(v, N, ressq):
        # expectation values
        tmp = v+ressq
        Etau = (v + 1)/tmp
        Elogtau = digamma(v + 1) - np.log(tmp)

        # derivatives of expectation value terms
        dEtau = 1.0/tmp - (v+1)/tmp**2
        dElogtau = polygamma(1, v + 1) - 1.0/(v + ressq)
        return N*np.log(v) - N*digamma(v) + np.sum(Elogtau) - np.sum(Etau), N/v - N*polygamma(1, v) + np.sum(dElogtau) - np.sum(dEtau)


    v, f, d = fmin_l_bfgs_b(func, v, args=(nrow, ressq), pgtol=1e-2, approx_grad=False, bounds=[(1e-3, 30)])
    Etau = (v + 1.0)/(v + ressq)  # used as new weights
    Lambda = np.mean(ressq*Etau, axis=0)
    return v, np.sqrt(Etau / Lambda[None, :])