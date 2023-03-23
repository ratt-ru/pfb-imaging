import numpy as np
from numba import njit, prange


def prox_21m(v, sigma, weight=None, axis=0):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape (nband, nbasis, ntot) where

    nband   - number of imaging bands
    nbasis  - number of orthogonal bases
    ntot    - total number of coefficients for each basis (must be equal)
    """
    l2_norm = np.sum(v, axis=axis)  # drops axis
    l2_soft = np.maximum(np.abs(l2_norm) - sigma * weight, 0.0) * np.sign(l2_norm)
    mask = l2_norm != 0
    ratio = np.zeros(mask.shape, dtype=v.dtype)
    ratio[mask] = l2_soft[mask] / l2_norm[mask]
    return v * np.expand_dims(ratio, axis=axis)  # restores axis


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def prox_21m_numba(v, result, lam, sigma=1.0, weight=None):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape (nband, nbasis, ntot) where

    nband   - number of imaging bands
    nbasis  - number of orthogonal bases
    ntot    - total number of coefficients for each basis (must be equal)
    """
    nband, nbasis, ntot = v.shape
    # sum over band axis upfront?
    # vsum = np.sum(v, axis=0)
    # result = np.zeros((nband, nbasis, ntot))
    for b in range(nbasis):
        vb = v[:, b]
        weightb = weight[b]
        resultb = result[:, b]
        for i in prange(ntot):
            vbisum = np.sum(vb[:, i])/sigma
            if not vbisum:
                resultb[:, i] = 0.0
                continue
            absvbi = np.abs(vbisum)
            softvbi = np.maximum(absvbi - lam*weightb[i]/sigma, 0.0) #* vbisum/absvbi
            resultb[:, i] = vb[:, i] * softvbi / absvbi /sigma
