import numpy as np
from numba import njit, prange


def prox_21m(v, sigma, weight=1.0, axis=0):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

     Assumed that v has shape (nband, nbasis, nymax, nxmax) where

    nband   - number of imaging bands
    nbasis  - number of orthogonal bases
    nxmax   - number of x coefficients for each basis (must be equal)
    nymax   - number of y coefficients for each basis (must be equal)

    sigma  - scalar setting overall threshold level
    weight  - (nbasis, nymax, nxmax) setting relative weights per components in the
              "MFS" cube
    """
    l2_norm = np.sum(v, axis=axis)  # drops axis
    l2_soft = np.maximum(np.abs(l2_norm) - sigma * weight, 0.0) * np.sign(l2_norm)
    mask = l2_norm != 0
    ratio = np.zeros(mask.shape, dtype=v.dtype)
    ratio[mask] = l2_soft[mask] / l2_norm[mask]
    return v * np.expand_dims(ratio, axis=axis)  # restores axis


@njit(nogil=True, cache=True, parallel=True)
def prox_21m_numba(v, result, lam, sigma=1.0, weight=None):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape (nband, nbasis, nymax, nxmax) where

    nband   - number of imaging bands
    nbasis  - number of orthogonal bases
    nxmax   - number of x coefficients for each basis (must be equal)
    nymax   - number of y coefficients for each basis (must be equal)

    """
    nband, nbasis, nymax, nxmax = v.shape
    # sum over band axis upfront?
    # vsum = np.sum(v, axis=0)
    # result = np.zeros((nband, nbasis, ntot))
    for b in range(nbasis):
        vb = v[:, b]
        weightb = weight[b]
        resultb = result[:, b]
        for i in prange(nymax):
            for j in range(nxmax):
                vbisum = np.sum(vb[:, i, j]) / sigma
                if not vbisum:
                    resultb[:, i, j] = 0.0
                    continue
                absvbi = np.abs(vbisum)
                softvbi = np.maximum(absvbi - lam * weightb[i, j] / sigma, 0.0)  # * vbisum/absvbi
                resultb[:, i, j] = vb[:, i, j] * softvbi / absvbi / sigma


def dual_update(v, x, psih, lam, sigma=1.0, weight=1.0):
    vp = v.copy()
    vout = np.zeros_like(v)
    psih(x, vout)
    vtilde = vp + sigma * vout
    # return vtilde
    v = vtilde - sigma * prox_21m(vtilde / sigma, lam / sigma, weight=weight)
    return v


@njit(nogil=True, cache=True, parallel=True)
def dual_update_numba(vp, v, lam, sigma=1.0, weight=None):
    """
    Computes dual update

    Assumed that v has shape (nband, nbasis, nymax, nxmax) where

    nband   - number of imaging bands
    nbasis  - number of orthogonal bases
    nxmax   - number of x coefficients for each basis (must be equal)
    nymax   - number of y coefficients for each basis (must be equal)

    v is initialised with psih(xp) and will be updated
    """
    nband, nbasis, nymax, nxmax = v.shape
    for b in range(nbasis):
        # select out basis
        # vtildeb = vp[:, b] + sigma * v[:, b]
        # weightb = weight[b]
        for i in prange(nymax):  # without the prange it segfaults when parallel=True?
            vtildebi = vp[:, b, i] + sigma * v[:, b, i]
            weightbi = weight[b, i]
            for j in range(nxmax):
                vtildebij = vtildebi[:, j]
                absvbijsum = np.abs(np.sum(vtildebij) / sigma)  # sum over band axis
                v[:, b, i, j] = vtildebij
                if absvbijsum:
                    softvbij = np.maximum(absvbijsum - lam * weightbi[j] / sigma, 0.0)
                    v[:, b, i, j] *= 1 - softvbij / absvbijsum


@njit(nogil=True, cache=True, parallel=True, fastmath=True)
def dual_update_numba_fast(vp, v, lam, sigma=1.0, weight=None):
    """
    Numerically stable + fastmath-safe version of dual_update_numba.

    Computes v = vtilde * min(1, lam*w / |sum(vtilde)|) where
    vtilde = vp + sigma * v. The sigma factors cancel algebraically
    so we avoid redundant divisions. The threshold/|sum| form avoids
    the catastrophic cancellation in 1 - softvbij/absvbijsum.

    v has shape (nband, nbasis, nymax, nxmax), initialised with psih(xp).
    """
    nband, nbasis, nymax, nxmax = v.shape
    for b in range(nbasis):
        weightbi_base = weight[b]
        for i in prange(nymax):
            weightbi = weightbi_base[i]
            for j in range(nxmax):
                # compute vtilde and band-axis sum inline
                band_sum = 0.0
                for k in range(nband):
                    vt = vp[k, b, i, j] + sigma * v[k, b, i, j]
                    v[k, b, i, j] = vt
                    band_sum += vt
                abs_band_sum = np.abs(band_sum)
                threshold = lam * weightbi[j]
                if abs_band_sum > threshold:
                    # stable form: threshold / abs_band_sum is always in (0, 1)
                    scale = threshold / abs_band_sum
                    for k in range(nband):
                        v[k, b, i, j] *= scale


# The l21 dual update split for a dual held per band in the band workers
# (PsiNocopytRay.dual_begin/dual_finish). The bands couple only through the
# band sum of vtilde = vp + sigma * Psi^T x, and Psi is the same linear map for
# every band, so that sum is sum_b vp_b + sigma * Psi^T(sum_b x_b): the driver
# needs one analysis of the band-summed image and the band-summed dual, never a
# per-band coefficient cube. The shrink factor does not depend on the band, so
# applying it per band reproduces dual_update_numba_fast exactly in exact
# arithmetic.


@njit(nogil=True, cache=True, parallel=True, fastmath=True)
def l21_dual_scale(vsum, c, lam, sigma, weight):
    """Band-summed half of the split l21 dual update, in place.

    On entry ``c`` is ``Psi^T(sum_b x_b)`` and ``vsum`` is ``sum_b v_b``. On
    exit ``c`` holds the per-coefficient shrink factor
    ``min(1, lam*w / |vsum + sigma*c|)`` and ``vsum`` the updated band-summed
    dual. All three arrays are ``(nbasis, n1, n2)``.
    """
    nbasis, n1, n2 = c.shape
    for b in range(nbasis):
        for i in prange(n1):
            for j in range(n2):
                s = vsum[b, i, j] + sigma * c[b, i, j]
                threshold = lam * weight[b, i, j]
                abs_s = np.abs(s)
                scale = threshold / abs_s if abs_s > threshold else 1.0
                c[b, i, j] = scale
                vsum[b, i, j] = s * scale


@njit(nogil=True, cache=True, parallel=True, fastmath=True)
def l21_dual_begin(c, v, sigma):
    """One band's ``vt = v + sigma * c`` (``c = Psi^T x_b``), into ``c``."""
    c_f = c.ravel()
    v_f = v.ravel()
    for i in prange(c_f.size):
        c_f[i] = v_f[i] + sigma * c_f[i]


@njit(nogil=True, cache=True, parallel=True, fastmath=True)
def l21_dual_finish(c, v, scale):
    """One band's shrink and extrapolation, in place over two buffers.

    ``c`` holds ``vt`` on entry and ``2 * v_new - v`` on exit, and ``v``
    becomes ``v_new = vt * scale``: no third coefficient buffer per band.
    """
    c_f = c.ravel()
    v_f = v.ravel()
    s_f = scale.ravel()
    for i in prange(c_f.size):
        vn = c_f[i] * s_f[i]
        c_f[i] = 2.0 * vn - v_f[i]
        v_f[i] = vn
