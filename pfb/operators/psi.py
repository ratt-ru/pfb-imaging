import numpy as np
import numba
import pywt
import dask.array as da
from pfb.wavelets.wavelets import wavedecn, waverecn, ravel_coeffs, unravel_coeffs


@numba.njit(nogil=True, fastmath=True, cache=True)
def pad(x, n):
    '''
    pad 1D array by n zeros
    '''
    return np.append(x, np.zeros(n, dtype=x.dtype))


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _coef2im_impl(alpha, bases, ntot, iy, sy, nx, ny):
    '''
    Per band coefficients to image
    '''
    nband, nbasis, _ = alpha.shape
    # not chunking over basis
    x = np.zeros((nband, nbasis, nx, ny), dtype=alpha.dtype)
    for l in numba.prange(nband):
        for b in range(nbasis):
            base = bases[b]
            a = alpha[l, b, 0:ntot[b]]
            if base == 'self':
                wave = a.reshape(nx, ny)
            else:
                alpha_rec = unravel_coeffs(
                    a, iy[base], sy[base], output_format='wavedecn')
                wave = waverecn(alpha_rec, base, mode='zero')

            x[l, b] = wave
    return np.sum(x, axis=1)

def _coef2im(alpha, bases, ntot, iy, sy, nx, ny):
    return _coef2im_impl_flat(alpha[0][0], bases, ntot,
                         iy, sy, nx, ny)

def coef2im(alpha, bases, ntot, iy, sy, nx, ny, compute=True):
    if not isinstance(alpha, da.Array):
        alpha = da.from_array(alpha, chunks=(1, -1, -1), name=False)

    graph = da.blockwise(_coef2im, ("band", "nx", "ny"),
                         alpha, ("band", "basis", "ntot"),
                         bases, None, #("basis",),
                         ntot, None, #("basis",),
                         iy, None,
                         sy, None,
                         nx, None,
                         ny, None,
                         new_axes={'nx': nx, 'ny': ny},
                         dtype=alpha.dtype,
                         align_arrays=False)
    if compute:
        return graph.compute()
    else:
        return graph


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _coef2im_impl_flat(alpha, bases, ntot, iy, sy, nx, ny):
    '''
    Per band coefficients to image
    '''
    nband, nbasis, _ = alpha.shape
    # not chunking over basis
    x = np.zeros((nband*nbasis, nx, ny), dtype=alpha.dtype)
    for i in numba.prange(nband*nbasis):
        l = i//nbasis
        b = i - l*nbasis
        base = bases[b]
        a = alpha[l, b, 0:ntot[b]]
        if base == 'self':
            wave = a.reshape(nx, ny)
        else:
            alpha_rec = unravel_coeffs(
                a, iy[base], sy[base], output_format='wavedecn')
            wave = waverecn(alpha_rec, base, mode='zero')

        x[i] = wave
    return np.sum(x.reshape(nband, nbasis, nx, ny), axis=1)


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _im2coef_impl(x, bases, ntot, nmax, nlevels):
    '''
    Per band image to coefficients
    '''
    nbasis = len(bases)
    nband, _, _ = x.shape
    alpha = np.zeros((nband, nbasis, nmax), dtype=x.dtype)
    for l in numba.prange(nband):
        for b in range(nbasis):
            base = bases[b]
            if base == 'self':
                # ravel and pad
                alpha[l, b] = pad(x[l].ravel(), nmax-ntot[b]) #, mode='constant')
            else:
                # decompose
                alpha_tmp = wavedecn(x[l], base, mode='zero', level=nlevels)
                # ravel and pad
                alpha_tmp, _, _ = ravel_coeffs(alpha_tmp)
                alpha[l, b] = pad(alpha_tmp, nmax-ntot[b])  #, mode='constant')

    return alpha

def _im2coef(x, bases, ntot, nmax, nlevels):
    return _im2coef_impl_flat(x[0][0], bases, ntot, nmax, nlevels)


def im2coef(x, bases, ntot, nmax, nlevels, compute=True):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1), name=False)

    graph = da.blockwise(_im2coef, ("band", "basis", "nmax"),
                         x, ("band", "nx", "ny"),
                         bases, None, #("basis",),
                         ntot, None, #("basis",),
                         nmax, None,
                         nlevels, None,
                         new_axes={'basis':len(bases), 'nmax':nmax},
                         dtype=x.dtype)

    if compute:
        return graph.compute()
    else:
        return graph


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _im2coef_impl_flat(x, bases, ntot, nmax, nlevels):
    '''
    Per band image to coefficients
    '''
    nbasis = len(bases)
    nband, _, _ = x.shape
    alpha = np.zeros((nband*nbasis, nmax), dtype=x.dtype)
    for i in numba.prange(nband*nbasis):
        l = i//nbasis
        b = i - l*nbasis
        base = bases[b]
        if base == 'self':
            # ravel and pad
            alpha[i] = pad(x[l].ravel(), nmax-ntot[b]) #, mode='constant')
        else:
            # decompose
            alpha_tmp = wavedecn(x[l], base, mode='zero', level=nlevels)
            # ravel and pad
            alpha_tmp, _, _ = ravel_coeffs(alpha_tmp)
            alpha[i] = pad(alpha_tmp, nmax-ntot[b])  #, mode='constant')

    return alpha.reshape(nband, nbasis, nmax)


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def im2coef_dist(x, bases, ntot, nmax, nlevels):
    '''
    Per band image to coefficients
    '''
    nbasis = len(bases)
    alpha = np.zeros((nbasis, nmax), dtype=x.dtype)
    for b in numba.prange(nbasis):
        base = bases[b]
        if base == 'self':
            # ravel and pad
            alpha[b] = pad(x.ravel(), nmax-ntot[b]) #, mode='constant')
        else:
            continue
            # # decompose
            # alpha_tmp = wavedecn(x, base, mode='zero', level=nlevels)
            # # ravel and pad
            # alpha_tmp, _, _ = ravel_coeffs(alpha_tmp)
            # alpha[b] = pad(alpha_tmp, nmax-ntot[b])  #, mode='constant')

    return alpha


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=True)
def coef2im_dist(alpha, bases, ntot, iy, sy, nx, ny):
    '''
    Per band coefficients to image
    '''
    nbasis = len(bases)
    x = np.zeros((nbasis, nx, ny), dtype=alpha.dtype)
    for b in numba.prange(nbasis):
        base = bases[b]
        a = alpha[b, 0:ntot[b]]
        if base == 'self':
            wave = a.reshape(nx, ny)
        else:
            continue
            # alpha_rec = unravel_coeffs(
            #     a, iy[base], sy[base], output_format='wavedecn')
            # wave = waverecn(alpha_rec, base, mode='zero')

        x[b] = wave
    return np.sum(x, axis=0)

