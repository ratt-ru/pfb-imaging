import numpy as np
import numba
import numexpr as ne
import concurrent.futures as cf
import pywt
import dask.array as da
from pfb.wavelets.wavelets import wavedecn, waverecn, ravel_coeffs, unravel_coeffs


@numba.njit(nogil=True, fastmath=True, cache=True)
def pad(x, n):
    '''
    pad 1D array by n zeros
    '''
    return np.append(x, np.zeros(n, dtype=x.dtype))


def _coef2im_impl(a, base, l, iy, sy, nx, ny):
    if base == 'self':
        wave = a.reshape(nx, ny)
    else:
        alpha_rec = unravel_coeffs(
            a, iy, sy, output_format='wavedecn')
        wave = waverecn(alpha_rec, base, mode='zero')
    return wave, l


def coef2im(alpha, x, bases, ntot, iy, sy, nx, ny, nthreads=1):
    '''
    Per band coefficients to image
    '''
    nband, nbasis, nmax = alpha.shape
    sqrtP = 1.0  #np.sqrt(nbasis)
    futures = []
    x[...] = 0.0
    with cf.ThreadPoolExecutor(max_workers=nthreads) as executor:
        for l in range(nband):
            for b in range(nbasis):
                base = bases[b]
                a = alpha[l, b, 0:ntot[b]]
                if base == 'self':
                    iyt = 1.0
                    syt = 1.0
                else:
                    iyt = iy[base]
                    syt = sy[base]
                fut = executor.submit(_coef2im_impl, a, base, l, iyt, syt, nx, ny)
                futures.append(fut)

        for f in cf.as_completed(futures):
            wave, l = f.result()
            ne.evaluate('x + wave', local_dict={
                        'x': x[l],
                        'wave': wave/sqrtP},
                        out=x[l], casting='same_kind')


def _im2coef_impl(x, base, l, b, nlevels):
    if base == 'self':
        wave = x.ravel()  #, mode='constant')
    else:
        wave = wavedecn(x, base, mode='zero', level=nlevels)
        wave, _, _ = ravel_coeffs(wave)
    return wave, l, b


def im2coef(x, alpha, bases, ntot, nmax, nlevels, nthreads=1):
    '''
    Per band image to coefficients
    '''
    nbasis = len(bases)
    sqrtP = 1.0  #np.sqrt(nbasis)
    nband, _, _ = x.shape
    alpha[...] = 0.0
    futures = []
    with cf.ThreadPoolExecutor(max_workers=nthreads) as executor:
        for l in range(nband):
            for b in range(nbasis):
                base = bases[b]
                fut = executor.submit(_im2coef_impl, x[l], base, l, b, nlevels)
                futures.append(fut)

        for f in cf.as_completed(futures):
            wave, l, b = f.result()
            alpha[l, b, 0:ntot[b]] = wave/sqrtP


# TODO - compare threadpool with dask versions
# def _coef2im(alpha, bases, ntot, iy, sy, nx, ny):
#     return _coef2im_impl(alpha[0][0], bases, ntot,
#                          iy, sy, nx, ny)

# def coef2im(alpha, bases, ntot, iy, sy, nx, ny, compute=True):
#     if not isinstance(alpha, da.Array):
#         alpha = da.from_array(alpha, chunks=(1, -1, -1), name=False)

#     graph = da.blockwise(_coef2im, ("band", "nx", "ny"),
#                          alpha, ("band", "basis", "ntot"),
#                          bases, None, #("basis",),
#                          ntot, None, #("basis",),
#                          iy, None,
#                          sy, None,
#                          nx, None,
#                          ny, None,
#                          new_axes={'nx': nx, 'ny': ny},
#                          dtype=alpha.dtype,
#                          align_arrays=False)
#     if compute:
#         return graph.compute()
#     else:
#         return graph


# def _im2coef(x, bases, ntot, nmax, nlevels):
#     return _im2coef_impl(x[0][0], bases, ntot, nmax, nlevels)


# def im2coef(x, bases, ntot, nmax, nlevels, compute=True):
#     if not isinstance(x, da.Array):
#         x = da.from_array(x, chunks=(1, -1, -1), name=False)

#     graph = da.blockwise(_im2coef, ("band", "basis", "nmax"),
#                          x, ("band", "nx", "ny"),
#                          bases, None, #("basis",),
#                          ntot, None, #("basis",),
#                          nmax, None,
#                          nlevels, None,
#                          new_axes={'basis':len(bases), 'nmax':nmax},
#                          dtype=x.dtype)

#     if compute:
#         return graph.compute()
#     else:
#         return graph


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

