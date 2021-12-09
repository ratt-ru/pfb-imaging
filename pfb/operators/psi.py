import numpy as np
import pywt
import dask.array as da


def _coef2im_impl(alpha, bases, padding, iy, sy, nx, ny):
    '''
    Per band coefficients to image
    '''
    nband, nbasis, _ = alpha.shape
    # not chunking over basis
    x = np.zeros((nband, nx, ny), dtype=alpha.dtype)
    for l in range(nband):
        for b, base in enumerate(bases):
            a = alpha[l, b, padding[b]]
            if base == 'self':
                wave = a.reshape(nx, ny)
            else:
                alpha_rec = pywt.unravel_coeffs(
                    a, iy[base], sy[base], output_format='wavedecn')
                wave = pywt.waverecn(alpha_rec, base, mode='zero')

            x[l] += wave
    return x

def _coef2im(alpha, bases, padding, iy, sy, nx, ny):
    return _coef2im_impl(alpha[0][0], bases[0], padding[0],
                         iy, sy, nx, ny)

def coef2im(alpha, bases, padding, iy, sy, nx, ny, compute=True):
    if not isinstance(alpha, da.Array):
        alpha = da.from_array(alpha, chunks=(1, -1, -1), name=False)
    graph = da.blockwise(_coef2im, ("band", "nx", "ny"),
                         alpha, ("band", "basis", "ntot"),
                         bases, ("basis",),
                         padding, ("basis",),
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

def _im2coef_impl(x, bases, ntot, nmax, nlevels):
    '''
    Per band image to coefficients
    '''
    nbasis = len(bases)
    nband, _, _ = x.shape
    alpha = np.zeros((nband, nbasis, nmax), dtype=x.dtype)
    for l in range(nband):
        for b, base in enumerate(bases):
            if base == 'self':
                # ravel and pad
                alpha[l, b] = np.pad((x[l]).ravel(), (0, nmax-ntot[b]), mode='constant')
            else:
                # decompose
                alpha_tmp = pywt.wavedecn(x[l], base, mode='zero', level=nlevels)
                # ravel and pad
                alpha_tmp, _, _ = pywt.ravel_coeffs(alpha_tmp)
                alpha[l, b] = np.pad(alpha_tmp, (0, nmax-ntot[b]), mode='constant')

    return alpha

def _im2coef(x, bases, ntot, nmax, nlevels):
    return _im2coef_impl(x[0][0], bases, ntot, nmax, nlevels)


def im2coef(x, bases, ntot, nmax, nlevels, compute=True):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1), name=False)
    graph = da.blockwise(_im2coef, ("band", "basis", "nmax"),
                         x, ("band", "nx", "ny"),
                         bases, ("basis",),
                         ntot, ("basis",),
                         nmax, None,
                         nlevels, None,
                         new_axes={'nmax':nmax},
                         dtype=x.dtype)
    if compute:
        return graph.compute()
    else:
        return graph
