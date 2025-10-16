import numpy as np
import dask.array as da
from ducc0.fft import r2c, c2r, c2c, good_size
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def _fft2d_impl(x, nthreads):
    return r2c(iFs(x, axes=(0, 1)), axes=(0, 1), nthreads=nthreads,
               forward=True, inorm=0)

def _fft2d(x, nthreads):
    return _fft2d_impl(x[0], nthreads)

def fft2d(x, nthreads=1):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(-1, -1), name=False)
    nyp = x.shape[-1]//2 + 1
    return da.blockwise(_fft2d, ('nx', 'nyo2'),
                        x, ('nx', 'ny'),
                        nthreads, None,
                        align_arrays=False,
                        new_axes={'nyo2':nyp},
                        dtype=da.result_type(x, np.complex64))


def _fft_cube_impl(x, nthreads):
    nb, nx, ny = x.shape
    nyp = x.shape[-1]//2 + 1
    xhat = np.zeros((nb, nx, nyp), dtype=np.result_type(x, np.complex64))
    for b in range(nb):
        xhat[b] = r2c(iFs(x[b], axes=(0, 1)), axes=(0, 1), nthreads=nthreads,
                      forward=True, inorm=0)
    return xhat

def _fft_cube(x, nthreads):
    return _fft_cube_impl(x, nthreads)

def fft_cube(x, nthreads=1):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1), name=False)
    nyp = x.shape[-1]//2 + 1
    return da.blockwise(_fft_cube, ('nb', 'nx', 'ny'),
                        x, ('nb', 'nx', 'ny'),
                        nthreads, None,
                        align_arrays=False,
                        adjust_chunks={'ny': nyp},
                        dtype=da.result_type(x, np.complex64))
