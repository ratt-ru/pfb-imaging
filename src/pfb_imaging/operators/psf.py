from functools import partial

import jax
import jax.numpy as jnp
from ducc0.fft import c2r, r2c


def psf_convolve_slice(
    xpad,  # preallocated array to store padded image
    xhat,  # preallocated array to store FTd image
    xout,  # preallocated array to store output image
    psfhat,
    lastsize,
    x,  # input image, not overwritten
    nthreads=1,
):
    nx, ny = x.shape
    xpad[...] = 0.0
    xpad[0:nx, 0:ny] = x
    r2c(xpad, axes=(0, 1), nthreads=nthreads, forward=True, inorm=0, out=xhat)
    xhat *= psfhat
    c2r(
        xhat,
        axes=(0, 1),
        forward=False,
        out=xpad,
        lastsize=lastsize,
        inorm=2,
        nthreads=nthreads,
        allow_overwriting_input=True,
    )
    xout[...] = xpad[0:nx, 0:ny]
    return xout


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def psf_convolve_slice_jax(nx, ny, nx_psf, ny_psf, psfhat, x):
    psfh = jax.lax.stop_gradient(psfhat)
    xhat = jnp.fft.rfft2(x, s=(nx_psf, ny_psf), norm="backward")
    xout = jnp.fft.irfft2(xhat * psfh, s=(nx_psf, ny_psf), norm="backward")[0:nx, 0:ny]
    return xout
