import numpy as np
import dask.array as da
from uuid import uuid4
from ducc0.fft import r2c, c2r


def psf_convolve_slice(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    psfhat,
                    lastsize,
                    x,  # input image, not overwritten
                    nthreads=1):
    nx, ny = x.shape
    xpad[...] = 0.0
    xpad[0:nx, 0:ny] = x
    r2c(xpad, axes=(0, 1), nthreads=nthreads,
        forward=True, inorm=0, out=xhat)
    xhat *= psfhat
    c2r(xhat, axes=(0, 1), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    xout[...] = xpad[0:nx, 0:ny]
    return xout


def psf_convolve_cube(xpad,    # preallocated array to store padded image
                      xhat,    # preallocated array to store FTd image
                      xout,    # preallocated array to store output image
                      psfhat,
                      lastsize,
                      x,       # input image, not overwritten
                      nthreads=1):
    '''
    The copyto is not necessarily faster it just allows us to see where time is spent
    '''
    _, nx, ny = x.shape
    xpad[...] = 0.0
    # np.copyto(xpad, 0.0)
    xpad[:, 0:nx, 0:ny] = x
    # np.copyto(xpad[:, 0:nx, 0:ny], x)
    r2c(xpad, axes=(1, 2), nthreads=nthreads,
        forward=True, inorm=0, out=xhat)
    xhat *= psfhat
    # ne.evaluate('xhat*psfhat', out=xhat, casting='unsafe')
    c2r(xhat, axes=(1, 2), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    xout[...] = xpad[:, 0:nx, 0:ny]
    # np.copyto(xout, xpad[:, 0:nx, 0:ny])
    return xout


def psf_convolve_xds(x, xds, psfopts, wsum, eta, mask,
                     compute=True, use_beam=True):
    '''
    Image space Hessian with reduction over dataset
    '''
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1), name=False)

    if not isinstance(mask, da.Array):
        mask = da.from_array(mask, chunks=(-1, -1), name=False)

    assert mask.ndim==2

    nband, nx, ny = x.shape

    convims = [da.zeros((nx, ny),
               chunks=(-1, -1), name="zeros-" + uuid4().hex)
               for _ in range(nband)]

    for ds in xds:
        psfhat = ds.PSFHAT.data
        b = ds.bandid
        if use_beam:
            beam = ds.BEAM.data * mask
        else:
            # TODO - separate implementation without
            # unnecessary beam application
            beam = mask

        convim = psf_convolve(x[b], psfhat, beam, psfopts)

        convims[b] += convim

    convim = da.stack(convims)/wsum

    if eta:
        convim += x * eta**2

    if compute:
        return convim.compute()
    else:
        return convim




##################### jax #####################################
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0,1,2,3))
def psf_convolve_slice_jax(
                    nx, ny,
                    nx_psf, ny_psf,
                    psfhat,
                    x):
    xhat = jnp.fft.rfft2(x,
                         s=(nx_psf, ny_psf),
                         norm='backward')
    xout = jnp.fft.irfft2(xhat*psfhat,
                          s=(nx_psf, ny_psf),
                          norm='backward')[0:nx, 0:ny]
    return xout