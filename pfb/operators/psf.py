import numpy as np
import numexpr as ne
import dask.array as da
from daskms.optimisation import inlined_array
from uuid import uuid4
from ducc0.fft import r2c, c2r, c2c, good_size
from ducc0.misc import roll_resize_roll as rrr
from uuid import uuid4
import gc


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
    # xpad[...] = 0.0
    np.copyto(xpad, 0.0)
    # xpad[:, 0:nx, 0:ny] = x
    np.copyto(xpad[:, 0:nx, 0:ny], x)
    r2c(xpad, axes=(1, 2), nthreads=nthreads,
        forward=True, inorm=0, out=xhat)
    # xhat *= psfhat
    ne.evaluate('xhat*psfhat', out=xhat, casting='unsafe')
    c2r(xhat, axes=(1, 2), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    # xout[...] = xpad[:, 0:nx, 0:ny]
    np.copyto(xout, xpad[:, 0:nx, 0:ny])
    return xout


def psf_convolve_xds(x, xds, psfopts, wsum, sigmainv, mask,
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

    if sigmainv:
        convim += x * sigmainv**2

    if compute:
        return convim.compute()
    else:
        return convim


def psf_convolve_cube_dask(x, psfhat, beam, psfopts,
                      wsum=1, sigmainv=None, compute=True):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1),
                          name="x-" + uuid4().hex)
    if not isinstance(psfhat, da.Array):
        psfhat = da.from_array(psfhat, chunks=(1, -1, -1),
                               name="psfhat-" + uuid4().hex)

    if beam is None:
        bout = None
    else:
        bout = ('nb', 'nx', 'ny')
        if not isinstance(beam, da.Array):
            beam = da.from_array(beam, chunks=(1, -1, -1),
                                 name="beam-" + uuid4().hex)

    convim = da.blockwise(_psf_convolve_cube, ('nb', 'nx', 'ny'),
                          x, ('nb', 'nx', 'ny'),
                          psfhat, ('nb', 'nx', 'ny'),
                          beam, bout,
                          psfopts, None,
                          align_arrays=False,
                          dtype=x.dtype)
    convim /= wsum

    if np.any(sigmainv):
        convim += x * sigmainv

    if compute:
        return convim.compute()
    else:
        return convim
