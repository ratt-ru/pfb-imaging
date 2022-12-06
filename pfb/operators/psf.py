import numpy as np
import dask.array as da
from daskms.optimisation import inlined_array
from uuid import uuid4
from ducc0.fft import r2c, c2r, c2c, good_size
from uuid import uuid4
from pfb.utils.misc import pad_and_shift, unpad_and_unshift
from pfb.utils.misc import pad_and_shift_cube, unpad_and_unshift_cube
import gc
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


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

def _psf_convolve_impl(x, psfhat, beam,
                       nthreads=None,
                       padding=None,
                       unpad_x=None,
                       unpad_y=None,
                       lastsize=None):
    nx, ny = x.shape
    xhat = x if beam is None else x * beam
    xhat = iFs(np.pad(xhat, padding, mode='constant'), axes=(0, 1))
    xhat = r2c(xhat, axes=(0, 1), nthreads=nthreads,
                forward=True, inorm=0)
    xhat = c2r(xhat * psfhat, axes=(0, 1), forward=False,
               lastsize=lastsize, inorm=2, nthreads=nthreads)
    convim = Fs(xhat, axes=(0, 1))[unpad_x, unpad_y]

    if beam is not None:
        convim *= beam

    return convim

def _psf_convolve(x, psfhat, beam, psfopts):
    return _psf_convolve_impl(x, psfhat, beam, **psfopts)

def psf_convolve(x, psfhat, beam, psfopts):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(-1, -1), name=False)
    if beam is None:
        bout = None
    else:
        bout = ('nx', 'ny')
    return da.blockwise(_psf_convolve, ('nx', 'ny'),
                        x, ('nx', 'ny'),
                        psfhat, ('nx', 'ny'),
                        beam, bout,
                        psfopts, None,
                        align_arrays=False,
                        dtype=x.dtype)


def _psf_convolve_cube_impl(x, psfhat, beam,
                            nthreads=None,
                            padding=None,
                            unpad_x=None,
                            unpad_y=None,
                            lastsize=None):
    nb, nx, ny = x.shape
    convim = np.zeros((nb, nx, ny), dtype=x.dtype)
    for b in range(nb):
        xhat = x[b] if beam is None else x[b] * beam[b]
        xhat = iFs(np.pad(xhat, padding, mode='constant'), axes=(0, 1))
        xhat = r2c(xhat, axes=(0, 1), nthreads=nthreads,
                   forward=True, inorm=0)
        xhat = c2r(xhat * psfhat[b], axes=(0, 1), forward=False,
                   lastsize=lastsize, inorm=2, nthreads=nthreads)
        convim[b] = Fs(xhat, axes=(0, 1))[unpad_x, unpad_y]

        if beam is not None:
            convim[b] *= beam[b]

    return convim

def _psf_convolve_cube(x, psfhat, beam, psfopts):
    return _psf_convolve_cube_impl(x, psfhat, beam, **psfopts)

def psf_convolve_cube(x, psfhat, beam, psfopts,
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


def _hessian_reg_psf(x, beam, psfhat,
                     nthreads=None,
                     sigmainv=None,
                     padding=None,
                     unpad_x=None,
                     unpad_y=None,
                     lastsize=None):
    """
    Tikhonov regularised Hessian approx
    """
    if isinstance(psfhat, da.Array):
        psfhat = psfhat.compute()
    if isinstance(beam, da.Array):
        beam = beam.compute()

    if beam is not None:
        xhat = iFs(np.pad(beam*x, padding, mode='constant'), axes=(1, 2))
    else:
        xhat = iFs(np.pad(x, padding, mode='constant'), axes=(1, 2))
    xhat = r2c(xhat, axes=(1, 2), nthreads=nthreads,
               forward=True, inorm=0)
    xhat = c2r(xhat * psfhat, axes=(1, 2), forward=False,
               lastsize=lastsize, inorm=2, nthreads=nthreads)
    im = Fs(xhat, axes=(1, 2))[:, unpad_x, unpad_y]

    if beam is not None:
        im *= beam

    if np.any(sigmainv):
        return im + x * sigmainv
    else:
        return im

def _hessian_reg_psf_slice(
                    x,  # input image, not overwritten
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    psfhat=None,
                    beam=None,
                    nthreads=1,
                    sigmainv=0,
                    lastsize=None,
                    wsum=1.0):
    """
    Tikhonov regularised Hessian approx
    """
    if beam is not None:
        pad_and_shift(x*beam, xpad)
    else:
        pad_and_shift(x*beam, xpad)
    r2c(xpad, axes=(0, 1), nthreads=nthreads,
        forward=True, inorm=0, out=xhat)
    xhat *= psfhat/wsum
    c2r(xhat, axes=(0, 1), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads)
    unpad_and_unshift(xpad, xout)

    if beam is not None:
        xout *= beam

    if np.any(sigmainv):
        return im + x * sigmainv
    else:
        return im

@profile
def psf_convolve_cube2(x,     # input image, not overwritten
                       xpad,  # preallocated array to store padded image
                       xhat,  # preallocated array to store FTd image
                       xout,  # preallocated array to store output image
                       psfhat,
                       lastsize,
                       nthreads=1):
    pad_and_shift_cube(x, xpad)
    r2c(xpad, axes=(1, 2), nthreads=nthreads,
        forward=True, inorm=0, out=xhat)
    xhat *= psfhat
    c2r(xhat, axes=(1, 2), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads)
    unpad_and_unshift_cube(xpad, xout)
    return xout
