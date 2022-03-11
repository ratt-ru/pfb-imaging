import numpy as np
import dask.array as da
from daskms.optimisation import inlined_array
from uuid import uuid4
from ducc0.fft import r2c, c2r, c2c, good_size
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
        x = da.from_array(x, chunks=(1, -1, -1), name=False)
    if not isinstance(psfhat, da.Array):
        psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)

    if beam is None:
        bout = None
    else:
        bout = ('nb', 'nx', 'ny')
        if not isinstance(beam, da.Array):
            beam = da.from_array(beam, chunks=(1, -1, -1), name=False)

    convim = da.blockwise(_psf_convolve_cube, ('nb', 'nx', 'ny'),
                          x, ('nb', 'nx', 'ny'),
                          psfhat, ('nb', 'nx', 'ny'),
                          beam, bout,
                          psfopts, None,
                          align_arrays=False,
                          dtype=x.dtype)/wsum
    if sigmainv:
        convim += x * sigmainv**2

    if compute:
        return convim.compute()
    else:
        return convim
