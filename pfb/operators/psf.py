import numpy as np
import dask.array as da
from daskms.optimisation import inlined_array
from ducc0.fft import r2c, c2r, c2c, good_size
from pfb.operators.psi import im2coef, coef2im
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def psf_convolve_alpha_xds(alpha, xds, psfopts, waveopts, wsum, sigmainv, compute=True):
    '''
    Image space Hessian reduction over dataset
    '''
    pmask = waveopts['pmask']
    bases = waveopts['bases']
    padding = waveopts['padding']
    iy = waveopts['iy']
    sy = waveopts['sy']
    nx = waveopts['nx']
    ny = waveopts['ny']
    ntot = waveopts['ntot']
    nmax = waveopts['nmax']
    nlevels = waveopts['nlevels']

    if not isinstance(alpha, da.Array):
        alpha = da.from_array(alpha, chunks=(1, -1, -1), name=False)

    # coeff to image
    x = coef2im(alpha, pmask, bases, padding, iy, sy, nx, ny)
    x = inlined_array(x, [pmask, alpha, bases, padding])

    convims = []
    for ds in xds:
        psfhat = ds.PSFHAT.data
        beam = ds.BEAM.data
        convim = psf_convolve(x, psfhat, beam, psfopts)
        convims.append(convim)

    # LB - it's not this simple when there are multiple spw's to consider
    convim = da.stack(convims).sum(axis=0)/wsum

    alpha_rec = im2coef(convim, pmask, bases, ntot, nmax, nlevels)
    alpha_rec = inlined_array(alpha_rec, [pmask, bases, ntot])

    if sigmainv:
        alpha_rec += alpha * sigmainv**2

    if compute:
        return alpha_rec.compute()
    else:
        return alpha_rec


def psf_convolve_xds(x, xds, psfopts, wsum, sigmainv, compute=True):
    '''
    Image space Hessian reduction over dataset
    '''
    convims = []
    for ds in xds:
        psfhat = ds.PSFHAT.data
        beam = ds.BEAM.data

        convim = psf_convolve(x, psfhat, beam, psfopts)

        convims.append(convim)

    # LB - it's not this simple when there are multiple spw's to consider
    convim = da.stack(convims).sum(axis=0)/wsum

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
    """
    Tikhonov regularised Hessian of coeffs
    """
    nband, nx, ny = x.shape
    convim = np.zeros_like(x)
    for b in range(nband):
        xhat = iFs(np.pad(x[b], padding, mode='constant'), axes=(0, 1))
        xhat = r2c(xhat, axes=(0, 1), nthreads=nthreads,
                    forward=True, inorm=0)
        xhat = c2r(xhat * psfhat[b], axes=(0, 1), forward=False,
                    lastsize=lastsize, inorm=2, nthreads=nthreads)
        convim[b] = Fs(xhat, axes=(0, 1))[unpad_x, unpad_y]

    return convim

def _psf_convolve(x, psfhat, beam, psfopts):
    return _psf_convolve_impl(x, psfhat, beam, **psfopts)

def psf_convolve(x, psfhat, beam, psfopts):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1), name=False)
    return da.blockwise(_psf_convolve, ('nband', 'nx', 'ny'),
                        x, ('nband', 'nx', 'ny'),
                        psfhat, ('nband', 'nx', 'ny'),
                        beam, ('nband', 'nx', 'ny'),
                        psfopts, None,
                        align_arrays=False,
                        dtype=x.dtype)
