import numpy as np
import dask.array as da
from ducc0.fft import r2c, c2r, c2c, good_size
from pfb.operators.psi import im2coef, coef2im
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def psf_convolve_xds(x, xdss, psfopts, wsum, sigmainv, compute=True):
    '''
    Image space Hessian reduction over dataset
    '''
    convims = []
    for xds in xdss:
        psf = xds.PSF.data
        beam = xds.BEAM.data

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
    return _psf_convolve_impl(x, psfhat, **psfopts)

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
