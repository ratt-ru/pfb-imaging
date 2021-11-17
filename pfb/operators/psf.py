import numpy as np
import dask.array as da
from ducc0.fft import r2c, c2r, c2c, good_size
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


class PSF(object):
    def __init__(self, psf, imsize, nthreads=1, backward_undersize=None):
        self.nthreads = nthreads
        self.nband, nx_psf, ny_psf = psf.shape
        _, nx, ny = imsize
        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        self.padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        self.ax = (1, 2)
        self.unpad_x = slice(npad_xl, -npad_xr)
        self.unpad_y = slice(npad_yl, -npad_yr)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True,
                          nthreads=nthreads, inorm=0)

        # LB - failed experiment?
        # self.psfhatinv = 1/(self.psfhat + 1.0)


        if backward_undersize is not None:
            # set up for backward step
            nx_psfb = good_size(int(backward_undersize * nx))
            ny_psfb = good_size(int(backward_undersize * ny))
            npad_xlb = (nx_psfb - nx)//2
            npad_xrb = nx_psfb - nx - npad_xlb
            npad_ylb = (ny_psfb - ny)//2
            npad_yrb = ny_psfb - ny - npad_ylb
            self.paddingb = ((0, 0), (npad_xlb, npad_xrb), (npad_ylb, npad_yrb))
            self.unpad_xb = slice(npad_xlb, -npad_xrb)
            self.unpad_yb = slice(npad_ylb, -npad_yrb)
            self.lastsizeb = ny + np.sum(self.paddingb[-1])

            xlb = (nx_psf - nx_psfb)//2
            xrb = nx_psf - nx_psfb - xlb
            ylb = (ny_psf - ny_psfb)//2
            yrb = ny_psf - ny_psfb - ylb
            psf_padb = iFs(psf[:, slice(xlb, -xrb), slice(ylb, -yrb)], axes=self.ax)
            self.psfhatb = r2c(psf_padb, axes=self.ax, forward=True,
                            nthreads=nthreads, inorm=0)
        else:
            self.paddingb = self.padding
            self.unpad_xb = self.unpad_x
            self.unpad_yb = self.unpad_y
            self.lastsizeb = self.lastsize
            self.psfhatb = self.psfhat


    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads,
                   forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False,
                   lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]


    def convolveb(self, x):
        xhat = iFs(np.pad(x, self.paddingb, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads,
                   forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhatb, axes=self.ax, forward=False,
                   lastsize=self.lastsizeb, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[:, self.unpad_xb, self.unpad_yb]

    # def iconvolve(self, x):
    #     xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
    #     xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads,
    #                forward=True, inorm=0)
    #     xhat = c2r(xhat * self.psfhatinv, axes=self.ax, forward=False,
    #                lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
    #     return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]


def _psf_convolve_impl(x, psfhat,
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

def _psf_convolve(x, psfhat, psfopts):
    return _psf_convolve_impl(x, psfhat, **psfopts)

def psf_convolve(x, psfhat, psfopts):
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1), name=False)
    return da.blockwise(_psf_convolve, ('nband', 'nx', 'ny'),
                        x, ('nband', 'nx', 'ny'),
                        psfhat, ('nband', 'nx', 'ny'),
                        psfopts, None,
                        align_arrays=False,
                        dtype=x.dtype)
