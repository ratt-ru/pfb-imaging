import numpy as np
from ducc0.fft import r2c, c2r, c2c
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

class PSF(object):
    def __init__(self, psf, nthreads=1, imsize=None, mask=None, beam=None):
        self.nthreads = nthreads
        self.nband, nx_psf, ny_psf = psf.shape
        if imsize is not None:
            _, nx, ny = imsize
            if nx > nx_psf or ny > ny_psf:
                raise ValueError("Image size can't be smaller than PSF size")
        else:
            # if imsize not passed in assume PSF is twice the size of image
            nx = nx_psf//2
            ny = ny_psf//2
        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        self.padding = ((0,0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        self.ax = (1,2)
        self.unpad_x = slice(npad_xl, -npad_xr)
        self.unpad_y = slice(npad_yl, -npad_yr)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)

        if mask is not None:
            self.mask = mask
        else:
            self.mask = lambda x: x

        if beam is not None:
            self.beam = beam
        else:
            self.beam = lambda x: x

    def convolve(self, x):
        xhat = iFs(np.pad(self.beam(self.mask(x)), self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return self.mask(self.beam(Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]))