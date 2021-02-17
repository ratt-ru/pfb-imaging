from numba import njit, prange
import numpy as np
import dask.array as da
from ducc0.fft import r2c, c2r, c2c
from africanus.gps.kernels import exponential_squared as expsq

iFs = np.fft.ifftshift
Fs = np.fft.fftshift


@njit(parallel=True, nogil=True, fastmath=True, inline='always')
def freqmul(A, x):
    nchan, npix = x.shape
    out = np.zeros((nchan, npix), dtype=x.dtype)
    for i in prange(npix):
        for j in range(nchan):
            for k in range(nchan):
                out[j, i] += A[j, k] * x[k, i]
    return out

@njit(parallel=True, nogil=True, fastmath=True, inline='always')
def make_kernel(nx_psf, ny_psf, sigma0, length_scale):
    K = np.zeros((1, nx_psf, ny_psf), dtype=np.float64)
    for j in range(nx_psf):
        for k in range(ny_psf):
            l = float(j - (nx_psf//2))
            m = float(k - (ny_psf//2))
            K[0,j,k] = sigma0**2*np.exp(-(l**2+m**2)/(2*length_scale**2))
    return K


def kron_matvec(A, b):
    D = len(A)
    N = b.size
    x = b

    for d in range(D):
        Gd = A[d].shape[0]
        NGd = N//Gd
        X = np.reshape(x, (Gd, NGd))
        Z = A[d].dot(X).T
        x = Z.ravel()
    return x


class mock_array(object):
    def __init__(self, n):
        self.n = n

    @property
    def size(self):
        return self.n**2

    @property
    def shape(self):
        return (self.n, self.n)

    @staticmethod
    def dot(x):
        return x


class PSF(object):
    def __init__(self, psf, nthreads=1, imsize=None, mask=None):
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

    def convolve(self, x):
        xhat = iFs(np.pad(self.mask(x), self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return self.mask(Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y])

class Gauss(object):
    def __init__(self, sigma0, nband, nx, ny, nthreads=8):
        self.nthreads = nthreads
        self.nx = nx
        self.ny = ny
        nx_psf = 2*self.nx
        npad_x = (nx_psf - nx)//2
        ny_psf = 2*self.ny
        npad_y = (ny_psf - ny)//2
        self.padding = ((0, 0), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (1,2)
        
        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])

        # set length scales
        length_scale = 0.5

        K = make_kernel(nx_psf, ny_psf, sigma0, length_scale)

        self.K = K
        K_pad = iFs(self.K, axes=self.ax)
        self.Khat = r2c(K_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)
        self.Khatinv = np.where(self.Khat.real > 1e-14, 1.0/self.Khat, 1e-14)


        # get covariance in each dimension
        # pixel coordinates
        self.Kv = mock_array(nband)  # np.eye(nband) * sigma0**2
        self.Kvinv = mock_array(nband) # np.eye(nband) / sigma0**2
        if nx == ny:
            l_coord = m_coord = np.arange(-(nx//2), nx//2)
            self.Kl = self.Km = expsq(l_coord, l_coord, 1.0, length_scale)
            self.Klinv = self.Kminv = np.linalg.pinv(self.Kl, hermitian=True, rcond=1e-12)
            self.Kl *= sigma0**2
            self.Klinv /= sigma0**2
        else:
            l_coord = np.arange(-(nx//2), nx//2)
            m_coord = np.arange(-(ny//2), ny//2)
        
            self.Kl = expsq(l_coord, l_coord, sigma0, length_scale)
            self.Km = expsq(m_coord, m_coord, 1.0, length_scale)
            self.Klinv = np.linalg.pinv(self.Kl, hermitian=True, rcond=1e-12)
            self.Kminv = np.linalg.pinv(self.Km, hermitian=True, rcond=1e-12)

        # Kronecker matrices for "fast" matrix vector products
        self.Kkron = (self.Kv, self.Kl, self.Km)
        self.Kinvkron = (self.Kvinv, self.Klinv, self.Kminv)

    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.Khat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        res = Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]
        return res

    def idot(self, x):
        return kron_matvec(self.Kinvkron, x.flatten()).reshape(*x.shape)

    def dot(self, x):
        return kron_matvec(self.Kkron, x.flatten()).reshape(*x.shape)


class Dirac(object):
    def __init__(self, nband, nx, ny, mask=None):
        """
        Models image as a sum of Dirac deltas i.e.
        
        x = H beta

        where H is a design matrix that maps the Dirac coefficients onto the image cube.

        Parameters
        ----------
        nband - number of bands
        nx - number of pixels in x-dimension
        ny - number of pixels in y-dimension
        mask - nx x my bool array containing locations of sources
        """
        self.nx = nx
        self.ny = ny
        self.nband = nband

        if mask is not None:
            self.mask = mask
        else:
            self.mask = lambda x: x

    def dot(self, x):
        """
        Components to image
        """
        return self.mask[None, :, :] * x

    def hdot(self, x):
        """
        Image to components
        """
        return self.mask[None, :, :] * x

    def update_locs(self, model):
        self.mask = np.logical_or(self.mask, mask)

    def trim_fat(self, model):
        self.mask = np.any(model, axis=0)
