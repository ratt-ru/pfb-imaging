import numpy as np
from africanus.gps.kernels import exponential_squared as expsq
from ducc0.fft import c2r, r2c
from numba import njit, prange

from pfb_imaging.utils.misc import kron_matvec

ifftshift = np.fft.ifftshift
fftshift = np.fft.fftshift


@njit(parallel=True, nogil=True, cache=True, inline="always")
def freqmul(aop, x):
    nchan, npix = x.shape
    out = np.zeros((nchan, npix), dtype=x.dtype)
    for i in prange(npix):
        for j in range(nchan):
            for k in range(nchan):
                out[j, i] += aop[j, k] * x[k, i]
    return out


@njit(parallel=True, nogil=True, cache=True, inline="always")
def make_kernel(nx_psf, ny_psf, sigma0, length_scale):
    cov = np.zeros((1, nx_psf, ny_psf), dtype=np.float64)
    for j in range(nx_psf):
        for k in range(ny_psf):
            l_coord = float(j - (nx_psf // 2))
            m_coord = float(k - (ny_psf // 2))
            cov[0, j, k] = sigma0**2 * np.exp(-(l_coord**2 + m_coord**2) / (2 * length_scale**2))
    return cov


class MockArray(object):
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


class Gauss(object):
    def __init__(self, sigma0, nband, nx, ny, nthreads=8):
        self.nthreads = nthreads
        self.nx = nx
        self.ny = ny
        nx_psf = 2 * self.nx
        npad_x = (nx_psf - nx) // 2
        ny_psf = 2 * self.ny
        npad_y = (ny_psf - ny) // 2
        self.padding = ((0, 0), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (1, 2)

        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])

        # set length scales
        length_scale = 0.5

        cov = make_kernel(nx_psf, ny_psf, sigma0, length_scale)

        self.cov = cov
        cov_pad = ifftshift(self.cov, axes=self.ax)
        self.covhat = r2c(cov_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)
        self.covhatinv = np.where(self.covhat.real > 1e-14, 1.0 / self.covhat, 1e-14)

        # get covariance in each dimension
        # pixel coordinates
        self.covnu = MockArray(nband)  # np.eye(nband) * sigma0**2
        self.covnuinv = MockArray(nband)  # np.eye(nband) / sigma0**2
        if nx == ny:
            l_coord = m_coord = np.arange(-(nx // 2), nx // 2)
            self.covl = self.covm = expsq(l_coord, l_coord, 1.0, length_scale)
            self.covlinv = self.covminv = np.linalg.pinv(self.covl, hermitian=True, rcond=1e-12)
            self.covl *= sigma0**2
            self.covlinv /= sigma0**2
        else:
            l_coord = np.arange(-(nx // 2), nx // 2)
            m_coord = np.arange(-(ny // 2), ny // 2)

            self.covl = expsq(l_coord, l_coord, sigma0, length_scale)
            self.covm = expsq(m_coord, m_coord, 1.0, length_scale)
            self.covlinv = np.linalg.pinv(self.covl, hermitian=True, rcond=1e-12)
            self.covminv = np.linalg.pinv(self.covm, hermitian=True, rcond=1e-12)

        # Kronecker matrices for "fast" matrix vector products
        self.covkron = (self.covnu, self.covl, self.covm)
        self.covinvkron = (self.covnuinv, self.covlinv, self.covminv)

    def convolve(self, x):
        xhat = ifftshift(np.pad(x, self.padding, mode="constant"), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(
            xhat * self.Khat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads
        )
        res = fftshift(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]
        return res

    def idot(self, x):
        return kron_matvec(self.covinvkron, x.flatten()).reshape(*x.shape)

    def dot(self, x):
        return kron_matvec(self.covkron, x.flatten()).reshape(*x.shape)
