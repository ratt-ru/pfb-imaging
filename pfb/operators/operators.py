from numba import njit, prange
import numpy as np
import dask.array as da
from daskms import xds_from_table
import pywt
from ducc0.wgridder import ms2dirty, dirty2ms
from ducc0.fft import r2c, c2r, c2c
from africanus.gps.kernels import exponential_squared as expsq

# from pfb.wavelets.wavelets import wavedecn, waverecn, ravel_coeffs, unravel_coeffs

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
def make_kernel(nv_psf, nx_psf, ny_psf, sigma0, length_scale):
    K = np.zeros((nv_psf, nx_psf, ny_psf), dtype=np.float64)
    for i in range(nv_psf):
        for j in range(nx_psf):
            for k in range(ny_psf):
                v = float(i - (nv_psf//2))
                l = float(j - (nx_psf//2))
                m = float(k - (ny_psf//2))
                K[i,j,k] = sigma0**2*np.exp(-(v**2+l**2+m**2)/(2*length_scale**2))
    return K


# eventually goes in africanus
@njit(fastmath=True, inline='always')
def kron_matvec(A, b):
    """
    Computes the matrix vector product of
    a kronecker matrix in linear time.
    Assumes A consists of kronecker product
    of square matrices.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        An array of arrays holding
        matrices [K0, K1, ...] where
        :math:`A = K_0 \\otimes K_1 \\otimes \\cdots`
    b : :class:`numpy.ndarray`
        The right hand side vector

    Returns
    -------
    x : :class:`numpy.ndarray`
        The result of :code:`A.dot(b)`
    """
    D = len(A)
    N = b.size
    x = b
    for d in range(D):
        Gd = A[d].shape[0]
        X = np.reshape(x, (Gd, N//Gd))
        Z = np.zeros((Gd, N//Gd), dtype=A[0].dtype)
        for i in range(Gd):
            for j in range(N//Gd):
                for k in range(Gd):
                    Z[i, j] += A[d][i, k] * X[k, j]
        x = Z.T.flatten()
    return x


class PSF(object):
    def __init__(self, psf, nthreads, sigma0=1.0):
        self.nthreads = nthreads
        self.nband, nx_psf, ny_psf = psf.shape
        nx = nx_psf//2
        ny = ny_psf//2
        npad_x = (nx_psf - nx)//2
        npad_y = (ny_psf - ny)//2
        self.padding = ((0,0), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (1,2)
        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)

    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]

class Prior(object):
    def __init__(self, sigma0, nband, nx, ny, nthreads=8):
        self.nthreads = nthreads
        self.nx = nx
        self.ny = ny
        self.nv = nband
        if nband > 1:
            nv_psf = 2*self.nv
            npad_v = (nv_psf - nband)//2
            self.unpad_v = slice(npad_v, -npad_v)
        else:
            nv_psf = 1
            npad_v = 0
            self.unpad_v = slice(None)
        nx_psf = 2*self.nx
        npad_x = (nx_psf - nx)//2
        ny_psf = 2*self.ny
        npad_y = (ny_psf - ny)//2
        self.padding = ((npad_v,npad_v), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (0, 1,2)
        
        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])
        
        # always work in pixel coordinates
        if nband > 1:
            v_coord = np.arange(-(nv_psf//2), nv_psf//2)
        else:
            v_coord = np.array([1.0])
        v_coord = np.arange(-(nv_psf//2), nv_psf//2)
        l_coord = np.arange(-(nx_psf//2), nx_psf//2)
        m_coord = np.arange(-(ny_psf//2), ny_psf//2)

        # set length scales
        length_scale = 1.5

        K = make_kernel(nv_psf, nx_psf, ny_psf, sigma0, length_scale)

        self.K = K
        K_pad = iFs(self.K, axes=self.ax)
        self.Khat = r2c(K_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)


        # get covariance in each dimension
        if nband > 1:
            v_coord = np.arange(-(nband//2), nband//2)
        else:
            v_coord = np.array([1.0])
        l_coord = np.arange(-(nx//2), nx//2)
        m_coord = np.arange(-(ny//2), ny//2)
        self.Kv = expsq(v_coord, v_coord, sigma0, length_scale) # + 1e-6*np.eye(nband)
        self.Kl = expsq(l_coord, l_coord, 1.0, length_scale) # + 1e-6*np.eye(nx)
        self.Km = expsq(m_coord, m_coord, 1.0, length_scale) # + 1e-6*np.eye(ny)
        
        # explicit inverses
        self.Kvinv = np.linalg.pinv(self.Kv)  
        self.Klinv = np.linalg.pinv(self.Kl)
        self.Kminv = np.linalg.pinv(self.Km)

        # # Kronecker matrices for fast matrix vector products
        self.Kkron = (self.Kv, self.Kl, self.Km)
        self.Kinvkron = (self.Kvinv, self.Klinv, self.Kminv)

    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.Khat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        res = Fs(xhat, axes=self.ax)[self.unpad_v, self.unpad_x, self.unpad_y]
        return res

    def iconvolve(self, x):
        from pfb.opt import pcg
        return pcg(self.convolve, 
                   x, np.zeros(x.shape), 
                   M=lambda x:x, tol=1e-5,
                   maxit=100, verbosity=1)

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
        self.mask = mask

    def dot(self, x):
        """
        Components to image
        """
        return np.where(self.mask[None, :, :], x, 0)

    def hdot(self, x):
        """
        Image to components
        """
        return np.where(self.mask[None, :, :], x, 0)

    def update_locs(self, mask):
        self.mask = np.logical_or(self.mask, mask)

    def trim_fat(self, model):
        self.mask = np.any(model, axis=0)
    

class PSI(object):
    def __init__(self, nband, nx, ny,
                 nlevels=2,
                 bases=['self', 'db1', 'db2', 'db3', 'db4']):
        """
        Sets up operators to move between wavelet coefficients
        in each basis and the image x.

        Parameters
        ----------
        nband - number of bands
        nx - number of pixels in x-dimension
        ny - number of pixels in y-dimension
        nlevels - The level of the decomposition. Default=2
        basis - List holding basis names.
                Default is db1-4 wavelets
                Supports any subset of
                ['self', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']

        Returns
        =======
        Psi - list of operators performing coeff to image where
            each entry corresponds to one of the basis elements.
        Psi_t - list of operators performing image to coeff where
                each entry corresponds to one of the basis elements.
        """
        self.real_type = np.float64
        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nlevels = nlevels
        self.P = len(bases)
        self.sqrtP = np.sqrt(self.P)
        self.bases = bases
        self.nbasis = len(bases)

        # do a mock decomposition to get max coeff size
        x = np.random.randn(nx, ny)
        self.ntot = []
        self.iy = {}
        self.sy = {}
        for i, b in enumerate(bases):
            if b=='self':
                alpha = x.flatten()
                y, iy, sy = x.flatten(), 0, 0
            else:
                alpha = pywt.wavedecn(x, b, mode='zero', level=self.nlevels)
                y, iy, sy = pywt.ravel_coeffs(alpha)
            self.iy[b] = iy
            self.sy[b] = sy
            self.ntot.append(y.size)
        
        # get padding info
        self.nmax = np.asarray(self.ntot).max()
        self.padding = []
        for i in range(self.nbasis):
            self.padding.append(slice(0, self.ntot[i]))


    def dot(self, alpha):
        """
        Takes array of coefficients to image.
        The input does not have the form expected by pywt
        so we have to reshape it. Comes in as a flat vector
        arranged as

        [cAn, cHn, cVn, cDn, ..., cH1, cV1, cD1]

        where each entry is a flattened array and n denotes the
        level of the decomposition. This has to be restructured as

        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)]

        where entries are arrays with size defined by set_index_scheme.
        """
        x = np.zeros((self.nband, self.nx, self.ny), dtype=self.real_type)
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                # unpad
                a = alpha[b, l, self.padding[b]]
                if base == 'self':
                    wave = a.reshape(self.nx, self.ny)
                else:
                    # unravel and rec
                    alpha_rec = pywt.unravel_coeffs(a, self.iy[b], self.sy[b])
                    wave = pywt.waverecn(alpha_rec, base, mode='zero')

                # accumulate
                x[l] += wave / self.sqrtP
        return x

    def hdot(self, x):
        """
        This implements the adjoint of Psi_func i.e. image to coeffs
        """
        alpha = np.zeros((self.nbasis, self.nband, self.nmax))
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                if base == 'self':
                    # just pad image to have same shape as flattened wavelet coefficients
                    alpha[b, l] = np.pad(x[l].reshape(self.nx*self.ny)/self.sqrtP, (0, self.nmax-self.ntot[b]), mode='constant')
                else:
                    # decompose
                    alphal = pywt.wavedecn(x[l], base, mode='zero', level=self.nlevels)
                    # ravel and pad
                    tmp, _, _ = pywt.ravel_coeffs(alphal)
                    alpha[b, l] = np.pad(tmp/self.sqrtP, (0, self.nmax-self.ntot[b]), mode='constant')
        return alpha


import dask.array as da

def _dot_internal(alpha, bases, padding, iy, sy, sqrtP, nx, ny, real_type):
    nbasis, nband, _ = alpha.shape
    # reduction over basis done externally since chunked
    x = np.zeros((nbasis, nband, nx, ny), dtype=real_type)
    for b in range(nbasis):
            base = bases[b]
            for l in range(nband):
                a = alpha[b, l, padding[b]]
                if base == 'self':
                    wave = a.reshape(nx, ny)
                else:
                    alpha_rec = pywt.unravel_coeffs(a, iy[base], sy[base], output_format='wavedecn')
                    wave = pywt.waverecn(alpha_rec, base, mode='zero')

                x[l] += wave / sqrtP
    return x

def _dot_internal_wrapper(alpha, bases, padding, iy, sy, sqrtP, nx, ny, real_type):
    return _dot_internal(alpha[0], bases, padding, iy, sy, sqrtP, nx, ny, real_type)

def _hdot_internal(x, bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type):
    nband = x.shape[0]
    nbasis = len(bases)
    alpha = np.zeros((nbasis, nband, nmax), dtype=real_type)
    for b in range(nbasis):
            base = bases[b]
            for l in range(nband):
                if base == 'self':
                    # ravel and pad
                    alpha[b, l] = np.pad(x[l].reshape(nx*ny)/sqrtP, (0, nmax-ntot[b]), mode='constant')
                else:
                    # decompose
                    alphal = pywt.wavedecn(x[l], base, mode='zero', level=nlevels)
                    # ravel and pad
                    tmp, _, _ = pywt.ravel_coeffs(alphal)
                    alpha[b, l] = np.pad(tmp/sqrtP, (0, nmax-ntot[b]), mode='constant')

    return alpha

def _hdot_internal_wrapper(x, bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type):
    return _hdot_internal(x[0][0], bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type)

class DaskPSI(PSI):
    def __init__(self, nband, nx, ny,
                 nlevels=2,
                 bases=['self', 'db1', 'db2', 'db3', 'db4'],
                 nthreads=8):
        PSI.__init__(self, nband, nx, ny, nlevels=nlevels,
                     bases=bases)
        # required to chunk over basis
        bases = np.array(self.bases, dtype=object)
        self.bases = da.from_array(bases, chunks=1)
        padding = np.array(self.padding, dtype=object)
        self.padding = da.from_array(padding, chunks=1)
        # iy = np.array(self.iy, dtype=object)
        # self.iy = da.from_array(iy, chunks=1)
        # sy = np.array(self.sy, dtype=object)
        # self.sy = da.from_array(sy, chunks=1)
        ntot = np.array(self.ntot, dtype=object)
        self.ntot = da.from_array(ntot, chunks=1)
        
    def dot(self, alpha):
        alpha_dask = da.from_array(alpha, chunks=(1, 1, self.nmax))
        x = da.blockwise(_dot_internal_wrapper, ("basis", "band", "nx", "ny"),
                         alpha_dask, ("basis", "band", "ntot"),
                         self.bases, ("basis",),
                         self.padding, ("basis",),
                         self.iy, None, #("basis",),
                         self.sy, None, # ("basis",),
                         self.sqrtP, None,
                         self.nx, None,
                         self.ny, None,
                         self.real_type, None,
                         new_axes={"nx": self.nx, "ny": self.ny},
                         dtype=self.real_type,
                         align_arrays=False)

        return x.sum(axis=0).compute()

    def hdot(self, x):
        xdask = da.from_array(x, chunks=(1, self.nx, self.ny))
        alpha = da.blockwise(_hdot_internal_wrapper, ("basis", "band", "nmax"),
                             xdask, ("band", "nx", "ny"),
                             self.bases, ("basis", ),
                             self.ntot,("basis", ),
                             self.nmax, None,
                             self.nlevels, None,
                             self.sqrtP, None,
                             self.nx, None,
                             self.ny, None,
                             self.real_type, None,
                             new_axes={"nmax": self.nmax},
                             dtype=self.real_type,
                             align_arrays=False)

        return alpha.compute()
