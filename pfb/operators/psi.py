import numpy as np
import pywt
import dask.array as da
# from pfb.wavelets.wavelets import wavedecn, waverecn, ravel_coeffs, unravel_coeffs

class PSI(object):
    def __init__(self, imsize=None,
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
        if imsize is None:
            raise ValueError("You must initialise imsize")
        else:
            self.nband, self.nx, self.ny = imsize
        self.nlevels = nlevels
        self.P = len(bases)
        self.sqrtP = np.sqrt(self.P)
        self.bases = bases
        self.nbasis = len(bases)

        # do a mock decomposition to get max coeff size
        x = np.random.randn(self.nx, self.ny)
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
        alpha comes in as a raveled array of coefficients and has to be unraveled
        before passing to waverecn.  
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
                    alpha_rec = pywt.unravel_coeffs(a, self.iy[base], self.sy[base])
                    wave = pywt.waverecn(alpha_rec, base, mode='zero')

                # accumulate
                x[l] += wave / self.sqrtP
        return x

    def hdot(self, x):
        """
        This implements the adjoint of Psi_func i.e. image to coeffs.
        Per basis and band coefficients are raveled padded so they can be stacked
        into a single array.
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
    def __init__(self, imsize=None,
                 nlevels=2,
                 bases=['self', 'db1', 'db2', 'db3', 'db4'],
                 nthreads=8):
        PSI.__init__(self, imsize, nlevels=nlevels,
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
        alpha_dask = da.from_array(alpha, chunks=(1, 1, self.nmax)) #, name='myname-' + hash(id(alpha)))
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

        return x.sum(axis=0).compute()  # scheduler='processes'

    def hdot(self, x):
        xdask = da.from_array(x, chunks=(1, self.nx, self.ny)) #, name='myname-' + hash(id(x)))
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

        return alpha.compute()  # scheduler='processes'