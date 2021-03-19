import numpy as np
import pywt
import dask.array as da
# from pfb.wavelets.wavelets import wavedecn, waverecn, ravel_coeffs, unravel_coeffs


class Theta(object):
    def __init__(self, nband, nx, ny):
        """
        Sets up operators to move between wavelet coefficients
        in each basis and the image x.

        Parameters
        ----------
        nband - number of bands
        nx - number of pixels in x-dimension
        ny - number of pixels in y-dimension
        nlevels - The level of the decomposition. Default=2
        bases - List holding basis names.
                Default is db1-8 wavelets
        """
        self.real_type = np.float64
        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nlevels = 3
        self.bases = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']
        self.P = len(self.bases)
        self.sqrtP = np.sqrt(self.P)
        self.nbasis = len(self.bases)

        # do a mock decomposition to get coefficient info
        x = np.random.randn(nx, ny)
        self.ntot = []
        self.iy = {}
        self.sy = {}
        for i, b in enumerate(self.bases):
            alpha = pywt.wavedecn(x, b, mode='zero', level=self.nlevels)
            y, iy, sy = pywt.ravel_coeffs(alpha)
            self.iy[b] = iy
            self.sy[b] = sy
            self.ntot.append(y.size)

        # get padding info
        self.nmax = np.asarray(self.ntot).max()
        self.dpadding = slice(0, nx*ny)  # Dirac slices/padding
        self.padding = []
        for i in range(self.nbasis):
            self.padding.append(slice(0, self.ntot[i]))

    def dot(self, coeffs):
        """
        Takes array of coefficients to image.
        alpha comes in as a raveled array of coefficients and has to be unraveled
        before passing to waverecn.  
        """
        x = np.zeros((2, self.nband, self.nx, self.ny), dtype=self.real_type)
        x[0] = coeffs[0, :, self.dpadding].reshape(
            self.nband, self.nx, self.ny)  # Dirac components
        alpha = coeffs[1::]  # wavelet coefficients
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                # unpad
                a = alpha[b, l, self.padding[b]]
                # unravel and rec
                alpha_rec = pywt.unravel_coeffs(
                    a, self.iy[base], self.sy[base])
                wave = pywt.waverecn(alpha_rec, base, mode='zero')

                # accumulate
                x[1, l] += wave / self.sqrtP
        return x

    def hdot(self, x):
        """
        Implements the adjoint i.e. image to coeffs.
        Per basis and band coefficients are raveled and padded so they can be stacked
        into a single array.
        """
        beta = x[0]
        alpha = x[1]
        coeffs = np.zeros((self.nbasis+1, self.nband, self.nmax))
        coeffs[0] = np.pad(beta.reshape(self.nband, self.nx*self.ny),
                           ((0, 0), (0, self.nmax-self.nx*self.ny)), mode='constant')
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                # decompose
                alphal = pywt.wavedecn(
                    alpha[l], base, mode='zero', level=self.nlevels)
                # ravel and pad
                tmp, _, _ = pywt.ravel_coeffs(alphal)
                coeffs[b+1, l] = np.pad(tmp/self.sqrtP,
                                        (0, self.nmax-self.ntot[b]), mode='constant')
        return coeffs


def _dot_internal(alpha, bases, padding, iy, sy, sqrtP, nx, ny, real_type):
    nbasis, nband, _ = alpha.shape
    # reduction over basis done externally since chunked
    x = np.zeros((nbasis, nband, nx, ny), dtype=real_type)
    for b in range(nbasis):
        base = bases[b]
        for l in range(nband):
            a = alpha[b, l, padding[b]]
            alpha_rec = pywt.unravel_coeffs(
                a, iy[base], sy[base], output_format='wavedecn')
            wave = pywt.waverecn(alpha_rec, base, mode='zero')

            x[b, l] += wave / sqrtP
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
            # decompose
            alphal = pywt.wavedecn(x[l], base, mode='zero', level=nlevels)
            # ravel and pad
            tmp, _, _ = pywt.ravel_coeffs(alphal)
            alpha[b, l] = np.pad(tmp/sqrtP, (0, nmax-ntot[b]), mode='constant')

    return alpha


def _hdot_internal_wrapper(x, bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type):
    return _hdot_internal(x[0][0], bases, ntot, nmax, nlevels, sqrtP, nx, ny, real_type)


class DaskTheta(Theta):
    def __init__(self, nband, nx, ny, nthreads=8):
        Theta.__init__(self, nband, nx, ny)
        # required to chunk over basis
        bases = np.array(self.bases, dtype=object)
        self.bases = da.from_array(bases, chunks=1)
        padding = np.array(self.padding, dtype=object)
        self.padding = da.from_array(padding, chunks=1)
        ntot = np.array(self.ntot, dtype=object)
        self.ntot = da.from_array(ntot, chunks=1)

    def dot(self, coeffs):
        x0 = coeffs[0, :, self.dpadding].reshape(
            self.nband, self.nx, self.ny)  # Dirac components
        alpha_dask = da.from_array(
            coeffs[1::], chunks=(1, self.nband, self.nmax))
        x1 = da.blockwise(_dot_internal_wrapper, ("basis", "band", "nx", "ny"),
                          alpha_dask, ("basis", "band", "ntot"),
                          self.bases, ("basis",),
                          self.padding, ("basis",),
                          self.iy, None,  # ("basis",),
                          self.sy, None,  # ("basis",),
                          self.sqrtP, None,
                          self.nx, None,
                          self.ny, None,
                          self.real_type, None,
                          new_axes={"nx": self.nx, "ny": self.ny},
                          dtype=self.real_type,
                          align_arrays=False)
        x1 = x1.sum(axis=0).compute(shedular='processes')
        return np.concatenate((x0[None], x1[None]), axis=0)

    def hdot(self, x):
        beta = np.pad(x[0].reshape(self.nband, self.nx*self.ny),
                      ((0, 0), (0, self.nmax-self.nx*self.ny)), mode='constant')
        xdask = da.from_array(x[1], chunks=(self.nband, self.nx, self.ny))
        alpha = da.blockwise(_hdot_internal_wrapper, ("basis", "band", "nmax"),
                             xdask, ("band", "nx", "ny"),
                             self.bases, ("basis", ),
                             self.ntot, ("basis", ),
                             self.nmax, None,
                             self.nlevels, None,
                             self.sqrtP, None,
                             self.nx, None,
                             self.ny, None,
                             self.real_type, None,
                             new_axes={"nmax": self.nmax},
                             dtype=self.real_type,
                             align_arrays=False)
        alpha = alpha.compute(shedular='processes')

        return np.concatenate((beta[None], alpha), axis=0)
