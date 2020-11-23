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
        x[0] = coeffs[0, :, self.dpadding].reshape(self.nband, self.nx, self.ny)  # Dirac components
        alpha = coeffs[1::]  # wavelet coefficients
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                # unpad
                a = alpha[b, l, self.padding[b]]
                # unravel and rec
                alpha_rec = pywt.unravel_coeffs(a, self.iy[base], self.sy[base])
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
        coeffs[0] = np.pad(beta.reshape(self.nband, self.nx*self.ny), ((0,0),(0, self.nmax-self.nx*self.ny)), mode='constant')
        for b in range(self.nbasis):
            base = self.bases[b]
            for l in range(self.nband):
                # decompose
                alphal = pywt.wavedecn(alpha[l], base, mode='zero', level=self.nlevels)
                # ravel and pad
                tmp, _, _ = pywt.ravel_coeffs(alphal)
                coeffs[b+1, l] = np.pad(tmp/self.sqrtP, (0, self.nmax-self.ntot[b]), mode='constant')
        return coeffs