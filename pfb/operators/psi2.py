import concurrent.futures as cf
import numpy as np
import pywt
from pfb.wavelets.wavelets_jsk import (get_buffer_size,
                                       dwt2d, idwt2d)


def dwt(x, buffer, dec_lo, dec_hi, nlevel, i):
    dwt2d(x, buffer, dec_lo, dec_hi, nlevel)
    return buffer, i


def idwt(buffer, rec_lo, rec_hi, nlevel, nx, ny, i):
    image = np.zeros((nx, ny), dtype=buffer.dtype)
    idwt2d(buffer, image, rec_lo, rec_hi, nlevel)
    return image, i


class psi_band(object):
    def __init__(self, bases, nlevel, nx, ny, nthreads):
        self.bases = bases
        self.nbasis = len(bases)
        self.nlevel = nlevel
        self.nx = nx
        self.ny = ny
        self.nthreads = nthreads
        self.dec_lo ={}
        self.dec_hi ={}
        self.rec_lo ={}
        self.rec_hi ={}
        self.buffer_size = {}
        self.buffer = {}
        self.nmax = 0
        for wavelet in bases:
            if wavelet=='self':
                self.buffer_size[wavelet] = nx*ny
                continue
            # Get the required filter banks from pywt.
            wvlt = pywt.Wavelet(wavelet)
            self.dec_lo[wavelet] = np.array(wvlt.filter_bank[0])  # Low pass, decomposition.
            self.dec_hi[wavelet] = np.array(wvlt.filter_bank[1])  # Hi pass, decomposition.
            self.rec_lo[wavelet] = np.array(wvlt.filter_bank[2])  # Low pass, recon.
            self.rec_hi[wavelet] = np.array(wvlt.filter_bank[3])  # Hi pass, recon.

            self.buffer_size[wavelet] = get_buffer_size((nx, ny),
                                                        self.dec_lo[wavelet].size,
                                                        nlevel)
            self.buffer[wavelet] = np.zeros(self.buffer_size[wavelet],
                                            dtype=np.float64)

            self.nmax = np.maximum(self.nmax, self.buffer_size[wavelet])


    def dot(self, x):
        '''
        signal to coeffs

        Input:
            x       - (nx, ny) input signal

        Output:
            alpha   - (nbasis, Nnmax) output coeffs
        '''
        alpha = np.zeros((self.nbasis, self.nmax),
                         dtype=x.dtype)

        with cf.ThreadPoolExecutor(max_workers=self.nthreads) as executor:
            futures = []
            for i, wavelet in enumerate(self.bases):
                if wavelet=='self':
                    alpha[i, 0:self.buffer_size[wavelet]] = x.ravel()
                    continue

                f = executor.submit(dwt, x,
                                    self.buffer[wavelet],
                                    self.dec_lo[wavelet],
                                    self.dec_hi[wavelet],
                                    self.nlevel,
                                    i)

                futures.append(f)

            for f in cf.as_completed(futures):
                buffer, i = f.result()
                alpha[i, 0:self.buffer_size[self.bases[i]]] = buffer

        return alpha

    def hdot(self, alpha):
        '''
        coeffs to signal

        Input:
            alpha   - (nbasis, Nxmax, Nymax) per basis output coeffs

        Output:
            x       - (nx, ny) output signal
        '''
        nx = self.nx
        ny = self.ny
        x = np.zeros((nx, ny), dtype=alpha.dtype)

        with cf.ThreadPoolExecutor(max_workers=self.nthreads) as executor:
            futures = []
            for i, wavelet in enumerate(self.bases):
                nmax = self.buffer_size[wavelet]
                if wavelet=='self':
                    x += alpha[i, 0:nmax].reshape(nx, ny)
                    continue

                f = executor.submit(idwt,
                                    alpha[i, 0:nmax],
                                    self.rec_lo[wavelet],
                                    self.rec_hi[wavelet],
                                    self.nlevel,
                                    self.nx,
                                    self.ny,
                                    i)


                futures.append(f)

            for f in cf.as_completed(futures):
                image, i = f.result()
                x += image

        return x
