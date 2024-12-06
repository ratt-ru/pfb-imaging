from collections import OrderedDict
import concurrent.futures as cf
import numpy as np
import numba
from numba import types, typed
from numba.experimental import jitclass
import pywt
from scipy.datasets import ascent
from pfb.wavelets import coeff_size, signal_size, dwt2d, idwt2d, copyT
from time import time


@numba.njit
def create_dict(items):
    return {k: v for k,v in items}


def psi_band_maker(nx, ny, bases, nlevel):
    Nbasis = len(bases)
    sqrtNbasis = np.sqrt(Nbasis)
    dec_lo = {}
    dec_hi = {}
    rec_lo = {}
    rec_hi = {}
    sx = {}
    sy = {}
    spx = {}
    spy = {}
    Ntotx = {}
    Ntoty = {}
    ix = {}
    iy = {}
    Nxmax = 0
    Nymax = 0
    for wavelet in bases:
        if wavelet == 'self':
            continue
        wvlt = pywt.Wavelet(wavelet)
        dec_lo[wavelet] = np.array(wvlt.filter_bank[0])
        dec_hi[wavelet] = np.array(wvlt.filter_bank[1])
        rec_lo[wavelet] = np.array(wvlt.filter_bank[2])
        rec_hi[wavelet] = np.array(wvlt.filter_bank[3])

        max_level = pywt.dwt_max_level(np.minimum(nx, ny), wavelet)
        if nlevel > max_level:
            raise ValueError(f"The requested decomposition level {nlevel} "
                             "is not possible")

        # bookeeping
        N2Cx = {}
        N2Cy = {}
        Nx = nx
        Ny = ny
        Nxm = 0
        Nym = 0
        sx[wavelet] = typed.List()
        sy[wavelet] = typed.List()
        spx[wavelet] = typed.List()
        spy[wavelet] = typed.List()
        F = int(wavelet[-1])*2  # filter length
        for k in range(nlevel):
            Cx = coeff_size(Nx, F)
            Cy = coeff_size(Ny, F)
            N2Cx[k] = (signal_size(Cx, F), Cx)
            N2Cy[k] = (signal_size(Cy, F), Cy)
            Nxm += Cx
            Nym += Cy
            sx[wavelet].append(Cx)
            sy[wavelet].append(Cy)
            Nx = Cx + Cx%2
            Ny = Cy + Cy%2
            spx[wavelet].append(signal_size(Cx, F))
            spy[wavelet].append(signal_size(Cy, F))
        Nxm += Cx  # last approx coeffs
        Nym += Cy
        Ntotx[wavelet] = Nxm
        Ntoty[wavelet] = Nym
        Nxmax = np.maximum(Nxmax, Nxm)
        Nymax = np.maximum(Nymax, Nym)

        ix[wavelet] = {}
        iy[wavelet] = {}
        lowx = N2Cx[nlevel-1][1]
        lowy = N2Cy[nlevel-1][1]
        ix[wavelet][nlevel-1] = (lowx, 2*lowx)
        iy[wavelet][nlevel-1] = (lowy, 2*lowy)
        lowx *= 2
        lowy *= 2
        for k in reversed(range(nlevel-1)):
            highx = N2Cx[k][1]
            highy = N2Cy[k][1]
            ix[wavelet][k] = (lowx, lowx + highx)
            iy[wavelet][k] = (lowy, lowy + highy)
            lowx += highx
            lowy += highy

        ix[wavelet] = create_dict(tuple(ix[wavelet].items()))
        iy[wavelet] = create_dict(tuple(iy[wavelet].items()))

    # this avoids slow constructor
    ix = create_dict(tuple(ix.items()))
    iy = create_dict(tuple(iy.items()))
    dec_lo = create_dict(tuple(dec_lo.items()))
    dec_hi = create_dict(tuple(dec_hi.items()))
    rec_lo = create_dict(tuple(rec_lo.items()))
    rec_hi = create_dict(tuple(rec_hi.items()))
    sx = create_dict(tuple(sx.items()))
    sy = create_dict(tuple(sy.items()))
    spx = create_dict(tuple(spx.items()))
    spy = create_dict(tuple(spy.items()))
    Ntotx = create_dict(tuple(Ntotx.items()))
    Ntoty = create_dict(tuple(Ntoty.items()))

    # set up buffers
    alpha = np.zeros((Nymax, Nxmax))  # avoid destroying coeff in
    cbuff = np.zeros((Nxmax, Nymax))
    cbuffT = np.zeros((Nymax, Nxmax))
    image = np.zeros((nx, ny))

    return psi_band(alpha, image, cbuff, cbuffT,
                    ix, iy, sx, sy, spx, spy,
                    dec_lo, dec_hi, rec_lo, rec_hi,
                    Ntotx, Ntoty, Nxmax, Nymax, Nbasis,
                    sqrtNbasis, typed.List(bases), nlevel, nx, ny)


kv_ty = (types.unicode_type, types.ListType(numba.int64))
kv_ty2 = (numba.int64, types.UniTuple(types.int64, 2))
kv_ty3 = (types.unicode_type, numba.float64[:])
spec = OrderedDict()
spec['alpha'] = numba.float64[:,:]
spec['image'] = numba.float64[:,:]
spec['cbuff'] = numba.float64[:,:]
spec['cbuffT'] = numba.float64[:,:]
spec['ix'] = types.DictType(types.unicode_type, types.DictType(*kv_ty2))
spec['iy'] = types.DictType(types.unicode_type, types.DictType(*kv_ty2))
spec['sx'] = types.DictType(*kv_ty)
spec['sy'] = types.DictType(*kv_ty)
spec['spx'] = types.DictType(*kv_ty)
spec['spy'] = types.DictType(*kv_ty)
spec['dec_lo'] = types.DictType(*kv_ty3)
spec['dec_hi'] = types.DictType(*kv_ty3)
spec['rec_lo'] = types.DictType(*kv_ty3)
spec['rec_hi'] = types.DictType(*kv_ty3)
spec['Ntotx'] = types.DictType(types.unicode_type, numba.int64)
spec['Ntoty'] = types.DictType(types.unicode_type, numba.int64)
spec['Nxmax'] = numba.int64
spec['Nymax'] = numba.int64
spec['Nbasis'] = numba.int64
spec['sqrtNbasis'] = numba.float64
spec['bases'] = types.ListType(types.unicode_type)
spec['Nlevel'] = numba.int64
spec['Nx'] = numba.int64
spec['Ny'] = numba.int64
@jitclass(spec)
class psi_band(object):
    def __init__(self, alpha, image, cbuff, cbuffT,
                 ix, iy, sx, sy, spx, spy,
                 dec_lo, dec_hi, rec_lo, rec_hi,
                 Ntotx, Ntoty, Nxmax, Nymax, Nbasis,
                 sqrtNbasis, bases, Nlevel, Nx, Ny):

        self.alpha = alpha
        self.image = image
        self.cbuff = cbuff
        self.cbuffT = cbuffT
        self.ix = ix
        self.iy = iy
        self.sx = sx
        self.sy = sy
        self.spx = spx
        self.spy = spy
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.Ntotx = Ntotx
        self.Ntoty = Ntoty
        self.Nxmax = Nxmax
        self.Nymax = Nymax
        self.Nbasis = Nbasis
        self.sqrtNbasis = sqrtNbasis
        self.bases = bases
        self.Nlevel = Nlevel
        self.Nx = Nx
        self.Ny = Ny

    def dot(self, x, alphao):
        '''
        signal to coeffs

        x       - (nx, ny) input signal
        alphao  - (nbasis, Nymax, Nxmax) per basis output coeffs
        '''

        alphao[...] = 0.0

        for i, wavelet in enumerate(self.bases):
            if wavelet=='self':
                copyT(x, alphao[i, 0:self.Ny, 0:self.Nx])
                # alphao[i, 0:self.Ny, 0:self.Nx] = x.T
                continue
            dec_lo = self.dec_lo[wavelet]
            dec_hi = self.dec_hi[wavelet]
            sx = self.sx[wavelet]
            sy = self.sy[wavelet]
            ix = self.ix[wavelet]
            iy = self.iy[wavelet]
            Ntotx = self.Ntotx[wavelet]
            Ntoty = self.Ntoty[wavelet]

            dwt2d(x,
                  alphao[i, 0:Ntoty, 0:Ntotx],
                  self.cbuff[0:Ntotx, 0:Ntoty],
                  self.cbuffT[0:Ntoty, 0:Ntotx],
                  ix, iy, sx, sy, dec_lo, dec_hi,
                  self.Nlevel)

        # alphao /= self.sqrtNbasis

        return alphao


    def hdot(self, alpha, xo):
        '''
        coeffs to signal

        alpha   - (nbasis, Nxmax, Nymax) per basis output coeffs
        xo      - (nx, ny) output signal
        '''
        xo[...] = 0.0  # accumulated
        for i, wavelet in enumerate(self.bases):
            if wavelet=='self':
                copyT(alpha[i, 0:self.Ny, 0:self.Nx], self.image)
                xo += self.image
                continue
            rec_lo = self.rec_lo[wavelet]
            rec_hi = self.rec_hi[wavelet]
            sx = self.sx[wavelet]
            sy = self.sy[wavelet]
            spx = self.spx[wavelet]
            spy = self.spy[wavelet]
            ix = self.ix[wavelet]
            iy = self.iy[wavelet]
            Ntotx = self.Ntotx[wavelet]
            Ntoty = self.Ntoty[wavelet]
            idwt2d(alpha[i, 0:Ntoty, 0:Ntotx],
                   self.image,
                   self.alpha[0:Ntoty, 0:Ntotx],
                   self.cbuff[0:Ntotx, 0:Ntoty],
                   self.cbuffT[0:Ntoty, 0:Ntotx],
                   ix, iy, sx, sy, spx, spy, rec_lo, rec_hi,
                   self.Nlevel)

            xo += self.image

        # xo /= self.sqrtNbasis

        return xo


def psi_dot_impl(x, alphao, psib, b, nthreads=1):
    numba.set_num_threads(nthreads)
    psib.dot(x, alphao)
    return b


def psi_hdot_impl(alpha, xo, psib, b, nthreads=1):
    numba.set_num_threads(nthreads)
    psib.hdot(alpha, xo)
    return b


class Psi(object):
    def __init__(self, nband, nx, ny, bases, nlevel, nthreads):

        self.psib = []
        for b in range(nband):
            self.psib.append(psi_band_maker(nx, ny, bases, nlevel))

        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nbasis = len(bases)
        self.nthreads = nthreads
        self.Nxmax = self.psib[0].Nxmax
        self.Nymax = self.psib[0].Nymax
        self.nthreads_per_band = np.maximum(1, nthreads//nband)

    def dot(self, x, alphao):
        '''
        image to coeffs
        '''
        futures = []
        with cf.ThreadPoolExecutor(max_workers=self.nband) as executor:
            for b in range(self.nband):
                f = executor.submit(psi_dot_impl, x[b], alphao[b], self.psib[b], b,
                                    nthreads=self.nthreads_per_band)
                futures.append(f)

            # wait for result
            for f in cf.as_completed(futures):
                b = f.result()

    def hdot(self, alpha, xo):
        '''
        coeffs to image
        '''
        futures = []
        with cf.ThreadPoolExecutor(max_workers=self.nband) as executor:
            for b in range(self.nband):
                f = executor.submit(psi_hdot_impl, alpha[b], xo[b], self.psib[b], b,
                                    nthreads=self.nthreads_per_band)
                futures.append(f)

            # wait for result
            for f in cf.as_completed(futures):
                b = f.result()
