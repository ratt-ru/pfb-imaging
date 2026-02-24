import concurrent.futures as cf
from collections import OrderedDict

import numba
import numpy as np
import pywt
from numba import typed, types
from numba.experimental import jitclass

from pfb_imaging.wavelets import coeff_size, copyt, dwt2d, idwt2d, signal_size


@numba.njit
def create_dict(items):
    return {k: v for k, v in items}


def psi_band_maker(nxi, nyi, bases, nlevel):
    nbasis = len(bases)
    sqrtnbasis = np.sqrt(nbasis)
    dec_lo = {}
    dec_hi = {}
    rec_lo = {}
    rec_hi = {}
    sx = {}
    sy = {}
    spx = {}
    spy = {}
    ntotx = {}
    ntoty = {}
    ix = {}
    iy = {}
    nxmax = 0
    nymax = 0
    for wavelet in bases:
        if wavelet == "self":
            continue
        wvlt = pywt.Wavelet(wavelet)
        dec_lo[wavelet] = np.array(wvlt.filter_bank[0])
        dec_hi[wavelet] = np.array(wvlt.filter_bank[1])
        rec_lo[wavelet] = np.array(wvlt.filter_bank[2])
        rec_hi[wavelet] = np.array(wvlt.filter_bank[3])

        max_level = pywt.dwt_max_level(np.minimum(nxi, nyi), wavelet)
        if nlevel > max_level:
            raise ValueError(f"The requested decomposition level {nlevel} is not possible")

        # bookeeping
        n2cx = {}
        n2cy = {}
        nx = nxi
        ny = nyi
        nxm = 0
        nym = 0
        sx[wavelet] = typed.List()
        sy[wavelet] = typed.List()
        spx[wavelet] = typed.List()
        spy[wavelet] = typed.List()
        filter_length = int(wavelet[-1]) * 2  # filter length
        for k in range(nlevel):
            cx = coeff_size(nx, filter_length)
            cy = coeff_size(ny, filter_length)
            n2cx[k] = (signal_size(cx, filter_length), cx)
            n2cy[k] = (signal_size(cy, filter_length), cy)
            nxm += cx
            nym += cy
            sx[wavelet].append(cx)
            sy[wavelet].append(cy)
            nx = cx + cx % 2
            ny = cy + cy % 2
            spx[wavelet].append(signal_size(cx, filter_length))
            spy[wavelet].append(signal_size(cy, filter_length))
        nxm += cx  # last approx coeffs
        nym += cy
        ntotx[wavelet] = nxm
        ntoty[wavelet] = nym
        nxmax = np.maximum(nxmax, nxm)
        nymax = np.maximum(nymax, nym)

        ix[wavelet] = {}
        iy[wavelet] = {}
        lowx = n2cx[nlevel - 1][1]
        lowy = n2cy[nlevel - 1][1]
        ix[wavelet][nlevel - 1] = (lowx, 2 * lowx)
        iy[wavelet][nlevel - 1] = (lowy, 2 * lowy)
        lowx *= 2
        lowy *= 2
        for k in reversed(range(nlevel - 1)):
            highx = n2cx[k][1]
            highy = n2cy[k][1]
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
    ntotx = create_dict(tuple(ntotx.items()))
    ntoty = create_dict(tuple(ntoty.items()))

    # set up buffers
    alpha = np.zeros((nymax, nxmax))  # avoid destroying coeff in
    cbuff = np.zeros((nxmax, nymax))
    cbufft = np.zeros((nymax, nxmax))
    image = np.zeros((nxi, nyi))

    return PsiBand(
        alpha,
        image,
        cbuff,
        cbufft,
        ix,
        iy,
        sx,
        sy,
        spx,
        spy,
        dec_lo,
        dec_hi,
        rec_lo,
        rec_hi,
        ntotx,
        ntoty,
        nxmax,
        nymax,
        nbasis,
        sqrtnbasis,
        typed.List(bases),
        nlevel,
        nxi,
        nyi,
    )


kv_ty = (types.unicode_type, types.ListType(numba.int64))
kv_ty2 = (numba.int64, types.UniTuple(types.int64, 2))
kv_ty3 = (types.unicode_type, numba.float64[:])
spec = OrderedDict()
spec["alpha"] = numba.float64[:, :]
spec["image"] = numba.float64[:, :]
spec["cbuff"] = numba.float64[:, :]
spec["cbufft"] = numba.float64[:, :]
spec["ix"] = types.DictType(types.unicode_type, types.DictType(*kv_ty2))
spec["iy"] = types.DictType(types.unicode_type, types.DictType(*kv_ty2))
spec["sx"] = types.DictType(*kv_ty)
spec["sy"] = types.DictType(*kv_ty)
spec["spx"] = types.DictType(*kv_ty)
spec["spy"] = types.DictType(*kv_ty)
spec["dec_lo"] = types.DictType(*kv_ty3)
spec["dec_hi"] = types.DictType(*kv_ty3)
spec["rec_lo"] = types.DictType(*kv_ty3)
spec["rec_hi"] = types.DictType(*kv_ty3)
spec["ntotx"] = types.DictType(types.unicode_type, numba.int64)
spec["ntoty"] = types.DictType(types.unicode_type, numba.int64)
spec["nxmax"] = numba.int64
spec["nymax"] = numba.int64
spec["nbasis"] = numba.int64
spec["sqrtnbasis"] = numba.float64
spec["bases"] = types.ListType(types.unicode_type)
spec["nlevel"] = numba.int64
spec["nx"] = numba.int64
spec["ny"] = numba.int64


@jitclass(spec)
class PsiBand(object):
    def __init__(
        self,
        alpha,
        image,
        cbuff,
        cbufft,
        ix,
        iy,
        sx,
        sy,
        spx,
        spy,
        dec_lo,
        dec_hi,
        rec_lo,
        rec_hi,
        ntotx,
        ntoty,
        nxmax,
        nymax,
        nbasis,
        sqrtnbasis,
        bases,
        nlevel,
        nx,
        ny,
    ):
        self.alpha = alpha
        self.image = image
        self.cbuff = cbuff
        self.cbufft = cbufft
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
        self.ntotx = ntotx
        self.ntoty = ntoty
        self.nxmax = nxmax
        self.nymax = nymax
        self.nbasis = nbasis
        self.sqrtnbasis = sqrtnbasis
        self.bases = bases
        self.nlevel = nlevel
        self.nx = nx
        self.ny = ny

    def dot(self, x, alphao):
        """
        signal to coeffs

        x       - (nx, ny) input signal
        alphao  - (nbasis, nymax, nxmax) per basis output coeffs
        """

        alphao[...] = 0.0

        for i, wavelet in enumerate(self.bases):
            if wavelet == "self":
                copyt(x, alphao[i, 0 : self.ny, 0 : self.nx])
                # alphao[i, 0:self.ny, 0:self.nx] = x.T
                continue
            dec_lo = self.dec_lo[wavelet]
            dec_hi = self.dec_hi[wavelet]
            sx = self.sx[wavelet]
            sy = self.sy[wavelet]
            ix = self.ix[wavelet]
            iy = self.iy[wavelet]
            ntotx = self.ntotx[wavelet]
            ntoty = self.ntoty[wavelet]

            dwt2d(
                x,
                alphao[i, 0:ntoty, 0:ntotx],
                self.cbuff[0:ntotx, 0:ntoty],
                self.cbufft[0:ntoty, 0:ntotx],
                ix,
                iy,
                sx,
                sy,
                dec_lo,
                dec_hi,
                self.nlevel,
            )

        # alphao /= self.sqrtnbasis

        return alphao

    def hdot(self, alpha, xo):
        """
        coeffs to signal

        alpha   - (nbasis, nxmax, nymax) per basis output coeffs
        xo      - (nx, ny) output signal
        """
        xo[...] = 0.0  # accumulated
        for i, wavelet in enumerate(self.bases):
            if wavelet == "self":
                copyt(alpha[i, 0 : self.ny, 0 : self.nx], self.image)
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
            ntotx = self.ntotx[wavelet]
            ntoty = self.ntoty[wavelet]
            idwt2d(
                alpha[i, 0:ntoty, 0:ntotx],
                self.image,
                self.alpha[0:ntoty, 0:ntotx],
                self.cbuff[0:ntotx, 0:ntoty],
                self.cbufft[0:ntoty, 0:ntotx],
                ix,
                iy,
                sx,
                sy,
                spx,
                spy,
                rec_lo,
                rec_hi,
                self.nlevel,
            )

            xo += self.image

        # xo /= self.sqrtnbasis

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
        self.nxmax = self.psib[0].nxmax
        self.nymax = self.psib[0].nymax
        self.nthreads_per_band = np.maximum(1, nthreads // nband)

    def dot(self, x, alphao):
        """
        image to coeffs
        """
        futures = []
        with cf.ThreadPoolExecutor(max_workers=self.nband) as executor:
            for b in range(self.nband):
                f = executor.submit(psi_dot_impl, x[b], alphao[b], self.psib[b], b, nthreads=self.nthreads_per_band)
                futures.append(f)

            # wait for result
            for f in cf.as_completed(futures):
                b = f.result()

    def hdot(self, alpha, xo):
        """
        coeffs to image
        """
        futures = []
        with cf.ThreadPoolExecutor(max_workers=self.nband) as executor:
            for b in range(self.nband):
                f = executor.submit(psi_hdot_impl, alpha[b], xo[b], self.psib[b], b, nthreads=self.nthreads_per_band)
                futures.append(f)

            # wait for result
            for f in cf.as_completed(futures):
                b = f.result()
