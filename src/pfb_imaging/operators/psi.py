import concurrent.futures as cf
from collections import OrderedDict
from typing import Protocol, runtime_checkable

import numba
import numpy as np
import pywt
from numba import typed, types
from numba.experimental import jitclass

from pfb_imaging.wavelets import (
    coeff_size,
    copyt,
    dwt2d,
    dwt2d_nocopyt,
    idwt2d,
    idwt2d_nocopyt,
    signal_size,
)

# ── Bookkeeping (shared by old and nocopyt factories) ────────────────


def _build_wavelet_bookkeeping(nxi, nyi, bases, nlevel):
    """Build per-wavelet, per-level bookkeeping arrays and filter lists.

    Returns a dict with all arrays needed by both PsiBand and PsiBandNocopyt.
    """
    nbasis = len(bases)
    wavelet_bases = [b for b in bases if b != "self"]
    nwavelet = len(wavelet_bases)

    is_self = np.zeros(nbasis, dtype=np.bool_)
    wavelet_map = np.full(nbasis, -1, dtype=np.int64)
    wi = 0
    for i, b in enumerate(bases):
        if b == "self":
            is_self[i] = True
        else:
            wavelet_map[i] = wi
            wi += 1

    arr_type = types.Array(types.float64, 1, "C")
    dec_lo_list = typed.List.empty_list(arr_type)
    dec_hi_list = typed.List.empty_list(arr_type)
    rec_lo_list = typed.List.empty_list(arr_type)
    rec_hi_list = typed.List.empty_list(arr_type)

    ix_arr = np.zeros((nwavelet, nlevel, 2), dtype=np.int64)
    iy_arr = np.zeros((nwavelet, nlevel, 2), dtype=np.int64)
    sx_arr = np.zeros((nwavelet, nlevel), dtype=np.int64)
    sy_arr = np.zeros((nwavelet, nlevel), dtype=np.int64)
    spx_arr = np.zeros((nwavelet, nlevel), dtype=np.int64)
    spy_arr = np.zeros((nwavelet, nlevel), dtype=np.int64)
    ntotx_arr = np.zeros(nwavelet, dtype=np.int64)
    ntoty_arr = np.zeros(nwavelet, dtype=np.int64)

    nxmax = 0
    nymax = 0
    for wi_idx, wavelet in enumerate(wavelet_bases):
        wvlt = pywt.Wavelet(wavelet)
        dec_lo_list.append(np.ascontiguousarray(wvlt.filter_bank[0]))
        dec_hi_list.append(np.ascontiguousarray(wvlt.filter_bank[1]))
        rec_lo_list.append(np.ascontiguousarray(wvlt.filter_bank[2]))
        rec_hi_list.append(np.ascontiguousarray(wvlt.filter_bank[3]))

        max_level = pywt.dwt_max_level(np.minimum(nxi, nyi), wavelet)
        if nlevel > max_level:
            raise ValueError(f"The requested decomposition level {nlevel} is not possible")

        n2cx = {}
        n2cy = {}
        nx = nxi
        ny = nyi
        nxm = 0
        nym = 0
        filter_length = int(wavelet[-1]) * 2
        for k in range(nlevel):
            cx = coeff_size(nx, filter_length)
            cy = coeff_size(ny, filter_length)
            n2cx[k] = (signal_size(cx, filter_length), cx)
            n2cy[k] = (signal_size(cy, filter_length), cy)
            nxm += cx
            nym += cy
            sx_arr[wi_idx, k] = cx
            sy_arr[wi_idx, k] = cy
            nx = cx + cx % 2
            ny = cy + cy % 2
            spx_arr[wi_idx, k] = signal_size(cx, filter_length)
            spy_arr[wi_idx, k] = signal_size(cy, filter_length)
        nxm += cx
        nym += cy
        ntotx_arr[wi_idx] = nxm
        ntoty_arr[wi_idx] = nym
        nxmax = max(nxmax, nxm)
        nymax = max(nymax, nym)

        lowx = n2cx[nlevel - 1][1]
        lowy = n2cy[nlevel - 1][1]
        ix_arr[wi_idx, nlevel - 1, 0] = lowx
        ix_arr[wi_idx, nlevel - 1, 1] = 2 * lowx
        iy_arr[wi_idx, nlevel - 1, 0] = lowy
        iy_arr[wi_idx, nlevel - 1, 1] = 2 * lowy
        lowx *= 2
        lowy *= 2
        for k in reversed(range(nlevel - 1)):
            highx = n2cx[k][1]
            highy = n2cy[k][1]
            ix_arr[wi_idx, k, 0] = lowx
            ix_arr[wi_idx, k, 1] = lowx + highx
            iy_arr[wi_idx, k, 0] = lowy
            iy_arr[wi_idx, k, 1] = lowy + highy
            lowx += highx
            lowy += highy

    nxmax = max(nxmax, nxi)
    nymax = max(nymax, nyi)

    return {
        "nbasis": nbasis,
        "nwavelet": nwavelet,
        "is_self": is_self,
        "wavelet_map": wavelet_map,
        "dec_lo": dec_lo_list,
        "dec_hi": dec_hi_list,
        "rec_lo": rec_lo_list,
        "rec_hi": rec_hi_list,
        "ix": ix_arr,
        "iy": iy_arr,
        "sx": sx_arr,
        "sy": sy_arr,
        "spx": spx_arr,
        "spy": spy_arr,
        "ntotx": ntotx_arr,
        "ntoty": ntoty_arr,
        "nxmax": nxmax,
        "nymax": nymax,
    }


# ══════════════════════════════════════════════════════════════════════
# Old (copyt) PsiBand
# ══════════════════════════════════════════════════════════════════════


def psi_band_maker(nxi, nyi, bases, nlevel):
    bk = _build_wavelet_bookkeeping(nxi, nyi, bases, nlevel)

    cbuff = np.zeros((bk["nxmax"], bk["nymax"]))
    cbufft = np.zeros((bk["nymax"], bk["nxmax"]))
    alpha = np.zeros((bk["nymax"], bk["nxmax"]))
    image = np.zeros((nxi, nyi))
    approx = np.zeros((bk["nxmax"], bk["nymax"]))

    return PsiBand(
        cbuff,
        cbufft,
        alpha,
        image,
        approx,
        bk["ix"],
        bk["iy"],
        bk["sx"],
        bk["sy"],
        bk["spx"],
        bk["spy"],
        bk["dec_lo"],
        bk["dec_hi"],
        bk["rec_lo"],
        bk["rec_hi"],
        bk["ntotx"],
        bk["ntoty"],
        bk["nxmax"],
        bk["nymax"],
        bk["nbasis"],
        bk["is_self"],
        bk["wavelet_map"],
        nlevel,
        nxi,
        nyi,
    )


arr1d_type = types.Array(numba.float64, 1, "C")
spec = OrderedDict()
spec["cbuff"] = numba.float64[:, :]
spec["cbufft"] = numba.float64[:, :]
spec["alpha"] = numba.float64[:, :]
spec["image"] = numba.float64[:, :]
spec["approx"] = numba.float64[:, :]
# per-wavelet, per-level arrays
spec["ix"] = numba.int64[:, :, :]
spec["iy"] = numba.int64[:, :, :]
spec["sx"] = numba.int64[:, :]
spec["sy"] = numba.int64[:, :]
spec["spx"] = numba.int64[:, :]
spec["spy"] = numba.int64[:, :]
# per-wavelet filter lists
spec["dec_lo"] = types.ListType(arr1d_type)
spec["dec_hi"] = types.ListType(arr1d_type)
spec["rec_lo"] = types.ListType(arr1d_type)
spec["rec_hi"] = types.ListType(arr1d_type)
# per-wavelet totals
spec["ntotx"] = numba.int64[:]
spec["ntoty"] = numba.int64[:]
# scalars
spec["nxmax"] = numba.int64
spec["nymax"] = numba.int64
spec["nbasis"] = numba.int64
# basis mapping
spec["is_self"] = numba.boolean[:]
spec["wavelet_map"] = numba.int64[:]
spec["nlevel"] = numba.int64
spec["nx"] = numba.int64
spec["ny"] = numba.int64


@jitclass(spec)
class PsiBand(object):
    def __init__(
        self,
        cbuff,
        cbufft,
        alpha,
        image,
        approx,
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
        is_self,
        wavelet_map,
        nlevel,
        nx,
        ny,
    ):
        self.cbuff = cbuff
        self.cbufft = cbufft
        self.alpha = alpha
        self.image = image
        self.approx = approx
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
        self.is_self = is_self
        self.wavelet_map = wavelet_map
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

        for i in range(self.nbasis):
            if self.is_self[i]:
                copyt(x, alphao[i, 0 : self.ny, 0 : self.nx])
                continue
            wi = self.wavelet_map[i]
            ntotx = self.ntotx[wi]
            ntoty = self.ntoty[wi]

            dwt2d(
                x,
                alphao[i, 0:ntoty, 0:ntotx],
                self.cbuff,
                self.cbufft,
                self.ix[wi],
                self.iy[wi],
                self.sx[wi],
                self.sy[wi],
                self.dec_lo[wi],
                self.dec_hi[wi],
                self.nlevel,
                self.approx,
            )

        return alphao

    def hdot(self, alpha, xo):
        """
        coeffs to signal

        alpha   - (nbasis, nymax, nxmax) per basis output coeffs
        xo      - (nx, ny) output signal
        """
        xo[...] = 0.0  # accumulated
        for i in range(self.nbasis):
            if self.is_self[i]:
                copyt(alpha[i, 0 : self.ny, 0 : self.nx], self.image)
                xo += self.image
                continue
            wi = self.wavelet_map[i]
            ntotx = self.ntotx[wi]
            ntoty = self.ntoty[wi]
            idwt2d(
                alpha[i, 0:ntoty, 0:ntotx],
                self.image,
                self.alpha[0:ntoty, 0:ntotx],
                self.cbuff,
                self.cbufft,
                self.ix[wi],
                self.iy[wi],
                self.sx[wi],
                self.sy[wi],
                self.spx[wi],
                self.spy[wi],
                self.rec_lo[wi],
                self.rec_hi[wi],
                self.nlevel,
            )

            xo += self.image

        return xo


# ══════════════════════════════════════════════════════════════════════
# Nocopyt PsiBand — polyphase axis-0 convolutions, no transpose
# ══════════════════════════════════════════════════════════════════════


def psi_band_maker_nocopyt(nxi, nyi, bases, nlevel):
    bk = _build_wavelet_bookkeeping(nxi, nyi, bases, nlevel)

    # cbuff only needs (nxmax, 2*symax) — no cbufft or approx needed
    symax = int(np.max(bk["sy"])) if bk["sy"].size > 0 else nyi
    cbuff = np.zeros((bk["nxmax"], 2 * symax))
    # alpha working copy for idwt2d_nocopyt
    alpha = np.zeros((bk["nxmax"], bk["nymax"]))
    image = np.zeros((nxi, nyi))

    return PsiBandNocopyt(
        cbuff,
        alpha,
        image,
        bk["ix"],
        bk["iy"],
        bk["sx"],
        bk["sy"],
        bk["spx"],
        bk["spy"],
        bk["dec_lo"],
        bk["dec_hi"],
        bk["rec_lo"],
        bk["rec_hi"],
        bk["ntotx"],
        bk["ntoty"],
        bk["nxmax"],
        bk["nymax"],
        bk["nbasis"],
        bk["is_self"],
        bk["wavelet_map"],
        nlevel,
        nxi,
        nyi,
    )


spec_nocopyt = OrderedDict()
spec_nocopyt["cbuff"] = numba.float64[:, :]
spec_nocopyt["alpha"] = numba.float64[:, :]
spec_nocopyt["image"] = numba.float64[:, :]
spec_nocopyt["ix"] = numba.int64[:, :, :]
spec_nocopyt["iy"] = numba.int64[:, :, :]
spec_nocopyt["sx"] = numba.int64[:, :]
spec_nocopyt["sy"] = numba.int64[:, :]
spec_nocopyt["spx"] = numba.int64[:, :]
spec_nocopyt["spy"] = numba.int64[:, :]
spec_nocopyt["dec_lo"] = types.ListType(arr1d_type)
spec_nocopyt["dec_hi"] = types.ListType(arr1d_type)
spec_nocopyt["rec_lo"] = types.ListType(arr1d_type)
spec_nocopyt["rec_hi"] = types.ListType(arr1d_type)
spec_nocopyt["ntotx"] = numba.int64[:]
spec_nocopyt["ntoty"] = numba.int64[:]
spec_nocopyt["nxmax"] = numba.int64
spec_nocopyt["nymax"] = numba.int64
spec_nocopyt["nbasis"] = numba.int64
spec_nocopyt["is_self"] = numba.boolean[:]
spec_nocopyt["wavelet_map"] = numba.int64[:]
spec_nocopyt["nlevel"] = numba.int64
spec_nocopyt["nx"] = numba.int64
spec_nocopyt["ny"] = numba.int64


@jitclass(spec_nocopyt)
class PsiBandNocopyt(object):
    def __init__(
        self,
        cbuff,
        alpha,
        image,
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
        is_self,
        wavelet_map,
        nlevel,
        nx,
        ny,
    ):
        self.cbuff = cbuff
        self.alpha = alpha
        self.image = image
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
        self.is_self = is_self
        self.wavelet_map = wavelet_map
        self.nlevel = nlevel
        self.nx = nx
        self.ny = ny

    def dot(self, x, alphao):
        """
        signal to coeffs

        x       - (nx, ny) input signal
        alphao  - (nbasis, nxmax, nymax) per basis output coeffs (x-first layout)
        """
        alphao[...] = 0.0

        for i in range(self.nbasis):
            if self.is_self[i]:
                alphao[i, 0 : self.nx, 0 : self.ny] = x
                continue
            wi = self.wavelet_map[i]
            ntotx = self.ntotx[wi]
            ntoty = self.ntoty[wi]

            dwt2d_nocopyt(
                x,
                alphao[i, 0:ntotx, 0:ntoty],
                self.cbuff,
                self.ix[wi],
                self.iy[wi],
                self.sx[wi],
                self.sy[wi],
                self.dec_lo[wi],
                self.dec_hi[wi],
                self.nlevel,
            )

        return alphao

    def hdot(self, alpha, xo):
        """
        coeffs to signal

        alpha   - (nbasis, nxmax, nymax) per basis coeffs (x-first layout)
        xo      - (nx, ny) output signal
        """
        xo[...] = 0.0
        for i in range(self.nbasis):
            if self.is_self[i]:
                self.image[:, :] = alpha[i, 0 : self.nx, 0 : self.ny]
                xo += self.image
                continue
            wi = self.wavelet_map[i]
            ntotx = self.ntotx[wi]
            ntoty = self.ntoty[wi]

            idwt2d_nocopyt(
                alpha[i, 0:ntotx, 0:ntoty],
                self.image,
                self.alpha[0:ntotx, 0:ntoty],
                self.cbuff,
                self.ix[wi],
                self.iy[wi],
                self.sx[wi],
                self.sy[wi],
                self.spx[wi],
                self.spy[wi],
                self.rec_lo[wi],
                self.rec_hi[wi],
                self.nlevel,
            )

            xo += self.image

        return xo


# ══════════════════════════════════════════════════════════════════════
# Python wrappers with thread pool for multi-band processing
# ══════════════════════════════════════════════════════════════════════


def _psi_dot_band(x, alphao, psib):
    """Call psib.dot — releases GIL during execution."""
    psib.dot(x, alphao)


def _psi_hdot_band(alpha, xo, psib):
    """Call psib.hdot — releases GIL during execution."""
    psib.hdot(alpha, xo)


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

        # set numba thread count from main thread (must not exceed
        # NUMBA_NUM_THREADS which is the process-wide ceiling)
        effective = min(nthreads, numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(effective)

        # thread pool only useful when nband > 1
        if nband > 1:
            self._executor = cf.ThreadPoolExecutor(max_workers=nband)
        else:
            self._executor = None

    def dot(self, x, alphao):
        """
        image to coeffs
        """
        if self._executor is None:
            # nband=1: call directly, no pool overhead
            self.psib[0].dot(x[0], alphao[0])
            return

        futures = []
        for b in range(self.nband):
            f = self._executor.submit(_psi_dot_band, x[b], alphao[b], self.psib[b])
            futures.append(f)

        for f in cf.as_completed(futures):
            f.result()

    def hdot(self, alpha, xo):
        """
        coeffs to image
        """
        if self._executor is None:
            self.psib[0].hdot(alpha[0], xo[0])
            return

        futures = []
        for b in range(self.nband):
            f = self._executor.submit(_psi_hdot_band, alpha[b], xo[b], self.psib[b])
            futures.append(f)

        for f in cf.as_completed(futures):
            f.result()


class PsiNocopyt(object):
    def __init__(self, nband, nx, ny, bases, nlevel, nthreads):
        self.psib = []
        for b in range(nband):
            self.psib.append(psi_band_maker_nocopyt(nx, ny, bases, nlevel))

        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nbasis = len(bases)
        self.nthreads = nthreads
        self.nxmax = self.psib[0].nxmax
        self.nymax = self.psib[0].nymax

        effective = min(nthreads, numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(effective)

        if nband > 1:
            self._executor = cf.ThreadPoolExecutor(max_workers=nband)
        else:
            self._executor = None

    def dot(self, x, alphao):
        """
        image to coeffs
        """
        if self._executor is None:
            self.psib[0].dot(x[0], alphao[0])
            return

        futures = []
        for b in range(self.nband):
            f = self._executor.submit(_psi_dot_band, x[b], alphao[b], self.psib[b])
            futures.append(f)

        for f in cf.as_completed(futures):
            f.result()

    def hdot(self, alpha, xo):
        """
        coeffs to image
        """
        if self._executor is None:
            self.psib[0].hdot(alpha[0], xo[0])
            return

        futures = []
        for b in range(self.nband):
            f = self._executor.submit(_psi_hdot_band, alpha[b], xo[b], self.psib[b])
            futures.append(f)

        for f in cf.as_completed(futures):
            f.result()


@runtime_checkable
class PsiOperatorProtocol(Protocol):
    def dot(self, x, alphao): ...

    def hdot(self, alpha, xo): ...
