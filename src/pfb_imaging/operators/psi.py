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
    copyt_seq,
    dwt2d,
    dwt2d_seq,
    idwt2d,
    idwt2d_seq,
    signal_size,
)


def psi_band_maker(nxi, nyi, bases, nlevel):
    nbasis = len(bases)

    # separate wavelet bases from "self"
    wavelet_bases = [b for b in bases if b != "self"]
    nwavelet = len(wavelet_bases)

    # basis mapping arrays
    is_self = np.zeros(nbasis, dtype=np.bool_)
    wavelet_map = np.full(nbasis, -1, dtype=np.int64)
    wi = 0
    for i, b in enumerate(bases):
        if b == "self":
            is_self[i] = True
        else:
            wavelet_map[i] = wi
            wi += 1

    # per-wavelet filter lists
    arr_type = types.Array(types.float64, 1, "C")
    dec_lo_list = typed.List.empty_list(arr_type)
    dec_hi_list = typed.List.empty_list(arr_type)
    rec_lo_list = typed.List.empty_list(arr_type)
    rec_hi_list = typed.List.empty_list(arr_type)

    # per-wavelet, per-level bookkeeping arrays
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

        # bookkeeping
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
        nxm += cx  # last approx coeffs
        nym += cy
        ntotx_arr[wi_idx] = nxm
        ntoty_arr[wi_idx] = nym
        nxmax = max(nxmax, nxm)
        nymax = max(nymax, nym)

        # build ix/iy arrays
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

    # ensure nxmax/nymax are at least as large as the image
    # (needed for "self" basis transpose)
    nxmax = max(nxmax, nxi)
    nymax = max(nymax, nyi)

    # set up buffers — cbuff/cbufft are 1D flat for per-level contiguous reshaping
    alpha = np.zeros((nymax, nxmax))
    cbuff = np.zeros(nxmax * nymax)
    cbufft = np.zeros(nymax * nxmax)
    image = np.zeros((nxi, nyi))
    approx = np.zeros((nxmax, nymax))

    return PsiBand(
        alpha,
        image,
        approx,
        cbuff,
        cbufft,
        ix_arr,
        iy_arr,
        sx_arr,
        sy_arr,
        spx_arr,
        spy_arr,
        dec_lo_list,
        dec_hi_list,
        rec_lo_list,
        rec_hi_list,
        ntotx_arr,
        ntoty_arr,
        nxmax,
        nymax,
        nbasis,
        is_self,
        wavelet_map,
        nlevel,
        nxi,
        nyi,
    )


arr1d_type = types.Array(numba.float64, 1, "C")
spec = OrderedDict()
spec["alpha"] = numba.float64[:, :]
spec["image"] = numba.float64[:, :]
spec["approx"] = numba.float64[:, :]
spec["cbuff"] = numba.float64[:]
spec["cbufft"] = numba.float64[:]
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
        alpha,
        image,
        approx,
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
        is_self,
        wavelet_map,
        nlevel,
        nx,
        ny,
    ):
        self.alpha = alpha
        self.image = image
        self.approx = approx
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

        alpha   - (nbasis, nxmax, nymax) per basis output coeffs
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


# ── PsiBasis: basis-parallel Psi operator ─────────────────────────────
# Instead of parallelizing within each dwt2d_level (row-level prange),
# this parallelizes over the basis axis with a single prange.
# Each basis gets its own contiguous scratch buffers, eliminating
# the ~54 fork/join barriers per dot call that PsiBand has.
# Trade-off: more memory (per-basis buffers) but fewer sync points.


@numba.njit(nogil=True, cache=True, parallel=True)
def _psi_basis_dot(
    x,
    alphao,
    is_self,
    wavelet_map,
    ix,
    iy,
    sx,
    sy,
    dec_lo,
    dec_hi,
    ntotx,
    ntoty,
    nlevel,
    nx,
    ny,
    nbasis,
    cbuff,
    cbufft,
    approx,
):
    """Forward transform parallelized over bases.

    x       - (nx, ny) input signal
    alphao  - (nbasis, nymax, nxmax) output coefficients
    cbuff   - (nbasis, nxmax*nymax) per-basis flat scratch
    cbufft  - (nbasis, nymax*nxmax) per-basis flat scratch
    approx  - (nbasis, nxmax, nymax) per-basis approx buffer
    """
    alphao[...] = 0.0
    for i in numba.prange(nbasis):
        if is_self[i]:
            copyt_seq(x, alphao[i, 0:ny, 0:nx])
        else:
            wi = wavelet_map[i]
            ntx = ntotx[wi]
            nty = ntoty[wi]
            dwt2d_seq(
                x,
                alphao[i, 0:nty, 0:ntx],
                cbuff[i],
                cbufft[i],
                ix[wi],
                iy[wi],
                sx[wi],
                sy[wi],
                dec_lo[wi],
                dec_hi[wi],
                nlevel,
                approx[i],
            )


@numba.njit(nogil=True, cache=True, parallel=True)
def _psi_basis_hdot(
    alpha,
    xo,
    is_self,
    wavelet_map,
    ix,
    iy,
    sx,
    sy,
    spx,
    spy,
    rec_lo,
    rec_hi,
    ntotx,
    ntoty,
    nlevel,
    nx,
    ny,
    nbasis,
    cbuff,
    cbufft,
    approx,
    image,
    alpha_buf,
):
    """Adjoint transform parallelized over bases.

    alpha     - (nbasis, nymax, nxmax) input coefficients
    xo        - (nx, ny) output signal (accumulated)
    cbuff     - (nbasis, nxmax*nymax) per-basis flat scratch
    cbufft    - (nbasis, nymax*nxmax) per-basis flat scratch
    image     - (nbasis, nx, ny) per-basis reconstruction scratch
    alpha_buf - (nbasis, nymax, nxmax) per-basis coeff copy buffer
    """
    for i in numba.prange(nbasis):
        if is_self[i]:
            copyt_seq(alpha[i, 0:ny, 0:nx], image[i])
        else:
            wi = wavelet_map[i]
            ntx = ntotx[wi]
            nty = ntoty[wi]
            idwt2d_seq(
                alpha[i, 0:nty, 0:ntx],
                image[i],
                alpha_buf[i, 0:nty, 0:ntx],
                cbuff[i],
                cbufft[i],
                ix[wi],
                iy[wi],
                sx[wi],
                sy[wi],
                spx[wi],
                spy[wi],
                rec_lo[wi],
                rec_hi[wi],
                nlevel,
            )
    # reduce per-basis images into output
    xo[...] = 0.0
    for i in range(nbasis):
        xo += image[i]


def _psi_basis_band_maker(nxi, nyi, bases, nlevel):
    """Build bookkeeping arrays and per-basis buffers for PsiBasis."""
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

    # per-basis buffers: each basis gets its own 1D flat scratch space
    # for per-level contiguous reshaping
    cbuff = np.zeros((nbasis, nxmax * nymax))
    cbufft = np.zeros((nbasis, nymax * nxmax))
    approx = np.zeros((nbasis, nxmax, nymax))
    image = np.zeros((nbasis, nxi, nyi))
    alpha_buf = np.zeros((nbasis, nymax, nxmax))

    return {
        "is_self": is_self,
        "wavelet_map": wavelet_map,
        "ix": ix_arr,
        "iy": iy_arr,
        "sx": sx_arr,
        "sy": sy_arr,
        "spx": spx_arr,
        "spy": spy_arr,
        "dec_lo": dec_lo_list,
        "dec_hi": dec_hi_list,
        "rec_lo": rec_lo_list,
        "rec_hi": rec_hi_list,
        "ntotx": ntotx_arr,
        "ntoty": ntoty_arr,
        "nxmax": nxmax,
        "nymax": nymax,
        "nlevel": nlevel,
        "nx": nxi,
        "ny": nyi,
        "nbasis": nbasis,
        "cbuff": cbuff,
        "cbufft": cbufft,
        "approx": approx,
        "image": image,
        "alpha_buf": alpha_buf,
    }


class PsiBasis:
    """Psi operator parallelized over the basis axis.

    Uses a single prange over bases instead of row-level prange within
    each wavelet level. Per-basis contiguous buffers are allocated once
    and reused across bands.
    """

    def __init__(self, nband, nx, ny, bases, nlevel, nthreads):
        d = _psi_basis_band_maker(nx, ny, bases, nlevel)

        # bookkeeping arrays (shared, read-only during transforms)
        self._is_self = d["is_self"]
        self._wavelet_map = d["wavelet_map"]
        self._ix = d["ix"]
        self._iy = d["iy"]
        self._sx = d["sx"]
        self._sy = d["sy"]
        self._spx = d["spx"]
        self._spy = d["spy"]
        self._dec_lo = d["dec_lo"]
        self._dec_hi = d["dec_hi"]
        self._rec_lo = d["rec_lo"]
        self._rec_hi = d["rec_hi"]
        self._ntotx = d["ntotx"]
        self._ntoty = d["ntoty"]
        self._nlevel = d["nlevel"]

        # per-basis scratch buffers (reused across bands)
        self._cbuff = d["cbuff"]
        self._cbufft = d["cbufft"]
        self._approx = d["approx"]
        self._image = d["image"]
        self._alpha_buf = d["alpha_buf"]

        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nbasis = d["nbasis"]
        self.nxmax = d["nxmax"]
        self.nymax = d["nymax"]
        self.nthreads = nthreads

        effective = min(nthreads, numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(effective)

    def dot(self, x, alphao):
        """Forward: image to coefficients."""
        for b in range(self.nband):
            _psi_basis_dot(
                x[b],
                alphao[b],
                self._is_self,
                self._wavelet_map,
                self._ix,
                self._iy,
                self._sx,
                self._sy,
                self._dec_lo,
                self._dec_hi,
                self._ntotx,
                self._ntoty,
                self._nlevel,
                self.nx,
                self.ny,
                self.nbasis,
                self._cbuff,
                self._cbufft,
                self._approx,
            )

    def hdot(self, alpha, xo):
        """Adjoint: coefficients to image."""
        for b in range(self.nband):
            _psi_basis_hdot(
                alpha[b],
                xo[b],
                self._is_self,
                self._wavelet_map,
                self._ix,
                self._iy,
                self._sx,
                self._sy,
                self._spx,
                self._spy,
                self._rec_lo,
                self._rec_hi,
                self._ntotx,
                self._ntoty,
                self._nlevel,
                self.nx,
                self.ny,
                self.nbasis,
                self._cbuff,
                self._cbufft,
                self._approx,
                self._image,
                self._alpha_buf,
            )


@runtime_checkable
class PsiOperatorProtocol(Protocol):
    def dot(self, x, alphao): ...

    def hdot(self, alpha, xo): ...
