# ruff: noqa
"""
Profile wavelet DWT optimizations: old (dict-based, no fastmath, .T.copy())
vs new (array-based, fastmath, pre-allocated approx buffer).

Measures both execution time and peak memory for:
  1. Low-level dwt2d + idwt2d round-trip
  2. Full Psi operator dot + hdot

Usage:
    python scripts/profile_wavelets.py [--nx 1024] [--ny 1024] [--nband 1] [--warmup 3] [--repeats 10]
"""

import ctypes
import importlib.metadata
import os

# parse --psi-nthreads early so we can set NUMBA_NUM_THREADS before import
_nthreads = os.cpu_count() or 1
os.environ["NUMBA_THREADING_LAYER"] = "tbb"
os.environ["NUMBA_NUM_THREADS"] = str(_nthreads)
dist = importlib.metadata.distribution("tbb")
tbb_path = None
for f in dist.files:
    if str(f).endswith("/libtbb.so"):
        tbb_path = str(dist.locate_file(f).resolve())
        ctypes.CDLL(tbb_path)
        break
if tbb_path is None:
    raise RuntimeError("Could not initialse TBB threading layer for numba.")

import argparse
import gc
import tracemalloc
from time import time

import numba
import numpy as np
import pywt

# ── old implementation (dict-based, no fastmath, .T.copy()) ─────────────


@numba.njit(nogil=True, cache=False, inline="always", fastmath=False, error_model="numpy")
def _old_downsampling_convolution(input, output, filter, step):
    i = step - 1
    o = 0
    input_size = input.shape[0]
    filter_size = filter.shape[0]

    while i < filter_size and i < input_size:
        fsum = input.dtype.type(0)
        j = 0
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    while i < input_size:
        fsum = input.dtype.type(0)
        j = 0
        while j < filter_size:
            fsum += input[i - j] * filter[j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    while i < filter_size:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    while i < input_size + filter_size - 1:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j < filter_size:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1


@numba.njit(nogil=True, cache=False, inline="always", fastmath=False, error_model="numpy")
def _old_upsampling_convolution_valid_sf(input, filter, output):
    input_size = input.shape[0]
    filter_size = filter.shape[0]
    output_size = output.shape[0]

    o = 0
    i = (filter_size // 2) - 1
    stopping_criteria = filter_size // 2

    while i < input_size and o < output_size:
        sum_even = input.dtype.type(0)
        sum_odd = input.dtype.type(0)
        j = 0
        j2 = 0
        while j < stopping_criteria:
            input_element = input[i - j]
            sum_even += filter[j2] * input_element
            sum_odd += filter[j2 + 1] * input_element
            j += 1
            j2 += 2
        output[o] += sum_even
        output[o + 1] += sum_odd
        i += 1
        o += 2


@numba.njit(nogil=True, cache=False, parallel=True, inline="always")
def _old_copyt(mat, out):
    block_size, tile_size = 256, 32
    n, m = mat.shape
    for tmp in numba.prange((m + block_size - 1) // block_size):
        i = tmp * block_size
        for j in range(0, n, block_size):
            timin, timax = i, min(i + block_size, m)
            tjmin, tjmax = j, min(j + block_size, n)
            for ti in range(timin, timax, tile_size):
                for tj in range(tjmin, tjmax, tile_size):
                    out[ti : ti + tile_size, tj : tj + tile_size] = mat[tj : tj + tile_size, ti : ti + tile_size].T


@numba.njit(nogil=True, cache=False)
def _old_coeff_size(nsignal, nfilter):
    return (nsignal + nfilter - 1) // 2


@numba.njit(nogil=True, cache=False)
def _old_signal_size(ncoeff, nfilter):
    return 2 * ncoeff - nfilter + 2


@numba.njit(nogil=True, cache=False, parallel=True)
def _old_dwt2d_level(image, coeffs, cbuff, cbufft, dec_lo, dec_hi):
    nx, ny = image.shape
    nay, nax = coeffs.shape

    midy = nay // 2
    for i in numba.prange(nx):
        _old_downsampling_convolution(image[i, :], cbuff[i, 0:midy], dec_lo, 2)
        _old_downsampling_convolution(image[i, :], cbuff[i, midy:], dec_hi, 2)

    _old_copyt(cbuff, cbufft)

    midx = nax // 2
    for i in numba.prange(nay):
        _old_downsampling_convolution(cbufft[i, 0:nx], coeffs[i, 0:midx], dec_lo, 2)
        _old_downsampling_convolution(cbufft[i, 0:nx], coeffs[i, midx:], dec_hi, 2)

    return coeffs[0:midy, 0:midx].T.copy()


@numba.njit(nogil=True, cache=False)
def _old_dwt2d(image, coeffs, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel):
    approx = image
    for i in range(nlevel):
        _, highx = ix[i]
        lowx = highx - 2 * sx[i]
        _, highy = iy[i]
        lowy = highy - 2 * sy[i]
        approx = _old_dwt2d_level(
            approx,
            coeffs[lowy:highy, lowx:highx],
            cbuff[lowx:highx, lowy:highy],
            cbufft[lowy:highy, lowx:highx],
            dec_lo,
            dec_hi,
        )


@numba.njit(nogil=True, cache=False, parallel=True)
def _old_idwt2d_level(coeffs, image, cbuff, cbufft, rec_lo, rec_hi):
    nay, nax = coeffs.shape
    nx, ny = image.shape

    cbufft[...] = 0.0
    image[...] = 0.0

    midx = nax // 2
    for i in numba.prange(nay):
        _old_upsampling_convolution_valid_sf(coeffs[i, 0:midx], rec_lo, cbufft[i, :])
        _old_upsampling_convolution_valid_sf(coeffs[i, midx:], rec_hi, cbufft[i, :])

    _old_copyt(cbufft, cbuff)

    midy = nay // 2
    for i in numba.prange(nx):
        _old_upsampling_convolution_valid_sf(cbuff[i, 0:midy], rec_lo, image[i, :])
        _old_upsampling_convolution_valid_sf(cbuff[i, midy:], rec_hi, image[i, :])


@numba.njit(nogil=True, cache=False)
def _old_idwt2d(coeffs, image, alpha, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel):
    nx, ny = image.shape
    alpha[...] = coeffs
    for i in range(nlevel - 1, -1, -1):
        nax = sx[i]
        nay = sy[i]
        _, highx = ix[i]
        lowx = highx - 2 * nax
        _, highy = iy[i]
        lowy = highy - 2 * nay
        nxo = spx[i]
        nyo = spy[i]
        if i < nlevel - 1:
            _old_copyt(image[0:nax, 0:nay], alpha[lowy : lowy + nay, lowx : lowx + nax])
        _old_idwt2d_level(
            alpha[lowy:highy, lowx:highx],
            image[0:nxo, 0:nyo],
            cbuff[0 : 2 * nax, 0 : 2 * nay],
            cbufft[0 : 2 * nay, 0 : 2 * nax],
            rec_lo,
            rec_hi,
        )


# ── bookkeeping helpers ─────────────────────────────────────────────────


def build_old_bookkeeping(nxi, nyi, wavelet, nlevel):
    """Build dict-based bookkeeping for old dwt2d/idwt2d."""
    filter_length = int(wavelet[-1]) * 2
    n2cx = {}
    n2cy = {}
    nx, ny = nxi, nyi
    ntotx = ntoty = 0
    sx = ()
    sy = ()
    spx = ()
    spy = ()
    for k in range(nlevel):
        cx = _old_coeff_size(nx, filter_length)
        cy = _old_coeff_size(ny, filter_length)
        n2cx[k] = (_old_signal_size(cx, filter_length), cx)
        n2cy[k] = (_old_signal_size(cy, filter_length), cy)
        ntotx += cx
        ntoty += cy
        sx += (cx,)
        sy += (cy,)
        nx = cx + cx % 2
        ny = cy + cy % 2
        spx += (_old_signal_size(cx, filter_length),)
        spy += (_old_signal_size(cy, filter_length),)
    ntotx += cx
    ntoty += cy

    ix = numba.typed.Dict()
    iy = numba.typed.Dict()
    lowx = n2cx[nlevel - 1][1]
    lowy = n2cy[nlevel - 1][1]
    ix[nlevel - 1] = (lowx, 2 * lowx)
    iy[nlevel - 1] = (lowy, 2 * lowy)
    lowx *= 2
    lowy *= 2
    for k in reversed(range(nlevel - 1)):
        highx = n2cx[k][1]
        highy = n2cy[k][1]
        ix[k] = (lowx, lowx + highx)
        iy[k] = (lowy, lowy + highy)
        lowx += highx
        lowy += highy
    return ix, iy, sx, sy, spx, spy, ntotx, ntoty


def build_new_bookkeeping(nxi, nyi, wavelet, nlevel):
    """Build array-based bookkeeping for new dwt2d/idwt2d."""
    from pfb_imaging.wavelets import coeff_size, signal_size

    filter_length = int(wavelet[-1]) * 2
    n2cx = {}
    n2cy = {}
    nx, ny = nxi, nyi
    ntotx = ntoty = 0
    sx = np.zeros(nlevel, dtype=np.int64)
    sy = np.zeros(nlevel, dtype=np.int64)
    spx = np.zeros(nlevel, dtype=np.int64)
    spy = np.zeros(nlevel, dtype=np.int64)
    for k in range(nlevel):
        cx = coeff_size(nx, filter_length)
        cy = coeff_size(ny, filter_length)
        n2cx[k] = (signal_size(cx, filter_length), cx)
        n2cy[k] = (signal_size(cy, filter_length), cy)
        ntotx += cx
        ntoty += cy
        sx[k] = cx
        sy[k] = cy
        nx = cx + cx % 2
        ny = cy + cy % 2
        spx[k] = signal_size(cx, filter_length)
        spy[k] = signal_size(cy, filter_length)
    ntotx += cx
    ntoty += cy

    ix = np.zeros((nlevel, 2), dtype=np.int64)
    iy = np.zeros((nlevel, 2), dtype=np.int64)
    lowx = n2cx[nlevel - 1][1]
    lowy = n2cy[nlevel - 1][1]
    ix[nlevel - 1, 0] = lowx
    ix[nlevel - 1, 1] = 2 * lowx
    iy[nlevel - 1, 0] = lowy
    iy[nlevel - 1, 1] = 2 * lowy
    lowx *= 2
    lowy *= 2
    for k in reversed(range(nlevel - 1)):
        highx = n2cx[k][1]
        highy = n2cy[k][1]
        ix[k, 0] = lowx
        ix[k, 1] = lowx + highx
        iy[k, 0] = lowy
        iy[k, 1] = lowy + highy
        lowx += highx
        lowy += highy
    return ix, iy, sx, sy, spx, spy, ntotx, ntoty


# ── profiling helpers ───────────────────────────────────────────────────


def profile_old_dwt(data, wavelet, nlevel, warmup, repeats):
    """Time the old dict-based dwt2d + idwt2d."""
    nxi, nyi = data.shape
    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])
    rec_lo = np.array(wvlt.filter_bank[2])
    rec_hi = np.array(wvlt.filter_bank[3])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_old_bookkeeping(nxi, nyi, wavelet, nlevel)

    alpha = np.zeros((ntoty, ntotx))
    cbuff = np.zeros((ntotx, ntoty))
    cbufft = np.zeros((ntoty, ntotx))

    # warmup (triggers JIT compilation)
    for _ in range(warmup):
        _old_dwt2d(data, alpha, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel)
        xrec = np.zeros((nxi, nyi))
        alpha_buf = np.zeros((ntoty, ntotx))
        _old_idwt2d(alpha, xrec, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

    # timed run
    gc.collect()
    tracemalloc.start()
    times = []
    for _ in range(repeats):
        t0 = time()
        _old_dwt2d(data, alpha, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel)
        xrec = np.zeros((nxi, nyi))
        alpha_buf = np.zeros((ntoty, ntotx))
        _old_idwt2d(alpha, xrec, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
        times.append(time() - t0)
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return np.array(times), mem_peak


def profile_new_dwt(data, wavelet, nlevel, warmup, repeats):
    """Time the new array-based dwt2d_seq + idwt2d_seq (serial, for kernel comparison)."""
    from pfb_imaging.wavelets import dwt2d, idwt2d

    nxi, nyi = data.shape
    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])
    rec_lo = np.array(wvlt.filter_bank[2])
    rec_hi = np.array(wvlt.filter_bank[3])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_new_bookkeeping(nxi, nyi, wavelet, nlevel)

    alpha = np.zeros((ntoty, ntotx))
    cbuff = np.zeros((ntotx, ntoty))
    cbufft = np.zeros((ntoty, ntotx))
    approx = np.zeros((ntotx, ntoty))

    # warmup
    for _ in range(warmup):
        dwt2d(data, alpha, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)
        xrec = np.zeros((nxi, nyi))
        alpha_buf = np.zeros((ntoty, ntotx))
        idwt2d(alpha, xrec, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

    # timed run
    gc.collect()
    tracemalloc.start()
    times = []
    for _ in range(repeats):
        t0 = time()
        dwt2d(data, alpha, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)
        xrec = np.zeros((nxi, nyi))
        alpha_buf = np.zeros((ntoty, ntotx))
        idwt2d(alpha, xrec, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
        times.append(time() - t0)
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return np.array(times), mem_peak


def print_comparison(label, times_old, mem_old, times_new, mem_new):
    med_old = np.median(times_old)
    med_new = np.median(times_new)
    min_old = np.min(times_old)
    min_new = np.min(times_new)

    print(f"\n{'=' * 64}")
    print(f"  {label}")
    print(f"{'=' * 64}")
    print(f"  {'':30s}  {'OLD':>10s}  {'NEW':>10s}")
    print(f"  {'─' * 54}")
    print(f"  {'Median time':30s}  {med_old * 1000:8.1f}ms  {med_new * 1000:8.1f}ms")
    print(f"  {'Best time':30s}  {min_old * 1000:8.1f}ms  {min_new * 1000:8.1f}ms")
    print(f"  {'Peak memory':30s}  {mem_old / 1e6:8.1f}MB  {mem_new / 1e6:8.1f}MB")
    print(f"  {'─' * 54}")

    speedup = med_old / med_new if med_new > 0 else float("inf")
    if speedup > 1:
        print(f"  Speedup (median): {speedup:.2f}x faster")
    else:
        print(f"  Slowdown (median): {1 / speedup:.2f}x slower")

    if mem_old > 0 and mem_new > 0:
        mem_ratio = mem_old / mem_new
        if mem_ratio > 1:
            print(f"  Memory reduction:  {mem_ratio:.2f}x less peak memory")
        elif mem_ratio < 1:
            print(f"  Memory increase:   {1 / mem_ratio:.2f}x more peak memory")
        else:
            print("  Memory: same")


def inspect_simd(func, sig, label):
    """Inspect a numba dispatcher's ASM for SIMD instructions.

    Returns a dict mapping instruction class to count.
    """
    import re

    # x86 SIMD instruction prefixes/patterns (SSE, AVX, AVX-512)
    simd_patterns = {
        "SSE (scalar double)": re.compile(r"\b(addsd|mulsd|subsd|divsd|sqrtsd|maxsd|minsd|fmasd)\b"),
        "SSE (packed double)": re.compile(r"\b(addpd|mulpd|subpd|divpd|sqrtpd|maxpd|minpd)\b"),
        "SSE (packed single)": re.compile(r"\b(addps|mulps|subps|divps|sqrtps|maxps|minps)\b"),
        "AVX (v-prefix double)": re.compile(
            r"\bv(addpd|mulpd|subpd|divpd|sqrtpd|maxpd|minpd|fmadd[0-9]*pd|fmsub[0-9]*pd|fnmadd[0-9]*pd)\b"
        ),
        "AVX (v-prefix single)": re.compile(
            r"\bv(addps|mulps|subps|divps|sqrtps|maxps|minps|fmadd[0-9]*ps|fmsub[0-9]*ps|fnmadd[0-9]*ps)\b"
        ),
        "AVX scalar double": re.compile(
            r"\bv(addsd|mulsd|subsd|divsd|sqrtsd|fmadd[0-9]*sd|fmsub[0-9]*sd|fnmadd[0-9]*sd)\b"
        ),
        "AVX-512 (zmm)": re.compile(r"\bzmm\d+\b"),
        "FMA": re.compile(r"\bv?fmadd[0-9]*(sd|ss|pd|ps)\b"),
        "FMA (fmsub)": re.compile(r"\bv?fmsub[0-9]*(sd|ss|pd|ps)\b"),
        "FMA (fnmadd)": re.compile(r"\bv?fnmadd[0-9]*(sd|ss|pd|ps)\b"),
    }

    # Also check LLVM IR for vectorization hints
    llvm_vec_patterns = {
        "LLVM vector ops": re.compile(r"<\d+ x double>"),
        "LLVM FMA intrinsic": re.compile(r"@llvm\.fma\."),
        "LLVM fmuladd": re.compile(r"@llvm\.fmuladd\."),
        "LLVM vector shuffle": re.compile(r"shufflevector"),
    }

    results = {}

    # get ASM
    try:
        func.compile(sig)
        asm_dict = func.inspect_asm()
        # inspect_asm returns dict keyed by signature
        asm_text = "\n".join(asm_dict.values())

        for name, pat in simd_patterns.items():
            count = len(pat.findall(asm_text))
            if count > 0:
                results[name] = count

        # count register usage to gauge vector width
        ymm_count = len(re.findall(r"\bymm\d+\b", asm_text))
        xmm_count = len(re.findall(r"\bxmm\d+\b", asm_text))
        zmm_count = len(re.findall(r"\bzmm\d+\b", asm_text))
        results["xmm register refs"] = xmm_count
        results["ymm register refs"] = ymm_count
        results["zmm register refs"] = zmm_count

    except Exception as e:
        results["ASM error"] = str(e)

    # get LLVM IR
    llvm_results = {}
    try:
        llvm_dict = func.inspect_llvm()
        llvm_text = "\n".join(llvm_dict.values())

        for name, pat in llvm_vec_patterns.items():
            count = len(pat.findall(llvm_text))
            if count > 0:
                llvm_results[name] = count

    except Exception as e:
        llvm_results["LLVM error"] = str(e)

    return results, llvm_results


def _get_conv_source():
    """Return the source code for convolution functions to recompile without cache."""
    # We re-define the NEW convolution functions with cache=False so that
    # numba's inspect_asm/inspect_llvm work. The production code uses cache=True
    # which disables inspection.

    @numba.njit(nogil=True, cache=False, inline="never", fastmath=True)
    def new_downsampling_convolution(input, output, filter, step):
        i = step - 1
        o = 0
        input_size = input.shape[0]
        filter_size = filter.shape[0]
        while i < filter_size and i < input_size:
            fsum = input.dtype.type(0)
            j = 0
            while j <= i:
                fsum += filter[j] * input[i - j]
                j += 1
            output[o] = fsum
            i += step
            o += 1
        while i < input_size:
            fsum = input.dtype.type(0)
            j = 0
            while j < filter_size:
                fsum += input[i - j] * filter[j]
                j += 1
            output[o] = fsum
            i += step
            o += 1
        while i < filter_size:
            fsum = input.dtype.type(0)
            j = i - input_size + 1
            while j <= i:
                fsum += filter[j] * input[i - j]
                j += 1
            output[o] = fsum
            i += step
            o += 1
        while i < input_size + filter_size - 1:
            fsum = input.dtype.type(0)
            j = i - input_size + 1
            while j < filter_size:
                fsum += filter[j] * input[i - j]
                j += 1
            output[o] = fsum
            i += step
            o += 1

    @numba.njit(nogil=True, cache=False, inline="never", fastmath=True)
    def new_upsampling_convolution_valid_sf(input, filter, output):
        input_size = input.shape[0]
        filter_size = filter.shape[0]
        output_size = output.shape[0]
        o = 0
        i = (filter_size // 2) - 1
        stopping_criteria = filter_size // 2
        while i < input_size and o < output_size:
            sum_even = input.dtype.type(0)
            sum_odd = input.dtype.type(0)
            j = 0
            j2 = 0
            while j < stopping_criteria:
                input_element = input[i - j]
                sum_even += filter[j2] * input_element
                sum_odd += filter[j2 + 1] * input_element
                j += 1
                j2 += 2
            output[o] += sum_even
            output[o + 1] += sum_odd
            i += 1
            o += 2

    return new_downsampling_convolution, new_upsampling_convolution_valid_sf


def print_simd_report():
    """Compile all key wavelet functions and report SIMD usage."""
    # Re-create new functions with cache=False and inline="never" for inspection
    new_ds, new_us = _get_conv_source()

    f64_1d = numba.types.Array(numba.float64, 1, "C")

    # trigger compilation with real data
    n = 64
    arr1d = np.zeros(n)
    out1d = np.zeros(n)
    filt = np.array([0.5, 0.5])

    new_ds(arr1d, out1d[:32], filt, 2)
    new_us(arr1d[:32], filt, out1d)
    _old_downsampling_convolution(arr1d, out1d[:32], filt, 2)
    _old_upsampling_convolution_valid_sf(arr1d[:32], filt, out1d)

    # also compile with longer filters to see if vectorization kicks in
    filt8 = np.zeros(8)  # db4 filter length
    filt10 = np.zeros(10)  # db5 filter length
    new_ds(arr1d, out1d[:32], filt8, 2)
    _old_downsampling_convolution(arr1d, out1d[:32], filt8, 2)
    new_ds(arr1d, out1d[:32], filt10, 2)
    _old_downsampling_convolution(arr1d, out1d[:32], filt10, 2)

    functions_to_inspect = [
        ("NEW downsampling_convolution (fastmath)", new_ds, (f64_1d, f64_1d, f64_1d, numba.int64)),
        (
            "OLD downsampling_convolution (no fastmath)",
            _old_downsampling_convolution,
            (f64_1d, f64_1d, f64_1d, numba.int64),
        ),
        ("NEW upsampling_convolution_valid_sf (fastmath)", new_us, (f64_1d, f64_1d, f64_1d)),
        (
            "OLD upsampling_convolution_valid_sf (no fastmath)",
            _old_upsampling_convolution_valid_sf,
            (f64_1d, f64_1d, f64_1d),
        ),
    ]

    from llvmlite import binding as llvm

    print(f"\n{'=' * 72}")
    print("  SIMD / Vectorization Analysis")
    print(f"{'=' * 72}")
    cpu_name = llvm.get_host_cpu_name()
    print(f"  CPU target: {cpu_name}")

    try:
        features = llvm.get_host_cpu_features()
        simd_features = sorted(
            k for k, v in features.items() if v and any(s in k for s in ["sse", "avx", "fma", "mmx"])
        )
        print(f"  SIMD features: {', '.join(simd_features)}")
    except Exception:
        pass

    print(f"  Numba version: {numba.__version__}")
    print()

    for label, func, sig in functions_to_inspect:
        asm_results, llvm_results = inspect_simd(func, sig, label)

        print(f"  {label}")
        print(f"  {'─' * 60}")

        if not asm_results and not llvm_results:
            print("    No SIMD instructions detected")
        else:
            if asm_results:
                print("    ASM:")
                for k, v in sorted(asm_results.items(), key=lambda x: -x[1]):
                    if v > 0:
                        print(f"      {k:35s}  {v:5d}")
            if llvm_results:
                print("    LLVM IR:")
                for k, v in sorted(llvm_results.items(), key=lambda x: -x[1]):
                    if v > 0:
                        print(f"      {k:35s}  {v:5d}")
        print()

    # Summary comparison
    print(f"  {'─' * 60}")
    print("  Key takeaways:")

    new_asm_results, _ = inspect_simd(
        new_ds,
        (f64_1d, f64_1d, f64_1d, numba.int64),
        "new_ds",
    )
    old_asm_results, _ = inspect_simd(
        _old_downsampling_convolution,
        (f64_1d, f64_1d, f64_1d, numba.int64),
        "old_ds",
    )

    new_fma = (
        new_asm_results.get("FMA", 0) + new_asm_results.get("FMA (fmsub)", 0) + new_asm_results.get("FMA (fnmadd)", 0)
    )
    old_fma = (
        old_asm_results.get("FMA", 0) + old_asm_results.get("FMA (fmsub)", 0) + old_asm_results.get("FMA (fnmadd)", 0)
    )
    new_ymm = new_asm_results.get("ymm register refs", 0)
    old_ymm = old_asm_results.get("ymm register refs", 0)
    new_packed = new_asm_results.get("SSE (packed double)", 0) + new_asm_results.get("AVX (v-prefix double)", 0)
    old_packed = old_asm_results.get("SSE (packed double)", 0) + old_asm_results.get("AVX (v-prefix double)", 0)

    print(
        f"    FMA instructions:  OLD={old_fma}  NEW={new_fma}  {'✓ fastmath enables FMA' if new_fma > old_fma else '✗ no FMA difference'}"
    )
    print(
        f"    Packed SIMD ops:   OLD={old_packed}  NEW={new_packed}  {'✓ vectorized' if new_packed > 0 else '✗ scalar only'}"
    )
    print(f"    YMM (256-bit) use: OLD={old_ymm}  NEW={new_ymm}  {'✓ AVX width' if new_ymm > old_ymm else '—'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Profile wavelet DWT optimizations")
    parser.add_argument("--nx", type=int, default=1024, help="Image x dimension")
    parser.add_argument("--ny", type=int, default=1024, help="Image y dimension")
    parser.add_argument("--nband", type=int, default=1, help="Number of frequency bands for Psi test")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (includes JIT)")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations")
    parser.add_argument("--nlevel", type=int, default=3, help="Wavelet decomposition levels")
    parser.add_argument("--simd-only", action="store_true", help="Only run SIMD analysis, skip timing")
    parser.add_argument("--psi-nthreads", type=int, default=1, help="Threads for Psi operator (overrides numba config)")
    args = parser.parse_args()

    nx, ny = args.nx, args.ny
    nlevel = args.nlevel
    nband = args.nband
    warmup = args.warmup
    repeats = args.repeats

    print(f"Numba threads: {numba.config.NUMBA_NUM_THREADS}")

    # ── 0. SIMD analysis ────────────────────────────────────────────

    print_simd_report()

    if args.simd_only:
        return

    print(f"Image shape: ({nx}, {ny})")
    print(f"Decomposition levels: {nlevel}")
    print(f"Warmup: {warmup}, Repeats: {repeats}")

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nx, ny))

    # ── 1. Low-level DWT round-trip ──────────────────────────────────

    for wavelet in ["db1", "db2", "db3", "db4", "db5"]:
        max_level = pywt.dwt_max_level(min(nx, ny), wavelet)
        nlev = min(nlevel, max_level)

        print(f"\nCompiling & warming up for {wavelet}, {nlev} levels...")
        times_old, mem_old = profile_old_dwt(data, wavelet, nlev, warmup, repeats)
        times_new, mem_new = profile_new_dwt(data, wavelet, nlev, warmup, repeats)

        print_comparison(
            f"dwt2d + idwt2d  |  {wavelet}  |  ({nx},{ny})  |  {nlev} levels",
            times_old,
            mem_old,
            times_new,
            mem_new,
        )

    # ── 2. Full Psi operator ─────────────────────────────────────────

    bases = ["self", "db1", "db2", "db3", "db4", "db5"]
    nbasis = len(bases)

    print(f"\n\nPsi operator: bases={bases}, nband={nband}")
    data_multi = rng.standard_normal((nband, nx, ny))

    from pfb_imaging.operators.psi import Psi, PsiBasis

    nlev_psi = min(nlevel, pywt.dwt_max_level(min(nx, ny), "db1"))

    # --- PsiBand (row-parallel within each level) ---
    psi = Psi(nband, nx, ny, bases, nlev_psi, nthreads=args.psi_nthreads)
    nxmax = psi.nxmax
    nymax = psi.nymax
    alpha = np.zeros((nband, nbasis, nymax, nxmax))
    xrec = np.zeros_like(data_multi)

    for _ in range(warmup):
        psi.dot(data_multi, alpha)
        psi.hdot(alpha, xrec)

    gc.collect()
    tracemalloc.start()
    times_psi = []
    for _ in range(repeats):
        t0 = time()
        psi.dot(data_multi, alpha)
        psi.hdot(alpha, xrec)
        times_psi.append(time() - t0)
    _, mem_psi = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    psi.dot(data_multi, alpha)
    psi.hdot(alpha, xrec)
    err = np.max(np.abs(nbasis * data_multi - xrec))

    print(f"\n{'=' * 64}")
    print(f"  PsiBand dot+hdot  |  {nbasis} bases  |  ({nband},{nx},{ny})  |  {nlev_psi} levels")
    print(f"{'=' * 64}")
    print("  Strategy: prange over rows within each level (many fork/join)")
    print(f"  Median time:  {np.median(times_psi) * 1000:.1f}ms")
    print(f"  Best time:    {np.min(times_psi) * 1000:.1f}ms")
    print(f"  Peak memory:  {mem_psi / 1e6:.1f}MB")
    print(f"  Round-trip max error: {err:.2e}")

    # --- PsiBasis (basis-parallel with per-basis buffers) ---
    psi_b = PsiBasis(nband, nx, ny, bases, nlev_psi, nthreads=args.psi_nthreads)
    alpha_b = np.zeros((nband, nbasis, psi_b.nymax, psi_b.nxmax))
    xrec_b = np.zeros_like(data_multi)

    for _ in range(warmup):
        psi_b.dot(data_multi, alpha_b)
        psi_b.hdot(alpha_b, xrec_b)

    gc.collect()
    tracemalloc.start()
    times_psi_b = []
    for _ in range(repeats):
        t0 = time()
        psi_b.dot(data_multi, alpha_b)
        psi_b.hdot(alpha_b, xrec_b)
        times_psi_b.append(time() - t0)
    _, mem_psi_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    psi_b.dot(data_multi, alpha_b)
    psi_b.hdot(alpha_b, xrec_b)
    err_b = np.max(np.abs(nbasis * data_multi - xrec_b))

    print(f"\n{'=' * 64}")
    print(f"  PsiBasis dot+hdot  |  {nbasis} bases  |  ({nband},{nx},{ny})  |  {nlev_psi} levels")
    print(f"{'=' * 64}")
    print("  Strategy: prange over bases (single fork/join per call)")
    print(f"  Median time:  {np.median(times_psi_b) * 1000:.1f}ms")
    print(f"  Best time:    {np.min(times_psi_b) * 1000:.1f}ms")
    print(f"  Peak memory:  {mem_psi_b / 1e6:.1f}MB")
    print(f"  Round-trip max error: {err_b:.2e}")

    # --- Comparison ---
    med_band = np.median(times_psi)
    med_basis = np.median(times_psi_b)
    if med_basis < med_band:
        print(f"\n  PsiBasis is {med_band / med_basis:.2f}x faster than PsiBand")
    else:
        print(f"\n  PsiBand is {med_basis / med_band:.2f}x faster than PsiBasis")

    # ── 3. Memory analysis (.T.copy() elimination) ─────────────────

    print(f"\n{'=' * 64}")
    print("  Memory savings from .T.copy() elimination")
    print("  (per dwt2d call, per basis — not visible to tracemalloc)")
    print(f"{'=' * 64}")

    for wavelet in ["db1", "db4", "db5"]:
        max_level = pywt.dwt_max_level(min(nx, ny), wavelet)
        nlev = min(nlevel, max_level)
        _, _, sx, sy, _, _, _, _ = build_new_bookkeeping(nx, ny, wavelet, nlev)

        alloc_bytes = 0
        for k in range(nlev):
            # old code: coeffs[0:midy, 0:midx].T.copy() at each level
            alloc_bytes += int(sx[k]) * int(sy[k]) * 8  # float64

        # new code: one pre-allocated approx buffer reused across levels
        ix_n, _, sx_n, sy_n, _, _, ntotx_n, ntoty_n = build_new_bookkeeping(nx, ny, wavelet, nlev)
        approx_bytes = ntotx_n * ntoty_n * 8

        print(f"  {wavelet} {nlev}L:")
        print(f"    Old: {nlev} allocations totalling {alloc_bytes / 1024:.1f} KB per dwt2d call")
        print(f"    New: 1 buffer of {approx_bytes / 1024:.1f} KB (reused, amortized to zero)")

    # ── 4. PyWavelets comparison ─────────────────────────────────────

    print(f"\n{'=' * 64}")
    print(f"  PyWavelets reference timing  |  ({nx},{ny})")
    print(f"{'=' * 64}")

    for wavelet in ["db1", "db4", "db5"]:
        max_level = pywt.dwt_max_level(min(nx, ny), wavelet)
        nlev = min(nlevel, max_level)

        # warmup
        for _ in range(warmup):
            c = pywt.wavedec2(data, wavelet, mode="zero", level=nlev)
            pywt.waverec2(c, wavelet, mode="zero")

        times_pw = []
        for _ in range(repeats):
            t0 = time()
            c = pywt.wavedec2(data, wavelet, mode="zero", level=nlev)
            pywt.waverec2(c, wavelet, mode="zero")
            times_pw.append(time() - t0)

        print(f"  {wavelet} {nlev}L:  median {np.median(times_pw) * 1000:.1f}ms  best {np.min(times_pw) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
