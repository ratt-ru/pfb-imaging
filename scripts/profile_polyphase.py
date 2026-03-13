# ruff: noqa
"""
Profile plain row-wise convolutions vs polyphase axis-0 convolutions.

The polyphase functions convolve along axis 0 without transposing the array.
If they are close to as fast as the plain row-wise convolutions, then the
nocopyt DWT implementation will be faster than the old copyt version (since
copyt was a significant fraction of compute time).

Compares:
  - ROW:  prange over rows, calling downsampling/upsampling_convolution per row
          (contiguous access along axis 1 — best-case cache behavior)
  - POLY: conv_downsample/upsample_axis0_polyphase_pair
          (strided access along axis 0 — same total FLOPs for square input)

For a square (N, N) input both approaches do the same number of multiply-adds,
so any difference is purely due to memory access pattern and parallelisation.

Usage:
    python scripts/profile_polyphase.py [--size 2048] [--warmup 5] [--repeats 30]
"""

import argparse
import gc
import shutil
from pathlib import Path
from time import time

import numba
import numpy as np
import pywt

from pfb_imaging.wavelets.convolutions import (
    conv_downsample_axis0_polyphase_pair,
    conv_upsample_axis0_polyphase_pair,
    downsampling_convolution,
    upsampling_convolution_valid_sf,
    upsampling_convolution_valid_sf_set,
)

# ── Clear stale Numba caches ────────────────────────────────────────
# Numba's file cache (`cache=True`) is keyed per-function by source hash.
# Functions decorated with `inline="always"` get compiled into their callers,
# but Numba does NOT track this cross-function dependency.  If an inlined
# function changes while the caller's source stays the same, the caller's
# cached machine code is stale and loading it can segfault. Dirt!
#
# Clearing __pycache__ dirs on every session start is cheap (~ms) and
# eliminates the problem entirely during development.
_src_root = Path(__file__).resolve().parent.parent / "src" / "pfb_imaging"
for _cache_dir in _src_root.rglob("__pycache__"):
    shutil.rmtree(_cache_dir, ignore_errors=True)


def coeff_size(nsignal, nfilter):
    return (nsignal + nfilter - 1) // 2


def signal_size(ncoeff, nfilter):
    return 2 * ncoeff - nfilter + 2


# ── Row-based wrappers (axis-1, contiguous) ──────────────────────────


@numba.njit(nogil=True, cache=False, parallel=True)
def row_downsample_pair(X, dec_lo, dec_hi, Y_lo, Y_hi):
    """Downsample each row with both lo and hi filters (prange over rows)."""
    N = X.shape[0]
    for i in numba.prange(N):
        downsampling_convolution(X[i, :], Y_lo[i, :], dec_lo, 2)
        downsampling_convolution(X[i, :], Y_hi[i, :], dec_hi, 2)


@numba.njit(nogil=True, cache=False, parallel=True)
def row_upsample_pair(X_lo, X_hi, rec_lo, rec_hi, Y):
    """Upsample each row from lo and hi subbands (prange over rows)."""
    N = Y.shape[0]
    for i in numba.prange(N):
        upsampling_convolution_valid_sf_set(X_lo[i, :], rec_lo, Y[i, :])
        upsampling_convolution_valid_sf(X_hi[i, :], rec_hi, Y[i, :])


# ── Polyphase wrappers (axis-0, strided) ─────────────────────────────


@numba.njit(nogil=True, cache=False, parallel=True)
def poly_downsample_pair(X, dec_lo, dec_hi, Y0, Y1):
    conv_downsample_axis0_polyphase_pair(X, dec_lo, dec_hi, Y0, Y1)


@numba.njit(nogil=True, cache=False, parallel=True)
def poly_upsample_pair(X0, X1, rec_lo, rec_hi, Y):
    conv_upsample_axis0_polyphase_pair(X0, X1, rec_lo, rec_hi, Y)


# ── Profiling ────────────────────────────────────────────────────────


def profile_downsample(size, wavelet, warmup, repeats):
    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.ascontiguousarray(wvlt.filter_bank[0])
    dec_hi = np.ascontiguousarray(wvlt.filter_bank[1])
    K = len(dec_lo)
    n_out = coeff_size(size, K)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((size, size))

    # Row-based outputs: (size, n_out)
    Y_lo_row = np.empty((size, n_out))
    Y_hi_row = np.empty((size, n_out))

    # Polyphase outputs: (n_out, size)
    Y_lo_poly = np.empty((n_out, size))
    Y_hi_poly = np.empty((n_out, size))

    # Warmup (JIT compile both paths)
    for _ in range(warmup):
        row_downsample_pair(X, dec_lo, dec_hi, Y_lo_row, Y_hi_row)
        poly_downsample_pair(X, dec_lo, dec_hi, Y_lo_poly, Y_hi_poly)

    # Time row-based
    gc.collect()
    times_row = []
    for _ in range(repeats):
        t0 = time()
        row_downsample_pair(X, dec_lo, dec_hi, Y_lo_row, Y_hi_row)
        times_row.append(time() - t0)

    # Time polyphase
    gc.collect()
    times_poly = []
    for _ in range(repeats):
        t0 = time()
        poly_downsample_pair(X, dec_lo, dec_hi, Y_lo_poly, Y_hi_poly)
        times_poly.append(time() - t0)

    return np.array(times_row), np.array(times_poly)


def profile_upsample(size, wavelet, warmup, repeats):
    wvlt = pywt.Wavelet(wavelet)
    rec_lo = np.ascontiguousarray(wvlt.filter_bank[2])
    rec_hi = np.ascontiguousarray(wvlt.filter_bank[3])
    K = len(rec_lo)
    n_coeff = coeff_size(size, K)
    n_signal = signal_size(n_coeff, K)

    rng = np.random.default_rng(42)

    # Row-based: (n_signal, n_coeff) subbands → (n_signal, size) output
    # (n_signal rows, each row reconstructed from n_coeff coefficients)
    X_lo_row = rng.standard_normal((n_signal, n_coeff))
    X_hi_row = rng.standard_normal((n_signal, n_coeff))
    Y_row = np.zeros((n_signal, size))

    # Polyphase: (n_coeff, n_signal) subbands → (n_signal, n_signal) output
    # (convolve axis-0 from n_coeff rows to ~2*n_coeff output rows)
    X_lo_poly = rng.standard_normal((n_coeff, n_signal))
    X_hi_poly = rng.standard_normal((n_coeff, n_signal))
    Y_poly = np.zeros((n_signal, n_signal))

    # Warmup
    for _ in range(warmup):
        row_upsample_pair(X_lo_row, X_hi_row, rec_lo, rec_hi, Y_row)
        poly_upsample_pair(X_lo_poly, X_hi_poly, rec_lo, rec_hi, Y_poly)

    # Time row-based
    gc.collect()
    times_row = []
    for _ in range(repeats):
        t0 = time()
        row_upsample_pair(X_lo_row, X_hi_row, rec_lo, rec_hi, Y_row)
        times_row.append(time() - t0)

    # Time polyphase
    gc.collect()
    times_poly = []
    for _ in range(repeats):
        t0 = time()
        poly_upsample_pair(X_lo_poly, X_hi_poly, rec_lo, rec_hi, Y_poly)
        times_poly.append(time() - t0)

    return np.array(times_row), np.array(times_poly)


def print_result(label, times_row, times_poly):
    med_row = np.median(times_row) * 1000
    med_poly = np.median(times_poly) * 1000
    min_row = np.min(times_row) * 1000
    min_poly = np.min(times_poly) * 1000
    ratio = med_poly / med_row

    print(f"  {label:30s}  {med_row:8.2f}ms ({min_row:7.2f})  {med_poly:8.2f}ms ({min_poly:7.2f})  {ratio:6.2f}")


def main():
    parser = argparse.ArgumentParser(description="Profile row-wise vs polyphase axis-0 convolutions")
    parser.add_argument("--size", type=int, default=2048, help="Square image dimension")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=30, help="Timed iterations")
    args = parser.parse_args()

    size = args.size
    warmup = args.warmup
    repeats = args.repeats
    wavelets = ["db1", "db2", "db3", "db4", "db5"]

    print(f"Image: ({size}, {size}),  warmup={warmup},  repeats={repeats}")
    print(f"Numba {numba.__version__},  threads={numba.config.NUMBA_NUM_THREADS}")
    print()
    print("ROW  = prange over rows, plain downsampling/upsampling_convolution (axis-1, contiguous)")
    print("POLY = conv_{down,up}sample_axis0_polyphase_pair (axis-0, strided)")
    print("ratio = poly / row  (< 1 means polyphase is faster, > 1 means slower)")
    print()

    # ── Downsampling ─────────────────────────────────────────────────
    print(f"{'=' * 90}")
    print("  Downsampling (lo + hi filter pair)")
    print(f"{'=' * 90}")
    print(f"  {'wavelet (K)':30s}  {'ROW median (min)':>20s}  {'POLY median (min)':>20s}  {'ratio':>6s}")
    print(f"  {'─' * 84}")

    for wavelet in wavelets:
        K = len(pywt.Wavelet(wavelet).filter_bank[0])
        times_row, times_poly = profile_downsample(size, wavelet, warmup, repeats)
        print_result(f"{wavelet} (K={K})", times_row, times_poly)

    # ── Upsampling ───────────────────────────────────────────────────
    print()
    print(f"{'=' * 90}")
    print("  Upsampling (lo + hi filter pair)")
    print(f"{'=' * 90}")
    print(f"  {'wavelet (K)':30s}  {'ROW median (min)':>20s}  {'POLY median (min)':>20s}  {'ratio':>6s}")
    print(f"  {'─' * 84}")

    for wavelet in wavelets:
        K = len(pywt.Wavelet(wavelet).filter_bank[0])
        times_row, times_poly = profile_upsample(size, wavelet, warmup, repeats)
        print_result(f"{wavelet} (K={K})", times_row, times_poly)

    print()


if __name__ == "__main__":
    main()
