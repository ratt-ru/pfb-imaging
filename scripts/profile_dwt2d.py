# ruff: noqa
"""
Profile dwt2d + idwt2d: old (copyt) vs new (nocopyt) implementations.

  - OLD:  dwt2d / idwt2d — uses copyt transposition at every level
  - NEW:  dwt2d_nocopyt / idwt2d_nocopyt — polyphase axis-0 convolutions,
          no transpose, fewer buffers

Usage:
    python scripts/profile_dwt2d.py [--nx 2048] [--ny 2048] [--nlevel 3] [--warmup 5] [--repeats 30]
"""

import ctypes
import importlib.metadata
import os

# parse --psi-nthreads early so we can set NUMBA_NUM_THREADS before import
# _nthreads = os.cpu_count() or 1
_nthreads = 8
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
from time import time

import numba
import numpy as np
import pywt
from pathlib import Path
import shutil

from pfb_imaging.wavelets import coeff_size, signal_size
from pfb_imaging.wavelets.wavelets import dwt2d, dwt2d_nocopyt, idwt2d, idwt2d_nocopyt

# ── Clear stale Numba caches ────────────────────────────────────────
# Numba's file cache (`cache=True`) is keyed per-function by source hash.
# Functions decorated with `inline="always"` get compiled into their callers,
# but Numba does NOT track this cross-function dependency.  If an inlined
# function changes while the caller's source stays the same, the caller's
# cached machine code is stale and loading it can segfault.
#
# Clearing __pycache__ dirs on every session start is cheap (~ms) and
# eliminates the problem entirely during development.
_src_root = Path(__file__).resolve().parent.parent / "src" / "pfb_imaging"
for _cache_dir in _src_root.rglob("__pycache__"):
    shutil.rmtree(_cache_dir, ignore_errors=True)


def build_bookkeeping(nxi, nyi, wavelet, nlevel):
    filter_length = int(wavelet[-1]) * 2
    n2cx = {}
    n2cy = {}
    nx, ny = nxi, nyi
    ntotx = ntoty = 0
    sx_arr = np.zeros(nlevel, dtype=np.int64)
    sy_arr = np.zeros(nlevel, dtype=np.int64)
    spx_arr = np.zeros(nlevel, dtype=np.int64)
    spy_arr = np.zeros(nlevel, dtype=np.int64)
    for k in range(nlevel):
        cx = coeff_size(nx, filter_length)
        cy = coeff_size(ny, filter_length)
        n2cx[k] = (signal_size(cx, filter_length), cx)
        n2cy[k] = (signal_size(cy, filter_length), cy)
        ntotx += cx
        ntoty += cy
        sx_arr[k] = cx
        sy_arr[k] = cy
        nx = cx + cx % 2
        ny = cy + cy % 2
        spx_arr[k] = signal_size(cx, filter_length)
        spy_arr[k] = signal_size(cy, filter_length)
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
    return ix, iy, sx_arr, sy_arr, spx_arr, spy_arr, ntotx, ntoty


def profile_dwt2d(data, wavelet, nlevel, warmup, repeats):
    nxi, nyi = data.shape
    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.ascontiguousarray(wvlt.filter_bank[0])
    dec_hi = np.ascontiguousarray(wvlt.filter_bank[1])
    rec_lo = np.ascontiguousarray(wvlt.filter_bank[2])
    rec_hi = np.ascontiguousarray(wvlt.filter_bank[3])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_bookkeeping(nxi, nyi, wavelet, nlevel)

    # Old (copyt) buffers — coeffs in (ntoty, ntotx) layout
    alpha_old = np.zeros((ntoty, ntotx))
    cbuff = np.zeros((ntotx, ntoty))
    cbufft = np.zeros((ntoty, ntotx))
    approx = np.zeros((ntotx, ntoty))

    # New (nocopyt) buffers — coeffs in (ntotx, ntoty) layout
    alpha_new = np.zeros((ntotx, ntoty))
    # cbuff for nocopyt: (nxmax, 2*symax) where nxmax >= nxi, symax = max sy
    symax = int(np.max(sy))
    cbuff_new = np.zeros((max(ntotx, nxi), 2 * symax))

    # Warmup (JIT compile all paths)
    for _ in range(warmup):
        dwt2d(data, alpha_old, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)
        xrec = np.zeros((nxi, nyi))
        alpha_buf = np.zeros((ntoty, ntotx))
        idwt2d(alpha_old, xrec, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

        dwt2d_nocopyt(data, alpha_new, cbuff_new, ix, iy, sx, sy, dec_lo, dec_hi, nlevel)
        xrec_n = np.zeros((nxi, nyi))
        alpha_buf_n = np.zeros((ntotx, ntoty))
        idwt2d_nocopyt(alpha_new, xrec_n, alpha_buf_n, cbuff_new, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

    # Verify: nocopyt coeffs == old coeffs transposed
    alpha_old[:] = 0.0
    alpha_new[:] = 0.0
    dwt2d(data, alpha_old, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)
    dwt2d_nocopyt(data, alpha_new, cbuff_new, ix, iy, sx, sy, dec_lo, dec_hi, nlevel)
    fwd_diff = np.max(np.abs(alpha_new - alpha_old.T))

    # Verify round-trip
    xrec_old = np.zeros((nxi, nyi))
    alpha_buf = np.zeros((ntoty, ntotx))
    idwt2d(alpha_old, xrec_old, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
    xrec_new = np.zeros((nxi, nyi))
    alpha_buf_n = np.zeros((ntotx, ntoty))
    idwt2d_nocopyt(alpha_new, xrec_new, alpha_buf_n, cbuff_new, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
    rt_err_old = np.max(np.abs(data - xrec_old))
    rt_err_new = np.max(np.abs(data - xrec_new))

    # Time forward (old)
    gc.collect()
    fwd_old_times = []
    for _ in range(repeats):
        t0 = time()
        dwt2d(data, alpha_old, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)
        fwd_old_times.append(time() - t0)

    # Time forward (new)
    gc.collect()
    fwd_new_times = []
    for _ in range(repeats):
        t0 = time()
        dwt2d_nocopyt(data, alpha_new, cbuff_new, ix, iy, sx, sy, dec_lo, dec_hi, nlevel)
        fwd_new_times.append(time() - t0)

    # Time inverse (old)
    gc.collect()
    inv_old_times = []
    for _ in range(repeats):
        xrec_old = np.zeros((nxi, nyi))
        alpha_buf = np.zeros((ntoty, ntotx))
        t0 = time()
        idwt2d(alpha_old, xrec_old, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
        inv_old_times.append(time() - t0)

    # Time inverse (new)
    gc.collect()
    inv_new_times = []
    for _ in range(repeats):
        xrec_new = np.zeros((nxi, nyi))
        alpha_buf_n = np.zeros((ntotx, ntoty))
        t0 = time()
        idwt2d_nocopyt(alpha_new, xrec_new, alpha_buf_n, cbuff_new, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
        inv_new_times.append(time() - t0)

    # Time round-trip (old)
    gc.collect()
    rt_old_times = []
    for _ in range(repeats):
        t0 = time()
        dwt2d(data, alpha_old, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)
        xrec_old = np.zeros((nxi, nyi))
        alpha_buf = np.zeros((ntoty, ntotx))
        idwt2d(alpha_old, xrec_old, alpha_buf, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
        rt_old_times.append(time() - t0)

    # Time round-trip (new)
    gc.collect()
    rt_new_times = []
    for _ in range(repeats):
        t0 = time()
        dwt2d_nocopyt(data, alpha_new, cbuff_new, ix, iy, sx, sy, dec_lo, dec_hi, nlevel)
        xrec_new = np.zeros((nxi, nyi))
        alpha_buf_n = np.zeros((ntotx, ntoty))
        idwt2d_nocopyt(alpha_new, xrec_new, alpha_buf_n, cbuff_new, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)
        rt_new_times.append(time() - t0)

    return {
        "fwd_diff": fwd_diff,
        "rt_err_old": rt_err_old,
        "rt_err_new": rt_err_new,
        "fwd_old": np.array(fwd_old_times),
        "fwd_new": np.array(fwd_new_times),
        "inv_old": np.array(inv_old_times),
        "inv_new": np.array(inv_new_times),
        "rt_old": np.array(rt_old_times),
        "rt_new": np.array(rt_new_times),
    }


def print_row(label, t_old, t_new):
    med_o = np.median(t_old) * 1000
    med_n = np.median(t_new) * 1000
    min_o = np.min(t_old) * 1000
    min_n = np.min(t_new) * 1000
    speedup = med_o / med_n if med_n > 0 else float("inf")
    print(f"  {label:20s}  {med_o:8.2f}ms  {med_n:8.2f}ms  {min_o:8.2f}ms  {min_n:8.2f}ms  {speedup:6.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Profile dwt2d: old (copyt) vs new (nocopyt)")
    parser.add_argument("--nx", type=int, default=2048, help="Image x dimension")
    parser.add_argument("--ny", type=int, default=2048, help="Image y dimension")
    parser.add_argument("--nlevel", type=int, default=3, help="Decomposition levels")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=30, help="Timed iterations")
    args = parser.parse_args()

    nx, ny = args.nx, args.ny
    nlevel = args.nlevel
    warmup = args.warmup
    repeats = args.repeats

    print(f"Numba {numba.__version__},  threads={numba.config.NUMBA_NUM_THREADS}")
    print(f"Image: ({nx}, {ny}),  levels={nlevel},  warmup={warmup},  repeats={repeats}")
    print()
    print("OLD = dwt2d / idwt2d (copyt transposition at every level)")
    print("NEW = dwt2d_nocopyt / idwt2d_nocopyt (polyphase axis-0, no transpose)")
    print("speedup = OLD / NEW  (> 1 means nocopyt is faster)")
    print()

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nx, ny))

    wavelets = ["db1", "db2", "db3", "db4", "db5"]

    for wavelet in wavelets:
        max_level = pywt.dwt_max_level(min(nx, ny), wavelet)
        nlev = min(nlevel, max_level)

        print(f"Compiling for {wavelet}, {nlev} levels ...")
        r = profile_dwt2d(data, wavelet, nlev, warmup, repeats)

        print(f"\n{'=' * 90}")
        print(f"  {wavelet}  |  ({nx},{ny})  |  {nlev} levels")
        print(f"{'=' * 90}")
        print(f"  Coeff diff (new vs old.T): {r['fwd_diff']:.1e}")
        print(f"  Round-trip error  old: {r['rt_err_old']:.1e}  new: {r['rt_err_new']:.1e}")
        print()
        print(f"  {'':20s}  {'OLD':>10s}  {'NEW':>10s}  {'OLD':>10s}  {'NEW':>10s}  {'speed':>6s}")
        print(f"  {'':20s}  {'median':>10s}  {'median':>10s}  {'best':>10s}  {'best':>10s}  {'OLD/NEW':>6s}")
        print(f"  {'─' * 84}")
        print_row("dwt2d (fwd)", r["fwd_old"], r["fwd_new"])
        print_row("idwt2d (inv)", r["inv_old"], r["inv_new"])
        print_row("round-trip", r["rt_old"], r["rt_new"])
        print()


if __name__ == "__main__":
    main()
