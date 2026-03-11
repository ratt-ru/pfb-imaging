# ruff: noqa
"""
Profile the full Psi operator: Psi (copyt) vs PsiNocopyt (polyphase)
vs PsiNocopytRay (Ray actor pool).

Measures execution time for dot, hdot, and round-trip (dot+hdot)
across multiple frequency bands, plus memory (RSS and USS) for each variant.

For lower-level profiling see:
  - profile_dwt_convolutions.py  (convolution kernels + SIMD analysis)
  - profile_polyphase.py         (row-wise vs polyphase axis-0 convolutions)
  - profile_dwt2d.py             (dwt2d copyt vs nocopyt)

Usage:
    python scripts/profile_wavelets.py [--nx 512] [--ny 512] [--nband 4] [--warmup 3] [--repeats 10]
"""

import ctypes
import importlib.metadata
import os

# set NUMBA_NUM_THREADS before importing numba
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
from time import time

import numba
import numpy as np
import psutil
import pywt
import ray

from pfb_imaging.operators.psi import Psi, PsiNocopyt, PsiNocopytRay


def get_rss_mb():
    """Return RSS of the current process in MB."""
    return psutil.Process().memory_info().rss / 1e6


def get_system_used_mb():
    """Return system-wide physical memory used in MB."""
    vm = psutil.virtual_memory()
    return vm.used / 1e6


def profile_psi(psi_op, data, alpha, xrec, warmup, repeats):
    """Profile dot and hdot separately, track peak RSS."""
    for _ in range(warmup):
        psi_op.dot(data, alpha)
        psi_op.hdot(alpha, xrec)

    gc.collect()
    rss_before = get_rss_mb()

    dot_times = []
    hdot_times = []
    rss_peak = rss_before
    for _ in range(repeats):
        t0 = time()
        psi_op.dot(data, alpha)
        t1 = time()
        psi_op.hdot(alpha, xrec)
        t2 = time()
        dot_times.append(t1 - t0)
        hdot_times.append(t2 - t1)
        rss_peak = max(rss_peak, get_rss_mb())

    rss_delta = rss_peak - rss_before

    # round-trip error
    psi_op.dot(data, alpha)
    psi_op.hdot(alpha, xrec)

    return np.array(dot_times), np.array(hdot_times), rss_delta


def print_results(label, entries):
    """Print a comparison table for multiple operator variants.

    entries: list of (name, dot_times, hdot_times, roundtrip_error, rss_delta, extra_mem_info)
    """
    print(f"\n{'=' * 78}")
    print(f"  {label}")
    print(f"{'=' * 78}")

    col_w = 16
    header = f"  {'':22s}"
    for name, _, _, _, _, _ in entries:
        header += f"  {name:>{col_w}s}"
    print(header)
    print(f"  {'─' * (22 + (col_w + 2) * len(entries))}")

    row_dot_med = f"  {'dot median':22s}"
    row_dot_best = f"  {'dot best':22s}"
    row_hdot_med = f"  {'hdot median':22s}"
    row_hdot_best = f"  {'hdot best':22s}"
    row_rt_med = f"  {'round-trip median':22s}"
    row_rt_best = f"  {'round-trip best':22s}"
    row_err = f"  {'round-trip error':22s}"
    row_rss = f"  {'main RSS delta':22s}"
    for _, dt, ht, err, rss_delta, _ in entries:
        rt = dt + ht
        row_dot_med += f"  {np.median(dt) * 1000:{col_w - 2}.1f}ms"
        row_dot_best += f"  {np.min(dt) * 1000:{col_w - 2}.1f}ms"
        row_hdot_med += f"  {np.median(ht) * 1000:{col_w - 2}.1f}ms"
        row_hdot_best += f"  {np.min(ht) * 1000:{col_w - 2}.1f}ms"
        row_rt_med += f"  {np.median(rt) * 1000:{col_w - 2}.1f}ms"
        row_rt_best += f"  {np.min(rt) * 1000:{col_w - 2}.1f}ms"
        row_err += f"  {err:{col_w}.2e}"
        row_rss += f"  {rss_delta:{col_w - 2}.1f}MB"
    print(row_dot_med)
    print(row_dot_best)
    print(row_hdot_med)
    print(row_hdot_best)
    print(row_rt_med)
    print(row_rt_best)
    print(row_err)
    print(row_rss)

    # extra memory info
    for name, _, _, _, _, extra in entries:
        if extra:
            print(f"  {name}: {extra}")

    # speedup relative to first entry
    print(f"  {'─' * (22 + (col_w + 2) * len(entries))}")
    base_rt = np.median(entries[0][1] + entries[0][2])
    base_name = entries[0][0]
    for name, dt, ht, _, _, _ in entries[1:]:
        med = np.median(dt + ht)
        speedup = base_rt / med if med > 0 else float("inf")
        if speedup > 1:
            print(f"  {name} is {speedup:.2f}x faster than {base_name}")
        else:
            print(f"  {name} is {1 / speedup:.2f}x slower than {base_name}")


def query_actor_memory(actors):
    """Query RSS and USS from all actors, return list of dicts."""
    return ray.get([a.get_memory_mb.remote() for a in actors])


def print_memory_breakdown(nx, ny, nband, psi_ray):
    """Print detailed memory breakdown including actor USS vs RSS."""
    print(f"\n{'=' * 78}")
    print("  Memory breakdown")
    print(f"{'=' * 78}")
    print(f"  System memory used: {get_system_used_mb():.0f}MB")
    print(f"  Main process RSS:   {get_rss_mb():.0f}MB")
    print(f"  Data arrays:        {nband} x ({nx},{ny}) x 8 bytes = {nband * nx * ny * 8 / 1e6:.0f}MB")

    actor_mem = query_actor_memory(psi_ray._actors)
    nactors = len(actor_mem)
    total_rss = sum(m["rss"] for m in actor_mem)
    total_uss = sum(m["uss"] for m in actor_mem)
    mean_rss = total_rss / nactors
    mean_uss = total_uss / nactors

    print(f"\n  Ray actors: {nactors} processes")
    print(f"  {'actor':>8s}  {'RSS':>8s}  {'USS':>8s}  {'shared':>8s}")
    print(f"  {'─' * 38}")
    for i, m in enumerate(actor_mem):
        shared = m["rss"] - m["uss"]
        print(f"  {i:>8d}  {m['rss']:7.0f}MB  {m['uss']:7.0f}MB  {shared:7.0f}MB")
    print(f"  {'─' * 38}")
    shared_total = total_rss - total_uss
    print(f"  {'total':>8s}  {total_rss:7.0f}MB  {total_uss:7.0f}MB  {shared_total:7.0f}MB")
    print(f"  {'mean':>8s}  {mean_rss:7.0f}MB  {mean_uss:7.0f}MB")
    print()
    print(f"  RSS  = Resident Set Size (includes shared pages — inflated)")
    print(f"  USS  = Unique Set Size (private pages only — true per-actor cost)")
    print(f"  shared = RSS - USS (memory-mapped object store, shared libs, etc.)")
    print(f"  Actual memory cost of actors: ~{total_uss:.0f}MB USS + shared pages (counted once)")


def main():
    parser = argparse.ArgumentParser(description="Profile Psi operator variants")
    parser.add_argument("--nx", type=int, default=512, help="Image x dimension")
    parser.add_argument("--ny", type=int, default=512, help="Image y dimension")
    parser.add_argument("--nband", type=int, default=4, help="Number of frequency bands")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (includes JIT)")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations")
    parser.add_argument("--nlevel", type=int, default=3, help="Wavelet decomposition levels")
    parser.add_argument("--psi-nthreads", type=int, default=1, help="Threads for Psi operators")
    parser.add_argument("--nactors", type=int, default=None, help="Number of Ray actors (default: nband)")
    parser.add_argument("--no-ray", action="store_true", help="Skip PsiNocopytRay benchmark")
    args = parser.parse_args()

    nx, ny = args.nx, args.ny
    nlevel = args.nlevel
    nband = args.nband
    warmup = args.warmup
    repeats = args.repeats

    bases = ["self", "db1", "db2", "db3", "db4", "db5"]
    nbasis = len(bases)
    nlev = min(nlevel, pywt.dwt_max_level(min(nx, ny), "db1"))

    # ── Init Ray early, before any numba work, to avoid fork-after-TBB warnings
    if not args.no_ray and not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    sys_mem_start = get_system_used_mb()
    print(f"Numba threads: {numba.config.NUMBA_NUM_THREADS}")
    print(f"Image shape: ({nx}, {ny})")
    print(f"Decomposition levels: {nlev}")
    print(f"Bases: {bases}")
    print(f"Bands: {nband}, Psi threads: {args.psi_nthreads}")
    print(f"Warmup: {warmup}, Repeats: {repeats}")
    print(f"System memory used: {sys_mem_start:.0f}MB")

    rng = np.random.default_rng(42)
    data = rng.standard_normal((nband, nx, ny))

    entries = []

    # ── Psi (copyt) ─────────────────────────────────────────────────
    sys_pre = get_system_used_mb()
    psi = Psi(nband, nx, ny, bases, nlev, nthreads=args.psi_nthreads)
    alpha_old = np.zeros((nband, nbasis, psi.nymax, psi.nxmax))
    xrec_old = np.zeros_like(data)

    print(f"\nCompiling Psi (copyt) ...")
    dt_old, ht_old, rss_d_old = profile_psi(psi, data, alpha_old, xrec_old, warmup, repeats)
    sys_post = get_system_used_mb()
    err_old = np.max(np.abs(nbasis * data - xrec_old))
    extra_old = f"system +{sys_post - sys_pre:.0f}MB"
    entries.append(("Psi", dt_old, ht_old, err_old, rss_d_old, extra_old))

    # ── PsiNocopyt ──────────────────────────────────────────────────
    sys_pre = get_system_used_mb()
    psi_nc = PsiNocopyt(nband, nx, ny, bases, nlev, nthreads=args.psi_nthreads)
    alpha_nc = np.zeros((nband, nbasis, psi_nc.nxmax, psi_nc.nymax))
    xrec_nc = np.zeros_like(data)

    print(f"Compiling PsiNocopyt ...")
    dt_nc, ht_nc, rss_d_nc = profile_psi(psi_nc, data, alpha_nc, xrec_nc, warmup, repeats)
    sys_post = get_system_used_mb()
    err_nc = np.max(np.abs(nbasis * data - xrec_nc))
    extra_nc = f"system +{sys_post - sys_pre:.0f}MB"
    entries.append(("PsiNocopyt", dt_nc, ht_nc, err_nc, rss_d_nc, extra_nc))

    # ── PsiNocopytRay ───────────────────────────────────────────────
    if not args.no_ray:
        sys_pre = get_system_used_mb()
        nactors = args.nactors
        nact_str = f"{nactors} actors" if nactors else f"{nband} actors (default)"
        print(f"Setting up PsiNocopytRay ({nact_str}) ...")
        psi_ray = PsiNocopytRay(
            nband,
            nx,
            ny,
            bases,
            nlev,
            nthreads=args.psi_nthreads,
            nactors=nactors,
        )
        # Verify threading in actor processes
        threading_info = ray.get([a.get_threading_info.remote() for a in psi_ray._actors])
        for i, info in enumerate(threading_info):
            print(
                f"  actor {i}: pid={info['pid']}  layer={info['threading_layer']}  "
                f"threads={info['num_threads']}/{info['NUMBA_NUM_THREADS']}"
            )

        alpha_ray = np.zeros((nband, nbasis, psi_ray.nxmax, psi_ray.nymax))
        xrec_ray = np.zeros_like(data)

        dt_ray, ht_ray, rss_d_ray = profile_psi(psi_ray, data, alpha_ray, xrec_ray, warmup, repeats)
        sys_post = get_system_used_mb()
        err_ray = np.max(np.abs(nbasis * data - xrec_ray))

        actor_mem = query_actor_memory(psi_ray._actors)
        total_uss = sum(m["uss"] for m in actor_mem)
        total_rss = sum(m["rss"] for m in actor_mem)
        extra_ray = (
            f"system +{sys_post - sys_pre:.0f}MB, "
            f"{psi_ray._nactors} actors: "
            f"USS {total_uss:.0f}MB total, RSS {total_rss:.0f}MB total"
        )
        entries.append(("PsiNocopytRay", dt_ray, ht_ray, err_ray, rss_d_ray, extra_ray))

    # ── Results ─────────────────────────────────────────────────────
    print_results(
        f"Psi operators  |  {nbasis} bases  |  ({nband},{nx},{ny})  |  {nlev} levels",
        entries,
    )

    # ── Memory breakdown ────────────────────────────────────────────
    if not args.no_ray:
        print_memory_breakdown(nx, ny, nband, psi_ray)


if __name__ == "__main__":
    main()
