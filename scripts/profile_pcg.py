"""
Profile PCG implementations: numexpr vs optimized numba.

Usage:
    python scripts/profile_pcg.py [--shape 8,8000,8000] [--maxit 50] [--warmup 5]
"""

import argparse
import gc
import tracemalloc
from time import time

import numpy as np


def make_toy_aop(shape, eta=0.1):
    """
    Toy Hessian-like operator: element-wise multiply by a diagonal + regularisation.
    aop(x) = diag * x + eta * x = (diag + eta) * x
    This is SPD (symmetric positive definite) as required by PCG.
    """
    rng = np.random.default_rng(42)
    # Positive diagonal to ensure SPD
    diag = rng.uniform(0.5, 2.0, size=shape).astype(np.float64)
    diag_plus_eta = diag + eta

    def aop(x):
        return diag_plus_eta * x

    return aop


def run_profile(pcg_fn, name, aop, b, x0, maxit, warmup):
    print(f"\n{'=' * 60}")
    print(f"Profiling: {name}")
    print(f"{'=' * 60}")

    # Warmup runs (also triggers JIT compilation for numba)
    print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        t0 = time()
        pcg_fn(
            aop,
            b,
            x0=x0.copy(),
            tol=1e-12,
            maxit=3,
            minit=1,
            verbosity=0,
            report_freq=100,
            return_resid=True,
        )
        print(f"  warmup {i + 1}: {time() - t0:.3f}s")

    # Timed run with memory tracking
    print(f"Timed run ({maxit} iterations)...")
    gc.collect()
    tracemalloc.start()
    t0 = time()
    x, r = pcg_fn(
        aop,
        b,
        x0=x0.copy(),
        tol=1e-12,
        maxit=maxit,
        minit=maxit,
        verbosity=2,
        report_freq=maxit + 1,
        return_resid=True,
    )
    elapsed = time() - t0
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Total:    {elapsed:.3f}s")
    print(f"  Per iter: {elapsed / maxit * 1000:.1f}ms")
    print(f"  ||x|| = {np.linalg.norm(x):.6e}")
    print(f"  ||r|| = {np.linalg.norm(r):.6e}")
    print(f"  Memory - current: {mem_current / 1e6:.1f} MB, peak: {mem_peak / 1e6:.1f} MB")
    return elapsed, x, r, mem_peak


def main():
    parser = argparse.ArgumentParser(description="Profile PCG implementations")
    parser.add_argument(
        "--shape",
        type=str,
        default="8,8000,8000",
        help="Array shape as comma-separated ints (default: 8,8000,8000)",
    )
    parser.add_argument("--maxit", type=int, default=50, help="Iterations for timed run")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    args = parser.parse_args()

    shape = tuple(int(s) for s in args.shape.split(","))
    nbytes = int(np.prod(shape)) * 8
    # PCG uses ~8 arrays of this size
    mem_gb = nbytes * 8 / 1e9
    print(f"Shape: {shape}")
    print(f"Array size: {nbytes / 1e9:.2f} GB (estimated total ~{mem_gb:.1f} GB)")

    rng = np.random.default_rng(123)
    b = rng.standard_normal(shape).astype(np.float64)
    x0 = np.zeros(shape, dtype=np.float64)
    aop = make_toy_aop(shape)

    from pfb_imaging.opt.pcg import pcg, pcg_numba

    t_ne, x_ne, r_ne, m_ne = run_profile(pcg, "numexpr (current)", aop, b, x0, args.maxit, args.warmup)
    t_nb, x_nb, r_nb, m_nb = run_profile(pcg_numba, "numba (optimized)", aop, b, x0, args.maxit, args.warmup)

    # Check agreement between implementations
    print(f"\n{'=' * 60}")
    print("Correctness check")
    print(f"{'=' * 60}")
    x_rdiff = np.linalg.norm(x_ne - x_nb) / max(np.linalg.norm(x_ne), 1e-12)
    r_rdiff = np.linalg.norm(r_ne - r_nb) / max(np.linalg.norm(r_ne), 1e-12)
    print(f"  Relative diff in x: {x_rdiff:.3e}")
    print(f"  Relative diff in r: {r_rdiff:.3e}")
    # fastmath allows FP reordering so we use a loose tolerance
    atol = 1e-7
    if x_rdiff < atol and r_rdiff < atol:
        print(f"  PASS (both < {atol:.0e})")
    else:
        print(f"  FAIL (tolerance {atol:.0e} exceeded)")

    print(f"\n{'=' * 60}")
    print("Performance summary")
    print(f"{'=' * 60}")
    print(f"  numexpr:  {t_ne:.3f}s ({t_ne / args.maxit * 1000:.1f} ms/iter)")
    print(f"  numba:    {t_nb:.3f}s ({t_nb / args.maxit * 1000:.1f} ms/iter)")
    speedup = t_ne / t_nb
    if speedup > 1:
        print(f"  numba is {speedup:.2f}x faster")
    else:
        print(f"  numexpr is {1 / speedup:.2f}x faster")

    print(f"\n{'=' * 60}")
    print("Memory summary (peak during timed run)")
    print(f"{'=' * 60}")
    print(f"  numexpr:  {m_ne / 1e6:.1f} MB")
    print(f"  numba:    {m_nb / 1e6:.1f} MB")
    if m_ne > 0 and m_nb > 0:
        ratio = m_ne / m_nb
        if ratio > 1:
            print(f"  numba uses {ratio:.2f}x less memory")
        else:
            print(f"  numexpr uses {1 / ratio:.2f}x less memory")


if __name__ == "__main__":
    main()
