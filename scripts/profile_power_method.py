"""
Profile power_method implementations: numpy vs optimized numba.

Usage:
    python scripts/profile_power_method.py [--shape 8,8000,8000] [--maxit 50] [--warmup 3]
"""

import argparse
import gc
import tracemalloc
from time import time

import numpy as np


def make_toy_aop(shape, eta=0.1):
    """
    Toy Hessian-like operator: element-wise multiply by a diagonal + regularisation.
    aop(x) = (diag + eta) * x
    SPD as required by power method.
    """
    rng = np.random.default_rng(42)
    diag = rng.uniform(0.5, 2.0, size=shape).astype(np.float64)
    diag_plus_eta = diag + eta

    def aop(x):
        return diag_plus_eta * x

    return aop


def run_profile(pm_fn, name, aop, shape, maxit, warmup):
    print(f"\n{'=' * 60}")
    print(f"Profiling: {name}")
    print(f"{'=' * 60}")

    # Use a fixed seed so both implementations start from the same b0
    rng = np.random.default_rng(999)
    b0 = rng.standard_normal(shape).astype(np.float64)

    # Warmup (triggers JIT compilation for numba)
    print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        t0 = time()
        pm_fn(aop, shape, b0=b0.copy(), tol=1e-30, maxit=3, verbosity=0)
        print(f"  warmup {i + 1}: {time() - t0:.3f}s")

    # Timed run with memory tracking
    print(f"Timed run ({maxit} iterations)...")
    gc.collect()
    tracemalloc.start()
    t0 = time()
    beta, b = pm_fn(
        aop,
        shape,
        b0=b0.copy(),
        tol=1e-30,
        maxit=maxit,
        verbosity=2,
        report_freq=maxit + 1,
    )
    elapsed = time() - t0
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Total:    {elapsed:.3f}s")
    print(f"  Per iter: {elapsed / maxit * 1000:.1f}ms")
    print(f"  beta = {beta:.10e}")
    print(f"  ||b||  = {np.linalg.norm(b):.10e}")
    print(f"  Memory - current: {mem_current / 1e6:.1f} MB, peak: {mem_peak / 1e6:.1f} MB")
    return elapsed, beta, b, mem_peak


def main():
    parser = argparse.ArgumentParser(description="Profile power_method implementations")
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
    mem_gb = nbytes * 3 / 1e9  # b, bp, aop(bp)
    print(f"Shape: {shape}")
    print(f"Array size: {nbytes / 1e9:.2f} GB (estimated total ~{mem_gb:.1f} GB)")

    aop = make_toy_aop(shape)

    from pfb_imaging.opt.power_method import power_method, power_method_numba

    t_np, beta_np, b_np, m_np = run_profile(
        power_method,
        "numpy (current)",
        aop,
        shape,
        args.maxit,
        args.warmup,
    )
    t_nb, beta_nb, b_nb, m_nb = run_profile(
        power_method_numba,
        "numba (optimized)",
        aop,
        shape,
        args.maxit,
        args.warmup,
    )

    # Correctness check
    print(f"\n{'=' * 60}")
    print("Correctness check")
    print(f"{'=' * 60}")
    beta_rdiff = np.abs(beta_np - beta_nb) / max(np.abs(beta_np), 1e-12)
    # b may differ in sign (eigenvector sign ambiguity), so compare magnitudes
    b_rdiff = min(
        np.linalg.norm(b_np - b_nb),
        np.linalg.norm(b_np + b_nb),
    ) / max(np.linalg.norm(b_np), 1e-12)
    print(f"  Relative diff in beta: {beta_rdiff:.3e}")
    print(f"  Relative diff in b:    {b_rdiff:.3e}")
    atol = 1e-7
    if beta_rdiff < atol and b_rdiff < atol:
        print(f"  PASS (both < {atol:.0e})")
    else:
        print(f"  FAIL (tolerance {atol:.0e} exceeded)")

    # Performance summary
    print(f"\n{'=' * 60}")
    print("Performance summary")
    print(f"{'=' * 60}")
    print(f"  numpy:  {t_np:.3f}s ({t_np / args.maxit * 1000:.1f} ms/iter)")
    print(f"  numba:  {t_nb:.3f}s ({t_nb / args.maxit * 1000:.1f} ms/iter)")
    speedup = t_np / t_nb
    if speedup > 1:
        print(f"  numba is {speedup:.2f}x faster")
    else:
        print(f"  numpy is {1 / speedup:.2f}x faster")

    # Memory summary
    print(f"\n{'=' * 60}")
    print("Memory summary (peak during timed run)")
    print(f"{'=' * 60}")
    print(f"  numpy:  {m_np / 1e6:.1f} MB")
    print(f"  numba:  {m_nb / 1e6:.1f} MB")
    if m_np > 0 and m_nb > 0:
        ratio = m_np / m_nb
        if ratio > 1:
            print(f"  numba uses {ratio:.2f}x less memory")
        else:
            print(f"  numpy uses {1 / ratio:.2f}x less memory")


if __name__ == "__main__":
    main()
