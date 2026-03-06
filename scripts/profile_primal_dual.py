"""
Profile primal_dual implementations:
  - primal_dual_optimised (numexpr + numba dual update)
  - primal_dual_numba (all-numba)

Usage:
    python scripts/profile_primal_dual.py [--shape 8,8000,8000] [--maxit 50] [--warmup 3]

The toy operators simulate a simple diagonal Hessian with wavelet-like
dual domain. psi/psih just copy x into v and back (identity in the
wavelet basis), so the profiling focuses on the PD loop overhead.
"""

import argparse
import gc
import tracemalloc
from time import time

import numpy as np


def make_toy_operators(shape_x, shape_v):
    """
    Build toy operators simulating a diagonal Hessian imaging problem.

    The data fidelity is 0.5 * ||D * x - dirty||^2 where D is a
    positive diagonal (simulating a beam-weighted PSF convolution)
    and dirty is a positive "observed" image. grad(x) = D*(D*x - dirty)
    so the optimum of the smooth part alone is x = dirty/D, which is
    positive — the positivity constraint and l1 regularisation push
    the solution slightly below this but it stays well above zero.
    """
    nband, nx, ny = shape_x
    _, nbasis, nymax, nxmax = shape_v
    rng = np.random.default_rng(42)

    # diagonal operator (simulates beam * PSF eigenvalues)
    diag = rng.uniform(0.5, 2.0, size=shape_x).astype(np.float64)
    diag_sq = diag * diag
    hessnorm = float(np.max(diag_sq))

    # positive "dirty image" that x should recover
    dirty = rng.uniform(1.0, 5.0, size=shape_x).astype(np.float64)
    diag_dirty = diag * dirty  # precompute D * dirty

    def grad(x):
        # gradient of 0.5 * ||D*x - dirty||^2 = D*(D*x - dirty)
        return diag_sq * x - diag_dirty

    def psi(x, v):
        """image to coeffs (in-place into v). Called as psi(xp, v)."""
        v[:] = 0.0
        v[:, 0, :ny, :nx] = x

    def psih(v, xout):
        """coeffs to image (in-place into xout). Called as psih(vp, xout)."""
        xout[:] = v[:, 0, :ny, :nx]

    # small l1 weights so regulariser doesn't dominate
    l1weight = rng.uniform(0.01, 0.1, size=(nbasis, nymax, nxmax)).astype(np.float64)

    return psi, psih, grad, l1weight, hessnorm


def run_profile(pd_fn, name, kwargs, maxit, warmup):
    print(f"\n{'=' * 60}")
    print(f"Profiling: {name}")
    print(f"{'=' * 60}")

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        kw = {**kwargs}
        kw["x"] = kwargs["x"].copy()
        kw["v"] = kwargs["v"].copy()
        kw["maxit"] = 3
        t0 = time()
        pd_fn(**kw, verbosity=0)
        print(f"  warmup {i + 1}: {time() - t0:.3f}s")

    # Timed run with memory tracking
    print(f"Timed run ({maxit} iterations)...")
    kw = {**kwargs}
    kw["x"] = kwargs["x"].copy()
    kw["v"] = kwargs["v"].copy()
    kw["maxit"] = maxit
    kw["verbosity"] = 2
    kw["report_freq"] = maxit + 1
    gc.collect()
    tracemalloc.start()
    t0 = time()
    x, v = pd_fn(**kw)
    elapsed = time() - t0
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Total:    {elapsed:.3f}s")
    print(f"  Per iter: {elapsed / maxit * 1000:.1f}ms")
    print(f"  ||x|| = {np.linalg.norm(x):.6e}")
    print(f"  ||v|| = {np.linalg.norm(v):.6e}")
    print(f"  Memory - current: {mem_current / 1e6:.1f} MB, peak: {mem_peak / 1e6:.1f} MB")
    return elapsed, x, v, mem_peak


def main():
    parser = argparse.ArgumentParser(description="Profile primal_dual implementations")
    parser.add_argument(
        "--shape",
        type=str,
        default="8,8000,8000",
        help="Image shape as comma-separated ints (default: 8,8000,8000)",
    )
    parser.add_argument("--nbasis", type=int, default=6, help="Number of wavelet bases")
    parser.add_argument("--maxit", type=int, default=50, help="Iterations for timed run")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    args = parser.parse_args()

    shape_x = tuple(int(s) for s in args.shape.split(","))
    nband, nx, ny = shape_x
    nbasis = args.nbasis
    nxmax, nymax = nx, ny
    shape_v = (nband, nbasis, nymax, nxmax)

    nbytes_x = int(np.prod(shape_x)) * 8
    nbytes_v = int(np.prod(shape_v)) * 8
    mem_gb = (nbytes_x * 4 + nbytes_v * 3) / 1e9
    print(f"Image shape: {shape_x}")
    print(f"Dual shape:  {shape_v}")
    print(f"Estimated memory: ~{mem_gb:.1f} GB")

    psi, psih, grad, l1weight, hessnorm = make_toy_operators(shape_x, shape_v)

    # start from zeros — realistic for imaging (no prior model)
    x0 = np.zeros(shape_x, dtype=np.float64)
    v0 = np.zeros(shape_v, dtype=np.float64)

    from pfb_imaging.opt.primal_dual import primal_dual_numba, primal_dual_optimised

    # In primal_dual_optimised/numba:
    #   psi(xp, v)     — image->coeffs
    #   psih(vp, xout) — coeffs->image
    kwargs = dict(
        x=x0,
        v=v0,
        lam=0.01,
        psi=psi,
        psih=psih,
        hessnorm=hessnorm,
        prox=None,
        l1weight=l1weight,
        reweighter=None,
        grad=grad,
        nu=1.0,
        tol=1e-30,
        positivity=1,
        report_freq=100,
        gamma=1.0,
    )

    t_opt, x_opt, v_opt, m_opt = run_profile(
        primal_dual_optimised,
        "primal_dual_optimised (numexpr)",
        kwargs,
        args.maxit,
        args.warmup,
    )
    t_nb, x_nb, v_nb, m_nb = run_profile(
        primal_dual_numba,
        "primal_dual_numba (numba)",
        kwargs,
        args.maxit,
        args.warmup,
    )

    # Correctness check
    print(f"\n{'=' * 60}")
    print("Correctness check (numba vs optimised)")
    print(f"{'=' * 60}")

    def rdiff(a, b):
        return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)

    x_d = rdiff(x_opt, x_nb)
    v_d = rdiff(v_opt, v_nb)
    print(f"  Relative diff in x: {x_d:.3e}")
    print(f"  Relative diff in v: {v_d:.3e}")
    atol = 1e-7
    worst = max(x_d, v_d)
    if worst < atol:
        print(f"  PASS (both < {atol:.0e})")
    else:
        print(f"  FAIL (worst {worst:.3e} exceeds {atol:.0e})")

    # Performance summary
    print(f"\n{'=' * 60}")
    print("Performance summary")
    print(f"{'=' * 60}")
    print(f"  optimised: {t_opt:.3f}s ({t_opt / args.maxit * 1000:.1f} ms/iter)")
    print(f"  numba:     {t_nb:.3f}s ({t_nb / args.maxit * 1000:.1f} ms/iter)")
    speedup = t_opt / t_nb
    if speedup > 1:
        print(f"  numba is {speedup:.2f}x faster")
    else:
        print(f"  optimised is {1 / speedup:.2f}x faster")

    # Memory summary
    print(f"\n{'=' * 60}")
    print("Memory summary (peak during timed run)")
    print(f"{'=' * 60}")
    print(f"  optimised: {m_opt / 1e6:.1f} MB")
    print(f"  numba:     {m_nb / 1e6:.1f} MB")
    if m_opt > 0 and m_nb > 0:
        ratio = m_opt / m_nb
        if ratio > 1:
            print(f"  numba uses {ratio:.2f}x less memory")
        else:
            print(f"  optimised uses {1 / ratio:.2f}x less memory")


if __name__ == "__main__":
    main()
