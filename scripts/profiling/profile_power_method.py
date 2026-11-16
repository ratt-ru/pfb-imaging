"""Profile the power-method implementations on a toy diagonal operator.

Compares ``power_method`` (numpy) with ``power_method_numba`` (fused numba
kernels). The operator is an elementwise multiply, timed separately, so
"solver" time is the normalisation and convergence bookkeeping.

Usage:
    uv run python scripts/profiling/profile_power_method.py [--shape 8,8000,8000] [--maxit 50] [--nthreads N]
"""

import argparse

from _common import ClockedOp, compare, parse_shape, rdiff, section, setup_threads, timed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", default="8,8000,8000", help="Cube shape, comma separated")
    parser.add_argument("--maxit", type=int, default=50, help="Iterations of the timed run")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs")
    parser.add_argument("--nthreads", type=int, default=None, help="numba threads (default: all cores)")
    args = parser.parse_args()
    nthreads = setup_threads(args.nthreads)

    # deferred: after setup_threads, which sets the BLAS/numba thread environment
    import numpy as np

    from pfb_imaging.opt.power_method import power_method, power_method_numba

    shape = parse_shape(args.shape)
    print(f"Shape {shape}, {int(np.prod(shape)) * 8 / 1e9:.2f} GB per cube, {nthreads} numba threads")

    rng = np.random.default_rng(42)
    diag = rng.uniform(0.5, 2.0, size=shape) + 0.1
    b0 = rng.standard_normal(shape)
    aop = ClockedOp(lambda x: diag * x)

    def run(solver):
        # tol far below reach pins the iteration count
        return lambda niter: solver(aop, shape, b0=b0.copy(), tol=1e-30, maxit=niter, verbosity=0)

    ref = timed(run(power_method), "power_method (numpy)", args.maxit, args.warmup, {"aop": aop})
    fused = timed(run(power_method_numba), "power_method_numba", args.maxit, args.warmup, {"aop": aop})

    section("Agreement")
    (beta_np, b_np), (beta_nb, b_nb) = ref[0], fused[0]
    d_beta = abs(beta_np - beta_nb) / abs(beta_np)
    # an eigenvector is defined up to sign
    d_b = min(rdiff(b_np, b_nb), rdiff(b_np, -b_nb))
    print(f"  largest eigenvalue: {beta_np:.6e} (true {diag.max():.6e} once converged)")
    verdict = "ok" if max(d_beta, d_b) < 1e-7 else "DIFFERENT"
    print(f"  relative difference in beta: {d_beta:.3e}, in b: {d_b:.3e}  ({verdict})")
    compare("numpy", ref, "numba", fused, args.maxit)


if __name__ == "__main__":
    main()
