"""Profile the conjugate-gradient solvers on a toy diagonal Hessian.

Compares the two CG loops deconv runs:

  - ``pcg_numba``:  the frozen legacy solver, run inside each band worker.
  - ``cg_inplace``: the driver's cube-level solve, which holds two cubes fewer (D58).

Both take the same steps without a preconditioner, so the iterates should agree
to rounding and any difference is loop overhead and memory. The operator is an
elementwise multiply, timed separately, so "solver" time is the CG bookkeeping.

Usage:
    uv run python scripts/profiling/profile_pcg.py [--shape 8,8000,8000] [--maxit 50] [--nthreads N]
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

    from pfb_imaging.opt.pcg import cg_inplace, pcg_numba

    shape = parse_shape(args.shape)
    cube_gb = int(np.prod(shape)) * 8 / 1e9
    print(f"Shape {shape}, {cube_gb:.2f} GB per cube, {nthreads} numba threads")

    rng = np.random.default_rng(42)
    # positive diagonal plus a ridge: symmetric positive definite, as CG needs
    diag = rng.uniform(0.5, 2.0, size=shape) + 0.1
    b = rng.standard_normal(shape)
    aop = ClockedOp(lambda x: diag * x)

    def run(solver):
        # tol=0 and minit=maxit pin the iteration count; the solvers stop on a stalled step otherwise
        return lambda niter: solver(aop, b, tol=0.0, maxit=niter, minit=niter, verbosity=0)

    legacy = timed(run(pcg_numba), "pcg_numba (band worker)", args.maxit, args.warmup, {"aop": aop})
    inplace = timed(run(cg_inplace), "cg_inplace (driver)", args.maxit, args.warmup, {"aop": aop})

    section("Agreement")
    d = rdiff(legacy[0], inplace[0])
    print(f"  relative difference in x: {d:.3e}  ({'ok' if d < 1e-7 else 'DIFFERENT'})")
    print(f"  residual ||A x - b|| / ||b||: {rdiff(b, diag * inplace[0]):.3e}")
    compare("pcg_numba", legacy, "cg_inplace", inplace, args.maxit)


if __name__ == "__main__":
    main()
