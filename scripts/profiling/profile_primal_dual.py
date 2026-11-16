"""Profile the primal-dual solvers on a toy diagonal Hessian with a real wavelet dictionary.

Compares the two loops that solve deconv's backward step:

  - ``primal_dual_numba``: the frozen legacy solver, kept as the test oracle.
  - ``PrimalDual`` + ``L21``: the composable solver `pfb deconv` runs.

The smooth term is ``0.5 ||D x - dirty||^2`` with a positive diagonal ``D``; the
regulariser is the weighted l21 norm over a SARA dictionary (``PsiNocopyt``,
in-process). Wavelet and gradient time are reported separately, so "solver"
time is the dual update, extrapolation, primal step and convergence test.

The two solvers take different primal steps unless the dictionary has a single
basis (the legacy one squares ``nu``, D57), so their iterates are only expected
to agree with ``--bases self``. ``PrimalDual`` also logs its own per-section
clock, which splits its time further.

Usage:
    uv run python scripts/profiling/profile_primal_dual.py [--shape 4,2048,2048] [--maxit 50] [--nthreads N]
"""

import argparse
from types import SimpleNamespace

from _common import ClockedOp, compare, parse_shape, rdiff, section, setup_threads, timed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", default="4,2048,2048", help="Image cube (nband, nx, ny), comma separated")
    parser.add_argument("--bases", default="self,db1,db2,db3", help="Wavelet bases, comma separated")
    parser.add_argument("--nlevel", type=int, default=3, help="Wavelet decomposition levels")
    parser.add_argument("--maxit", type=int, default=50, help="Iterations of the timed run")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs")
    parser.add_argument("--nthreads", type=int, default=None, help="numba threads (default: all cores)")
    args = parser.parse_args()
    nthreads = setup_threads(args.nthreads)

    # deferred: after setup_threads, which sets the BLAS/numba thread environment
    import numpy as np

    from pfb_imaging.operators.psi import PsiNocopyt
    from pfb_imaging.opt.primal_dual import PrimalDual, primal_dual_numba
    from pfb_imaging.prox.l21 import L21
    from pfb_imaging.prox.positivity import positivity

    nband, nx, ny = parse_shape(args.shape)
    bases = tuple(args.bases.split(","))
    nbasis = len(bases)
    psi = PsiNocopyt(nband, nx, ny, list(bases), args.nlevel, nthreads)
    shape_x = (nband, nx, ny)
    shape_v = (nband, nbasis, psi.nymax, psi.nxmax)
    gb = (int(np.prod(shape_x)) * 4 + int(np.prod(shape_v)) * 2) * 8 / 1e9
    print(f"Image {shape_x}, dual {shape_v}, bases {bases}, {nthreads} numba threads, ~{gb:.1f} GB")

    rng = np.random.default_rng(42)
    diag = rng.uniform(0.5, 2.0, size=shape_x)
    diag_sq = diag * diag
    hessnorm = float(diag_sq.max())
    diag_dirty = diag * rng.uniform(1.0, 5.0, size=shape_x)
    l1weight = rng.uniform(0.01, 0.1, size=shape_v[1:])
    lam = 0.01
    nu = float(nbasis)  # squared frame bound of a concatenation of orthonormal bases

    grad = ClockedOp(lambda x: diag_sq * x - diag_dirty)
    dot = ClockedOp(psi.dot)
    hdot = ClockedOp(psi.hdot)
    ops = {"grad": grad, "psi.dot": dot, "psi.hdot": hdot}

    def run_legacy(niter):
        # naming trap in the legacy signature: its `psih` is synthesis, its `psi` analysis
        return primal_dual_numba(
            np.zeros(shape_x),
            np.zeros(shape_v),
            lam,
            hdot,
            dot,
            hessnorm,
            prox=None,
            l1weight=l1weight,
            reweighter=None,
            grad=grad,
            nu=nu,
            tol=0.0,
            maxit=niter,
            positivity=1,
            verbosity=0,
        )

    # psi with its two operators clocked; PrimalDual and L21 read the sizes off it
    clocked_psi = SimpleNamespace(
        nband=nband, nx=nx, ny=ny, nbasis=nbasis, nymax=psi.nymax, nxmax=psi.nxmax, dot=dot, hdot=hdot
    )

    def run_composed(niter):
        reg = L21(clocked_psi, bases=bases, nu=nu)
        reg.l1weight = l1weight
        pd = PrimalDual(tol=0.0, maxit=niter, verbosity=1 if niter > 3 else 0, primal_prox=positivity)
        pd.setup(reg, hessnorm)
        pd.set_grad(grad)
        return pd.solve(np.zeros(shape_x), lam), pd._v

    legacy = timed(run_legacy, "primal_dual_numba (legacy oracle)", args.maxit, args.warmup, ops)
    composed = timed(run_composed, "PrimalDual + L21 (pfb deconv)", args.maxit, args.warmup, ops)

    section("Agreement")
    d_x = rdiff(legacy[0][0], composed[0][0])
    d_v = rdiff(legacy[0][1], composed[0][1])
    expected = "expected to agree" if nbasis == 1 else "steps differ for more than one basis (D57)"
    print(f"  relative difference in x: {d_x:.3e}, in v: {d_v:.3e}  ({expected})")
    compare("legacy", legacy, "PrimalDual", composed, args.maxit)


if __name__ == "__main__":
    main()
