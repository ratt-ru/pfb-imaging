import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.deconv.sara_pd import L21PrimalDual
from pfb_imaging.opt.primal_dual import PrimalDual, primal_dual_numba

pmp = pytest.mark.parametrize


# ---------------------------------------------------------------------------
# Simple PsiOperatorProtocol implementations for testing
# ---------------------------------------------------------------------------


class IdentityPsi:
    """Identity operator satisfying PsiOperatorProtocol."""

    def dot(self, x, v):
        np.copyto(v, x)

    def hdot(self, v, xout):
        np.copyto(xout, v)


class SlicePsi:
    """Psi operator that embeds/extracts via slicing (for L21 tests)."""

    def __init__(self, nx, ny, nbasis, nymax, nxmax):
        self.nx = nx
        self.ny = ny
        self.nbasis = nbasis
        self.nymax = nymax
        self.nxmax = nxmax

    def dot(self, x, v):
        v[:] = 0.0
        v[:, 0, : self.ny, : self.nx] = x

    def hdot(self, v, xout):
        xout[:] = v[:, 0, : self.ny, : self.nx]


# ---------------------------------------------------------------------------
# QuadraticL1PrimalDual — test-only subclass for the LASSO problem
#
#   min_x  0.5 * ||x - b||^2  +  lam * ||x||_1
#
# Analytic solution:  x* = sign(b) * max(|b| - lam, 0)  (soft-threshold)
# ---------------------------------------------------------------------------


class QuadraticL1PrimalDual(PrimalDual):
    """Primal-dual solver for LASSO with identity measurement operator.

    Uses pure-numpy dual step (no numba kernels needed for testing).

    psi.dot  = identity: copies x into v
    psi.hdot = identity: copies v into xout
    """

    def __init__(self, b, lam, **kwargs):
        # hessnorm = 1 for A = I
        super().__init__(IdentityPsi(), hessnorm=1.0, **kwargs)
        self.lam = lam
        self.set_grad(lambda x: x - b)

    def dual_step(self, xp, v, vp):
        # v_new = clip(vp + sigma * Ψ†xp, -lam, lam)
        self.psi.dot(xp, v)  # v = xp
        np.clip(vp + self.sigma * v, -self.lam, self.lam, out=v)

    def prox_primal(self, x):
        pass  # no positivity constraint for LASSO test


# ---------------------------------------------------------------------------
# Helper: toy operators for L21 test (matches profiling script style)
# ---------------------------------------------------------------------------


def make_l21_problem(shape_x, shape_v, seed=42):
    """Build a small diagonal-Hessian imaging problem."""
    nband, nx, ny = shape_x
    _, nbasis, nymax, nxmax = shape_v
    rng = np.random.default_rng(seed)

    diag = rng.uniform(0.5, 2.0, size=shape_x).astype(np.float64)
    diag_sq = diag * diag
    hessnorm = float(np.max(diag_sq))
    dirty = rng.uniform(1.0, 5.0, size=shape_x).astype(np.float64)
    diag_dirty = diag * dirty

    def grad(x):
        return diag_sq * x - diag_dirty

    psi_op = SlicePsi(nx, ny, nbasis, nymax, nxmax)

    # bare functions for primal_dual_numba (legacy interface)
    def psi_func(x, v):
        v[:] = 0.0
        v[:, 0, :ny, :nx] = x

    def psih_func(v, xout):
        xout[:] = v[:, 0, :ny, :nx]

    l1weight = rng.uniform(0.01, 0.1, size=(nbasis, nymax, nxmax)).astype(np.float64)

    return psi_op, psi_func, psih_func, grad, l1weight, hessnorm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pmp("n", [50, 200])
@pmp("lam", [0.1, 1.0, 5.0])
def test_lasso_identity_analytic(n, lam):
    """QuadraticL1PrimalDual recovers the analytic soft-threshold solution."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n).astype(np.float64)
    x_star = np.sign(b) * np.maximum(np.abs(b) - lam, 0.0)

    x0 = np.zeros(n, dtype=np.float64)
    v0 = np.zeros(n, dtype=np.float64)

    solver = QuadraticL1PrimalDual(b, lam, tol=1e-10, maxit=5000, verbosity=0)
    x, _ = solver.solve(x0, v0)

    assert_allclose(x, x_star, atol=1e-4)


@pmp("nband", [1, 3])
@pmp("nx", [16, 32])
@pmp("positivity", [0, 1])
def test_l21_matches_primal_dual_numba(nband, nx, positivity):
    """L21PrimalDual and primal_dual_numba produce identical output."""
    ny = nx
    nbasis = 2
    nymax = nxmax = nx
    shape_x = (nband, nx, ny)
    shape_v = (nband, nbasis, nymax, nxmax)

    psi_op, psi_func, psih_func, grad, l1weight, hessnorm = make_l21_problem(shape_x, shape_v)

    lam = 0.05
    tol = 1e-8
    maxit = 30

    x0 = np.zeros(shape_x, dtype=np.float64)
    v0 = np.zeros(shape_v, dtype=np.float64)

    # --- reference: primal_dual_numba ---
    # NOTE: primal_dual_numba signature has psih (4th) before psi (5th),
    # but in the body psi(xp,v)=analysis and psih(vp,xout)=synthesis.
    x_ref, v_ref = primal_dual_numba(
        x0.copy(),
        v0.copy(),
        lam,
        psih_func,  # 4th positional = psih param (synthesis in the body)
        psi_func,  # 5th positional = psi param (analysis in the body)
        hessnorm,
        prox=None,
        l1weight=l1weight,
        reweighter=None,
        grad=grad,
        nu=1.0,
        tol=tol,
        maxit=maxit,
        positivity=positivity,
        verbosity=0,
    )

    # --- new class ---
    solver = L21PrimalDual(
        psi_op,
        hessnorm,
        lam=lam,
        l1weight=l1weight,
        reweighter=None,
        positivity=positivity,
        nu=1.0,
        tol=tol,
        maxit=maxit,
        verbosity=0,
    )
    solver.set_grad(grad)
    x_cls, v_cls = solver.solve(x0.copy(), v0.copy())

    def rdiff(a, b):
        return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)

    assert rdiff(x_cls, x_ref) < 1e-10, f"x differs: {rdiff(x_cls, x_ref):.3e}"
    assert rdiff(v_cls, v_ref) < 1e-10, f"v differs: {rdiff(v_cls, v_ref):.3e}"


def test_solve_raises_without_grad():
    """solve() raises RuntimeError when set_grad has not been called."""

    class _MinimalPD(PrimalDual):
        def dual_step(self, xp, v, vp):
            self.psi.dot(xp, v)
            np.clip(vp + self.sigma * v, -1.0, 1.0, out=v)

        def prox_primal(self, x):
            pass

    solver = _MinimalPD(IdentityPsi(), hessnorm=1.0, verbosity=0)
    with pytest.raises(RuntimeError, match="set_grad"):
        solver.solve(np.zeros(5), np.zeros(5))
