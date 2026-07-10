"""Concrete PrimalDual solver: composition, Moreau/fused equivalence, legacy oracle."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.operators.psi import IdentityPsi
from pfb_imaging.opt import BackwardSolver
from pfb_imaging.opt.primal_dual import PrimalDual, primal_dual_numba
from pfb_imaging.prox.l1 import L1
from pfb_imaging.prox.l21 import L21
from pfb_imaging.prox.positivity import positivity
from tests.test_regularisers import SlicePsi

pmp = pytest.mark.parametrize


def _l21_problem(nband, nx, ny, nbasis=2, npad=0, seed=42):
    """Small diagonal-Hessian imaging problem (mirrors the old test helper)."""
    nymax, nxmax = nx + npad, ny + npad
    rng = np.random.default_rng(seed)
    diag = rng.uniform(0.5, 2.0, size=(nband, nx, ny))
    diag_sq = diag * diag
    hessnorm = float(np.max(diag_sq))
    dirty = rng.uniform(1.0, 5.0, size=(nband, nx, ny))
    diag_dirty = diag * dirty

    def grad(x):
        return diag_sq * x - diag_dirty

    psi = SlicePsi(nband, nx, ny, nbasis, nymax, nxmax)
    l1weight = rng.uniform(0.01, 0.1, size=(nbasis, nymax, nxmax))
    return psi, grad, l1weight, hessnorm


@pmp("n", [50, 200])
@pmp("lam", [0.1, 1.0])
def test_lasso_identity_analytic(n, lam):
    """PD + L1 + IdentityPsi recovers the soft-threshold solution (Moreau path)."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((1, n, 1))
    x_star = np.sign(b) * np.maximum(np.abs(b) - lam, 0.0)

    reg = L1(IdentityPsi(1, n, 1))
    pd = PrimalDual(tol=1e-10, maxit=5000, verbosity=0)
    assert isinstance(pd, BackwardSolver)
    pd.setup(reg, hessnorm=1.0)
    pd.set_grad(lambda x: x - b)
    x = pd.solve(np.zeros_like(b), lam)
    assert_allclose(x, x_star, atol=1e-4)


@pmp("nband", [1, 3])
@pmp("nx", [16, 32])
@pmp("positivity_mode", [0, 1])
def test_l21_matches_primal_dual_numba(nband, nx, positivity_mode):
    """New PrimalDual + L21 reproduces the legacy primal_dual_numba trajectories."""
    ny = nx
    psi, grad, l1weight, hessnorm = _l21_problem(nband, nx, ny)
    lam, tol, maxit = 0.05, 1e-8, 30

    def psi_func(x, v):
        psi.dot(x, v)

    def psih_func(v, xout):
        psi.hdot(v, xout)

    shape_v = (nband, psi.nbasis, psi.nymax, psi.nxmax)
    x_ref, v_ref = primal_dual_numba(
        np.zeros((nband, nx, ny)),
        np.zeros(shape_v),
        lam,
        psih_func,  # 4th positional (named psih) is the SYNTHESIS op in the body
        psi_func,  # 5th positional (named psi) is the ANALYSIS op in the body
        hessnorm,
        prox=None,
        l1weight=l1weight,
        reweighter=None,
        grad=grad,
        nu=1.0,
        tol=tol,
        maxit=maxit,
        positivity=positivity_mode,
        verbosity=0,
    )

    reg = L21(psi, bases=("self", "db1"))
    reg.l1weight = l1weight
    pd = PrimalDual(tol=tol, maxit=maxit, verbosity=0, primal_prox=positivity if positivity_mode else None)
    pd.setup(reg, hessnorm)
    pd.set_grad(grad)
    x_new = pd.solve(np.zeros((nband, nx, ny)), lam)

    def rdiff(a, b):
        return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)

    assert rdiff(x_new, x_ref) < 1e-10
    assert rdiff(pd._v, v_ref) < 1e-10


def test_fused_and_moreau_paths_agree():
    """PD via reg.dual_update (fused) == PD via generic reg.prox (Moreau)."""
    nband, nx = 2, 16
    psi, grad, l1weight, hessnorm = _l21_problem(nband, nx, nx)
    reg = L21(psi, bases=("self", "db1"))
    reg.l1weight = l1weight

    class MoreauOnly:
        """Same regulariser with the fused fast path hidden."""

        def __init__(self, inner):
            self.psi = inner.psi
            self.nu = inner.nu
            self.prox = inner.prox

    lam, tol, maxit = 0.05, 1e-8, 25
    x0 = np.zeros((nband, nx, nx))

    pd1 = PrimalDual(tol=tol, maxit=maxit, verbosity=0)
    pd1.setup(reg, hessnorm)
    pd1.set_grad(grad)
    x_fused = pd1.solve(x0.copy(), lam)

    pd2 = PrimalDual(tol=tol, maxit=maxit, verbosity=0)
    pd2.setup(MoreauOnly(reg), hessnorm)
    pd2.set_grad(grad)
    x_moreau = pd2.solve(x0.copy(), lam)

    assert_allclose(x_fused, x_moreau, rtol=1e-10, atol=1e-12)


def test_dual_warm_start_and_reset():
    nband, nx = 1, 8
    psi, grad, l1weight, hessnorm = _l21_problem(nband, nx, nx)
    reg = L21(psi, bases=("self", "db1"))
    reg.l1weight = l1weight
    pd = PrimalDual(tol=1e-8, maxit=20, verbosity=0)
    pd.setup(reg, hessnorm)
    pd.set_grad(grad)
    pd.solve(np.zeros((nband, nx, nx)), 0.05)
    assert np.any(pd._v)  # dual retained for warm start
    pd.reset()
    assert not np.any(pd._v)


def test_solve_raises_without_setup_or_grad():
    pd = PrimalDual(verbosity=0)
    with pytest.raises(RuntimeError, match="setup"):
        pd.solve(np.zeros((1, 2, 2)), 1.0)
    pd.setup(L1(IdentityPsi(1, 2, 2)), hessnorm=1.0)
    with pytest.raises(RuntimeError, match="set_grad"):
        pd.solve(np.zeros((1, 2, 2)), 1.0)
