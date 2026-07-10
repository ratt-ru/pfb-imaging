"""Concrete ForwardBackward solver (composition, no subclassing)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.operators.psi import IdentityPsi
from pfb_imaging.opt import BackwardSolver
from pfb_imaging.opt.forward_backward import ForwardBackward
from pfb_imaging.prox.l1 import L1

pmp = pytest.mark.parametrize


@pmp("acceleration", [True, False])
@pmp("lam", [0.1, 1.0])
def test_lasso_analytic(acceleration, lam):
    """min 0.5||x - b||^2 + lam||x||_1 -> soft threshold of b."""
    nband, nx, ny = 1, 50, 4
    rng = np.random.default_rng(0)
    b = rng.standard_normal((nband, nx, ny))

    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-10, maxit=5000, verbosity=0, gamma=0.45, acceleration=acceleration)
    assert isinstance(fb, BackwardSolver)
    fb.setup(reg, hessnorm=1.0)  # step = 0.9 < 1/L
    fb.set_grad(lambda x: x - b)
    x = fb.solve(np.zeros_like(b), lam)

    x_star = np.sign(b) * np.maximum(np.abs(b) - lam, 0.0)
    assert_allclose(x, x_star, atol=1e-4)


def test_generic_tight_frame_matches_handcoded_l21():
    """The generic tight-frame prox reproduces the old L21ForwardBackward.prox."""
    from pfb_imaging.prox.l21 import L21
    from pfb_imaging.prox.prox_21m import prox_21m_numba
    from tests.test_regularisers import SlicePsi

    nband, nx, ny, nbasis = 2, 8, 8, 2
    psi = SlicePsi(nband, nx, ny, nbasis, nx + 4, ny + 4)
    reg = L21(psi, bases=("self", "db1"))
    nu = 2.0
    reg.nu = nu

    fb = ForwardBackward(verbosity=0)
    fb.setup(reg, hessnorm=1.0)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((nband, nx, ny))
    lam = 0.3

    # reference: the deleted L21ForwardBackward.prox, hand-coded
    alpha = np.zeros((nband, nbasis, psi.nymax, psi.nxmax))
    buf = np.zeros_like(alpha)
    xout = np.zeros_like(x)
    psi.dot(x, alpha)
    prox_21m_numba(alpha, buf, fb.step * lam, sigma=1.0, weight=reg.l1weight)
    buf -= alpha
    psi.hdot(buf, xout)
    want = x + xout / nu

    got = fb._apply_prox(x.copy(), lam)
    assert_allclose(got, want, rtol=1e-13)


def test_on_converge_continues_iteration():
    calls = []

    def cb(x, k, eps):
        calls.append(k)
        return len(calls) >= 3  # keep going twice, stop on the third event

    nband, nx, ny = 1, 10, 1
    b = np.ones((nband, nx, ny))
    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-6, maxit=1000, verbosity=0, gamma=0.45, on_converge=cb)
    fb.setup(reg, hessnorm=1.0)
    fb.set_grad(lambda x: x - b)
    fb.solve(np.zeros_like(b), 0.1)
    assert len(calls) == 3


def test_primal_prox_applied():
    from pfb_imaging.prox.positivity import positivity

    nband, nx, ny = 1, 20, 1
    b = np.linspace(-1, 1, nx).reshape(nband, nx, ny)
    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-10, maxit=2000, verbosity=0, gamma=0.45, primal_prox=positivity)
    fb.setup(reg, hessnorm=1.0)
    fb.set_grad(lambda x: x - b)
    x = fb.solve(np.zeros_like(b), 0.05)
    assert (x >= 0).all()


def test_solve_raises_without_setup_or_grad():
    fb = ForwardBackward(verbosity=0)
    with pytest.raises(RuntimeError, match="setup"):
        fb.solve(np.zeros((1, 2, 2)), 1.0)
    fb.setup(L1(IdentityPsi(1, 2, 2)), hessnorm=1.0)
    with pytest.raises(RuntimeError, match="set_grad"):
        fb.solve(np.zeros((1, 2, 2)), 1.0)
