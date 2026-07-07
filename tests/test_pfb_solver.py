"""PFBSolver: composition of (hess, forward, backward, prox) behind DeconvSolver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.deconv import DeconvSolver
from pfb_imaging.operators.psi import IdentityPsi
from pfb_imaging.opt.forward_backward import ForwardBackward
from pfb_imaging.opt.pcg import PCG
from pfb_imaging.prox.l1 import L1


class IdOp:
    """Identity Hessian."""

    def dot(self, x):
        return x.copy()

    def hdot(self, x):
        return x.copy()


def _solver(b, lam_unused=None, **kwargs):
    from pfb_imaging.deconv.pfb import PFBSolver

    nband, nx, ny = b.shape
    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-10, maxit=5000, verbosity=0, gamma=0.45)
    return PFBSolver(
        IdOp(),
        PCG(tol=1e-12, maxit=200, minit=1, verbosity=0),
        fb,
        reg,
        model=np.zeros_like(b),
        update=np.zeros_like(b),
        gamma=1.0,
        hessnorm=1.0,
        l1_reweight_from=-1,  # reweighting disabled
        **kwargs,
    )


def test_satisfies_deconv_solver_protocol():
    b = np.zeros((1, 4, 4))
    assert isinstance(_solver(b), DeconvSolver)


def test_one_major_cycle_identity_hess():
    """With H = I and model0 = 0: update = residual = b, xtilde = b,
    and backward solves min 0.5||x-b||^2 + lam||x||_1 -> soft(b, lam)."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((1, 30, 4))
    lam = 0.3

    solver = _solver(b)
    solver.first(b)
    update = solver.forward(b)
    assert_allclose(update, b, atol=1e-10)  # I^{-1} b

    model = solver.backward(lam)
    assert_allclose(model, np.sign(b) * np.maximum(np.abs(b) - lam, 0.0), atol=1e-4)

    solver.last()  # reweighting disabled: must be a no-op
    assert solver.reweight_active is False


def test_power_method_when_hessnorm_none():
    b = np.zeros((1, 8, 8))
    from pfb_imaging.deconv.pfb import PFBSolver

    reg = L1(IdentityPsi(1, 8, 8))
    solver = PFBSolver(
        IdOp(),
        PCG(verbosity=0),
        ForwardBackward(verbosity=0),
        reg,
        model=b.copy(),
        update=b.copy(),
        hessnorm=None,
        pm_verbose=0,
        l1_reweight_from=-1,
    )
    # spectral norm of I is 1; the 1.05 safety factor is applied
    assert solver.hess_norm == pytest.approx(1.05, rel=1e-2)


def test_reweight_on_converge_counter():
    from pfb_imaging.deconv.pfb import ReweightOnConverge

    class StubReg:
        reweight_active = True

        def __init__(self):
            self.calls = 0

        def update_weights(self, x):
            self.calls += 1

    reg = StubReg()
    cb = ReweightOnConverge(reg, maxreweight=2, verbosity=0)
    x = np.zeros(3)
    assert cb(x, 10, 1e-6) is False and reg.calls == 1  # first reweight, continue
    assert cb(x, 11, 1e-6) is False and reg.calls == 2  # consecutive -> count 1
    assert cb(x, 12, 1e-6) is False and reg.calls == 3  # consecutive -> count 2 == max
    assert cb(x, 13, 1e-6) is True  # cap reached -> stop
    cb.reset()
    assert cb(x, 20, 1e-6) is False  # counter cleared

    reg.reweight_active = False
    cb2 = ReweightOnConverge(reg, maxreweight=2, verbosity=0)
    assert cb2(x, 0, 1e-6) is True  # not armed -> stop at convergence


def test_trigger_reweight_arms_last():
    rng = np.random.default_rng(1)
    b = rng.standard_normal((1, 16, 16))
    from pfb_imaging.deconv.pfb import PFBSolver
    from pfb_imaging.prox.l21 import L21
    from tests.test_regularisers import SlicePsi

    psi = SlicePsi(1, 16, 16, 2, 20, 20)
    reg = L21(psi, bases=("self", "db1"))
    solver = PFBSolver(
        IdOp(),
        PCG(tol=1e-10, maxit=100, minit=1, verbosity=0),
        ForwardBackward(tol=1e-8, maxit=500, verbosity=0, gamma=0.45),
        reg,
        model=np.zeros_like(b),
        update=np.zeros_like(b),
        hessnorm=1.0,
        l1_reweight_from=100,  # far in the future
    )
    solver.first(b)
    solver.forward(b)
    solver.backward(0.1)
    solver.last()
    assert not solver.reweight_active  # threshold not reached
    solver.trigger_reweight()
    solver.last()
    assert solver.reweight_active  # armed by trigger
