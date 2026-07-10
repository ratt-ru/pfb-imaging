"""PCG ForwardSolver: generic path and duck-typed distributed fast path."""

import numpy as np
from numpy.testing import assert_allclose

from pfb_imaging.opt import ForwardSolver
from pfb_imaging.opt.pcg import PCG


class DiagOp:
    def __init__(self, d):
        self.d = d

    def dot(self, x):
        return self.d * x

    def hdot(self, x):
        return self.dot(x)


def test_pcg_solves_diagonal_system():
    rng = np.random.default_rng(0)
    d = rng.uniform(1.0, 3.0, size=(2, 8, 8))
    b = rng.standard_normal((2, 8, 8))
    pcg = PCG(tol=1e-12, maxit=500, minit=1, verbosity=0)
    assert isinstance(pcg, ForwardSolver)
    x = pcg.solve(DiagOp(d), b)
    assert_allclose(x, b / d, atol=1e-8)


def test_pcg_delegates_to_operator_cg():
    class FakeHess:
        def __init__(self):
            self.called_with = None

        def dot(self, x):  # pragma: no cover - must not be used
            raise AssertionError("generic path used despite cg fast path")

        def cg(self, rhs, x0=None, tol=None, maxit=None, minit=None):
            self.called_with = (tol, maxit, minit)
            return rhs * 2.0

    hess = FakeHess()
    pcg = PCG(tol=1e-4, maxit=77, minit=3, verbosity=0)
    out = pcg.solve(hess, np.ones((1, 2, 2)))
    assert hess.called_with == (1e-4, 77, 3)
    assert_allclose(out, 2.0)
