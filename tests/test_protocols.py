"""Protocol conformance for the composable deconv framework (issue #185)."""

import numpy as np

from pfb_imaging.deconv import DeconvSolver, Regulariser
from pfb_imaging.operators import LinearOperator, PsiOperator
from pfb_imaging.operators.hessian import HessianTree, HessPSF
from pfb_imaging.operators.psi import Psi
from pfb_imaging.opt import BackwardSolver, ForwardSolver


def _delta_part(nx_psf, ny_psf):
    """Single partition with a delta-function psfhat (identity convolution)."""
    psfhat = np.ones((1, nx_psf, ny_psf // 2 + 1))
    beam = np.ones((1, nx_psf // 2, ny_psf // 2))
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([1.0])}


def test_hessians_satisfy_linear_operator():
    nx = ny = 8
    hess = HessianTree([_delta_part(2 * nx, 2 * ny)], nx, ny, 2 * nx, 2 * ny)
    assert isinstance(hess, LinearOperator)
    abspsf = np.ones((1, 2 * nx, ny + 1))
    assert isinstance(HessPSF(nx, ny, abspsf, eta=1.0, taper_width=2), LinearOperator)


def test_psi_satisfies_psi_operator():
    psi = Psi(1, 32, 32, ("self", "db1"), 2, 1)
    assert isinstance(psi, PsiOperator)


def test_protocols_reject_nonconforming():
    class Empty:
        pass

    assert not isinstance(Empty(), LinearOperator)
    assert not isinstance(Empty(), PsiOperator)
    assert not isinstance(Empty(), Regulariser)
    assert not isinstance(Empty(), ForwardSolver)
    assert not isinstance(Empty(), BackwardSolver)
    assert not isinstance(Empty(), DeconvSolver)


def test_structural_regulariser_conformance():
    class Toy:
        def __init__(self):
            self.psi = None
            self.nu = 1.0

        def prox(self, v, vout, lam, sigma=1.0):
            np.copyto(vout, v)

    assert isinstance(Toy(), Regulariser)
