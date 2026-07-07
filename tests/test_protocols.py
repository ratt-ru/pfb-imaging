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


def test_identity_psi_roundtrip():
    from pfb_imaging.operators.psi import IdentityPsi

    nband, nx, ny = 2, 8, 6
    psi = IdentityPsi(nband, nx, ny)
    assert isinstance(psi, PsiOperator)
    assert (psi.nbasis, psi.nymax, psi.nxmax) == (1, nx, ny)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((nband, nx, ny))
    alpha = np.zeros((nband, 1, nx, ny))
    xo = np.zeros_like(x)
    psi.dot(x, alpha)
    psi.hdot(alpha, xo)
    np.testing.assert_array_equal(xo, x)


def test_positivity_prox_modes():
    from pfb_imaging.prox.positivity import positivity, positivity_band, positivity_prox

    assert positivity_prox(0) is None
    assert positivity_prox(1) is positivity
    assert positivity_prox(2) is positivity_band

    x = np.array([[[1.0, -1.0], [2.0, -0.5]]])  # (1, 2, 2)
    positivity(x)
    np.testing.assert_array_equal(x, [[[1.0, 0.0], [2.0, 0.0]]])

    # band mode: zero the pixel across all bands if non-positive in any band
    y = np.stack([np.full((2, 2), 1.0), np.array([[1.0, -1.0], [1.0, 1.0]])])
    positivity_band(y)
    assert y[0, 0, 1] == 0.0 and y[1, 0, 1] == 0.0
    assert y[0, 0, 0] == 1.0 and y[1, 1, 1] == 1.0


# ---------------------------------------------------------------------------
# Protocol enforcement: seams raise clear TypeErrors on non-conforming pieces
# ---------------------------------------------------------------------------


class _Bad:
    """Conforms to nothing."""


def test_require_protocol_error_names_missing_members():
    import pytest

    from pfb_imaging.operators import require_protocol

    with pytest.raises(TypeError, match=r"hess must satisfy the LinearOperator Protocol.*missing.*dot"):
        require_protocol(_Bad(), LinearOperator, "hess")
    # conforming objects pass silently
    require_protocol(Psi(1, 32, 32, ("self",), 1, 1), PsiOperator, "psi")


def test_pfb_solver_rejects_nonconforming_pieces():
    import pytest

    from pfb_imaging.deconv.pfb import PFBSolver
    from pfb_imaging.operators.psi import IdentityPsi
    from pfb_imaging.opt.forward_backward import ForwardBackward
    from pfb_imaging.opt.pcg import PCG
    from pfb_imaging.prox.l1 import L1

    class IdOp:
        def dot(self, x):
            return x.copy()

        def hdot(self, x):
            return x.copy()

    b = np.zeros((1, 4, 4))
    reg = L1(IdentityPsi(1, 4, 4))

    def kwargs():
        return dict(model=b.copy(), update=b.copy(), hessnorm=1.0, l1_reweight_from=-1)

    with pytest.raises(TypeError, match="hess must satisfy the LinearOperator Protocol"):
        PFBSolver(_Bad(), PCG(), ForwardBackward(verbosity=0), reg, **kwargs())
    with pytest.raises(TypeError, match="forward_alg must satisfy the ForwardSolver Protocol"):
        PFBSolver(IdOp(), _Bad(), ForwardBackward(verbosity=0), reg, **kwargs())
    with pytest.raises(TypeError, match="backward_alg must satisfy the BackwardSolver Protocol"):
        PFBSolver(IdOp(), PCG(), _Bad(), reg, **kwargs())
    with pytest.raises(TypeError, match="prox must satisfy the Regulariser Protocol"):
        PFBSolver(IdOp(), PCG(), ForwardBackward(verbosity=0), _Bad(), **kwargs())


def test_backward_solvers_validate_regulariser_in_setup():
    import pytest

    from pfb_imaging.opt.forward_backward import ForwardBackward
    from pfb_imaging.opt.primal_dual import PrimalDual

    with pytest.raises(TypeError, match="prox must satisfy the Regulariser Protocol"):
        ForwardBackward(verbosity=0).setup(_Bad(), 1.0)
    with pytest.raises(TypeError, match="prox must satisfy the Regulariser Protocol"):
        PrimalDual(verbosity=0).setup(_Bad(), 1.0)

    class BadPsiReg:
        def __init__(self):
            self.psi = _Bad()  # psi lacking the PsiOperator surface
            self.nu = 1.0

        def prox(self, v, vout, lam, sigma=1.0):
            np.copyto(vout, v)

    with pytest.raises(TypeError, match=r"prox\.psi must satisfy the PsiOperator Protocol"):
        ForwardBackward(verbosity=0).setup(BadPsiReg(), 1.0)
    with pytest.raises(TypeError, match=r"prox\.psi must satisfy the PsiOperator Protocol"):
        PrimalDual(verbosity=0).setup(BadPsiReg(), 1.0)


def test_regularisers_validate_psi():
    import pytest

    from pfb_imaging.prox.l1 import L1
    from pfb_imaging.prox.l21 import L21

    with pytest.raises(TypeError, match="psi must satisfy the PsiOperator Protocol"):
        L1(_Bad())
    with pytest.raises(TypeError, match="psi must satisfy the PsiOperator Protocol"):
        L21(_Bad(), bases=("self",))


def test_pcg_generic_path_validates_hess():
    import pytest

    from pfb_imaging.opt.pcg import PCG

    with pytest.raises(TypeError, match="hess must satisfy the LinearOperator Protocol"):
        PCG(verbosity=0).solve(_Bad(), np.ones((1, 2, 2)))
