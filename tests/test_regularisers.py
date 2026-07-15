"""Regulariser implementations against the Regulariser Protocol contract."""

import numpy as np
from numpy.testing import assert_allclose

from pfb_imaging.deconv import Regulariser
from pfb_imaging.operators.psi import IdentityPsi


def test_l1_is_soft_threshold():
    from pfb_imaging.prox.l1 import L1

    nband, nx, ny = 2, 6, 5
    reg = L1(IdentityPsi(nband, nx, ny))
    assert isinstance(reg, Regulariser)

    rng = np.random.default_rng(1)
    v = rng.standard_normal((nband, 1, nx, ny))
    vout = np.zeros_like(v)
    lam, sigma = 0.3, 2.0
    reg.prox(v, vout, lam, sigma=sigma)

    # prox_{(lam/sigma) l1}(v/sigma) elementwise
    vs = v / sigma
    expected = np.sign(vs) * np.maximum(np.abs(vs) - lam / sigma, 0.0)
    assert_allclose(vout, expected, rtol=1e-14)


def test_l1_weighting():
    from pfb_imaging.prox.l1 import L1

    reg = L1(IdentityPsi(1, 2, 2))
    reg.weight[...] = 10.0  # threshold everything
    v = np.ones((1, 1, 2, 2))
    vout = np.zeros_like(v)
    reg.prox(v, vout, 1.0)
    assert_allclose(vout, 0.0)


class SlicePsi:
    """Embeds the image into a larger coefficient grid (test helper)."""

    def __init__(self, nband, nx, ny, nbasis, nymax, nxmax):
        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nbasis = nbasis
        self.nymax = nymax
        self.nxmax = nxmax

    def dot(self, x, v):
        v[:] = 0.0
        for b in range(self.nbasis):
            v[:, b, : self.nx, : self.ny] = x

    def hdot(self, v, xout):
        xout[:] = v[:, :, : self.nx, : self.ny].sum(axis=1)


def _l21_reg(nband=2, nx=8, ny=8, nbasis=2, npad=4):
    from pfb_imaging.prox.l21 import L21

    psi = SlicePsi(nband, nx, ny, nbasis, nx + npad, ny + npad)
    return L21(psi, bases=("self", "db1"), rmsfactor=1.0, alpha=2.0), psi


def test_l21_prox_matches_kernel():
    from pfb_imaging.prox.prox_21m import prox_21m_numba

    reg, psi = _l21_reg()
    rng = np.random.default_rng(2)
    v = rng.standard_normal((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))
    got = np.zeros_like(v)
    want = np.zeros_like(v)
    reg.prox(v, got, 0.7, sigma=1.5)
    prox_21m_numba(v, want, 0.7, sigma=1.5, weight=reg.l1weight)
    assert_allclose(got, want, rtol=1e-14)


def test_l21_dual_update_matches_kernel():
    from pfb_imaging.prox.prox_21m import dual_update_numba_fast

    reg, psi = _l21_reg()
    rng = np.random.default_rng(3)
    shape = (psi.nband, psi.nbasis, psi.nymax, psi.nxmax)
    vp, v = rng.standard_normal(shape), rng.standard_normal(shape)
    vp2, v2 = vp.copy(), v.copy()
    reg.dual_update(vp, v, 0.4, sigma=2.0)
    dual_update_numba_fast(vp2, v2, 0.4, sigma=2.0, weight=reg.l1weight)
    assert_allclose(v, v2, rtol=1e-14)


def test_l21_reweighting_lifecycle():
    reg, psi = _l21_reg()
    assert not reg.reweight_active
    w0 = reg.l1weight.copy()

    rng = np.random.default_rng(4)
    update = rng.standard_normal((psi.nband, psi.nx, psi.ny))
    model = np.abs(rng.standard_normal((psi.nband, psi.nx, psi.ny)))

    reg.init_reweighting(update)
    assert reg.reweight_active
    reg.update_weights(model)
    assert reg.l1weight.shape == w0.shape
    assert not np.array_equal(reg.l1weight, w0)
    # high-SNR coefficients get weights near (1+rmsfactor)/large, i.e. weights in (0, 1+rmsfactor]
    assert (reg.l1weight > 0).all() and (reg.l1weight <= 1.0 + reg.rmsfactor).all()


def test_l21_reweighting_survives_zero_update():
    """All-zero update (empty per-basis coefficients) must not poison the weights with NaN."""
    reg, psi = _l21_reg()
    reg.init_reweighting(np.zeros((psi.nband, psi.nx, psi.ny)))
    assert reg.reweight_active

    rng = np.random.default_rng(7)
    model = np.abs(rng.standard_normal((psi.nband, psi.nx, psi.ny)))
    reg.update_weights(model)
    assert np.isfinite(reg.l1weight).all()
