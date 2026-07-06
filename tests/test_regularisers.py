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
