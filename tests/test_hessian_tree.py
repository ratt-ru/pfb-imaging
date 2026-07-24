"""HessianTree sum-over-partitions PSF-convolution operator (casacore-free)."""

import numpy as np
from ducc0.fft import r2c

from pfb_imaging.operators.hessian import HessianTree

ifftshift = np.fft.ifftshift


def _delta_psf_partition(nx, ny, nx_psf, ny_psf, wsum, seed=0):
    """A partition whose PSF is a delta of integral ``wsum`` and unit beam."""
    psf = np.zeros((1, nx_psf, ny_psf))
    psf[0, nx_psf // 2, ny_psf // 2] = wsum  # delta scaled by wsum
    psfhat = r2c(ifftshift(psf, axes=(1, 2)), axes=(1, 2), forward=True, inorm=0)
    beam = np.ones((1, nx, ny))
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([float(wsum)])}


def test_delta_psf_is_identity_up_to_eta():
    nx = ny = 8
    nx_psf = ny_psf = 16
    parts = [_delta_psf_partition(nx, ny, nx_psf, ny_psf, wsum=3.0)]
    hess = HessianTree(parts, nx, ny, nx_psf, ny_psf, eta=0.0)
    x = np.zeros((1, nx, ny))
    x[0, 4, 4] = 1.0
    out = hess.dot(x)
    # normalised delta-PSF convolution is the identity
    np.testing.assert_allclose(out, x, atol=1e-6)


def test_eta_adds_tikhonov():
    nx = ny = 8
    nx_psf = ny_psf = 16
    parts = [_delta_psf_partition(nx, ny, nx_psf, ny_psf, wsum=3.0)]
    eta = 0.5
    hess = HessianTree(parts, nx, ny, nx_psf, ny_psf, eta=eta)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, nx, ny))
    out = hess.dot(x)
    np.testing.assert_allclose(out, (1.0 + eta) * x, atol=1e-6)


def test_two_identical_partitions_equal_one():
    nx = ny = 8
    nx_psf = ny_psf = 16
    one = [_delta_psf_partition(nx, ny, nx_psf, ny_psf, 3.0)]
    two = [_delta_psf_partition(nx, ny, nx_psf, ny_psf, 3.0), _delta_psf_partition(nx, ny, nx_psf, ny_psf, 3.0)]
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, nx, ny))
    o1 = HessianTree(one, nx, ny, nx_psf, ny_psf, eta=0.1).dot(x)
    o2 = HessianTree(two, nx, ny, nx_psf, ny_psf, eta=0.1).dot(x)
    # averaging two identical (wsum-weighted) partitions is the same single Hessian
    np.testing.assert_allclose(o1, o2, atol=1e-6)


def test_accepts_2d_input():
    nx = ny = 8
    nx_psf = ny_psf = 16
    parts = [_delta_psf_partition(nx, ny, nx_psf, ny_psf, 2.0)]
    hess = HessianTree(parts, nx, ny, nx_psf, ny_psf, eta=0.0)
    x = np.zeros((nx, ny))
    x[3, 3] = 1.0
    out = hess.dot(x)
    assert out.shape == (1, nx, ny)
    np.testing.assert_allclose(out[0], x, atol=1e-6)


def test_hessian_tree_nonsquare_yx():
    """(Y, X) semantics: a (ny, nx) = (12, 16) delta-PSF Hessian is identity."""
    nx, ny = 16, 12
    nx_psf, ny_psf = 32, 24
    psf = np.zeros((1, ny_psf, nx_psf))
    psf[0, ny_psf // 2, nx_psf // 2] = 2.0
    psfhat = np.abs(r2c(ifftshift(psf, axes=(1, 2)), axes=(1, 2), forward=True, inorm=0))
    parts = [{"psfhat": psfhat, "beam": np.ones((1, ny, nx)), "wsum": np.array([2.0])}]
    hess = HessianTree(parts, nx, ny, nx_psf, ny_psf, eta=0.0)
    x = np.random.default_rng(0).standard_normal((1, ny, nx))
    out = hess.dot(x)
    assert out.shape == (1, ny, nx)
    np.testing.assert_allclose(out, x, rtol=1e-10, atol=1e-12)
