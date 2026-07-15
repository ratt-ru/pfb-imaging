"""HessTreeRay: Ray band-actor Hessian vs local HessianTree/HessPSF (tier 2)."""

import numpy as np
import pytest
from ducc0.fft import r2c
from numpy.testing import assert_allclose

from pfb_imaging.operators import LinearOperator
from pfb_imaging.operators.hessian import HessianTree, HessPSF, HessTreeRay

pmp = pytest.mark.parametrize


def _rand_part(rng, nx, ny, nx_psf, ny_psf, wsum=1.0):
    """Partition dict with a positive real psfhat (|FT of a random psf|)."""
    psf = rng.uniform(0.0, 1.0, size=(nx_psf, ny_psf))
    psfhat = np.abs(r2c(psf, axes=(0, 1), forward=True, inorm=0))[None]
    beam = np.ones((1, nx, ny))
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([wsum])}


def test_wsum_override():
    rng = np.random.default_rng(0)
    nx = ny = 8
    part = _rand_part(rng, nx, ny, 2 * nx, 2 * ny, wsum=4.0)
    x = rng.standard_normal((1, nx, ny))
    default = HessianTree([part], nx, ny, 2 * nx, 2 * ny).dot(x)
    overridden = HessianTree([part], nx, ny, 2 * nx, 2 * ny, wsum=8.0).dot(x)
    assert_allclose(overridden, default / 2.0, rtol=1e-13)


@pmp("nband", [1, 3])
def test_dot_matches_local_hessian_tree(nband):
    rng = np.random.default_rng(1)
    nx = ny = 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny) for _ in range(2)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.01)
    assert isinstance(hess, LinearOperator)

    x = rng.standard_normal((nband, nx, ny))
    got = hess.dot(x)
    want = np.zeros_like(x)
    for b in range(nband):
        local = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=0.01)
        want[b] = local.dot(x[b])[0]
    assert_allclose(got, want, rtol=1e-12)


def test_dot_matches_hess_psf_single_partition():
    """Single partition, unit wsum, no beam, eta=0: HessTreeRay == HessPSF."""
    rng = np.random.default_rng(2)
    nband, nx, ny = 2, 16, 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    abspsf = np.concatenate([p[0]["psfhat"] for p in parts], axis=0)

    tree = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.0)
    # taper_width only affects HessPSF.idot (unused here); the default (32)
    # errors for nx=ny=16 (taperf slices [:taper_width] on a length-nx axis),
    # so shrink it as tests/test_protocols.py already does for small images.
    ref = HessPSF(nx, ny, abspsf, eta=0.0, taper_width=2)

    x = rng.standard_normal((nband, nx, ny))
    assert_allclose(tree.dot(x), ref.dot(x).copy(), rtol=1e-12)


@pmp("nband", [1, 3])
def test_cg_matches_local_pcg(nband):
    from pfb_imaging.opt.pcg import pcg_numba

    rng = np.random.default_rng(3)
    nx = ny = 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.5, cg_tol=1e-8, cg_maxit=200)

    rhs = rng.standard_normal((nband, nx, ny))
    got = hess.cg(rhs)
    for b in range(nband):
        local = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=0.5)
        want_b = pcg_numba(lambda z: local.dot(z)[0], rhs[b], tol=1e-8, maxit=200, minit=1, verbosity=0)
        assert_allclose(got[b], want_b, rtol=1e-6, atol=1e-9)
