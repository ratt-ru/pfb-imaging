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


def test_prior_term_matches_the_dense_congruence():
    """M_gp x == M_data x + eta * (Cinv @ x) when eta is uniform.

    With eta_mode unset the congruence D^.5 Cinv D^.5 collapses to eta*Cinv, so
    the driver-side term is exactly eta*(Cinv - I) applied over the band axis.
    """
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(3)
    nband, nx, ny = 3, 8, 8
    eta = 1e-2
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)

    base = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=eta)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=eta, freq_prec=kinv)

    x = rng.standard_normal((nband, nx, ny))
    want = base.dot(x) - eta * x + eta * np.einsum("bc,cyx->byx", kinv, x)
    assert_allclose(gp.dot(x), want, rtol=1e-11, atol=1e-13)


def test_prior_is_a_no_op_when_freq_prec_is_none():
    """The default path must be bit-identical to today (regression guard)."""
    rng = np.random.default_rng(4)
    nband, nx, ny = 2, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2, freq_prec=None)
    x = rng.standard_normal((nband, nx, ny))
    want = np.zeros_like(x)
    for b in range(nband):
        want[b] = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=1e-2).dot(x[b])[0]
    assert_allclose(hess.dot(x), want, rtol=0, atol=0)


def test_prior_hessian_stays_symmetric_and_positive_definite():
    """CG requires both; the driver-side correction alone is only NSD."""
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(5)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 1.0, cap=50.0)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2, freq_prec=kinv)

    for _ in range(10):
        u = rng.standard_normal((nband, nx, ny))
        v = rng.standard_normal((nband, nx, ny))
        assert_allclose(np.vdot(u, gp.dot(v)), np.vdot(gp.dot(u), v), rtol=1e-10)
        assert np.vdot(u, gp.dot(u)) > 0.0


def test_wrong_freq_prec_shape_is_rejected():
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(6)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, 4), 0.5)  # 4 bands, not 3
    with pytest.raises(ValueError, match="freq_prec"):
        HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2, freq_prec=kinv)


def test_cg_solves_the_coupled_system_when_the_prior_is_on():
    """The real contract: whichever branch runs, cg must invert the operator dot applies."""
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(7)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-1, freq_prec=kinv, cg_tol=1e-10, cg_maxit=500)

    rhs = rng.standard_normal((nband, nx, ny))
    u = gp.cg(rhs)
    assert_allclose(gp.dot(u), rhs, rtol=1e-5, atol=1e-7)


def test_cg_without_the_prior_still_uses_the_band_parallel_pool_path():
    """The in-worker fast path is one Ray dispatch per solve; do not lose it."""
    rng = np.random.default_rng(8)
    nband, nx, ny = 2, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-1)

    calls = []
    original = hess._pool.hess_cg

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    hess._pool.hess_cg = spy
    u = hess.cg(rng.standard_normal((nband, nx, ny)))
    assert calls == [1], "the band-parallel pool path was bypassed"
    assert u.shape == (nband, nx, ny)


def test_cg_with_the_prior_bypasses_the_band_parallel_pool_path():
    """Band-parallel CG cannot solve a band-coupled operator."""
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(9)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-1, freq_prec=kinv)

    def boom(*args, **kwargs):
        raise AssertionError("hess_cg must not be called when bands are coupled")

    gp._pool.hess_cg = boom
    gp.cg(rng.standard_normal((nband, nx, ny)))
