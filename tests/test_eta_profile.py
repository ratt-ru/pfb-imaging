"""Spatially varying Tikhonov profiles for the PSF-convolution preconditioner.

``eta_profile`` (issue #287, E1) raises ``eta`` where the PSF approximation to
the Hessian is worst. Because ``lambda_max(M^-1 H)`` is monotone decreasing in
``M`` in the PSD order, that lowers the smallest diverging ``gamma``. The
invariants that make it a *strict* win are the ones tested here:

* the floor is ``eta`` itself, so ``lambda_min(M)`` -- hence the CG iteration
  count -- cannot degrade;
* the ceiling is ``eta*cap``, so the beam skirt cannot blow ``lambda_max(M)`` up;
* ``M`` stays symmetric positive definite, so CG still applies;
* ``mode=None`` is bit-for-bit the old uniform behaviour.
"""

import numpy as np
import pytest
from ducc0.fft import r2c

from pfb_imaging.operators.hessian import ETA_MODES, HessianTree, eta_profile

NX = NY = 32
NX_PSF = NY_PSF = 64
ifftshift = np.fft.ifftshift


def _partition(beam, wsum=3.0):
    psf = np.zeros((1, NY_PSF, NX_PSF))
    psf[0, NY_PSF // 2, NX_PSF // 2] = wsum
    psfhat = r2c(ifftshift(psf, axes=(1, 2)), axes=(1, 2), forward=True, inorm=0)
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([float(wsum)])}


def _tapered_beam(peak_at, width=0.35):
    """A gaussian beam peaking at pixel ``peak_at``, as a single-pointing mosaic member."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    r2 = ((xx - peak_at[1]) / (width * NX)) ** 2 + ((yy - peak_at[0]) / (width * NY)) ** 2
    return np.exp(-r2)[None]


def test_none_is_the_uniform_scalar():
    parts = [_partition(_tapered_beam((NY // 2, NX // 2)))]
    assert eta_profile(parts, 1e-3, None, NY, NX) == 1e-3


@pytest.mark.parametrize("mode", ETA_MODES)
def test_floor_is_eta_and_ceiling_is_eta_times_cap(mode):
    """Both bounds matter: the floor protects CG, the ceiling protects lambda_max(M)."""
    eta, cap = 1e-3, 50.0
    # two offset pointings so the mosaic beam is not radially symmetric
    parts = [_partition(_tapered_beam((NY // 2, NX // 3))), _partition(_tapered_beam((NY // 2, 2 * NX // 3)))]
    e = eta_profile(parts, eta, mode, NY, NX, cap=cap)
    assert e.shape == (1, NY, NX)
    assert e.min() >= eta * (1.0 - 1e-12), f"{mode} dips below eta -- lambda_min(M) would degrade"
    assert e.max() <= eta * cap * (1.0 + 1e-12), f"{mode} exceeds eta*cap"
    # and it is not trivially constant: the point is the variation
    assert e.max() > 2.0 * e.min()


def test_invbeam_tracks_the_wsum_weighted_mosaic_of_b_squared():
    """The operator carries B on both sides (D23), so the profile follows B^2, not B."""
    eta, cap = 1.0, 1e6  # cap out of the way; eta=1 so the profile IS the multiplier
    b1, b2 = _tapered_beam((NY // 2, NX // 3)), _tapered_beam((NY // 2, 2 * NX // 3))
    w1, w2 = 1.0, 3.0
    parts = [_partition(b1, wsum=w1), _partition(b2, wsum=w2)]
    b2eff = (w1 * b1**2 + w2 * b2**2) / (w1 + w2)
    b2eff /= b2eff.max()
    want = np.clip(1.0 / np.maximum(np.sqrt(b2eff), 1.0 / cap), 1.0, cap)
    np.testing.assert_allclose(eta_profile(parts, eta, "invbeam", NY, NX, cap=cap), want, rtol=1e-12)
    # invbeam2 is the same profile with twice the log-slope
    got2 = eta_profile(parts, eta, "invbeam2", NY, NX, cap=cap)
    np.testing.assert_allclose(got2, want**2, rtol=1e-12)


def test_radial_reaches_the_cap_at_one_field_half_width():
    eta, cap = 2.0, 11.0
    parts = [_partition(np.ones((1, NY, NX)))]
    e = eta_profile(parts, eta, "radial", NY, NX, cap=cap) / eta
    assert e[0, NY // 2, NX // 2] == pytest.approx(1.0)
    # one half-width along x from the centre
    assert e[0, NY // 2, -1] == pytest.approx(1.0 + (cap - 1.0) * ((NX / 2 - 1) / (NX / 2)) ** 2, rel=1e-12)
    assert e.max() == pytest.approx(cap)  # the corners saturate


def test_profiled_hessian_is_symmetric_and_positive_definite():
    """A profile must not break the two properties CG relies on."""
    parts = [_partition(_tapered_beam((NY // 2, NX // 3))), _partition(_tapered_beam((NY // 3, 2 * NX // 3)))]
    hess = HessianTree(parts, NX, NY, NX_PSF, NY_PSF, eta=1e-2, eta_mode="radial-invbeam", eta_cap=100.0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, NY, NX))
    y = rng.standard_normal((1, NY, NX))
    assert np.vdot(x, hess.dot(y)) == pytest.approx(np.vdot(hess.dot(x), y), rel=1e-10)
    assert float(np.vdot(x, hess.dot(x)).real) > 0.0


def test_profile_only_ever_increases_the_denominator():
    """The PSD-monotonicity claim, checked directly: v'Mv can only grow."""
    parts = [_partition(_tapered_beam((NY // 2, NX // 3)))]
    kw = dict(eta=1e-3, nthreads=1)
    uniform = HessianTree(parts, NX, NY, NX_PSF, NY_PSF, **kw)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, NY, NX))
    base = float(np.vdot(x, uniform.dot(x)).real)
    for mode in ETA_MODES:
        prof = HessianTree(parts, NX, NY, NX_PSF, NY_PSF, eta_mode=mode, eta_cap=100.0, **kw)
        assert float(np.vdot(x, prof.dot(x)).real) >= base, f"{mode} shrank v'Mv"


def test_profile_survives_the_ray_facade():
    """Each worker must build the profile from ITS OWN band's beams.

    The profile is deliberately not shipped through Ray, so this is the check
    that ``eta_mode``/``eta_cap`` actually reach the per-band ``HessianTree``
    and that per-band beams give per-band profiles.
    """
    # deferred: HessTreeRay pulls in ray for nband > 1
    from pfb_imaging.operators.hessian import HessTreeRay

    rng = np.random.default_rng(3)
    # band 0's mosaic peaks left of centre, band 1's right of it
    parts = [[_partition(_tapered_beam((NY // 2, NX // 3)))], [_partition(_tapered_beam((NY // 2, 2 * NX // 3)))]]
    kw = dict(etas=1e-2, eta_mode="invbeam", eta_cap=25.0)
    hess = HessTreeRay(parts, NX, NY, NX_PSF, NY_PSF, wsums=3.0, **kw)
    x = rng.standard_normal((2, NY, NX))
    got = hess.dot(x)
    for b in range(2):
        local = HessianTree(parts[b], NX, NY, NX_PSF, NY_PSF, eta=1e-2, wsum=3.0, eta_mode="invbeam", eta_cap=25.0)
        np.testing.assert_allclose(got[b], local.dot(x[b])[0], rtol=1e-12)
    # and the two bands really do differ (else the assertion above is vacuous)
    assert not np.allclose(got[0], hess.dot(x[::-1])[1])


def test_radial_is_identical_across_bands_and_beam_modes_are_not():
    """Band-to-band uniformity of ``e`` matters even though it cannot bias the fixed point.

    ``e`` is preconditioner-only, so the converged model is unchanged either way.
    But bands couple *only* through the L21 prox (D3), so a band-dependent ``e``
    damps some bands' updates more than others and the joint sparsity decision is
    taken on a model whose spectral shape is still converging -- a bias at any
    finite iteration count. ``radial`` is pure image geometry and so is identical
    across bands by construction; the beam-driven modes are frequency-dependent
    and are not. Pinning both halves so neither drifts silently.
    """
    b_lo = [_partition(_tapered_beam((NY // 2, NX // 2), width=0.40))]  # wider beam = lower band
    b_hi = [_partition(_tapered_beam((NY // 2, NX // 2), width=0.30))]
    for mode, want_same in (("radial", True), ("invbeam", False), ("invbeam2", False), ("radial-invbeam", False)):
        lo = eta_profile(b_lo, 1e-3, mode, NY, NX, cap=100.0)
        hi = eta_profile(b_hi, 1e-3, mode, NY, NX, cap=100.0)
        same = np.allclose(lo, hi)
        assert same is want_same, f"{mode}: band-identical={same}, expected {want_same}"


def test_bad_mode_and_cap_are_rejected():
    parts = [_partition(np.ones((1, NY, NX)))]
    with pytest.raises(ValueError, match="unknown eta_mode"):
        eta_profile(parts, 1e-3, "beam", NY, NX)
    with pytest.raises(ValueError, match="eta_cap"):
        eta_profile(parts, 1e-3, "radial", NY, NX, cap=0.5)
