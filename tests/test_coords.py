"""Casacore-free coordinate helpers used by the imager mosaic path."""

import numpy as np
from numpy.testing import assert_allclose

from pfb_imaging.utils.misc import (
    parse_sky_coords,
    radec_barycentre,
    radec_to_lm,
    to_mjd_time,
    to_unix_time,
)


def test_mjd_unix_roundtrip():
    t_mjd = 5.1e9  # MJD seconds, present-day epoch
    assert to_mjd_time(to_unix_time(t_mjd)) == t_mjd


def test_radec_to_lm_matches_africanus():
    # africanus is the reference implementation (pulls casacore -> test-only import)
    from africanus.coordinates import radec_to_lm as af_radec_to_lm

    rng = np.random.default_rng(42)
    radec0 = np.array([1.2, -0.6])
    for _ in range(5):
        radec = radec0 + rng.uniform(-0.05, 0.05, 2)
        expected = af_radec_to_lm(radec[None, :], radec0)[0]
        assert_allclose(radec_to_lm(radec, radec0), expected, atol=1e-12)


def test_radec_to_lm_zero_at_tangent_point():
    radec0 = np.array([0.3, 0.4])
    assert_allclose(radec_to_lm(radec0, radec0), [0.0, 0.0], atol=1e-15)


def test_barycentre_simple_mean_small_offsets():
    # two fields straddling ra=1.0 at dec=-0.5: RA is exact by symmetry; the
    # spherical mean lies a second-order ~ (dra/2)^2 * tan(dec) poleward of
    # the common declination (great circles bulge poleward of parallels)
    radecs = np.array([[1.0 - 1e-3, -0.5], [1.0 + 1e-3, -0.5]])
    bc = radec_barycentre(radecs)
    assert_allclose(bc[0], 1.0, atol=1e-12)
    assert_allclose(bc[1], -0.5, atol=1e-6)
    assert bc[1] <= -0.5  # poleward, never equatorward


def test_barycentre_wraps_ra():
    # fields at RA 359 deg and 1 deg -> barycentre RA at 0 (not 180), in [0, 2pi)
    radecs = np.deg2rad(np.array([[359.0, -30.0], [1.0, -30.0]]))
    bc = radec_barycentre(radecs)
    assert 0.0 <= bc[0] < 2 * np.pi
    # RA is exact by symmetry (modulo the [0, 2pi) seam)
    assert min(bc[0], 2 * np.pi - bc[0]) < 1e-9

    # equidistance + coplanarity pin the geodesic midpoint without
    # re-deriving the implementation formula
    def unit(rd):
        return np.array([np.cos(rd[1]) * np.cos(rd[0]), np.cos(rd[1]) * np.sin(rd[0]), np.sin(rd[1])])

    b, p1, p2 = unit(bc), unit(radecs[0]), unit(radecs[1])
    assert_allclose(np.dot(b, p1), np.dot(b, p2), atol=1e-12)  # equidistant
    assert_allclose(np.dot(np.cross(p1, p2), b), 0.0, atol=1e-12)  # on their great circle
    assert bc[1] < np.deg2rad(-30.0)  # poleward bulge


def test_parse_sky_coords():
    # 12h == 180 deg
    radec = parse_sky_coords("12:00:00,-30:00:00")
    assert_allclose(radec, [np.pi, np.deg2rad(-30.0)], atol=1e-12)
