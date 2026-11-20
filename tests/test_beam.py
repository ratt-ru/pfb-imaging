import numpy as np
import pytest

pmp = pytest.mark.parametrize


def test_beam_grid_cell_bounds_the_placement_error():
    """The cell is sqrt(tol) * lambda_min / D and the axis is centred on a node (#365)."""
    from scipy.constants import c as lightspeed

    from pfb_imaging.utils.beam import beam_grid

    lm = beam_grid(1.7e9, 15.0, 1.2, tol=1e-4)
    cell = np.rad2deg(1e-2 * lightspeed / 1.7e9 / 15.0)
    np.testing.assert_allclose(np.diff(lm), cell, rtol=1e-12)
    np.testing.assert_allclose(lm, -lm[::-1], atol=1e-15)
    assert lm.size % 2 == 1 and lm[lm.size // 2] == 0.0
    # reaches the radius, with at most one extra cell beyond the one it needs
    assert 1.2 + cell <= lm[-1] < 1.2 + 2 * cell


class _CubeOnly:
    """Stands in for a BeamWizard: field_beam_axis reads only the cube's axes."""

    def __init__(self, half_deg):
        import xarray as xr

        ax = np.arange(-half_deg, half_deg, 0.0625)
        self.bds = xr.Dataset(coords={"X": ax, "Y": ax})


def test_field_beam_axis_reaches_the_requested_distance_unless_the_cube_caps_it():
    """The axis depends on the reach, not on the image; the cube's inscribed disk caps it (#354)."""
    from pfb_imaging.core.imager import field_beam_axis

    wizards = {"MM": _CubeOnly(2.0), "MPM": _CubeOnly(2.0)}
    lm, axis, cover = field_beam_axis(wizards, 1.2, 1.7e9)
    assert axis == 1.2 and lm[-1] >= axis
    # the cube's inscribed radius less two cube pixels for its spline edge
    assert cover == pytest.approx(1.9375 - 2 * 0.0625)
    # the cross group's 15 m dish sets the cell, not MM's 13.5 m
    cell = np.rad2deg(1e-2 * 299792458.0 / 1.7e9 / 15.0)
    np.testing.assert_allclose(lm[1] - lm[0], cell)

    # asked for more than the cube covers: capped one diagonal cell inside the limit
    lm, axis, cover = field_beam_axis(wizards, 3.0, 1.7e9)
    np.testing.assert_allclose(axis, cover - np.sqrt(2) * cell)
    assert lm[-1] >= axis


def test_pointing_reach_adds_the_farthest_field_to_the_radius():
    """A mosaic field off the tangent point needs its beam out to R plus its own offset."""
    from pfb_imaging.core.imager import pointing_reach
    from pfb_imaging.utils.misc import radec_to_lm

    tangent = np.array([1.0, -0.5])
    near = tangent + np.array([0.0, np.deg2rad(0.2)])
    far = tangent + np.array([np.deg2rad(0.4), np.deg2rad(-0.3)])
    r = np.deg2rad(1.5)
    lf, mf = radec_to_lm(far, tangent)
    np.testing.assert_allclose(pointing_reach([near, far], tangent, r), np.rad2deg(np.hypot(lf, mf) + r), rtol=1e-12)
    np.testing.assert_allclose(pointing_reach([tangent], tangent, r), 1.5, rtol=1e-12)


def test_check_image_extent_refuses_an_image_past_the_max_field_of_view():
    """The image's farthest corner from the tangent point, --target offsets included, is bounded by R."""
    from pfb_imaging.core.imager import check_image_extent

    cell = np.deg2rad(1.0 / 3600)
    nx = ny = 2048  # 0.40 deg half-diagonal
    half_diag = np.rad2deg(0.5 * np.hypot(nx, ny) * cell)
    r = np.deg2rad(1.0)
    np.testing.assert_allclose(check_image_extent(nx, ny, cell, {(0.0, 0.0)}, r), half_diag)
    # an exactly-fitting image is accepted
    check_image_extent(nx, ny, cell, {(0.0, 0.0)}, np.deg2rad(half_diag))
    with pytest.raises(ValueError, match="beyond --max-field-of-view 1.000 deg"):
        check_image_extent(nx, ny, cell, {(0.0, 0.0), (np.deg2rad(0.7), 0.0)}, r)


def test_check_beam_reach_measures_from_each_pointing_and_refuses_past_the_axis():
    """The image corner farthest from any pointing, --target offsets included, must lie on the axis."""
    from pfb_imaging.core.imager import check_beam_reach, field_beam_axis
    from pfb_imaging.utils.misc import radec_to_lm

    cell = np.deg2rad(1.0 / 3600)
    nx = ny = 1024
    half_diag = np.rad2deg(0.5 * np.hypot(nx, ny) * cell)
    tangent = np.array([1.0, -0.5])
    far = tangent + np.array([np.deg2rad(0.4), np.deg2rad(-0.3)])
    lf, mf = radec_to_lm(far, tangent)
    dist = np.rad2deg(np.hypot(lf, mf))

    # an image offset towards the field needs only its half-diagonal, away from it twice the offset more
    toward, away = (float(lf), float(mf)), (-float(lf), -float(mf))
    check_beam_reach([far], tangent, {toward}, nx, ny, cell, half_diag * (1 + 1e-9), np.inf)
    check_beam_reach([far], tangent, {away}, nx, ny, cell, (2 * dist + half_diag) * (1 + 1e-9), np.inf)
    with pytest.raises(ValueError, match="beyond the 9.00 deg the beam cube covers"):
        check_beam_reach([far], tangent, {away}, nx, ny, cell, 2 * dist + half_diag - 1e-6, 9.0)

    # inside the cube's inscribed disk but within two cube pixels of its edge: refused
    wizards = {"MM": _CubeOnly(2.0), "MPM": _CubeOnly(2.0)}
    _, axis, cover = field_beam_axis(wizards, 3.0, 1.7e9)
    n_edge = int(2 * 1.85 / np.rad2deg(cell) / np.sqrt(2))
    with pytest.raises(ValueError, match="two-pixel margin"):
        check_beam_reach([tangent], tangent, {(0.0, 0.0)}, n_edge, n_edge, cell, axis, cover)
