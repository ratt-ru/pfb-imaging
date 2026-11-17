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


def test_field_beam_axis_reaches_the_farthest_pixel_and_refuses_past_the_cube():
    """The axis covers the image corner farthest from any pointing; beyond the cube it refuses."""
    from pfb_imaging.core.imager import field_beam_axis

    cell = np.deg2rad(1.0 / 3600)
    radec = np.array([1.0, -0.5])
    nx = ny = 2048  # 0.57 deg half-width, 0.80 deg half-diagonal
    wizards = {"MM": _CubeOnly(2.0), "MPM": _CubeOnly(2.0)}
    lm, radius, cover = field_beam_axis(wizards, [radec], radec, {(0.0, 0.0)}, nx, ny, cell, 1.7e9)
    np.testing.assert_allclose(radius, np.rad2deg(0.5 * np.hypot(nx, ny) * cell))
    assert lm[-1] >= radius
    # the cube's inscribed radius less two cube pixels for its spline edge
    assert cover == pytest.approx(1.9375 - 2 * 0.0625)
    # the cross group's 15 m dish sets the cell, not MM's 13.5 m
    np.testing.assert_allclose(lm[1] - lm[0], np.rad2deg(1e-2 * 299792458.0 / 1.7e9 / 15.0))

    # a --target offset of 0.5 deg pushes the far corner out by as much
    off = (np.deg2rad(0.5), 0.0)
    _, shifted, _ = field_beam_axis(wizards, [radec], radec, {(0.0, 0.0), off}, nx, ny, cell, 1.7e9)
    np.testing.assert_allclose(shifted, radius + 0.5, rtol=1e-6)

    with pytest.raises(ValueError, match="beyond the 1.81 deg the beam cube covers"):
        field_beam_axis(wizards, [radec], radec, {(0.0, 0.0)}, 6 * nx, 6 * ny, cell, 1.7e9)
    # inside the inscribed disk but within two cube pixels of its edge: refused
    n_edge = int(2 * 1.85 / np.rad2deg(cell) / np.sqrt(2))
    with pytest.raises(ValueError, match="two-pixel margin"):
        field_beam_axis(wizards, [radec], radec, {(0.0, 0.0)}, n_edge, n_edge, cell, 1.7e9)


def test_field_beam_axis_measures_from_each_pointing_in_a_mosaic():
    """A field off the tangent point adds its own offset; the farthest field sets the radius."""
    from pfb_imaging.core.imager import field_beam_axis
    from pfb_imaging.utils.misc import radec_to_lm

    cell = np.deg2rad(1.0 / 3600)
    nx = ny = 1024
    half_diag = np.rad2deg(0.5 * np.hypot(nx, ny) * cell)
    tangent = np.array([1.0, -0.5])
    near = tangent + np.array([0.0, np.deg2rad(0.2)])
    far = tangent + np.array([np.deg2rad(0.4), np.deg2rad(-0.3)])
    wizards = {None: _CubeOnly(2.0)}

    _, radius, _ = field_beam_axis(wizards, [near, far], tangent, {(0.0, 0.0)}, nx, ny, cell, 1.4e9)
    lf, mf = radec_to_lm(far, tangent)
    np.testing.assert_allclose(radius, np.rad2deg(np.hypot(lf, mf)) + half_diag, rtol=1e-12)
    assert np.rad2deg(np.hypot(lf, mf)) > 0.4  # the far field, not the near one

    # an image offset towards a field shrinks its distance; away from it grows it
    lo = (float(lf), float(mf))
    _, toward, _ = field_beam_axis({None: _CubeOnly(2.0)}, [far], tangent, {lo}, nx, ny, cell, 1.4e9)
    np.testing.assert_allclose(toward, half_diag, rtol=1e-9)
    away = (-float(lf), -float(mf))
    _, back, _ = field_beam_axis({None: _CubeOnly(2.0)}, [far], tangent, {away}, nx, ny, cell, 1.4e9)
    np.testing.assert_allclose(back, 2 * np.rad2deg(np.hypot(lf, mf)) + half_diag, rtol=1e-9)
