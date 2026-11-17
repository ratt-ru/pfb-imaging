import numpy as np
import pytest
from katbeam import JimBeam

from pfb_imaging.utils.beam import eval_beam, interp_beam

pmp = pytest.mark.parametrize


@pmp("beam_model", ("kbl", "kbuhf", None))
@pmp("nx", (24, 128))
@pmp("ny", (64, 92))
def test_beam(beam_model, nx, ny):
    freq = 1e9
    cell_deg = 1e-2
    beam, l_coord, m_coord = interp_beam(freq, nx, ny, cell_deg, beam_model)
    ll, mm = np.meshgrid(l_coord, m_coord, indexing="ij")
    bvals = eval_beam(beam, l_coord, m_coord, ll, mm)

    if beam_model is None:
        assert (bvals == 1).all()
    elif beam_model == "kbl":
        jb = JimBeam("MKAT-AA-L-JIM-2020").I
        assert np.allclose(bvals, jb(ll, mm, freq / 1e6), atol=1e-10)
    elif beam_model == "kbuhf":
        jb = JimBeam("MKAT-AA-UHF-JIM-2020").I
        assert np.allclose(bvals, jb(ll, mm, freq / 1e6), atol=1e-10)


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
    assert cover == pytest.approx(1.9375)
    # the cross group's 15 m dish sets the cell, not MM's 13.5 m
    np.testing.assert_allclose(lm[1] - lm[0], np.rad2deg(1e-2 * 299792458.0 / 1.7e9 / 15.0))

    # a --target offset of 0.5 deg pushes the far corner out by as much
    off = (np.deg2rad(0.5), 0.0)
    _, shifted, _ = field_beam_axis(wizards, [radec], radec, {(0.0, 0.0), off}, nx, ny, cell, 1.7e9)
    np.testing.assert_allclose(shifted, radius + 0.5, rtol=1e-6)

    with pytest.raises(ValueError, match="beyond the 1.94 deg the beam cube covers"):
        field_beam_axis(wizards, [radec], radec, {(0.0, 0.0)}, 6 * nx, 6 * ny, cell, 1.7e9)
