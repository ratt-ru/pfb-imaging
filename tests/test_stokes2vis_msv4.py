"""Unit tests for pure helpers in utils/stokes2vis_msv4."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pfb_model_spec.utils.degrid import pixel_lm

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.beam import reproject_and_interp_scat_beam
from pfb_imaging.utils.stokes2vis_msv4 import beam_on_image_grid, integration_edges, real_beam_maps


def test_real_maps_pass_through_with_zero_ratio():
    maps = [np.ones((3, 4)), 2.0 * np.ones((3, 4))]
    out, ratio = real_beam_maps(maps)
    assert out.shape == (2, 3, 4)
    assert out.dtype == np.float64
    assert ratio == 0.0
    assert_allclose(out[1], 2.0)


def test_complex_maps_keep_only_the_real_part():
    m = np.full((2, 2), 3.0 + 4.0j, dtype=np.complex64)
    out, ratio = real_beam_maps([m])
    assert out.dtype == np.float64
    assert_allclose(out[0], 3.0)
    assert ratio == pytest.approx(4.0 / 3.0)


def test_ratio_is_the_max_over_correlations():
    a = np.full((2, 2), 1.0 + 0.1j, dtype=np.complex64)
    b = np.full((2, 2), 1.0 + 0.5j, dtype=np.complex64)
    _, ratio = real_beam_maps([a, b])
    assert ratio == pytest.approx(0.5)


def test_zero_real_part_does_not_divide_by_zero():
    m = np.full((2, 2), 1.0j, dtype=np.complex64)
    out, ratio = real_beam_maps([m])
    assert_allclose(out[0], 0.0)
    assert np.isfinite(ratio)


def test_mixed_real_and_complex_maps():
    out, ratio = real_beam_maps([np.ones((2, 2)), np.full((2, 2), 1.0 + 0.25j)])
    assert out.shape == (2, 2, 2)
    assert ratio == pytest.approx(0.25)


def test_integration_edges_use_each_rows_own_interval():
    """A long edge integration widens the bound; one median interval would clip it (#327)."""
    time = np.array([10.0, 20.0, 30.0, np.nan])
    interval = np.array([8.0, 2.0, 2.0, np.nan])  # a long first integration, then a NaN fill row
    assert integration_edges(time, interval) == (6.0, 31.0)
    # no finite interval: fall back to the centroids' extent
    assert integration_edges(time[:3], np.full(3, np.nan)) == (10.0, 30.0)


def _tilted_beam(l_deg, m_deg):
    """Linear in l and m, so bilinear interpolation is exact and any mirror is a large error."""
    return 1.0 + 30.0 * l_deg + 10.0 * m_deg


@pytest.mark.parametrize("nx, ny", [(64, 64), (64, 48)])
def test_single_field_beam_orientation_matches_pixel_lm_and_reproject(nx, ny):
    """The single-field BEAM is East at low x, like the wgridder image and degrid (#355).

    It must agree with the sky (l, m) of each pixel under pfb's wgridder conventions
    (`pixel_lm`, which degrid evaluates its beam on) and with the mosaic reproject branch.
    """
    cell_rad = 1e-5
    l_beam = np.linspace(-0.05, 0.05, 1025)
    m_beam = np.linspace(-0.04, 0.04, 801)
    mm, ll = np.meshgrid(m_beam, l_beam, indexing="ij")
    beam_small = _tilted_beam(ll, mm)[None]
    radec = np.deg2rad([45.0, -30.0])

    beam = beam_on_image_grid(beam_small, l_beam, m_beam, radec, radec, cell_rad, nx, ny, ["I"])

    flip_u, flip_v, _, x0, y0 = wgridder_conventions(0.0, 0.0)
    geom = {"nx": nx, "ny": ny, "cell_rad": cell_rad, "x0": x0, "y0": y0, "flip_u": flip_u, "flip_v": flip_v}
    l_px, m_px = pixel_lm(geom)
    expected = _tilted_beam(np.rad2deg(l_px)[None, :], np.rad2deg(m_px)[:, None])
    assert beam[0, ny // 2, 4] > 1.0  # x = 4 is East of centre
    assert_allclose(beam[0], expected, rtol=0, atol=1e-9)

    reprojected = reproject_and_interp_scat_beam(
        beam_small, l_beam, m_beam, radec, radec, np.rad2deg(cell_rad), nx, ny, "I"
    )
    assert_allclose(beam, reprojected, rtol=0, atol=1e-9)
