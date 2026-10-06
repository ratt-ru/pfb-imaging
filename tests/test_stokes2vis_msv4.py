"""Unit tests for pure helpers in utils/stokes2vis_msv4."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.utils.stokes2vis_msv4 import real_beam_maps


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
