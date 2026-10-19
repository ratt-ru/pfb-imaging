"""utils.inplace: the driver's parallel in-place cube kernels."""

import numpy as np
from numpy.testing import assert_allclose

from pfb_imaging.utils import inplace


def test_kernels_match_numpy_and_accept_read_only_sources():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, 3, 5, 7))
    x.flags.writeable = False  # Ray results arrive read-only
    dst = np.empty_like(y)

    inplace.copy(dst, x)
    assert_allclose(dst, x, rtol=0, atol=0)
    inplace.add(dst, y)
    assert_allclose(dst, x + y, rtol=1e-12, atol=1e-14)
    inplace.scale(dst, -2.0)
    assert_allclose(dst, -2.0 * (x + y), rtol=1e-12, atol=1e-14)
    inplace.axpby(dst, 0.5, x, -3.0, y)
    assert_allclose(dst, 0.5 * x - 3.0 * y, rtol=1e-12, atol=1e-14)


def test_axpby_may_alias_its_output():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((2, 4, 6))
    want = 2.0 * x + y
    inplace.axpby(x, 2.0, x, 1.0, y)
    assert_allclose(x, want, rtol=1e-12, atol=1e-14)
