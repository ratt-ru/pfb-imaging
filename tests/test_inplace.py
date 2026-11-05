"""utils.inplace: the driver's parallel in-place cube kernels."""

import numpy as np
import pytest
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


def test_rel_diff_matches_numpy():
    rng = np.random.default_rng(3)
    x, y, z = (rng.standard_normal((2, 5, 7)) for _ in range(3))
    want = np.linalg.norm(x - (y - 0.7 * z)) / np.linalg.norm(x)
    assert_allclose(inplace.rel_diff(x, y, z, -0.7), want, rtol=1e-12)


@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_kernels_refuse_a_destination_they_would_not_write(layout):
    """numba's ravel copies a non-C array, so a write through it would vanish; raise instead."""
    base = np.zeros((4, 6))
    dst = np.asfortranarray(base) if layout == "fortran" else np.zeros((8, 6))[::2]
    assert not dst.flags.c_contiguous
    src = np.ones((4, 6))
    for kernel, args in (
        (inplace.copy, (dst, src)),
        (inplace.add, (dst, src)),
        (inplace.scale, (dst, 2.0)),
        (inplace.axpby, (dst, 1.0, src, 1.0, src)),
    ):
        with pytest.raises(ValueError, match="C-contiguous"):
            kernel(*args)
    # a non-C source is fine: it is only read
    out = np.zeros((4, 6))
    inplace.copy(out, np.asfortranarray(src))
    assert np.all(out == 1.0)
