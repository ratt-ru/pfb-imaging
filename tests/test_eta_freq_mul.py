"""eta_freq_mul: the fused frequency congruence against its four-pass numpy form.

The kernel is a candidate replacement for the driver-side coupling term in
``HessTreeRay.dot`` (wiki D30) and is not wired in yet, so these tests hold it
to the numpy expression it would replace.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.operators.gauss import eta_freq_mul

pmp = pytest.mark.parametrize


def _numpy_reference(out, dcinv, s, x):
    """Exactly what HessTreeRay.dot does today."""
    nband = dcinv.shape[0]
    buf = x * s
    buf2 = np.empty_like(x)
    np.matmul(dcinv, buf.reshape(nband, -1), out=buf2.reshape(nband, -1))
    return out + buf2 * s


def _setup(nband, ny, nx, seed=0, profile=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((nband, ny, nx))
    out = rng.standard_normal((nband, ny, nx))
    # a symmetric coupling with the shape freq_precision produces
    q = np.linalg.qr(rng.standard_normal((nband, nband)))[0]
    dcinv = q @ np.diag(rng.uniform(0.1, 1.0, nband)) @ q.T - np.eye(nband)
    s = rng.uniform(0.5, 2.0, (nband, ny, nx)) if profile else rng.uniform(0.5, 2.0, (nband, 1, 1))
    return out, dcinv, s, x


@pmp("nband", [1, 2, 3, 8])
@pmp("nchunk", [1, 4])
def test_matches_numpy_for_a_uniform_eta(nband, nchunk):
    out, dcinv, s, x = _setup(nband, 12, 10)
    want = _numpy_reference(out, dcinv, s, x)
    got = eta_freq_mul(out.copy(), dcinv, s, x, nchunk=nchunk)
    assert_allclose(got, want, rtol=1e-12, atol=1e-14)


@pmp("nband", [1, 3, 8])
@pmp("nchunk", [1, 4])
def test_matches_numpy_for_a_spatially_varying_eta(nband, nchunk):
    out, dcinv, s, x = _setup(nband, 12, 10, profile=True)
    want = _numpy_reference(out, dcinv, s, x)
    got = eta_freq_mul(out.copy(), dcinv, s, x, nchunk=nchunk)
    assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_per_band_s_accepts_either_shape():
    """(nband,) and (nband, 1, 1) are the same thing to the caller."""
    out, dcinv, s, x = _setup(3, 12, 10)
    flat = eta_freq_mul(out.copy(), dcinv, s.reshape(3), x)
    cube = eta_freq_mul(out.copy(), dcinv, s, x)
    assert_allclose(flat, cube, rtol=0, atol=0)


def test_accumulates_in_place_and_returns_the_same_array():
    out, dcinv, s, x = _setup(3, 12, 10)
    before = out.copy()
    got = eta_freq_mul(out, dcinv, s, x)
    assert got is out
    assert np.abs(out - before).max() > 0.0


def test_result_is_independent_of_the_chunk_count():
    """Chunking splits pixels, never bands, so the arithmetic per pixel is identical."""
    out, dcinv, s, x = _setup(4, 31, 29, profile=True)  # sizes coprime with the tile
    ref = eta_freq_mul(out.copy(), dcinv, s, x, nchunk=1)
    for nchunk in (2, 3, 7, 64):
        assert_allclose(eta_freq_mul(out.copy(), dcinv, s, x, nchunk=nchunk), ref, rtol=0, atol=0)


def test_spans_more_than_one_tile():
    """The tile loop must cover every pixel, including a ragged final tile."""
    from pfb_imaging.operators.gauss import _ETA_FREQ_TILE

    npix = 3 * _ETA_FREQ_TILE + 17
    out, dcinv, s, x = _setup(2, 1, npix)
    want = _numpy_reference(out, dcinv, s, x)
    assert_allclose(eta_freq_mul(out.copy(), dcinv, s, x, nchunk=3), want, rtol=1e-12, atol=1e-14)


def test_zero_coupling_is_a_no_op():
    out, dcinv, s, x = _setup(3, 12, 10)
    got = eta_freq_mul(out.copy(), np.zeros_like(dcinv), s, x)
    assert_allclose(got, out, rtol=0, atol=0)


def test_bad_shapes_are_rejected():
    out, dcinv, s, x = _setup(3, 12, 10)
    with pytest.raises(ValueError, match="square"):
        eta_freq_mul(out.copy(), dcinv[:, :2], s, x)
    with pytest.raises(ValueError, match="shape"):
        eta_freq_mul(out[:2].copy(), dcinv, s, x)
    with pytest.raises(ValueError, match="s must have"):
        eta_freq_mul(out.copy(), dcinv, np.ones((2, 2)), x)


def test_non_contiguous_arrays_are_rejected():
    """reshape would silently copy, so the accumulation would be thrown away."""
    out, dcinv, s, x = _setup(3, 12, 20)
    with pytest.raises(ValueError, match="C-contiguous"):
        eta_freq_mul(out[:, :, ::2].copy(order="F"), dcinv, s, x[:, :, ::2].copy())
    with pytest.raises(ValueError, match="C-contiguous"):
        eta_freq_mul(out.copy(), dcinv, s, np.asfortranarray(x))
