"""Tests for ``pfb restore`` on the imager ``.dt`` DataTree (issue #303).

Unit tests exercise the pure array functions in ``utils/restoration.py``.
Driver tests build a synthetic ``.dt`` in-process (``_write_restore_dt``, no
MS or imager needed), in the style of ``tests/test_fits_tree.py``. One
integration test runs ``imager -> deconv -> restore`` against the
``sky_truth`` fixture.
"""

import numpy as np
import pytest


def test_restore_products_algebra():
    """The three products differ only in where the beam is applied.

    With a spatially constant beam the closed forms collapse to scalar
    relations, so the test pins the definitions rather than the numerics of
    the convolution.
    """
    from pfb_imaging.utils.restoration import restore_products

    nx = ny = 64
    rng = np.random.default_rng(0)
    model = np.zeros((1, ny, nx))
    model[0, ny // 2, nx // 2] = 2.0
    residual = rng.standard_normal((1, ny, nx)) * 0.01
    beam = np.full((1, ny, nx), 0.5)
    gpar = np.array([[4.0, 4.0, 0.0]])

    out = restore_products(model, residual, beam, gpar, products=("a", "i", "k"), pb_min=0.1)

    mconv = out["k"] - residual  # m (x) G, recovered from the mixed product
    np.testing.assert_allclose(out["i"], mconv + residual / 0.5, rtol=0, atol=1e-10)
    np.testing.assert_allclose(out["a"], 0.5 * mconv + residual, rtol=0, atol=1e-10)
    # the relation the mosaic case depends on
    np.testing.assert_allclose(out["a"] / 0.5, out["i"], rtol=0, atol=1e-10)


def test_restore_products_pb_min_zeroes_low_beam():
    """The intrinsic image is zeroed below pb_min, following utils/spi.py:31."""
    from pfb_imaging.utils.restoration import restore_products

    nx = ny = 32
    model = np.zeros((1, ny, nx))
    model[0, ny // 2, nx // 2] = 1.0
    residual = np.full((1, ny, nx), 0.1)
    beam = np.ones((1, ny, nx))
    beam[0, :, : nx // 2] = 0.05  # below the floor
    gpar = np.array([[3.0, 3.0, 0.0]])

    out = restore_products(model, residual, beam, gpar, products=("i",), pb_min=0.1)

    assert np.all(out["i"][0, :, : nx // 2] == 0.0)
    assert np.any(out["i"][0, :, nx // 2 :] != 0.0)


@pytest.mark.parametrize("pa_deg", [20.0, 70.0])
def test_restore_products_preserves_position_angle(pa_deg):
    """Restoring a delta with an elliptical beam reproduces that beam on the
    (y, x) raster. Dropping convolve2gaussres(yx_order=True) mirrors the PA
    and this fails (wiki D19/D20).
    """
    from pfb_imaging.utils.misc import fitcleanbeam
    from pfb_imaging.utils.restoration import restore_products

    nx = ny = 128
    gpar = np.array([[10.0, 4.0, np.deg2rad(pa_deg)]])
    model = np.zeros((1, ny, nx))
    model[0, ny // 2, nx // 2] = 1.0
    residual = np.zeros((1, ny, nx))
    beam = np.ones((1, ny, nx))

    out = restore_products(model, residual, beam, gpar, products=("k",))

    emaj, emin, pa = fitcleanbeam(out["k"] / out["k"].max(), yx_order=True)[0]
    np.testing.assert_allclose(emaj, 10.0, rtol=0.05)
    np.testing.assert_allclose(emin, 4.0, rtol=0.05)
    np.testing.assert_allclose(pa, np.deg2rad(pa_deg), atol=np.deg2rad(3.0))


def test_clean_beam_is_weighted_and_not_the_mean_of_per_band_fits():
    """G_mfs is the fit to the wsum-weighted average PSF, so a band carrying
    almost no weight barely moves it. The mean of the per-band fitted
    Gaussians -- the legacy MFS beam -- is weight-blind and lands far away.
    """
    from pfb_imaging.utils.misc import gaussian2d
    from pfb_imaging.utils.restoration import clean_beam

    n = 128
    coord = -(n // 2) + np.arange(n)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    shapes = [(6.0, 6.0, 0.0), (18.0, 18.0, 0.0)]
    wsums = np.array([[100.0], [1.0]])
    # stored PSFs are un-normalised (shape * wsum), as core/imager.py writes them
    psfs = np.stack([w[0] * gaussian2d(xx, yy, g, normalise=False).T[None] for g, w in zip(shapes, wsums)])

    per_band = np.stack([clean_beam(p, w) for p, w in zip(psfs, wsums)])
    mfs = clean_beam(psfs.sum(axis=0), wsums.sum(axis=0))

    np.testing.assert_allclose(per_band[0, 0, 0], 6.0, rtol=0.05)
    np.testing.assert_allclose(per_band[1, 0, 0], 18.0, rtol=0.05)
    # the heavily-weighted narrow band dominates the summed PSF
    assert mfs[0, 0] < 8.0
    # ... and that is nowhere near the weight-blind mean of the two fits (12.0)
    assert abs(mfs[0, 0] - per_band[:, 0, 0].mean()) > 3.0


def test_clean_beam_zero_wsum_is_nan_not_a_crash():
    """A fully flagged band has wsum 0; the fit must degrade to NaN so the
    caller's nanmax/nanmean skip it rather than dividing by zero.
    """
    from pfb_imaging.utils.restoration import clean_beam

    psf = np.zeros((1, 32, 32))
    out = clean_beam(psf, np.array([0.0]))
    assert np.isnan(out).all()


def test_lowest_resolution_takes_max_axes_and_mean_pa():
    from pfb_imaging.utils.restoration import lowest_resolution

    gp = np.array([[[6.0, 3.0, 0.2]], [[4.0, 5.0, 0.6]]])
    np.testing.assert_allclose(lowest_resolution(gp), np.array([[6.0, 5.0, 0.4]]))
