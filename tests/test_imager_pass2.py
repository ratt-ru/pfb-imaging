"""Pass-2 per-partition gridding (casacore-free, synthetic data)."""

import numpy as np
import xarray as xr

from pfb_imaging.operators.gridder import grid_partition, residual_from_partitions
from pfb_imaging.utils.weighting import _compute_counts


def _synth_partition(nrow=200, seed=0):
    """A synthetic single-correlation partition with spread-out uvw."""
    rng = np.random.default_rng(seed)
    uvw = rng.standard_normal((nrow, 3)) * 100.0
    freq = np.array([1.0e9])
    vis = rng.standard_normal((1, nrow, 1)) + 1j * rng.standard_normal((1, nrow, 1))
    wgt = np.abs(rng.standard_normal((1, nrow, 1))) + 0.1
    mask = np.ones((nrow, 1), dtype=np.uint8)
    beam = np.ones((1, 3, 3))
    return xr.Dataset(
        {
            "VIS": (("corr", "row", "chan"), vis),
            "WEIGHT": (("corr", "row", "chan"), wgt),
            "MASK": (("row", "chan"), mask),
            "UVW": (("row", "three"), uvw),
            "FREQ": (("chan",), freq),
            "BEAM": (("corr", "l_beam", "m_beam"), beam),
        },
        coords={"corr": ["I"], "l_beam": np.array([-1.0, 0.0, 1.0]), "m_beam": np.array([-1.0, 0.0, 1.0])},
    )


def test_grid_partition_shapes_and_wsum():
    part = _synth_partition()
    out = grid_partition(part, None, nx=16, ny=16, nx_psf=32, ny_psf=32, cell_rad=1.0e-6, robustness=None)
    assert out["DIRTY"].shape == (1, 16, 16)
    assert out["PSF"].shape == (1, 32, 32)
    assert out["PSFHAT"].shape == (1, 32, 32 // 2 + 1)
    assert out["BEAM"].shape == (1, 16, 16)
    assert out["WSUM"].shape == (1,)
    expected = (part.WEIGHT.values[0] * part.MASK.values).sum()
    np.testing.assert_allclose(out["WSUM"][0], expected, rtol=1e-6)
    assert np.isfinite(out["DIRTY"]).all()


def test_grid_partition_row_additivity():
    """Gridding is linear over rows: cat(p0, p1) dirty == dirty(p0) + dirty(p1).

    This is exactly the property that makes the sum-over-partitions Hessian
    correct for partitions sharing a phase centre and beam.
    """
    p0 = _synth_partition(nrow=120, seed=0)
    p1 = _synth_partition(nrow=80, seed=1)
    cat = xr.concat([p0[["VIS", "WEIGHT", "MASK", "UVW"]], p1[["VIS", "WEIGHT", "MASK", "UVW"]]], dim="row")
    cat = cat.assign(FREQ=p0.FREQ, BEAM=p0.BEAM)

    kw = dict(nx=16, ny=16, nx_psf=32, ny_psf=32, cell_rad=1.0e-6, robustness=None)
    o_cat = grid_partition(cat, None, **kw)
    o0 = grid_partition(p0, None, **kw)
    o1 = grid_partition(p1, None, **kw)

    np.testing.assert_allclose(o_cat["DIRTY"], o0["DIRTY"] + o1["DIRTY"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(o_cat["PSF"], o0["PSF"] + o1["PSF"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(o_cat["WSUM"], o0["WSUM"] + o1["WSUM"], rtol=1e-6)


def test_grid_partition_robust_reweights():
    part = _synth_partition()
    nx_pad = ny_pad = 32
    # counts must match the convention used inside grid_partition:
    # wgridder_conventions(0,0) -> flip_u=False (usign=-1), flip_v=True (vsign=+1)
    counts = _compute_counts(
        part.UVW.values,
        part.FREQ.values,
        part.MASK.values,
        part.WEIGHT.values,
        nx_pad,
        ny_pad,
        1.0e-6,
        1.0e-6,
        part.WEIGHT.values.dtype,
        ngrid=1,
        usign=-1.0,
        vsign=1.0,
    )
    out = grid_partition(
        part, counts, nx=16, ny=16, nx_psf=32, ny_psf=32, cell_rad=1.0e-6, robustness=-2.0, nx_pad=nx_pad, ny_pad=ny_pad
    )
    assert out["WEIGHT"].shape == part.WEIGHT.shape
    # robust/uniform weighting downweights dense cells -> never exceeds the natural max
    assert out["WEIGHT"].max() <= part.WEIGHT.values.max() + 1e-9
    # natural-weight call leaves weights untouched
    nat = grid_partition(part, None, nx=16, ny=16, nx_psf=32, ny_psf=32, cell_rad=1.0e-6, robustness=None)
    np.testing.assert_allclose(nat["WEIGHT"], part.WEIGHT.values)


def _image_beam_partition(nx, ny, nrow=200, seed=0, beam_val=1.0, l0=0.0, m0=0.0):
    """A partition with the beam already on the image grid (as stored in pass 2)."""
    rng = np.random.default_rng(seed)
    uvw = rng.standard_normal((nrow, 3)) * 100.0
    freq = np.array([1.0e9])
    wgt = np.abs(rng.standard_normal((1, nrow, 1))) + 0.1
    mask = np.ones((nrow, 1), dtype=np.uint8)
    beam = np.full((1, nx, ny), float(beam_val))
    return xr.Dataset(
        {
            "WEIGHT": (("corr", "row", "chan"), wgt),
            "MASK": (("row", "chan"), mask),
            "UVW": (("row", "three"), uvw),
            "FREQ": (("chan",), freq),
            "BEAM": (("corr", "x", "y"), beam),
        },
        coords={"corr": ["I"]},
        attrs={"l0": l0, "m0": m0},
    )


def test_residual_zero_model_returns_dirty():
    nx = ny = 16
    part = _image_beam_partition(nx, ny, seed=0)
    dirty = np.random.default_rng(5).standard_normal((1, nx, ny))
    model = np.zeros((1, nx, ny))
    res = residual_from_partitions(dirty, [part], model, cell_rad=1.0e-6)
    np.testing.assert_allclose(res, dirty, atol=1e-12)


def test_residual_partition_additivity():
    """convim summed over [p0, p1] equals convim(p0) + convim(p1)."""
    nx = ny = 16
    p0 = _image_beam_partition(nx, ny, nrow=120, seed=0)
    p1 = _image_beam_partition(nx, ny, nrow=80, seed=1)
    rng = np.random.default_rng(7)
    dirty = rng.standard_normal((1, nx, ny))
    model = rng.standard_normal((1, nx, ny))
    c01 = dirty - residual_from_partitions(dirty, [p0, p1], model, 1.0e-6)
    c0 = dirty - residual_from_partitions(dirty, [p0], model, 1.0e-6)
    c1 = dirty - residual_from_partitions(dirty, [p1], model, 1.0e-6)
    np.testing.assert_allclose(c01, c0 + c1, rtol=1e-5, atol=1e-8)


def test_residual_beam_applied_once():
    """Doubling the beam doubles the model term (beam applied once on degrid side)."""
    nx = ny = 16
    p1 = _image_beam_partition(nx, ny, seed=0, beam_val=1.0)
    p2 = _image_beam_partition(nx, ny, seed=0, beam_val=2.0)
    rng = np.random.default_rng(9)
    dirty = np.zeros((1, nx, ny))
    model = rng.standard_normal((1, nx, ny))
    c1 = dirty - residual_from_partitions(dirty, [p1], model, 1.0e-6)
    c2 = dirty - residual_from_partitions(dirty, [p2], model, 1.0e-6)
    np.testing.assert_allclose(c2, 2.0 * c1, rtol=1e-5, atol=1e-8)
