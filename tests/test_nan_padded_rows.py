"""xarray-ms NaN-UVW padding must never reach a degrid or the ``.dt``.

xarray-ms lays MSv4 data out on a regular ``(time, baseline)`` grid and fills
absent samples with **NaN UVW** (``core/imager.py`` calls these out via its
``uvw_mask``).  Those rows are fully flagged and zero-weighted, so anything that
respects the mask is unaffected -- but ducc derives its w range from *every* row
it is handed, so a NaN row gives a NaN w extent and either the
``too many w planes`` assertion or outright heap corruption.  Two independent
guards, because the two fixes are independent:

1. Every ducc **degrid** on the deconv path passes ``mask=`` (the grid direction
   always did), so a partition that still contains padding is safe. Needed for
   ``.dt`` stores written before the pass-1 fix.
2. Pass 1 drops the padded rows outright, so new ``.dt`` stores never carry
   them. This also keeps africanus' BDA mapper away from them: it derives a
   bin's central UVW from the bin's first and last row irrespective of flags,
   and one padded row poisons that to NaN and trips
   ``assert fracsizeChanBlockMin >= 1`` (``max(nan, 1)`` is ``nan``).

Regression for the wgridder/BDA failures in issue #287.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.operators.gridder import residual_from_partitions

FREQ = np.array([1.0e9, 1.1e9])
CELL_RAD = 1.0e-5
NPIX = 64


def _partition(uvw, mask, weight, beam):
    """A partition child of the shape ``residual_from_partitions`` consumes."""
    ds = xr.Dataset(
        {
            "UVW": (("row", "uvw"), uvw),
            "WEIGHT": (("corr", "row", "chan"), weight),
            "MASK": (("row", "chan"), mask),
            "FREQ": (("chan",), FREQ),
            "BEAM": (("corr", "y", "x"), beam),
        }
    )
    ds.attrs.update({"l0": 0.0, "m0": 0.0})
    return ds


def test_residual_from_partitions_ignores_nan_padded_rows():
    """A padded partition must degrid as if the padding were not there.

    Note: with the ``mask=`` argument missing from the degrid this does not fail
    politely -- ducc either raises ``too many w planes`` or corrupts the heap and
    takes the interpreter with it. Either way the test does not pass.
    """
    rng = np.random.default_rng(42)
    nrow, nchan = 400, FREQ.size
    uvw = rng.normal(0.0, 400.0, (nrow, 3))
    uvw[:, 2] *= 3.0  # a w spread wide enough for the plane count to matter
    mask = np.ones((nrow, nchan), dtype=np.uint8)
    weight = np.ones((1, nrow, nchan))

    # emulate the padding: NaN UVW, fully masked, zero weight
    pad = np.zeros(nrow, dtype=bool)
    pad[::4] = True
    uvw[pad] = np.nan
    mask[pad] = 0
    weight[:, pad] = 0.0

    beam = np.ones((1, NPIX, NPIX))
    model = np.zeros((1, NPIX, NPIX))
    model[0, NPIX // 2, NPIX // 2] = 1.0
    model[0, NPIX // 3, NPIX // 4] = 0.5
    dirty = np.zeros((1, NPIX, NPIX))

    padded = _partition(uvw, mask, weight, beam)
    keep = np.flatnonzero(~pad)
    trimmed = _partition(uvw[keep], mask[keep], weight[:, keep], beam)

    got, bgot = residual_from_partitions(
        dirty, [padded], model, CELL_RAD, nthreads=1, epsilon=1e-7, do_wgridding=True, bdirty=dirty
    )
    want, bwant = residual_from_partitions(
        dirty, [trimmed], model, CELL_RAD, nthreads=1, epsilon=1e-7, do_wgridding=True, bdirty=dirty
    )

    assert np.isfinite(got).all(), "padded rows leaked NaN into the residual"
    assert np.isfinite(bgot).all(), "padded rows leaked NaN into the gradient residual"
    assert np.abs(want).max() > 0, "degenerate test: the trimmed degrid produced nothing"
    scale = np.abs(want).max()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(bgot, bwant, rtol=0, atol=1e-12 * scale)


@pytest.fixture(scope="module")
def holed_ms(ms_name, tmp_path_factory):
    """A copy of the test MS with a (baseline, time) hole punched in it.

    Removing rows outright is refused by the test MS's storage managers, so the
    hole is made with a TaQL selection copied deeply into a fresh MS. xarray-ms
    then pads the missing slots, which is what we need to exercise.
    """
    from casacore.tables import table

    dst = Path(tmp_path_factory.mktemp("holed")) / "holes.MS"
    if dst.exists():
        shutil.rmtree(dst)
    with table(str(ms_name), ack=False) as tb:
        utime = np.unique(tb.getcol("TIME"))
        tcut = float(utime[utime.size // 2])
        sel = tb.query(f"NOT (ANTENNA1==0 && ANTENNA2==3 && TIME<{tcut!r})")
        assert sel.nrows() < tb.nrows(), "TaQL selection removed nothing"
        sel.copy(str(dst), deep=True)
        sel.close()
    return str(dst)


def test_imager_drops_nan_padded_rows(holed_ms, image_geometry, tmp_path):
    """Pass 1 must not store xarray-ms's NaN-UVW padding in the ``.dt``."""
    from pfb_imaging.core.imager import get_engine, imager

    # guard against a vacuous test: the raw MSv4 view must actually be padded
    src = xr.open_datatree(holed_ms, **get_engine(holed_ms, None))
    npad = sum(int(np.isnan(n.ds.UVW.values).any(axis=-1).sum()) for n in src.children.values() if "UVW" in n.ds)
    assert npad > 0, "no NaN padding in the source view -- test would prove nothing"

    outname = str(tmp_path / "padded")
    imager(
        [Path(holed_ms)],
        outname,
        channels_per_image=-1,
        integrations_per_image=-1,
        product="I",
        nx=NPIX,
        ny=NPIX,
        cell_size=image_geometry.cell_size,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
        progressbar=False,
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    bands = [n for n in dt.children if n.startswith("band")]
    assert bands, "imager produced no band nodes"
    checked = 0
    for bname in bands:
        for pname, part in dt[bname].children.items():
            uvw = part.ds.UVW.values
            assert np.isfinite(uvw).all(), f"{bname}/{pname} stored {np.isnan(uvw).sum()} NaN UVW values"
            # padding is fully masked, so its survival would also show up here
            fully_masked = ~part.ds.MASK.values.astype(bool).any(axis=1)
            assert not fully_masked.any(), f"{bname}/{pname} stored {fully_masked.sum()} fully flagged rows"
            checked += 1
    assert checked, "no partition children to check"
