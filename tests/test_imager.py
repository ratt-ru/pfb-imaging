"""End-to-end smoke test for the MSv4 DataTree imager (.dt product).

This module loads the arcae ``xarray-ms:msv2`` engine. As of arcae 0.5.2
(ratt-ru/arcae#211, #212) arcae and python-casacore coexist in one process, so
this file runs in the same ``pytest tests/`` session as the casacore-based
tests, and the equivalence test calls the legacy ``init``+``grid`` reference
in-process (it previously ran them in a subprocess for arcae#72 isolation).
"""

import glob
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from pfb_imaging.core.imager import imager as imager_core


def _describe(name, num, den, per_node_wsum):
    """Compact diagnostics for an (un-normalised) MFS numerator / wsum pair.

    Printed before the equivalence assertion so a CI failure shows *which*
    image went bad and why (empty store, zero wsum, NaNs in DIRTY, ...).
    """
    nnan = int(np.isnan(num).sum())
    finite = num[np.isfinite(num)]
    lo = float(finite.min()) if finite.size else float("nan")
    hi = float(finite.max()) if finite.size else float("nan")
    print(
        f"[{name}] nodes={len(per_node_wsum)} den(sum wsum)={den:.6g} "
        f"per-node wsum={per_node_wsum} DIRTY nan={nnan}/{num.size} "
        f"finite-min/max={lo:.4g}/{hi:.4g}",
        flush=True,
    )


def _mfs_dirty_from_dt(store):
    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    nodes = [dt[n].ds for n in dt.children if n.startswith("band")]
    num = sum(ds.DIRTY.values[0] for ds in nodes)  # corr 0 == Stokes I
    per_node_wsum = [float(ds.WSUM.values[0]) for ds in nodes]
    den = sum(per_node_wsum)
    _describe(f"imager .dt {store}", num, den, per_node_wsum)
    return num / den


def _mfs_dirty_from_dds(store):
    from pfb_imaging.utils.naming import xds_from_url

    dds, _ = xds_from_url(store)
    num = sum(ds.DIRTY.values[0] for ds in dds)
    per_node_wsum = [float(ds.WSUM.values[0]) for ds in dds]
    den = sum(per_node_wsum)
    _describe(f"legacy .dds {store}", num, den, per_node_wsum)
    return num / den


def test_imager_writes_dt_tree(ms_name, tmp_path):
    """imager() runs both passes and writes a unified .dt DataTree plus FITS."""
    outname = str(tmp_path / "test_imager")

    imager_core(
        [Path(ms_name)],
        outname,
        integrations_per_image=15,
        channels_per_image=2,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        fits_mfs=True,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)

    # output-image nodes: band{b}_time{t}
    image_names = sorted(n for n in dt.children if n.startswith("band"))
    assert image_names, "no output-image nodes written"

    band = dt[image_names[0]]
    for v in ("DIRTY", "RESIDUAL", "PSF", "WSUM", "PSFPARSN"):
        assert v in band.ds, f"band node missing {v}"
    assert np.isfinite(band.ds.DIRTY.values).all()
    # fresh image: residual == dirty (no model yet)
    np.testing.assert_array_equal(band.ds.RESIDUAL.values, band.ds.DIRTY.values)

    # partition children with vis-space + per-partition image-space products
    part_names = [n for n in band.children]
    assert part_names, "band node has no partition children"
    part = band[part_names[0]]
    for v in ("VIS", "WEIGHT", "MASK", "UVW", "FREQ", "PSF", "PSFHAT", "BEAM"):
        assert v in part.ds, f"partition missing {v}"
    assert part.ds.attrs["baseline_group"] == "all"

    # MFS FITS written for DIRTY
    assert glob.glob(str(tmp_path / "*dirty*mfs.fits")), "no MFS dirty FITS written"


def test_scratch_retained_by_default(ms_name, tmp_path):
    """The pass-1 .scratch store is kept by default for re-gridding without re-read."""
    from daskms.fsspec_store import DaskMSStore  # casacore-free fsspec store

    outname = str(tmp_path / "cache")
    imager_core(
        [Path(ms_name)],
        outname,
        channels_per_image=2,
        product="I",
        field_of_view=1.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )
    assert DaskMSStore(outname + "_I.scratch").exists()
    assert DaskMSStore(outname + "_I.dt").exists()


def test_imager_matches_init_grid_single_field(ms_name, tmp_path):
    """imager .dt dirty matches the legacy init+grid .dds dirty (natural weights).

    As of arcae 0.5.2 (ratt-ru/arcae#211, #212) arcae and python-casacore coexist
    in one process, so the casacore-based init+grid reference now runs **in-process**
    alongside the arcae-based imager (it used to run in a subprocess purely to keep
    casacore out of the arcae process). All three calls share the session Ray
    cluster, so each passes keep_ray_alive=True to avoid tearing it down mid-suite.

    KNOWN LIMITATION (FIXME): the shared test MS downloaded from Google Drive
    currently has an all-zero DATA column, so in CI both pipelines produce an
    identically-zero dirty image and this test only verifies that the two paths
    *agree* (and run end to end) -- it is a no-op on the actual gridding maths.
    It becomes a real equivalence check only when the MS carries visibilities
    (older locally-cached copies still do, which is why it is meaningful there).
    To make it meaningful everywhere, populate DATA with a deterministic model
    (predict point sources with ducc0 ``dirty2vis`` as test_kclean/test_sara do)
    -- ideally once per session in conftest. Fully flagging one band at the same
    time would additionally exercise the band-dropping path. Until then the
    comparison is deliberately divide-by-zero-safe (see below) so a zero-signal MS
    does not masquerade as a NaN failure.
    """
    from pfb_imaging.core.grid import grid
    from pfb_imaging.core.init import init

    nx = ny = 256

    base_leg = str(tmp_path / "legacy")
    init(
        [Path(ms_name)],
        base_leg,
        channels_per_image=2,
        integrations_per_image=-1,
        product="I",
        overwrite=True,
        keep_ray_alive=True,
    )
    grid(
        base_leg,
        nx=nx,
        ny=ny,
        psf=False,
        residual=False,
        noise=False,
        beam=False,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    base_new = str(tmp_path / "imager")
    imager_core(
        [Path(ms_name)],
        base_new,
        channels_per_image=2,
        integrations_per_image=-1,
        product="I",
        nx=nx,
        ny=ny,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    leg = _mfs_dirty_from_dds(base_leg + "_I_main.dds")
    new = _mfs_dirty_from_dt(base_new + "_I.dt")
    assert new.shape == leg.shape == (nx, ny)

    # finiteness is checked per-image first so a CI failure names the offending
    # path (legacy vs imager) instead of collapsing both into a single nan diff
    assert np.isfinite(leg).all(), "legacy init+grid MFS dirty contains NaN/Inf (see _describe output above)"
    assert np.isfinite(new).all(), "imager .dt MFS dirty contains NaN/Inf (see _describe output above)"

    # Both images are already wsum-normalised (sum DIRTY / sum WSUM), summed over
    # all output-image nodes, so the comparison is MFS and independent of how the
    # bands are indexed (imager keeps the original band id, init+grid reindexes
    # contiguously when a band is dropped). init+grid and imager share the wsum
    # convention, so they agree to ~1e-12 on real data. Use the 1 + x shift idiom
    # (cf. test_sara) so the assertion stays well-defined when both images are
    # identically zero -- an MS with no signal in DATA, or a fully-flagged band --
    # rather than dividing by a zero peak.
    assert_allclose(1 + new, 1 + leg, rtol=1e-4, atol=1e-4)


def test_imager_concat_row_collapses_time(ms_name, tmp_path):
    """concat_row=True collapses the time axis into one band node and agrees
    with concat_row=False on the MFS dirty.

    Both runs pin weight_grouping="per-band" and robustness=0.0 so the per-row
    imaging weights are identical; vis2dirty is linear in the rows
    (grid(A∪B) == grid(A) + grid(B) up to fp), so the wsum-normalised MFS dirty
    must match. The shared MS is one scan of 60 integrations, so
    integrations_per_image=15 splits it into 4 time blocks (4 timeids) to make
    the concat_row=False granularity meaningful.
    """
    from collections import Counter

    common = dict(
        channels_per_image=2,
        integrations_per_image=15,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        weight_grouping="per-band",
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    base_true = str(tmp_path / "concat_true")
    imager_core([Path(ms_name)], base_true, concat_row=True, **common)
    base_false = str(tmp_path / "concat_false")
    imager_core([Path(ms_name)], base_false, concat_row=False, **common)

    dt_true = xr.open_datatree(base_true + "_I.dt", engine="zarr", chunks=None)
    dt_false = xr.open_datatree(base_false + "_I.dt", engine="zarr", chunks=None)
    bands_true = sorted(n for n in dt_true.children if n.startswith("band"))
    bands_false = sorted(n for n in dt_false.children if n.startswith("band"))

    # concat_row=True: exactly one time0000 node per distinct band
    assert bands_true, "no band nodes written"
    assert all(n.endswith("_time0000") for n in bands_true)
    bandids_true = {int(dt_true[n].ds.attrs["bandid"]) for n in bands_true}
    assert len(bands_true) == len(bandids_true)

    # concat_row=False: multiple time nodes for at least one band (4 blocks here)
    per_band = Counter(int(dt_false[n].ds.attrs["bandid"]) for n in bands_false)
    assert max(per_band.values()) > 1, "expected multiple time nodes per band at ipi=15"

    def mfs(dt, names):
        num = sum(dt[n].ds.DIRTY.values[0] for n in names)
        den = sum(float(dt[n].ds.WSUM.values[0]) for n in names)
        return num / den

    a = mfs(dt_true, bands_true)
    b = mfs(dt_false, bands_false)
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert_allclose(1 + a, 1 + b, rtol=1e-4, atol=1e-4)
