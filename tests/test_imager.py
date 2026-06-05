"""End-to-end smoke test for the MSv4 DataTree imager (.dt product).

This module loads the arcae ``xarray-ms:msv2`` engine, which cannot coexist
with python-casacore in one process (arcae#72). CI runs it in its own pytest
invocation. The equivalence test therefore runs the casacore-based
``init``+``grid`` reference in a subprocess and compares on disk.
"""

import glob
import os
import subprocess
import sys
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

    init+grid load python-casacore, so they run in a subprocess; imager loads
    arcae in-process. Both are pinned to the same image size and compared as
    peak-normalised MFS dirty images.
    """
    nx = ny = 256
    pfb = os.path.join(os.path.dirname(sys.executable), "pfb")
    env = os.environ.copy()

    base_leg = str(tmp_path / "legacy")

    def _run(label, argv):
        """Run a pfb subprocess, surfacing its output so CI logs the casacore
        reference path (the init/grid Ray runtime is the prime suspect when the
        reference image comes back empty/NaN on a resource-constrained runner)."""
        res = subprocess.run([pfb, *argv], env=env, capture_output=True, text=True)
        print(f"\n===== {label} rc={res.returncode} =====", flush=True)
        print(f"[{label} stdout tail]\n{res.stdout[-3000:]}", flush=True)
        print(f"[{label} stderr tail]\n{res.stderr[-3000:]}", flush=True)
        res.check_returncode()

    _run(
        "pfb init",
        [
            "init",
            "--ms",
            str(ms_name),
            "--output-filename",
            base_leg,
            "--channels-per-image",
            "2",
            "--integrations-per-image",
            "-1",
            "--product",
            "I",
            "--overwrite",
        ],
    )
    _run(
        "pfb grid",
        [
            "grid",
            "--output-filename",
            base_leg,
            "--nx",
            str(nx),
            "--ny",
            str(ny),
            "--no-psf",
            "--no-residual",
            "--no-noise",
            "--no-beam",
            "--no-fits-mfs",
            "--no-fits-cubes",
            "--overwrite",
        ],
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
    assert np.isfinite(leg).all(), "legacy init+grid MFS dirty contains NaN/Inf (see diagnostics above)"
    assert np.isfinite(new).all(), "imager .dt MFS dirty contains NaN/Inf (see diagnostics above)"

    # Both images are already wsum-normalised (sum DIRTY / sum WSUM), summed over
    # all output-image nodes, so the comparison is MFS and independent of how the
    # bands are indexed (imager keeps the original band id, init+grid reindexes
    # contiguously when a band is dropped). init+grid and imager share the wsum
    # convention, so they agree to ~1e-12 on real data. Use the 1 + x shift idiom
    # (cf. test_sara) so the assertion stays well-defined when both images are
    # identically zero -- an MS with no signal in DATA, or a fully-flagged band --
    # rather than dividing by a zero peak.
    assert_allclose(1 + new, 1 + leg, rtol=1e-4, atol=1e-4)
