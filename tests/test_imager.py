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

from pfb_imaging.core.imager import imager as imager_core


def _mfs_dirty_from_dt(store):
    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    nodes = [dt[n].ds for n in dt.children if n.startswith("band")]
    num = sum(ds.DIRTY.values[0] for ds in nodes)  # corr 0 == Stokes I
    den = sum(float(ds.WSUM.values[0]) for ds in nodes)
    return num / den


def _mfs_dirty_from_dds(store):
    from pfb_imaging.utils.naming import xds_from_url

    dds, _ = xds_from_url(store)
    num = sum(ds.DIRTY.values[0] for ds in dds)
    den = sum(float(ds.WSUM.values[0]) for ds in dds)
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
    subprocess.run(
        [
            pfb,
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
        check=True,
        env=env,
        capture_output=True,
    )
    subprocess.run(
        [
            pfb,
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
        check=True,
        env=env,
        capture_output=True,
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

    # peak-normalised comparison is robust to small absolute-scale differences
    a = new / np.abs(new).max()
    b = leg / np.abs(leg).max()
    assert np.abs(a - b).max() < 5e-2, f"max abs diff {np.abs(a - b).max():.3e} exceeds tolerance"
