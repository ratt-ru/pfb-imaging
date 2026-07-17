"""End-to-end smoke test for the MSv4 DataTree imager (.dt product).

This module loads the arcae ``xarray-ms:msv2`` engine. As of arcae 0.5.2
(ratt-ru/arcae#211, #212) arcae and python-casacore coexist in one process, so
this file runs in the same ``pytest tests/`` session as the casacore-based
tests. Correctness is pinned against the injected ``sky_truth`` fixture (WCS
positions/fluxes and a brute-force DFT oracle), not against the legacy
``init``+``grid`` path (retired, #277).
"""

import glob
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from pfb_imaging.core.imager import imager as imager_core


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


def test_sky_truth_fixture_writes_ms(sky_truth, ms_name, ms_meta):
    """The fixture's DATA/FLAG writes land in the MS and are deterministic."""
    from daskms import xds_from_ms

    xds = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1, "corr": -1})[0]
    data = xds.DATA.values
    flag = xds.FLAG.values
    assert data.any(), "fixture wrote all-zero DATA"
    np.testing.assert_array_equal(flag, sky_truth.flag)
    assert flag[:, sky_truth.flagged_chan, :].all()
    # XX == YY (Stokes I only), cross-hands zero
    np.testing.assert_array_equal(data[:, :, 0], data[:, :, -1])
    assert not data[:, :, 1].any() and not data[:, :, 2].any()


def _peak_yx(da_2d):
    """(iy, ix) of the max of a 2D DataArray with dims ('x','y') or ('y','x')."""
    arr = da_2d.values
    iflat = int(np.argmax(arr))
    i0, i1 = np.unravel_index(iflat, arr.shape)
    if da_2d.dims == ("y", "x"):
        return i0, i1
    if da_2d.dims == ("x", "y"):
        return i1, i0
    raise AssertionError(f"unexpected image dims {da_2d.dims}")


def test_imager_groundtruth(sky_truth, ms_name, tmp_path):
    """Injected sources land at their (RA, Dec) through the FITS WCS, at the
    right flux; the .dt arrays agree via their dims names.

    Written order-agnostically (WCS + dims, never raw index order) so it
    passes identically before and after the (Y, X) switch -- it is the safety
    net for that switch.
    """
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    outname = str(tmp_path / "gt")
    imager_core(
        [Path(ms_name)],
        outname,
        channels_per_image=2,
        integrations_per_image=-1,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=None,
        fits_mfs=True,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    # --- FITS: WCS positions and fluxes ---
    fits_files = glob.glob(str(tmp_path / "*dirty*mfs.fits"))
    assert len(fits_files) == 1
    with afits.open(fits_files[0]) as hdul:
        img = hdul[0].data.squeeze()  # (ny, nx) FITS layout
        w = WCS(hdul[0].header).celestial
    assert img.shape == (sky_truth.ny, sky_truth.nx)

    # brightest source: global argmax must sit at its WCS-predicted pixel
    order = np.argsort(sky_truth.ref_flux)[::-1]
    s0 = order[0]
    px, py = w.world_to_pixel(sky_truth.sky_coords[s0])
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    assert (ix, iy) == (int(round(float(px))), int(round(float(py))))

    # every source: flux at its own WCS pixel (MFS dirty is wsum-normalised
    # Jy/beam; expected peak = ref_flux / n). Tolerance is 15%: the dirty
    # peaks carry the other sources' PSF sidelobes and the 52% fractional
    # bandwidth's spectral averaging -- measured contamination is 9.3% on the
    # faintest source (its pixel value matches a brute-force DFT to 6
    # decimals, so this is physics, not a pipeline bias). The precise
    # numerical check is test_imager_matches_dft.
    for s in range(sky_truth.lpix.size):
        px, py = w.world_to_pixel(sky_truth.sky_coords[s])
        val = img[int(round(float(py))), int(round(float(px)))]
        expected = sky_truth.ref_flux[s] / sky_truth.nvals[s]
        assert abs(val - expected) < 0.15 * expected, f"source {s}: {val} vs {expected}"

    # --- .dt arrays: dims-aware peak position agrees with the FITS ---
    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    nodes = [dt[n].ds for n in dt.children if n.startswith("band")]
    num = sum(ds.DIRTY[0] for ds in nodes)
    den = sum(float(ds.WSUM.values[0]) for ds in nodes)
    iy_dt, ix_dt = _peak_yx(num)
    assert (iy_dt, ix_dt) == (iy, ix)
    np.testing.assert_allclose((num.values / den).max(), img.max(), rtol=1e-6)


def test_imager_matches_dft(sky_truth, ms_name, tmp_path):
    """Gridded DIRTY matches a brute-force DFT of the stored partition inputs.

    Fully independent of the wgridder: the absolute-maths check that replaces
    the init+grid equivalence oracle.
    """
    from africanus.constants import c as lightspeed

    outname = str(tmp_path / "dft")
    imager_core(
        [Path(ms_name)],
        outname,
        channels_per_image=2,
        integrations_per_image=-1,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=None,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    band = dt[nodes[0]]
    nx, ny = sky_truth.nx, sky_truth.ny
    cell = sky_truth.cell_rad

    # a dozen probe pixels: the three source pixels + fixed scattered ones
    rng = np.random.default_rng(99)
    probes = [(nx // 2 - int(lp), ny // 2 + int(mp)) for lp, mp in zip(sky_truth.lpix, sky_truth.mpix)]
    probes += [tuple(int(v) for v in p) for p in rng.integers(8, min(nx, ny) - 8, size=(9, 2))]

    dirty = band.ds.DIRTY[0]  # un-normalised, dims-aware access below
    peak = float(np.abs(dirty.values).max())

    for ixm, iym in probes:
        # x-major pixel -> (l, m) per the pinned wgridder convention
        # (test_wgridder_image_orientation)
        l_p = (nx // 2 - ixm) * cell
        m_p = (iym - ny // 2) * cell
        nlm = np.sqrt(1.0 - l_p * l_p - m_p * m_p)
        val_dft = 0.0
        for cname in band.children:
            p = band[cname].ds
            uvw = p.UVW.values
            freq = p.FREQ.values
            vis = p.VIS.values[0]
            wgt = p.WEIGHT.values[0]
            mask = p.MASK.values.astype(bool)
            phase = (
                -2j
                * np.pi
                * freq[None, :]
                / lightspeed
                * (uvw[:, 0:1] * l_p + uvw[:, 1:2] * m_p + uvw[:, 2:] * (nlm - 1.0))
            )
            val_dft += float(np.sum(wgt[mask] * np.real(vis[mask] * np.exp(phase)[mask])))
        val_grid = float(dirty.isel(x=ixm, y=iym).values)
        assert abs(val_grid - val_dft) < 1e-5 * peak, f"pixel ({ixm},{iym}): {val_grid} vs {val_dft}"
