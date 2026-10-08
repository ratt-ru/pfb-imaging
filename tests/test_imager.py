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
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_imaging.core.imager import imager as imager_core


@pytest.mark.slow
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
    for v in ("DIRTY", "PSF", "WSUM", "PSFPARSN", "BEAM"):
        assert v in band.ds, f"band node missing {v}"
    assert np.isfinite(band.ds.DIRTY.values).all()
    # residual is only computed when a model is passed in (future feature)
    assert "RESIDUAL" not in band.ds
    # band BEAM = wsum-weighted mean of the partition beams (linear-mosaic response)
    pdss = [band[p].ds for p in band.children]
    num = sum(np.asarray(p.attrs["wsum"])[:, None, None] * p.BEAM.values for p in pdss)
    den = sum(np.asarray(p.attrs["wsum"]) for p in pdss)[:, None, None]
    assert_allclose(band.ds.BEAM.values, num / den, rtol=1e-12, atol=0)
    # BDIRTY = sum_p B_p * dirty_p, the model-free term of the exact deconv
    # gradient (D23); for this MS each band has a single partition, so it must
    # equal that partition's beam times the summed DIRTY
    assert "BDIRTY" in band.ds, "band node missing BDIRTY"
    assert len(pdss) == 1
    assert_allclose(band.ds.BDIRTY.values, pdss[0].BEAM.values * band.ds.DIRTY.values, rtol=1e-12, atol=0)

    # partition children with vis-space + per-partition image-space products
    part_names = [n for n in band.children]
    assert part_names, "band node has no partition children"
    part = band[part_names[0]]
    for v in ("VIS", "WEIGHT", "MASK", "UVW", "FREQ", "PSF", "PSFHAT", "BEAM"):
        assert v in part.ds, f"partition missing {v}"
    assert part.ds.attrs["baseline_group"] == "all"

    # MFS FITS written for DIRTY
    assert glob.glob(str(tmp_path / "fits" / "*dirty*mfs.fits")), "no MFS dirty FITS written"

    # beam FITS: dimensionless, D22 header card, wsum-weighted mean over bands
    from astropy.io import fits as afits

    beam_hits = glob.glob(str(tmp_path / "fits" / "*_beam_time*_mfs.fits"))
    assert beam_hits, "no beam MFS FITS written"
    with afits.open(beam_hits[0]) as hdul:
        assert hdul[0].header["BEAMINCN"], "missing D22 n-term card"
        assert hdul[0].header["BUNIT"] == ""
        bimg = np.squeeze(hdul[0].data).astype(np.float64)
    nodes = [dt[n].ds for n in image_names]
    num = sum(float(ds.WSUM.values[0]) * ds.BEAM.values[0] for ds in nodes)
    den = sum(float(ds.WSUM.values[0]) for ds in nodes)
    assert_allclose(bimg, num / den, rtol=1e-5, atol=1e-7)


@pytest.mark.slow
def test_scratch_retained_by_default(ms_name, tmp_path):
    """The pass-1 .scratch store is kept by default for re-gridding without re-read."""
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
    assert Path(outname + "_I.scratch").exists()
    assert Path(outname + "_I.dt").exists()


@pytest.mark.slow
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
    from tests.conftest import require_daskms

    require_daskms()

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


@pytest.fixture(scope="module")
def gt_imager_fits(ms_name, sky_truth, tmp_path_factory):
    """One ground-truth imager run with both MFS and per-partition FITS on.

    test_imager_groundtruth and test_imager_fits_per_partition differ only in
    which FITS they ask for (fits_mfs vs fits_per_partition) and assert on, so
    one run with both flags serves both. Returns (outname, fits_dir).
    """
    root = tmp_path_factory.mktemp("gt_imager_fits")
    outname = str(root / "gtfits")
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
        fits_per_partition=True,
        overwrite=True,
        keep_ray_alive=True,
    )
    return outname, root / "fits"


@pytest.mark.slow
def test_imager_groundtruth(sky_truth, gt_imager_fits):
    """Injected sources land at their (RA, Dec) through the FITS WCS, at the
    right flux; the .dt arrays agree via their dims names.

    Written order-agnostically (WCS + dims, never raw index order) so it
    passes identically before and after the (Y, X) switch -- it is the safety
    net for that switch.
    """
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    outname, fits_dir = gt_imager_fits

    # --- FITS: WCS positions and fluxes ---
    fits_files = glob.glob(str(fits_dir / "*dirty*mfs.fits"))
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


def _open_first_vis_node(ms_name):
    from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

    from pfb_imaging.core.imager import get_engine

    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    for node in dt.children.values():
        if node.attrs.get("type") in VISIBILITY_XDS_TYPES:
            return node
    raise RuntimeError("no visibility node in test MS")


def _run_stokes_vis(ms_name, scratch, radec_new=None, beam_model=None, cell_rad=1e-5, flag_chans=None):
    """Drive stokes_vis directly (no Ray) on the test MS's first node.

    flag_chans: optional channel slice to flag before conversion, used to shift
    the piece's effective frequency away from the nominal band centre (#296).
    """
    from pfb_imaging.utils.stokes2vis_msv4 import stokes_vis

    node = _open_first_vis_node(ms_name)
    dc1 = node.ds.attrs["data_groups"]["base"]["correlated_data"]
    freq = node.ds.frequency.values
    if flag_chans is not None:
        # FLAG is (time, baseline_id, frequency, polarization). Assign through
        # the .dataset setter on a copy: rebuilding the node with
        # xr.DataTree(dataset=...) would drop antenna_xds and the other
        # children stokes_vis reads.
        ds_mod = node.to_dataset()
        flg = ds_mod.FLAG.values.copy()
        flg[:, :, flag_chans, :] = True
        ds_mod["FLAG"] = (ds_mod.FLAG.dims, flg)
        node = node.copy()
        node.dataset = ds_mod
    uvw = node.ds.UVW.values.reshape(-1, 3)
    uvw = uvw[~np.isnan(uvw).all(axis=-1)]
    max_blength = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max()

    bandid, timeid, piece = stokes_vis(
        dc1=dc1,
        node_dt=node,
        scratch_store=scratch,
        bandid=0,
        timeid=0,
        msid=0,
        freq_nominal=float(freq.mean()),
        product="I",
        beam_model=beam_model,
        max_blength=float(max_blength),
        max_freq=float(freq.max()),
        nx_pad=64,
        ny_pad=64,
        cell_rad=cell_rad,
        nx=64,
        ny=64,
        radec_new=radec_new,
        nthreads=1,
    )
    ds = xr.open_datatree(scratch, engine="zarr", chunks=None)[f"band{bandid:04d}_time{timeid:04d}/{piece}"].ds
    return ds.load()


def test_stokes_vis_rephases_to_new_centre(sky_truth, ms_name, tmp_path, needs_rephasing):
    """Rephasing changes phases and UVW only; attrs record both centres.

    sky_truth guarantees non-zero DATA (the phases-differ assertion is
    vacuous on a zero-signal MS).
    """
    ref = _run_stokes_vis(ms_name, str(tmp_path / "ref.scratch"))
    # no rephasing: tangent point == field pointing
    assert ref.attrs["ra"] == ref.attrs["ra0"]
    assert ref.attrs["dec"] == ref.attrs["dec0"]

    # rephase 60 arcsec north of the field centre
    radec_new = np.array([ref.attrs["ra0"], ref.attrs["dec0"] + np.deg2rad(60.0 / 3600.0)])
    new = _run_stokes_vis(ms_name, str(tmp_path / "new.scratch"), radec_new=radec_new)

    assert_allclose([new.attrs["ra"], new.attrs["dec"]], radec_new, atol=1e-12)
    assert new.attrs["ra0"] == ref.attrs["ra0"] and new.attrs["dec0"] == ref.attrs["dec0"]
    # phase-only data change: amplitudes preserved, phases not
    assert_allclose(np.abs(new.VIS.values), np.abs(ref.VIS.values), rtol=1e-9)
    assert not np.allclose(new.VIS.values, ref.VIS.values)
    # weights and mask untouched; UVW re-synthesized towards the new centre
    assert_allclose(new.WEIGHT.values, ref.WEIGHT.values, rtol=1e-12)
    np.testing.assert_array_equal(new.MASK.values, ref.MASK.values)
    assert not np.allclose(new.UVW.values, ref.UVW.values)


@pytest.mark.slow
def test_imager_rephase_roundtrip(sky_truth, ms_name, tmp_path, needs_rephasing):
    """Rephasing to an offset phase_dir with target back at the original field
    centre reproduces the unrephased image -- compared projection-aware.

    The two runs live in DIFFERENT SIN projections (tangent at the field
    centre vs at the offset phase_dir), whose frames are rotated w.r.t. each
    other by ~dra*sin(dec) to first order (2.3 arcmin here; a 0.14 px
    displacement at the fov edge, measured to match the analytic rotation).
    A full-field pixel-wise comparison therefore CANNOT converge -- that was
    misdiagnosed as an "RA-axis geometry bug" on the abandoned
    imager_rephase_and_interp_beam branch. Valid oracles, used below:

    1. WSUM: natural weights are phase-invariant -> tight.
    2. Central box (+/-20 px), where the inter-frame displacement is < 0.01
       px: pixel-wise DIRTY/PSF agreement at the numerical floor (measured
       1.9e-4 of the dirty peak within +/-10 px; atol 2e-3 * psf_peak leaves
       margin for per-band floors).
    3. Ground truth through the WCS: every injected source must land at its
       true (RA, Dec) in the round-trip FITS (whose header carries the
       CRPIX-shifted target convention) at its injected flux.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    common = dict(
        channels_per_image=2,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=None,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )
    base_ref = str(tmp_path / "ref")
    imager_core([Path(ms_name)], base_ref, fits_mfs=False, **common)
    dt_ref = xr.open_datatree(base_ref + "_I.dt", engine="zarr", chunks=None)
    names = sorted(n for n in dt_ref.children if n.startswith("band"))
    ra0 = dt_ref[names[0]].attrs["ra"]
    dec0 = dt_ref[names[0]].attrs["dec"]

    def fmt(ra_rad, dec_rad):
        c = SkyCoord(ra_rad * u.rad, dec_rad * u.rad, frame="fk5")
        ra_str = c.ra.to_string(u.hour, sep=":", precision=8)
        dec_str = c.dec.to_string(u.deg, sep=":", precision=8)
        return f"{ra_str},{dec_str}"

    # rephase 3 arcmin north and 4 arcmin east (diagonal, to pin the RA/l
    # axis as well as Dec/m); ask for the image centred back on the field
    base_new = str(tmp_path / "rephased")
    ra_offset = ra0 + np.deg2rad(4.0 / 60.0) / np.cos(dec0)
    dec_offset = dec0 + np.deg2rad(3.0 / 60.0)
    imager_core(
        [Path(ms_name)],
        base_new,
        phase_dir=fmt(ra_offset, dec_offset),
        target=fmt(ra0, dec0),
        fits_mfs=True,
        **common,
    )
    dt_new = xr.open_datatree(base_new + "_I.dt", engine="zarr", chunks=None)

    half = 20  # central box: inter-frame displacement < 0.01 px here
    ys = slice(sky_truth.ny // 2 - half, sky_truth.ny // 2 + half + 1)
    xs = slice(sky_truth.nx // 2 - half, sky_truth.nx // 2 + half + 1)
    for name in names:
        ref = dt_ref[name].ds.load()
        new = dt_new[name].ds.load()
        # natural weights are phase-invariant
        assert_allclose(new.WSUM.values, ref.WSUM.values, rtol=1e-9)
        wsum = ref.WSUM.values[0]
        psf_peak = ref.PSF.values[0].max() / wsum
        d_new = new.DIRTY.isel(y=ys, x=xs).values[0] / wsum
        d_ref = ref.DIRTY.isel(y=ys, x=xs).values[0] / wsum
        assert_allclose(d_new, d_ref, atol=2e-3 * psf_peak)
        p_new = new.PSF.isel(y_psf=ys, x_psf=xs).values[0] / wsum
        p_ref = ref.PSF.isel(y_psf=ys, x_psf=xs).values[0] / wsum
        assert_allclose(p_new, p_ref, atol=2e-3 * psf_peak)
        # attrs record the rephased tangent point and the target offset
        assert new.attrs["dec"] > ref.attrs["dec"]
        assert new.attrs["ra"] > ref.attrs["ra"]
        assert abs(new.attrs["m0"]) > 0.0
        assert abs(new.attrs["l0"]) > 0.0

    # ground truth through the WCS of the round-trip FITS (CRPIX-shifted
    # target convention): sources land at their true positions and fluxes
    fits_files = glob.glob(str(tmp_path / "fits" / "*rephased*dirty*mfs.fits"))
    assert len(fits_files) == 1
    with afits.open(fits_files[0]) as hdul:
        img = hdul[0].data.squeeze()
        w = WCS(hdul[0].header).celestial
    for src in range(sky_truth.lpix.size):
        px, py = w.world_to_pixel(sky_truth.sky_coords[src])
        iy, ix = int(round(float(py))), int(round(float(px)))
        # peak within 1 px of the WCS-predicted position
        box = img[iy - 1 : iy + 2, ix - 1 : ix + 2]
        by, bx = np.unravel_index(int(np.argmax(box)), box.shape)
        val = float(box[by, bx])
        expected = sky_truth.ref_flux[src] / sky_truth.nvals[src]
        assert abs(val - expected) < 0.15 * expected, f"source {src}: {val} vs {expected}"


def test_stokes_vis_beam_on_image_grid(ms_name, tmp_path, needs_rephasing):
    """Pass-1 places the BEAM on the image grid; under rephasing its peak
    stays at the FIELD pointing (where the antennas point), not the tangent.

    cell is enlarged so 10 px is a measurable fraction of the katbeam width
    (at the default 1e-5 rad cell the beam is flat to ~1e-6 over the fov and
    the argmax is noise).
    """
    cell = 5.0e-4  # rad; 64 px fov ~ 1.8 deg
    ref = _run_stokes_vis(ms_name, str(tmp_path / "b0.scratch"), beam_model="katbeam", cell_rad=cell)
    assert ref.BEAM.dims == ("corr", "y", "x")
    assert ref.BEAM.shape == (1, 64, 64)
    assert ref.attrs["beam_includes_n"] is True

    # beam_model=None stores exactly the folded 1/n (D22), not ones
    none_ds = _run_stokes_vis(ms_name, str(tmp_path / "bn.scratch"), cell_rad=cell)
    coords = (-(64 / 2) + np.arange(64)) * cell
    yy_n, xx_n = np.meshgrid(coords, coords, indexing="ij")
    nlm = np.sqrt(1.0 - xx_n**2 - yy_n**2)
    np.testing.assert_allclose(none_ds.BEAM.values[0], 1.0 / nlm, rtol=1e-6)
    iy, ix = np.unravel_index(int(np.argmax(ref.BEAM.values[0])), (64, 64))
    assert (iy, ix) == (32, 32), "unrephased beam must peak at the image centre"

    # tangent 10 px north of the field pointing: the beam peak moves 10 px
    # south of the image centre (the field is south of the new tangent)
    radec_new = np.array([ref.attrs["ra0"], ref.attrs["dec0"] + 10 * cell])
    new = _run_stokes_vis(
        ms_name, str(tmp_path / "b1.scratch"), radec_new=radec_new, beam_model="katbeam", cell_rad=cell
    )
    assert new.BEAM.shape == (1, 64, 64)
    iy, ix = np.unravel_index(int(np.argmax(new.BEAM.values[0])), (64, 64))
    assert (iy, ix) == (22, 32), f"rephased beam peak at ({iy},{ix}), expected (22,32)"
    # reprojection fills 0 outside the small-grid coverage; interior peak ~1
    assert new.BEAM.values[0, 22, 32] > 0.99


def test_imager_no_psf_quicklook(ms_name, tmp_path):
    """psf=False skips PSF gridding everywhere; deconv refuses the tree cleanly."""
    from pfb_imaging.core.deconv import deconv

    outname = str(tmp_path / "quicklook")
    imager_core(
        [Path(ms_name)],
        outname,
        channels_per_image=2,
        product="I",
        field_of_view=1.0,
        psf=False,
        fits_mfs=True,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    # the per-task progress telemetry reaches the log file, not just the terminal (#348)
    (logname,) = glob.glob(str(tmp_path / "logs" / "imager_*.log"))
    text = Path(logname).read_text()
    assert "Completed: " in text and "Gridded: " in text

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    bands = [n for n in dt.children if n.startswith("band")]
    assert bands
    for b in bands:
        assert "PSF" not in dt[b].ds and "PSFPARSN" not in dt[b].ds
        for p in dt[b].children:
            pds = dt[b][p].ds
            for v in ("PSF", "PSFHAT", "PSFPARSN"):
                assert v not in pds, f"{b}/{p} has {v} despite psf=False"

    # dirty FITS still written, no psf FITS (glob on _psf_ to avoid the outname)
    assert glob.glob(str(tmp_path / "fits" / "*dirty*mfs.fits"))
    assert not glob.glob(str(tmp_path / "fits" / "*_psf_*")), "psf FITS written despite psf=False"

    with pytest.raises(ValueError, match="re-run pfb imager with --psf"):
        deconv(outname, log_directory=str(tmp_path), nthreads=1)


@pytest.mark.slow
def test_imager_fits_per_partition(sky_truth, gt_imager_fits):
    """Per-partition sanity FITS: one file per computed variable per partition,
    with a WCS that puts the injected sources where they belong."""
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    outname, fits_dir = gt_imager_fits

    pdir = fits_dir / "gtfits_I_partitions"
    assert pdir.is_dir(), "partitions FITS subdirectory not created"

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    bands = [n for n in dt.children if n.startswith("band")]
    nparts = sum(len(dt[b].children) for b in bands)
    for var in ("dirty", "psf", "beam"):
        hits = glob.glob(str(pdir / f"{var}_band*_part*.fits"))
        assert len(hits) == nparts, f"{var}: {len(hits)} FITS for {nparts} partitions"

    # orientation sanity: brightest truth source sits at its WCS pixel
    hits = sorted(glob.glob(str(pdir / "dirty_band*_part0000_*.fits")))
    with afits.open(hits[0]) as hdul:
        img = np.squeeze(hdul[0].data)
        w = WCS(hdul[0].header).celestial
    assert img.shape == (sky_truth.ny, sky_truth.nx)
    s0 = np.argsort(sky_truth.ref_flux)[::-1][0]
    px, py = w.world_to_pixel(sky_truth.sky_coords[s0])
    iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)
    assert (ix, iy) == (int(round(float(px))), int(round(float(py))))

    # beam FITS carries the D22 card
    bhit = sorted(glob.glob(str(pdir / "beam_band*_part0000_*.fits")))[0]
    with afits.open(bhit) as hdul:
        assert hdul[0].header["BEAMINCN"]


def test_stokes_vis_effective_freq_is_weighted(ms_name, tmp_path):
    """freq_out is the weight-weighted mean over surviving channels, not the
    nominal band centre handed in by the driver (issue #296)."""
    ds = _run_stokes_vis(ms_name, str(tmp_path / "f0.scratch"), flag_chans=slice(4, 8))

    # stored WEIGHT is the natural weight (robust weights arrive in pass 2)
    w_chan = (ds.WEIGHT.values * ds.MASK.values[None]).sum(axis=(0, 1))
    expected = (w_chan * ds.FREQ.values).sum() / w_chan.sum()
    assert_allclose(ds.attrs["freq_out"], expected, rtol=1e-10)
    assert_allclose(ds.attrs["wsum_nat"], w_chan.sum(), rtol=1e-10)

    # nominal is the unweighted mean of all 8 channels (1.35 GHz); flagging the
    # top half must pull the effective frequency well below it
    assert_allclose(ds.attrs["freq_nominal"], 1.35e9, rtol=1e-6)
    assert ds.attrs["freq_out"] < ds.attrs["freq_nominal"] - 5.0e7


def test_stokes_vis_beam_follows_effective_freq(ms_name, tmp_path):
    """The effective frequency must reach the beam evaluation, not just the
    attrs. cell is enlarged for the same reason as
    test_stokes_vis_beam_on_image_grid: at the default 1e-5 rad cell katbeam is
    flat to ~1e-6 across the fov and any comparison is noise.
    """
    cell = 5.0e-4
    lo = _run_stokes_vis(
        ms_name, str(tmp_path / "lo.scratch"), beam_model="katbeam", cell_rad=cell, flag_chans=slice(4, 8)
    )
    hi = _run_stokes_vis(
        ms_name, str(tmp_path / "hi.scratch"), beam_model="katbeam", cell_rad=cell, flag_chans=slice(0, 4)
    )
    assert lo.attrs["freq_out"] < hi.attrs["freq_out"]
    assert not np.allclose(lo.BEAM.values, hi.BEAM.values, rtol=1e-3)
    # katbeam narrows with frequency, so the higher-frequency beam encloses
    # less total response over the same fov
    assert hi.BEAM.values.sum() < lo.BEAM.values.sum()


@pytest.mark.slow
def test_imager_effective_freq_uneven_bands(ms_name, tmp_path):
    """8 channels binned 3/3/2: the band-edge centres miss the true channel
    groups by 17-50 MHz. freq_out must track the channels actually gridded
    (issue #296); freq_nominal preserves the assignment grid.
    """
    outname = str(tmp_path / "eff_freq")
    imager_core(
        [Path(ms_name)],
        outname,
        integrations_per_image=15,
        channels_per_image=3,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    band_names = sorted(n for n in dt.children if n.startswith("band"))
    bands = sorted((dt[n].ds for n in band_names), key=lambda d: d.attrs["bandid"])
    assert len({b.attrs["bandid"] for b in bands}) == 3

    nominal = [b.attrs["freq_nominal"] for b in bands]
    effective = [b.attrs["freq_out"] for b in bands]

    # band_edges = linspace(0.95, 1.75, 4) GHz -> midpoints
    assert_allclose(sorted(set(nominal)), [1.0833333e9, 1.35e9, 1.6166667e9], rtol=1e-6)

    # the actual channel groups are 1.0-1.2, 1.3-1.5 and 1.6-1.7 GHz
    for eff, nom in zip(effective, nominal):
        lo, hi = {0: (1.0e9, 1.2e9), 1: (1.3e9, 1.5e9), 2: (1.6e9, 1.7e9)}[
            int(np.argmin(np.abs(np.array([1.0833333e9, 1.35e9, 1.6166667e9]) - nom)))
        ]
        assert lo <= eff <= hi, f"effective freq {eff} outside its channel group"
        assert abs(eff - nom) > 1.0e7

    # each partition carries its own effective frequency, and the band value is
    # the wsum-weighted mean over them
    for name in band_names:
        parts = [dt[name][p].ds for p in dt[name].children]
        w = np.array([np.asarray(p.attrs["wsum"]).sum() for p in parts])
        f = np.array([p.attrs["freq_out"] for p in parts])
        assert_allclose(dt[name].ds.attrs["freq_out"], (w * f).sum() / w.sum(), rtol=1e-10)


def test_gain_table_is_refused_before_ray_init(ms_name, tmp_path, monkeypatch):
    """--gain-table is refused, not silently ignored (#333).

    The MSv4 imager never carried the MSv2 path's gain application across, so a
    calibrated-imaging run produced uncalibrated images with nothing in the log
    to say so. Refusing is the interim contract until the gains are threaded
    into `stokes_vis`; the monkeypatch pins that it happens before a cluster is
    stood up, so the failure costs nothing.
    """
    import pfb_imaging.core.imager as imager_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)

    with pytest.raises(NotImplementedError, match="--gain-table"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "nope"),
            gain_table=[Path(tmp_path / "gains.qc")],
            overwrite=True,
            keep_ray_alive=True,
        )


def test_baseline_groups_refuses_non_meerkat_before_ray_init(ms_name, tmp_path, monkeypatch):
    """A non-MeerKAT array is rejected up front: we have no group beam for it.

    The monkeypatch is the real assertion -- the guard must fire before a Ray
    cluster is stood up, the way deconv's --psf guard does.
    """
    import pfb_imaging.core.imager as imager_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)

    with pytest.raises(ValueError, match="vla"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "nope"),
            baseline_groups=True,
            beam_model="L",
            overwrite=True,
            keep_ray_alive=True,
        )


def test_baseline_groups_without_beam_model_skips_the_telescope_guard(ms_name, tmp_path, monkeypatch):
    """With no beam model there is no group beam to get wrong, so the split is
    pure data partitioning and is telescope-agnostic (spec §4 ruling)."""
    import pfb_imaging.core.imager as imager_mod

    def _boom(*a, **kw):
        raise RuntimeError("reached init_ray")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)

    with pytest.raises(RuntimeError, match="reached init_ray"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "nope"),
            baseline_groups=True,
            antenna_groups="vla-0*,vla-[12]*",
            overwrite=True,
            keep_ray_alive=True,
        )


def test_baseline_groups_refuses_katbeam(ms_name, tmp_path, monkeypatch):
    """katbeam has no MeerKAT+ model and gives only power beams."""
    import pfb_imaging.core.imager as imager_mod

    monkeypatch.setattr(imager_mod, "init_ray", lambda *a, **kw: None)

    with pytest.raises(ValueError, match="katbeam"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "nope"),
            baseline_groups=True,
            beam_model="katbeam",
            overwrite=True,
            keep_ray_alive=True,
        )


def test_baseline_groups_refuses_non_l_band(ms_name, tmp_path, monkeypatch):
    """meerkat-beams serves groups for L band only (its design-decisions D15)."""
    import pfb_imaging.core.imager as imager_mod

    monkeypatch.setattr(imager_mod, "init_ray", lambda *a, **kw: None)

    with pytest.raises(ValueError, match="L band"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "nope"),
            baseline_groups=True,
            beam_model="UHF",
            overwrite=True,
            keep_ray_alive=True,
        )


def test_bad_antenna_groups_spec_is_refused_early(ms_name, tmp_path, monkeypatch):
    import pfb_imaging.core.imager as imager_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)

    with pytest.raises(ValueError, match="exactly two"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "nope"),
            baseline_groups=True,
            antenna_groups="m*",
            overwrite=True,
            keep_ray_alive=True,
        )


def _total_rows(dt_path):
    """Total vis rows across every partition of every band."""
    dt = xr.open_datatree(dt_path, engine="zarr", chunks=None)
    return sum(dt[b][p].ds.sizes["row"] for b in dt.children if b.startswith("band") for p in dt[b].children)


def _summed_band_products(dt_path):
    """Per-band summed DIRTY/PSF/WSUM, keyed by band node name."""
    dt = xr.open_datatree(dt_path, engine="zarr", chunks=None)
    out = {}
    for name in sorted(n for n in dt.children if n.startswith("band")):
        band = dt[name].ds
        out[name] = (band.DIRTY.values, band.PSF.values, band.WSUM.values)
    return out


@pytest.mark.slow
def test_baseline_groups_sum_to_the_ungrouped_image(ms_name, tmp_path):
    """Summing the three groups reproduces the ungrouped image exactly.

    Imaging weights are reduced per band, not per partition, so both runs see
    identical weights; gridding is then linear in rows, so the split is
    algebraically a no-op. The two patterns cover 10 and 17 of the MS's 27
    antennas -- exhaustive, no overlap.
    """
    common = dict(
        integrations_per_image=-1,
        channels_per_image=4,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
        # Pinned, not inherited from the default: the tolerance below is tied to
        # this value, and the default moved to 1e-5 in #340.
        epsilon=1e-7,
    )

    plain = str(tmp_path / "plain")
    imager_core([Path(ms_name)], plain, **common)

    grouped = str(tmp_path / "grouped")
    imager_core(
        [Path(ms_name)],
        grouped,
        baseline_groups=True,
        antenna_groups="vla-0*,vla-[12]*",
        **common,
    )

    # The equivalence below is vacuous unless the split actually happened: an
    # unsplit "grouped" run is the plain run, and would match trivially.
    gdt = xr.open_datatree(grouped + "_I.dt", engine="zarr", chunks=None)
    gband = gdt[sorted(n for n in gdt.children if n.startswith("band"))[0]]
    assert sorted(gband[p].ds.attrs["baseline_group"] for p in gband.children) == ["MM", "MPM", "MPMP"]

    # Rows are partitioned, not duplicated or dropped: the masks must index the
    # node's own baseline axis (Review Focus 3).
    assert _total_rows(grouped + "_I.dt") == _total_rows(plain + "_I.dt")

    ref = _summed_band_products(plain + "_I.dt")
    got = _summed_band_products(grouped + "_I.dt")
    assert set(ref) == set(got), "grouped run produced different band nodes"

    for name in ref:
        rd, rp, rw = ref[name]
        gd, gp, gw = got[name]
        # wsum is a plain sum of weights and must agree to roundoff.
        assert_allclose(gw, rw, rtol=1e-10, atol=0)
        # DIRTY/PSF are compared against the image peak, not per pixel: summing
        # three partial griddings is algebraically identical but not bitwise
        # associative, and a relative tolerance would be dominated by pixels
        # near zero. The bound is the wgridder's OWN accuracy target -- the
        # epsilon=1e-7 pinned above -- with a 10x margin; comparing tighter than
        # epsilon compares beyond what the gridder promises.
        # Measured worst-pixel agreement here: 7.8e-8 relative, 6.7e-8 of peak.
        assert_allclose(gd, rd, rtol=0, atol=1e-6 * np.abs(rd).max())
        assert_allclose(gp, rp, rtol=0, atol=1e-6 * np.abs(rp).max())


@pytest.mark.slow
def test_baseline_groups_write_three_partitions_with_the_right_row_counts(ms_name, tmp_path):
    """Three partitions per band, each sliced on its own node's baseline axis."""
    outname = str(tmp_path / "grouped")
    imager_core(
        [Path(ms_name)],
        outname,
        integrations_per_image=-1,
        channels_per_image=4,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
        baseline_groups=True,
        antenna_groups="vla-0*,vla-[12]*",
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    band = dt[sorted(n for n in dt.children if n.startswith("band"))[0]]

    labels = sorted(band[p].ds.attrs["baseline_group"] for p in band.children)
    assert labels == ["MM", "MPM", "MPMP"]

    # Every group must carry data. Exact row counts are NOT ntime*nbl: pass 1
    # drops fully-flagged rows, and flagging is not uniform across baselines.
    # Row conservation across the split is pinned in the equivalence test.
    rows = {band[p].ds.attrs["baseline_group"]: band[p].ds.sizes["row"] for p in band.children}
    assert set(rows) == {"MM", "MPM", "MPMP"}
    assert all(v > 0 for v in rows.values()), f"a group carries no rows: {rows}"
    # 10 vs 17 antennas: MM has the fewest baselines (45), MPM the most (170).
    assert rows["MM"] < rows["MPMP"] < rows["MPM"], f"group sizes look wrong: {rows}"

    # D46's diagnostic must reach the .dt, not die in the scratch store.
    for p in band.children:
        assert "beam_imre_ratio" in band[p].ds.attrs

    # Review Focus 2: no partition may be written with zero wsum -- the band
    # BEAM is a wsum-weighted mean and would divide by zero.
    for p in band.children:
        assert np.all(np.asarray(band[p].ds.attrs["wsum"]) > 0)
    assert np.isfinite(band.ds.BEAM.values).all()
    assert np.isfinite(band.ds.BDIRTY.values).all()


@pytest.mark.slow
def test_single_class_array_writes_one_partition_not_three(ms_name, tmp_path):
    """Review Focus 1: an empty group must not become an empty partition."""
    outname = str(tmp_path / "onegroup")
    imager_core(
        [Path(ms_name)],
        outname,
        integrations_per_image=-1,
        channels_per_image=4,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
        baseline_groups=True,
        antenna_groups="vla-*,nosuchantenna*",
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    band = dt[sorted(n for n in dt.children if n.startswith("band"))[0]]
    assert len(band.children) == 1
    assert band[next(iter(band.children))].ds.attrs["baseline_group"] == "MM"


def test_one_beam_wizard_is_built_per_baseline_group(ms_name, tmp_path, monkeypatch):
    """Three groups -> three wizards, each constructed with its own group label."""
    import pfb_imaging.core.imager as imager_mod

    built = []

    class _FakeWizard:
        def __init__(self, band=None, group=None):
            built.append((band, group))

    monkeypatch.setattr(imager_mod, "BeamWizard", _FakeWizard)
    monkeypatch.setattr(imager_mod, "check_telescope_is_meerkat", lambda *a, **kw: None)

    def _stop(*a, **kw):
        raise RuntimeError("stop after beam construction")

    # zarr.open_group is the first call after the beam block; set_image_size
    # runs BEFORE it and would stop too early to observe the wizards.
    monkeypatch.setattr(imager_mod.zarr, "open_group", _stop)

    with pytest.raises(RuntimeError, match="stop after beam construction"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "beams"),
            baseline_groups=True,
            antenna_groups="vla-0*,vla-[12]*",
            beam_model="L",
            field_of_view=1.0,
            overwrite=True,
            keep_ray_alive=True,
        )

    assert sorted(g for _, g in built) == ["MM", "MPM", "MPMP"]
    assert {b for b, _ in built} == {"L"}


def test_partition_fits_names_carry_the_group_when_split(tmp_path):
    """With grouping on, three partitions per field would otherwise collide."""
    from pfb_imaging.core.imager import _partition_fits

    nx = ny = 8
    prod = {
        "DIRTY": np.ones((1, ny, nx)),
        "WSUM": np.ones(1),
        "BEAM": np.ones((1, ny, nx)),
    }
    meta = {"ra": 0.1, "dec": -0.5, "time_out": 1.6e9, "l0": 0.0, "m0": 0.0}

    _partition_fits(
        str(tmp_path),
        "band0000_time0000",
        2,
        "FIELD_A",
        "MPM",
        prod,
        meta,
        1.4e9,
        1e-6,
        do_psf=False,
        do_beam=True,
    )
    assert glob.glob(str(tmp_path / "dirty_band0000_time0000_part0002_FIELD_A_MPM.fits"))


def test_partition_fits_names_are_unchanged_when_not_split(tmp_path):
    """baseline_group 'all' must not rename today's output."""
    from pfb_imaging.core.imager import _partition_fits

    nx = ny = 8
    prod = {
        "DIRTY": np.ones((1, ny, nx)),
        "WSUM": np.ones(1),
        "BEAM": np.ones((1, ny, nx)),
    }
    meta = {"ra": 0.1, "dec": -0.5, "time_out": 1.6e9, "l0": 0.0, "m0": 0.0}

    _partition_fits(
        str(tmp_path),
        "band0000_time0000",
        2,
        "FIELD_A",
        "all",
        prod,
        meta,
        1.4e9,
        1e-6,
        do_psf=False,
        do_beam=True,
    )
    assert glob.glob(str(tmp_path / "dirty_band0000_time0000_part0002_FIELD_A.fits"))


def test_concat_pieces_reports_the_worst_beam_imre_ratio():
    """The Re(B) diagnostic must survive the piece reduction as a max.

    _concat_pieces keeps piece 0's attrs, so a naive reduction would report
    whichever piece happened to be first rather than the worst one (wiki D46).
    """
    from pfb_imaging.core.imager import _concat_pieces

    def _piece(ratio, wsum):
        return xr.Dataset(
            {
                "VIS": (("corr", "row", "chan"), np.ones((1, 2, 1), dtype=np.complex64)),
                "WEIGHT": (("corr", "row", "chan"), np.ones((1, 2, 1), dtype=np.float32)),
                "MASK": (("row", "chan"), np.ones((2, 1), dtype=np.uint8)),
                "UVW": (("row", "three"), np.zeros((2, 3))),
                "FREQ": (("chan",), np.array([1.4e9])),
                "BEAM": (("corr", "y", "x"), np.ones((1, 2, 2), dtype=np.float32)),
            },
            attrs={"wsum_nat": wsum, "freq_out": 1.4e9, "beam_imre_ratio": ratio},
        )

    part = _concat_pieces([_piece(0.01, 1.0), _piece(0.25, 1.0), _piece(0.05, 1.0)])
    assert part.attrs["beam_imre_ratio"] == pytest.approx(0.25)


def test_antenna_groups_without_baseline_groups_is_refused(ms_name, tmp_path, monkeypatch):
    """An ignored option must say so rather than silently doing nothing."""
    import pfb_imaging.core.imager as imager_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)

    with pytest.raises(ValueError, match="--baseline-groups"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "nope"),
            antenna_groups="m*,e*",
            overwrite=True,
            keep_ray_alive=True,
        )


def test_only_the_groups_present_in_the_data_get_a_wizard(ms_name, tmp_path, monkeypatch):
    """Building a wizard for an absent group would stage beams nothing uses.

    The MdV-2026 group products need hand-staging, so an eager wizard for a
    group with no baselines kills the run in meerkat_beams.cache for a beam no
    task would ever consume.
    """
    import pfb_imaging.core.imager as imager_mod

    built = []

    class _FakeWizard:
        def __init__(self, band=None, group=None):
            built.append((band, group))

    monkeypatch.setattr(imager_mod, "BeamWizard", _FakeWizard)
    monkeypatch.setattr(imager_mod, "check_telescope_is_meerkat", lambda *a, **kw: None)

    def _stop(*a, **kw):
        raise RuntimeError("stop after beam construction")

    monkeypatch.setattr(imager_mod.zarr, "open_group", _stop)

    with pytest.raises(RuntimeError, match="stop after beam construction"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "onegroup"),
            baseline_groups=True,
            antenna_groups="vla-*,nosuchantenna*",
            beam_model="L",
            field_of_view=1.0,
            overwrite=True,
            keep_ray_alive=True,
        )

    assert [g for _, g in built] == ["MM"], f"built wizards for absent groups: {built}"


@pytest.mark.slow
def test_real_group_beams_are_complex_only_for_the_cross_group():
    """D46 rests on MPM being complex and MM/MPMP being real. Check that against
    the actual meerkat-beams group datasets rather than a mock.

    The MdV-2026 products are unpublished (placeholder gdrive IDs in
    meerkat_beams.cache), so they cannot be downloaded and CI never runs this --
    it skips unless they have been staged by hand into MBEAMS_CACHE_DIR, which
    pfb points at /tmp/mbeams-cache-<uid> (see pfb_imaging/__init__.py), NOT at
    meerkat-beams' own ~/.cache default.

    Measured on staged products at 1.28 GHz: max|Im|/max|Re| = 2.1e-2.
    """
    from pathlib import Path as _Path

    cache = pytest.importorskip("meerkat_beams.cache")
    for product in ("MeerKAT_L_mdv2026", "MKE_L"):
        if not _Path(cache.bds_path_for_product(product)).exists():
            pytest.skip(f"group beam product {product!r} not staged under {cache.cache_root()}")

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from meerkat_beams.utils import BeamWizard

    from pfb_imaging.utils.stokes2vis_msv4 import real_beam_maps

    lm = np.linspace(-1.0, 1.0, 32)
    times = Time(np.array([60000.0]), format="mjd")
    ratios = {}
    for group in ("MM", "MPM", "MPMP"):
        bw = BeamWizard(band="L", group=group)
        bw.set_field_centre(SkyCoord(ra=0.0 * u.rad, dec=-0.5 * u.rad))
        bmap, _ = bw.get_rotation_averaged_beam(
            l=lm,
            m=lm,
            times=times,
            freq=np.atleast_1d(1.28e9),
            time_stepping=1,
            pixel_stepping=1,
            var="nstokes",
            i="I",
            j="I",
            verbose=0,
        )
        maps, ratio = real_beam_maps([bmap])
        assert maps.dtype == np.float64
        assert np.isfinite(maps).all()
        ratios[group] = ratio

    assert ratios["MM"] == 0.0, "MeerKAT-MeerKAT beam should be real"
    assert ratios["MPMP"] == 0.0, "MeerKAT+-MeerKAT+ beam should be real"
    # the cross group has no single Jones matrix, so its Stokes beam is complex
    assert 0.0 < ratios["MPM"] < 0.1, f"unexpected cross-group |Im|/|Re|: {ratios['MPM']}"


@pytest.mark.slow
def test_pass2_returns_no_images_and_mfs_beam_matches_dt(ms_name, tmp_path, monkeypatch):
    """Pass-2 results stay light, and the MFS beam is fitted to the .dt's PSFs (#339).

    Image-sized task returns sit in the Ray object store and are counted in
    every process that touched them, so the driver reads band PSFs from the
    .dt instead. The FITS assertion pins the MFS beam to that source.
    """
    import ray
    from astropy.io import fits as afits

    import pfb_imaging.core.imager as core_imager
    from pfb_imaging.utils.misc import fitcleanbeam

    results = []
    real_get = ray.get

    def spy(refs, *args, **kwargs):
        out = real_get(refs, *args, **kwargs)
        if isinstance(out, dict) and "timeid" in out:
            results.append(out)
        return out

    monkeypatch.setattr(core_imager.ray, "get", spy)
    outname = str(tmp_path / "img")
    imager_core(
        [Path(ms_name)],
        outname,
        channels_per_image=2,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        fits_mfs=True,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    assert results, "spy saw no pass-2 results"
    for res in results:
        big = [k for k, v in res.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
        assert not big, f"pass-2 result carries image arrays {big}"

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    bands = [b for b in dt.children if b.startswith("band")]
    psf = sum(dt[b].ds.PSF.values for b in bands)
    wsum = sum(dt[b].ds.WSUM.values for b in bands)
    want = fitcleanbeam(psf / wsum[:, None, None], yx_order=True)[0]
    cell_deg = np.rad2deg(dt.attrs["cell_rad"])
    (mfs,) = glob.glob(str(tmp_path / "fits" / "*dirty*mfs.fits"))
    hdr = afits.getheader(mfs)
    assert_allclose([hdr["BMAJ"], hdr["BMIN"]], want[:2] * cell_deg, rtol=1e-6)
