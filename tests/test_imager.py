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


def _open_first_vis_node(ms_name):
    from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

    from pfb_imaging.core.imager import get_engine

    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    for node in dt.children.values():
        if node.attrs.get("type") in VISIBILITY_XDS_TYPES:
            return node
    raise RuntimeError("no visibility node in test MS")


def _run_stokes_vis(ms_name, scratch, radec_new=None, beam_model=None, cell_rad=1e-5):
    """Drive stokes_vis directly (no Ray) on the test MS's first node."""
    from pfb_imaging.utils.stokes2vis_msv4 import stokes_vis

    node = _open_first_vis_node(ms_name)
    dc1 = node.ds.attrs["data_groups"]["base"]["correlated_data"]
    freq = node.ds.frequency.values
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
        freq_out=float(freq.mean()),
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


def test_stokes_vis_rephases_to_new_centre(sky_truth, ms_name, tmp_path):
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


def test_imager_rephase_roundtrip(sky_truth, ms_name, tmp_path):
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
    fits_files = glob.glob(str(tmp_path / "*rephased*dirty*mfs.fits"))
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


def test_stokes_vis_beam_on_image_grid(ms_name, tmp_path):
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
