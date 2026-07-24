"""Tests for the hci sub-command.

FITS-header unit tests plus in-process runs of the Ray-distributed
batch_stokes_image/stokes_image path on the small test MS: channels-per-bin
invariance, output structure, transient injection, and a weighting-mode
smoke test (minvar is the mode that exercised the weight_data recompile bug
in production, issue #273).
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from africanus.coordinates import radec_to_lm
from astropy.io import fits
from daskms import xds_from_table
from numpy.testing import assert_allclose

from pfb_imaging.core.hci import (
    _FITS_AXIS_KEYWORDS,
    _STOKES_FITS_INDEX,
    _make_fits_header,
)
from pfb_imaging.core.hci import (
    hci as hci_core,
)
from pfb_imaging.utils.misc import set_image_size
from pfb_imaging.utils.transients import generate_transient_spectra

pmp = pytest.mark.parametrize


def _lm_to_radec(lcoord, mcoord, ra0, dec0):
    """Invert the SIN (orthographic) projection.

    Args:
        lcoord, mcoord: direction cosines relative to the phase centre.
        ra0, dec0: phase-centre RA/Dec in radians.

    Returns:
        (ra, dec) of the (l, m) point in radians.
    """
    n = np.sqrt(1.0 - lcoord**2 - mcoord**2)
    dec = np.arcsin(mcoord * np.cos(dec0) + n * np.sin(dec0))
    ra = ra0 + np.arctan2(lcoord, n * np.cos(dec0) - mcoord * np.sin(dec0))
    return ra, dec


def _make_base_fits_header():
    """Build a 5-axis FITS header mirroring the layout stored by hci.

    Axis order matches make_dummy_dataset: (X, Y, TIME/INTEGRATION, FREQ, STOKES).
    """
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 5
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    hdr["NAXIS3"] = 5
    hdr["NAXIS4"] = 3
    hdr["NAXIS5"] = 1
    hdr["EXTEND"] = True
    hdr["BSCALE"] = 1.0
    hdr["BZERO"] = 0.0
    hdr["BUNIT"] = "Jy/beam"
    hdr["EQUINOX"] = 2000.0
    hdr["BTYPE"] = "Intensity"
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CTYPE3"] = "INTEGRATION"
    hdr["CTYPE4"] = "FREQ"
    hdr["CTYPE5"] = "STOKES"
    hdr["CRVAL1"] = 10.0
    hdr["CRVAL2"] = -30.0
    hdr["CRVAL3"] = 0.0
    hdr["CRVAL4"] = 1.0e9
    hdr["CRVAL5"] = 1
    hdr["CRPIX1"] = 33
    hdr["CRPIX2"] = 33
    hdr["CRPIX3"] = 1
    hdr["CRPIX4"] = 1
    hdr["CRPIX5"] = 1
    hdr["CDELT1"] = -1e-5
    hdr["CDELT2"] = 1e-5
    hdr["CDELT3"] = 60.0
    hdr["CDELT4"] = 1e6
    hdr["CDELT5"] = 1
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CUNIT3"] = "s"
    hdr["CUNIT4"] = "Hz"
    hdr["CUNIT5"] = ""
    hdr["WCSAXES"] = 5
    return dict(hdr)


def test_stokes_fits_index_codes():
    """The module-level Stokes -> FITS code mapping must match the FITS standard."""
    assert _STOKES_FITS_INDEX == {"I": 1, "Q": 2, "U": 3, "V": 4}


def test_fits_axis_keywords_complete():
    """All axis-numbered FITS keywords we care about are present in the rewrite set."""
    for kw in ("CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT", "NAXIS"):
        assert kw in _FITS_AXIS_KEYWORDS


def test_make_fits_header_drop_time_axis():
    """Dropping the TIME axis (FITS axis 3) yields a 4-axis (X, Y, FREQ, STOKES) header.

    The FREQ axis must be renumbered to axis 3 and STOKES to axis 4, with
    mandatory keywords appearing first as required by StreamingHDU.
    """
    base = _make_base_fits_header()
    stokes = ["I"]
    # numpy shape (STOKES, FREQ, Y, X) -> FITS axes (X=1, Y=2, FREQ=3, STOKES=4)
    shape = (1, 3, 64, 64)
    hdr = _make_fits_header(base, shape, stokes, drop_axes={3})

    assert hdr["NAXIS"] == 4
    assert hdr["NAXIS1"] == 64
    assert hdr["NAXIS2"] == 64
    assert hdr["NAXIS3"] == 3
    assert hdr["NAXIS4"] == 1
    assert hdr["CTYPE1"] == "RA---SIN"
    assert hdr["CTYPE2"] == "DEC--SIN"
    assert hdr["CTYPE3"] == "FREQ"
    assert hdr["CTYPE4"] == "STOKES"
    assert hdr["CRVAL3"] == 1.0e9
    assert hdr["CDELT3"] == 1e6
    # Stokes axis values come from the FITS code table, not the base header
    assert hdr["CRVAL4"] == 1
    assert hdr["CDELT4"] == 1
    assert hdr["CUNIT4"] == ""
    # mandatory keywords appear first in key order (StreamingHDU requirement)
    keys = list(hdr.keys())
    expected_prefix = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4"]
    assert keys[: len(expected_prefix)] == expected_prefix
    assert hdr["WCSAXES"] == 4
    # the old axis-5 (STOKES) keywords must not leak through
    for kw in ("CTYPE5", "CRVAL5", "CRPIX5", "CDELT5", "CUNIT5", "NAXIS5"):
        assert kw not in hdr


def test_make_fits_header_drop_freq_axis():
    """Dropping the FREQ axis (FITS axis 4) yields a 4-axis (X, Y, TIME, STOKES) header."""
    base = _make_base_fits_header()
    # numpy shape (STOKES, TIME, Y, X)
    shape = (1, 5, 64, 64)
    hdr = _make_fits_header(base, shape, ["I"], drop_axes={4})

    assert hdr["NAXIS"] == 4
    assert hdr["NAXIS3"] == 5
    assert hdr["CTYPE3"] == "INTEGRATION"
    assert hdr["CRVAL3"] == 0.0
    assert hdr["CDELT3"] == 60.0
    assert hdr["CTYPE4"] == "STOKES"


@pmp(
    "stokes,expected_crval,expected_cdelt",
    [
        (["I"], 1, 1),
        (["Q", "U"], 2, 1),
        (["I", "Q", "U", "V"], 1, 1),
    ],
)
def test_make_fits_header_stokes_axis(stokes, expected_crval, expected_cdelt):
    """The STOKES axis CRVAL/CDELT are derived from the Stokes code table."""
    base = _make_base_fits_header()
    n_stokes = len(stokes)
    shape = (n_stokes, 3, 64, 64)
    hdr = _make_fits_header(base, shape, stokes, drop_axes={3})

    assert hdr["NAXIS4"] == n_stokes
    assert hdr["CTYPE4"] == "STOKES"
    assert hdr["CRVAL4"] == expected_crval
    assert hdr["CDELT4"] == expected_cdelt


def test_make_fits_header_does_not_mutate_input():
    """_make_fits_header must not mutate the caller's dict."""
    base = _make_base_fits_header()
    snapshot = dict(base)
    _make_fits_header(base, (1, 3, 64, 64), ["I"], drop_axes={3})
    assert base == snapshot


def test_hci_channels_per_bin_invariance_no_beam(ms_name, tmp_path):
    """channels_per_bin must not change hci output when no beam model is supplied.

    With channels_per_image=-1 (a single image per spw) and no beam correction,
    the per-bin sub-division is only relevant for beam application and
    deconvolution. The final cube is a weight-normalised sum over bins of
    linear wgridder outputs, which telescopes back to the single-bin result.
    We therefore expect cube, cube_mean and weight to be numerically
    equivalent across channels_per_bin=-1 and channels_per_bin=4.
    """
    out_full = str(tmp_path / "hci_full.zarr")
    out_bin4 = str(tmp_path / "hci_bin4.zarr")

    common_kwargs = dict(
        product="I",
        channels_per_image=-1,
        integrations_per_image=10,
        images_per_chunk=6,
        max_simul_chunks=1,
        field_of_view=0.5,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        beam_model=None,
        robustness=0.0,
        epsilon=1e-7,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    hci_core([ms_name], out_full, channels_per_bin=-1, **common_kwargs)
    hci_core([ms_name], out_bin4, channels_per_bin=4, **common_kwargs)

    ds_full = xr.open_zarr(out_full)
    ds_bin4 = xr.open_zarr(out_bin4)

    # output geometry must match so we can compare element-wise
    assert ds_full.cube.shape == ds_bin4.cube.shape
    assert ds_full.cube_mean.shape == ds_bin4.cube_mean.shape
    assert ds_full.weight.shape == ds_bin4.weight.shape

    cube_full = ds_full.cube.values
    cube_bin4 = ds_bin4.cube.values
    mean_full = ds_full.cube_mean.values
    mean_bin4 = ds_bin4.cube_mean.values
    weight_full = ds_full.weight.values
    weight_bin4 = ds_bin4.weight.values

    # sanity: the run actually produced data
    assert np.any(weight_full > 0)
    assert np.any(weight_bin4 > 0)

    # weights telescope exactly: sum_b max(PSF_b) at the central pixel equals
    # max(PSF_full) since each band's PSF max sits at the same pixel.
    assert_allclose(weight_bin4, weight_full, rtol=1e-6, atol=1e-6)

    cube_scale = float(max(np.abs(cube_full).max(), np.abs(cube_bin4).max(), 1.0))
    mean_scale = float(max(np.abs(mean_full).max(), np.abs(mean_bin4).max(), 1.0))
    assert_allclose(cube_bin4, cube_full, rtol=1e-5, atol=1e-5 * cube_scale)
    assert_allclose(mean_bin4, mean_full, rtol=1e-5, atol=1e-5 * mean_scale)


def test_hci_produces_expected_output_structure(ms_name, tmp_path):
    """hci writes a zarr dataset with the documented variables and coords."""
    out = str(tmp_path / "hci_structure.zarr")
    hci_core(
        [ms_name],
        out,
        product="I",
        channels_per_image=-1,
        integrations_per_image=20,
        images_per_chunk=3,
        max_simul_chunks=1,
        field_of_view=0.5,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        robustness=0.0,
        epsilon=1e-7,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    ds = xr.open_zarr(out)
    for name in ("cube", "cube_mean", "rms", "weight", "nonzero", "flag", "channel_width"):
        assert name in ds, f"{name} missing from hci output"
    assert list(ds.coords["STOKES"].values) == ["I"]
    assert ds.cube.dims == ("STOKES", "FREQ", "TIME", "Y", "X")
    assert ds.cube_mean.dims == ("STOKES", "FREQ", "Y", "X")
    # single spw + channels_per_image=-1 -> single frequency bin in the cube
    assert ds.sizes["FREQ"] == 1
    # 60 integrations / integrations_per_image=20 -> 3 time bins
    assert ds.sizes["TIME"] == 3
    # channel_width should be populated with the per-band bandwidth
    assert np.all(ds.channel_width.values > 0)


def test_hci_rejects_existing_output_without_overwrite(ms_name, tmp_path):
    """hci must refuse to clobber an existing output unless overwrite=True."""
    out = str(tmp_path / "hci_exists.zarr")
    Path(out).mkdir()  # pretend a previous run left its output in place
    # write a marker file so we can tell if the directory was removed
    (Path(out) / "marker").write_text("do not delete")

    with pytest.raises(RuntimeError, match="exists"):
        hci_core(
            [ms_name],
            out,
            product="I",
            channels_per_image=-1,
            integrations_per_image=60,
            images_per_chunk=1,
            max_simul_chunks=1,
            field_of_view=0.5,
            super_resolution_factor=2.0,
            nworkers=1,
            nthreads=1,
            overwrite=False,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )

    assert (Path(out) / "marker").exists()


def test_hci_inject_transients(ms_name, ms_meta, tmp_path):
    """Injected transient lands at the expected pixel with the expected dynamic spectrum.

    The base visibilities are zeroed via data_column="DATA-DATA" so the cube
    contains only the injected transient. The transient is placed on an exact
    pixel centre by choosing integer (l, m) offsets and inverting the SIN
    projection, with natural weighting so the per-bin flux is analytic.

    The analytic expectation assumes every sample participates, so clear any
    persistent flags another test's fixture left in the shared MS (sky_truth
    writes ~10% flags plus a dead channel, which shifts the per-bin
    normalisation by ~0.1% and breaks the tight tolerances here).
    """
    import dask
    import dask.array as da
    from daskms import xds_from_ms, xds_to_table

    xds0 = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1, "corr": -1})[0]
    nrow_, nchan_, ncorr_ = xds0.FLAG.shape
    xds0 = xds0.assign(
        FLAG=(("row", "chan", "corr"), da.zeros((nrow_, nchan_, ncorr_), dtype=bool, chunks=(-1, -1, -1))),
        FLAG_ROW=(("row",), da.zeros(nrow_, dtype=bool, chunks=-1)),
    )
    dask.compute(xds_to_table(xds0, ms_name, columns=["FLAG", "FLAG_ROW"]))

    out = str(tmp_path / "hci_transients.zarr")

    # --- geometry: must match what hci computes internally ---
    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, 0.5, 2.0)

    # phase centre of the test field (radians)
    field = xds_from_table(f"{ms_name}::FIELD")[0]
    ra0, dec0 = (float(v) for v in field.PHASE_DIR.values.squeeze())

    # place the transient an integer number of pixels off centre (well inside
    # the ~0.257 deg half-FOV) so it sits exactly on a pixel centre.
    di, dj = -20, 16  # X (l) and Y (m) pixel offsets from the central pixel
    lcoord = di * cell_rad
    mcoord = dj * cell_rad
    ra, dec = _lm_to_radec(lcoord, mcoord, ra0, dec0)

    # guard the inversion against the projection the code actually uses
    lm_check = radec_to_lm(np.array([[ra, dec]]), np.array([ra0, dec0])).squeeze()
    assert_allclose(lm_check, [lcoord, mcoord], atol=1e-12)

    # --- transient config written into tmp_path (sidecar zarr stays out of repo) ---
    transient = {
        "name": "test_transient",
        "time": {"peak_time": 1200.0, "duration": 600.0, "shape": "gaussian"},
        "frequency": {"peak_flux": 2.0, "reference_freq": 1.4e9, "spectral_index": -1.5},
        "position": {"ra": float(np.rad2deg(ra)), "dec": float(np.rad2deg(dec))},
    }
    config_path = tmp_path / "transient.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"transients": [transient]}, f)

    ipi = 4  # integrations per image -> 60 / 4 = 15 time bins
    cpi = 2  # channels per image    -> 8  / 2 = 4  freq bins
    hci_core(
        [ms_name],
        out,
        product="I",
        data_column="DATA-DATA",
        inject_transients=str(config_path),
        channels_per_image=cpi,
        channels_per_bin=-1,
        integrations_per_image=ipi,
        images_per_chunk=15,
        max_simul_chunks=1,
        field_of_view=0.5,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        beam_model=None,
        robustness=None,
        epsilon=1e-7,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    ds = xr.open_zarr(out)
    assert ds.sizes["FREQ"] == ms_meta.nchan // cpi
    assert ds.sizes["TIME"] == ms_meta.ntime // ipi

    # The transient was injected at direction cosines (l, m) = (di, dj) * cell_rad.
    # The wgridder places this on an exact pixel centre offset from the central
    # pixel (nx//2, ny//2) by -di in X and +dj in Y (flip_v convention).
    # We predict from (l, m) rather than from ds.X/ds.Y because the cube's RA
    # labels are linear in RA and omit the cos(dec) projection factor, so they
    # disagree with the gridded position by ~1/cos(dec) for an l-offset at dec!=0.
    ix = nx // 2 - di
    iy = ny // 2 + dj

    # expected time/freq profiles, evaluated on the MS sampling
    tprofile, fprofile = generate_transient_spectra(ms_meta.utime, ms_meta.freq, transient)
    n_tbin = ms_meta.ntime // ipi
    n_fbin = ms_meta.nchan // cpi
    expected_t = tprofile.reshape(n_tbin, ipi).mean(axis=1)
    expected_f = fprofile.reshape(n_fbin, cpi).mean(axis=1)

    # position: argmax of the brightest (time, freq) slice must be the predicted pixel
    t_peak = int(np.argmax(expected_t))
    f_peak = int(np.argmax(expected_f))
    img = ds.cube.values[0, f_peak, t_peak]  # (Y, X)
    peak_iy, peak_ix = np.unravel_index(np.argmax(img), img.shape)
    assert (int(peak_iy), int(peak_ix)) == (iy, ix)

    # verify our contiguous-block binning matches the cube's coordinates
    assert_allclose(ds.TIME.values, ms_meta.utime.reshape(n_tbin, ipi).mean(axis=1), atol=1e-6)
    assert_allclose(ds.FREQ.values, ms_meta.freq.reshape(n_fbin, cpi).mean(axis=1), atol=1e-3)

    # flux at the source pixel as a function of (freq, time). For a source on an
    # exact pixel centre under natural weighting, the PSF-peak normalisation
    # recovers the injected flux directly, so the cube equals the binned dynamic
    # spectrum. Observed agreement is <0.1%; 1% leaves margin for float32 cube
    # storage and gridder (epsilon) numerics across platforms.
    src = ds.cube.values[0, :, :, iy, ix]  # (FREQ, TIME)
    expected_ft = expected_f[:, None] * expected_t[None, :]  # (FREQ, TIME)
    peak = float(expected_ft.max())

    assert_allclose(src, expected_ft, rtol=1e-2, atol=1e-2 * peak)


def test_hci_inject_transients_location_vs_distance(ms_name, ms_meta, tmp_path):
    """Injected sources stay at their true sky coordinate out to the field edge.

    Reproduces the controlled experiment in ratt-ru/breifast#263 (sources placed
    radially at increasing angular distance from the phase centre, all other
    parameters fixed) on the small test MS, single field / no rephasing.

    For each source we check two things through the *cube's own SIN WCS* (the
    same header breifast reads), so the test catches both a pixel-placement
    error and a coordinate-labelling error, and would expose either one growing
    with distance:

    1. the recovered peak lands on the predicted pixel, and
    2. that pixel maps back (via the RA---SIN/DEC--SIN WCS) to the injected
       ``(ra, dec)`` on the sphere, to well under a pixel.

    The base visibilities are zeroed (``DATA-DATA``) so only the injected
    sources are imaged, and any stale flags in the shared MS are cleared so the
    per-sample analytic expectation holds.
    """
    import dask
    import dask.array as da
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from daskms import xds_from_ms, xds_to_table

    xds0 = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1, "corr": -1})[0]
    nrow_, nchan_, ncorr_ = xds0.FLAG.shape
    xds0 = xds0.assign(
        FLAG=(("row", "chan", "corr"), da.zeros((nrow_, nchan_, ncorr_), dtype=bool, chunks=(-1, -1, -1))),
        FLAG_ROW=(("row",), da.zeros(nrow_, dtype=bool, chunks=-1)),
    )
    dask.compute(xds_to_table(xds0, ms_name, columns=["FLAG", "FLAG_ROW"]))

    out = str(tmp_path / "hci_location.zarr")

    # wide field so the sources span a real range of angular distances
    fov = 1.0
    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, fov, 2.0)

    field = xds_from_table(f"{ms_name}::FIELD")[0]
    ra0, dec0 = (float(v) for v in field.PHASE_DIR.values.squeeze())

    # sources on a diagonal ray at increasing distance from the phase centre,
    # each on an exact pixel centre (integer (l, m) offsets -> exact SIN radec).
    offsets = [(20, 15), (60, 45), (100, 75), (140, 105), (170, 128)]
    injected = []  # (di, dj, ra_rad, dec_rad)
    transients = []
    for k, (di, dj) in enumerate(offsets):
        ra, dec = _lm_to_radec(di * cell_rad, dj * cell_rad, ra0, dec0)
        injected.append((di, dj, ra, dec))
        transients.append(
            {
                "name": f"src{k}",
                "time": {"peak_time": 1800.0, "duration": 3000.0, "shape": "gaussian"},
                "frequency": {"peak_flux": 2.0, "reference_freq": 1.4e9, "spectral_index": 0.0},
                "position": {"ra": float(np.rad2deg(ra)), "dec": float(np.rad2deg(dec))},
            }
        )
    config_path = tmp_path / "transients_location.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"transients": transients}, f)

    hci_core(
        [ms_name],
        out,
        product="I",
        data_column="DATA-DATA",
        inject_transients=str(config_path),
        channels_per_image=-1,
        channels_per_bin=-1,
        integrations_per_image=ms_meta.ntime,  # single time bin
        images_per_chunk=1,
        max_simul_chunks=1,
        field_of_view=fov,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        beam_model=None,
        robustness=None,
        epsilon=1e-8,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    ds = xr.open_zarr(out)
    img = ds.cube.values[0, 0, 0]  # (Y, X)

    # rebuild the celestial WCS from the header hci stored in the cube attrs
    hdr = fits.Header()
    for key, val in ds.attrs["fits_header"]:
        hdr[key] = val
    wcs = WCS(hdr).celestial

    cell_deg = np.rad2deg(cell_rad)
    for di, dj, ra_inj, dec_inj in injected:
        ix_pred = nx // 2 - di
        iy_pred = ny // 2 + dj
        lo_y, lo_x = max(0, iy_pred - 6), max(0, ix_pred - 6)
        win = img[lo_y : iy_pred + 7, lo_x : ix_pred + 7]
        ly, lx = np.unravel_index(np.argmax(win), win.shape)
        iy_f, ix_f = lo_y + ly, lo_x + lx

        # 1. pixel placement
        pix_err = np.hypot(ix_f - ix_pred, iy_f - iy_pred)
        assert pix_err <= 1.0, f"source ({di},{dj}) landed {pix_err:.1f} px from prediction"

        # 2. cube WCS at the recovered pixel must match the injected sky position
        world = wcs.pixel_to_world(ix_f, iy_f)
        injected_coord = SkyCoord(ra_inj * u.rad, dec_inj * u.rad, frame="fk5")
        sep_arcsec = world.separation(injected_coord).to_value("arcsec")
        # exact-pixel sources: the WCS should reproduce the injected radec to a
        # small fraction of a pixel (cell ~ 8.8 arcsec here); 0.2 px is generous.
        assert sep_arcsec <= 0.2 * cell_deg * 3600.0, (
            f"source ({di},{dj}) at distance "
            f"{np.hypot(di, dj) * cell_deg:.3f} deg: cube WCS position is "
            f"{sep_arcsec:.2f} arcsec from the injected coordinate"
        )


def test_hci_inject_transients_rephased(ms_name, ms_meta, tmp_path):
    """Injected sources land at their true position when the image is rephased.

    Exercises the mosaic path (``--phase-dir`` set to a tangent point that
    differs from the MS field centre). The transient is built in the original
    frame and carried to the rephased frame with the chgcentre w-difference; a
    sign error there displaced every injected source by a constant
    ``-2 * (field -> tangent)`` translation, which in a real mosaic grows with
    each field's distance from the common tangent (ratt-ru/breifast#263).

    Sources are injected at increasing offsets *about the rephasing centre*, and
    each must reappear at its predicted pixel there (the image is centred on the
    rephasing centre). Before the fix the central source alone landed ~90 px
    away; here we require sub-pixel accuracy at every offset.
    """
    import dask
    import dask.array as da
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from daskms import xds_from_ms, xds_to_table

    xds0 = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1, "corr": -1})[0]
    nrow_, nchan_, ncorr_ = xds0.FLAG.shape
    xds0 = xds0.assign(
        FLAG=(("row", "chan", "corr"), da.zeros((nrow_, nchan_, ncorr_), dtype=bool, chunks=(-1, -1, -1))),
        FLAG_ROW=(("row",), da.zeros(nrow_, dtype=bool, chunks=-1)),
    )
    dask.compute(xds_to_table(xds0, ms_name, columns=["FLAG", "FLAG_ROW"]))

    out = str(tmp_path / "hci_rephased.zarr")

    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, 0.5, 2.0)

    field = xds_from_table(f"{ms_name}::FIELD")[0]
    ra0, dec0 = (float(v) for v in field.PHASE_DIR.values.squeeze())

    # rephasing centre ~0.1 deg from the field centre so w_diff != 0; round-trip
    # the phase-dir string through SkyCoord exactly as hci parses it so our
    # predicted pixels use the same centre the workers do.
    ra_c = ra0 + np.deg2rad(0.1) / np.cos(dec0)
    dec_c = dec0 + np.deg2rad(0.05)
    cc = SkyCoord(ra_c * u.rad, dec_c * u.rad, frame="fk5")
    phase_dir = f"{cc.ra.to_string(unit=u.hourangle, sep=':')},{cc.dec.to_string(unit=u.deg, sep=':')}"
    c2 = SkyCoord(*phase_dir.split(","), frame="fk5", unit=(u.hourangle, u.deg))
    ra_c = np.deg2rad(c2.ra.value)
    dec_c = np.deg2rad(c2.dec.value)

    offsets = [(0, 0), (-25, -18), (25, 18), (-30, 25)]
    transients = []
    for k, (di, dj) in enumerate(offsets):
        ra, dec = _lm_to_radec(di * cell_rad, dj * cell_rad, ra_c, dec_c)
        transients.append(
            {
                "name": f"src{k}",
                "time": {"peak_time": 1800.0, "duration": 3000.0, "shape": "gaussian"},
                "frequency": {"peak_flux": 2.0, "reference_freq": 1.4e9, "spectral_index": 0.0},
                "position": {"ra": float(np.rad2deg(ra)), "dec": float(np.rad2deg(dec))},
            }
        )
    config_path = tmp_path / "transients_rephased.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"transients": transients}, f)

    hci_core(
        [ms_name],
        out,
        product="I",
        data_column="DATA-DATA",
        inject_transients=str(config_path),
        phase_dir=phase_dir,
        channels_per_image=-1,
        channels_per_bin=-1,
        integrations_per_image=ms_meta.ntime,  # single time bin
        images_per_chunk=1,
        max_simul_chunks=1,
        field_of_view=0.5,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        beam_model=None,
        robustness=None,
        epsilon=1e-8,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    ds = xr.open_zarr(out)
    assert ds.sizes["FREQ"] == 1
    assert ds.sizes["TIME"] == 1
    img = ds.cube.values[0, 0, 0]  # (Y, X)

    for di, dj in offsets:
        ix_pred = nx // 2 - di
        iy_pred = ny // 2 + dj
        lo_y, lo_x = max(0, iy_pred - 5), max(0, ix_pred - 5)
        win = img[lo_y : iy_pred + 6, lo_x : ix_pred + 6]
        ly, lx = np.unravel_index(np.argmax(win), win.shape)
        iy_f, ix_f = lo_y + ly, lo_x + lx
        err = np.hypot(ix_f - ix_pred, iy_f - iy_pred)
        assert err <= 1.0, f"rephased source at offset ({di},{dj}) landed {err:.1f} px from prediction"
        assert win.max() > 0.0, f"no flux recovered for rephased source at offset ({di},{dj})"


@pmp("wgt_mode", ("l2", "minvar"))
def test_hci_writes_cube(wgt_mode, ms_name, tmp_path):
    outname = str(tmp_path / f"test_hci_{wgt_mode}.zarr")

    hci_core(
        [Path(ms_name)],
        outname,
        product="I",
        data_column="DATA",
        integrations_per_image=8,
        images_per_chunk=4,
        max_simul_chunks=2,
        field_of_view=1.0,
        precision="single",
        epsilon=1e-4,  # ducc0 single-precision kernels need epsilon >~ 1e-5
        wgt_mode=wgt_mode,
        overwrite=True,
        keep_ray_alive=True,
    )

    ds = xr.open_zarr(outname, chunks=None)
    cube = ds.cube.values  # (STOKES, FREQ, TIME, Y, X)
    wsum = ds.weight.values  # (STOKES, FREQ, TIME)

    assert (wsum > 0).any(), "no data was imaged"
    imaged = cube[wsum > 0]
    assert np.isfinite(imaged).all()
    assert np.abs(imaged).max() > 0
