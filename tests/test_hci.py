"""pfb hci end to end: output layout, gains, transients, rephasing, targets, errors.

Every run of the command starts fresh worker actors that compile their kernels,
so even a four-image run costs ~4 s warm: the end-to-end tests are slow (D45).
"""

import shutil

import arcae
import numpy as np
import pytest
import xarray as xr
import yaml
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from pfb_imaging.core import hci as hci_module
from pfb_imaging.core.hci import hci as hci_core
from pfb_imaging.utils.misc import radec_barycentre, radec_to_lm, set_image_size, to_mjd_time
from pfb_imaging.utils.msv4 import get_engine
from pfb_imaging.utils.transients import generate_transient_spectra

COMMON = dict(
    product="I",
    nx=64,
    ny=64,
    cell_size=20.0,
    integrations_per_image=1,
    robustness=0.0,
    psf_out=True,
    nthreads=1,
    nworkers=2,
    epsilon=1e-7,
    precision="double",
    overwrite=True,
    keep_ray_alive=True,
)


def _vis_node(ms_name):
    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    return next(n for n in dt.subtree if n.attrs.get("type") == "visibility")


def _gain_term(path, node, gain_time, value, gain_freq=None):
    """Write a one-direction diagonal QuartiCal-style gain term for ``node``.

    ``value`` is a scalar or a ``(gain_time, gain_freq)`` array, the same for every antenna.
    """
    ants = [str(a) for a in node["antenna_xds"].antenna_name.values]
    gain_freq = node.ds.frequency.values[:1] if gain_freq is None else gain_freq
    v = np.broadcast_to(np.asarray(value, dtype=np.complex128), (gain_time.size, gain_freq.size))
    g = np.ascontiguousarray(np.broadcast_to(v[:, :, None, None, None], v.shape + (len(ants), 1, 2)))
    xr.Dataset(
        {
            "gains": (("gain_time", "gain_freq", "antenna", "direction", "correlation"), g),
            "gain_flags": (("gain_time", "gain_freq", "antenna", "direction"), np.zeros(g.shape[:-1], np.int8)),
        },
        coords={
            "gain_time": gain_time,
            "gain_freq": gain_freq,
            "antenna": ants,
            "direction": [0],
            "correlation": ["XX", "YY"],
        },
        attrs={"FIELD_ID": 0, "DATA_DESC_ID": 0, "SCAN_NUMBER": int(np.unique(node.ds.scan_name.values).item())},
    ).to_zarr(path, mode="w")


@pytest.mark.slow
def test_hci_output_layout(ms_name, ms_meta, tmp_path):
    out = str(tmp_path / "cube.zarr")
    hci_core([ms_name], out, images_per_chunk=4, **COMMON)
    ds = xr.open_zarr(out)
    assert ds.cube.dims == ("STOKES", "FREQ", "TIME", "Y", "X")
    assert ds.cube_mean.dims == ("STOKES", "FREQ", "Y", "X")
    assert list(ds.STOKES.values) == ["I"]
    # one spw and channels_per_image=-1: one band; integrations_per_image=1: one image per integration
    assert (ds.sizes["FREQ"], ds.sizes["TIME"]) == (1, ms_meta.ntime)
    assert np.all(ds.channel_width.values > 0)
    assert ds.cube.encoding["chunks"] == (1, 1, 4, 64, 64)
    assert ds.rms.encoding["chunks"] == (1, 1, 4)
    for v in ("cube_mean", "psf", "psf2", "flag", "weight", "nonzero", "channel_width"):
        assert v in ds
    assert np.isfinite(ds.cube.values).all()
    assert ds.TIME.values[0] > 4e9  # MJD seconds, as the retired dask hci


@pytest.mark.slow
def test_hci_constant_gain_scales_cube(ms_name, sky_truth, tmp_path):
    """Applying a constant diagonal gain g divides the normalised image by |g|^2."""
    node = _vis_node(ms_name)
    _gain_term(f"{tmp_path}/gains.qc/NET/NET_0", node, to_mjd_time(node.ds.time.values[:1]), 2.0 + 0j)
    plain, gained = str(tmp_path / "plain.zarr"), str(tmp_path / "gained.zarr")
    hci_core([ms_name], plain, **COMMON)
    hci_core([ms_name], gained, gain_table=[f"{tmp_path}/gains.qc::NET"], **COMMON)
    a, b = xr.open_zarr(plain).cube.values, xr.open_zarr(gained).cube.values
    assert np.abs(a).max() > 0
    assert_allclose(b, a / 4.0, rtol=1e-6, atol=1e-9 * np.abs(a).max())


@pytest.mark.slow
def test_hci_data_resolution_gains_with_several_bands(ms_name, sky_truth, tmp_path):
    """Per-time, per-channel gains with channels_per_image < nchan: each image scales by 1/|g|^2.

    |g| is constant within an image but differs between images and bands, and the phase varies
    per integration and channel (the same for every antenna, so it cancels): a misaligned time
    or channel axis, or a channel subset that fails to select, changes the result.
    """
    node = _vis_node(ms_name)
    ntime, nchan = node.ds.sizes["time"], node.ds.sizes["frequency"]
    ipi, cpi = 10, 3
    assert nchan > cpi
    t_amp = 1.0 + 0.1 * (np.arange(ntime) // ipi)
    b_amp = 1.0 + 0.5 * (np.arange(nchan) // cpi)
    phase = 0.3 * np.arange(ntime)[:, None] - 0.2 * np.arange(nchan)[None, :]
    gains = t_amp[:, None] * b_amp[None, :] * np.exp(1j * phase)
    _gain_term(
        f"{tmp_path}/gains.qc/NET/NET_0",
        node,
        to_mjd_time(node.ds.time.values),
        gains,
        gain_freq=node.ds.frequency.values,
    )
    opts = {**COMMON, "integrations_per_image": ipi, "channels_per_image": cpi}
    plain, gained = str(tmp_path / "plain.zarr"), str(tmp_path / "gained.zarr")
    hci_core([ms_name], plain, **opts)
    hci_core([ms_name], gained, gain_table=[f"{tmp_path}/gains.qc::NET"], **opts)
    a, b = xr.open_zarr(plain).cube.values, xr.open_zarr(gained).cube.values
    nband = -(-nchan // cpi)
    assert np.abs(a).max() > 0
    assert a.shape[1:3] == (nband, -(-ntime // ipi))
    scale = (t_amp[::ipi][None, :] * b_amp[::cpi][:, None]) ** 2  # (band, time)
    assert_allclose(b, a / scale[None, :, :, None, None], rtol=1e-6, atol=1e-9 * np.abs(a).max())


def test_hci_gain_table_count_mismatch(ms_name, tmp_path):
    with pytest.raises(ValueError, match="one gain table per"):
        hci_core([ms_name], str(tmp_path / "x.zarr"), gain_table=["a::NET", "b::NET"], **COMMON)


def test_hci_refuses_off_grid_gains_before_ray(ms_name, tmp_path, monkeypatch):
    """An off-grid gain term is refused by the driver from coordinates alone, before Ray starts.

    Worker-error propagation itself is covered by tests/test_hci_dispatch.py.
    """
    node = _vis_node(ms_name)
    t = to_mjd_time(node.ds.time.values) + 3.0  # 3 s off the data grid
    _gain_term(f"{tmp_path}/bad.qc/NET/NET_0", node, t, 1.0 + 0j)

    def no_ray(*args, **kwargs):
        raise AssertionError("Ray was started")

    monkeypatch.setattr(hci_module, "init_ray", no_ray)
    with pytest.raises(ValueError, match=r"bad\.qc/NET does not apply to .* node .*on-grid only"):
        hci_core([ms_name], str(tmp_path / "x.zarr"), gain_table=[f"{tmp_path}/bad.qc::NET"], **COMMON)


def test_hci_refuses_weight_grid_out(ms_name, tmp_path):
    with pytest.raises(NotImplementedError, match="weight-grid-out"):
        hci_core([ms_name], str(tmp_path / "x.zarr"), weight_grid_out=True, **COMMON)


@pytest.mark.slow
def test_hci_two_ms_centre_on_the_barycentre(ms_name, tmp_path, pctable):
    """#320's configuration: one field per MS, different pointings and times."""
    second = str(tmp_path / "second.ms")
    shutil.copytree(ms_name, second)
    with pctable(f"{second}::FIELD", readonly=False, ack=False) as t:
        pd = t.getcol("PHASE_DIR")
        pd[:, 0, 1] += np.deg2rad(0.2)
        t.putcol("PHASE_DIR", pd)
        for col in ("DELAY_DIR", "REFERENCE_DIR"):
            t.putcol(col, pd)
    with pctable(second, readonly=False, ack=False) as t:
        t.putcol("TIME", t.getcol("TIME") + 1.0e5)
        t.putcol("TIME_CENTROID", t.getcol("TIME_CENTROID") + 1.0e5)
    with pctable(f"{ms_name}::FIELD", ack=False) as t:
        pd0 = t.getcol("PHASE_DIR")[0, 0]
    out = str(tmp_path / "cube.zarr")
    hci_core([ms_name, second], out, **COMMON)
    ds = xr.open_zarr(out)
    ra0, dec0 = np.rad2deg(radec_barycentre(np.array([pd0, pd0 + [0, np.deg2rad(0.2)]])))
    nx, ny = ds.sizes["X"], ds.sizes["Y"]
    assert_allclose(ds.X.values[nx // 2], ra0, atol=1e-9)
    assert_allclose(ds.Y.values[ny // 2], dec0, atol=1e-9)


def test_hci_refuses_non_positive_max_simul_chunks(ms_name, tmp_path):
    with pytest.raises(ValueError, match="max_simul_chunks must be >= 1"):
        hci_core([ms_name], str(tmp_path / "x.zarr"), max_simul_chunks=0, **COMMON)


def test_hci_plans_one_ms_at_a_time(ms_name, tmp_path, monkeypatch):
    """Each MS's tree is closed before the next opens, and on a planning error (#241)."""
    events = []
    real_open, real_close = xr.open_datatree, xr.DataTree.close

    def opening(*args, **kwargs):
        events.append("open")
        return real_open(*args, **kwargs)

    def closing(self):
        events.append("close")
        return real_close(self)

    monkeypatch.setattr(xr, "open_datatree", opening)
    monkeypatch.setattr(xr.DataTree, "close", closing)
    # the same MS twice maps two partitions to each output image: planning fails after both are read
    with pytest.raises(ValueError, match="same output image"):
        # one worker plans in this process; more scan in a process pool (plan_ms)
        hci_core([ms_name, ms_name], str(tmp_path / "x.zarr"), **{**COMMON, "nworkers": 1})
    assert events == ["open", "close", "open", "close"]


def test_hci_refuses_unknown_column(ms_name, tmp_path):
    with pytest.raises(ValueError, match="NOPE"):
        hci_core([ms_name], str(tmp_path / "x.zarr"), data_column="DATA-NOPE", **COMMON)


@pytest.mark.parametrize(
    "over, err, match",
    [
        ({"nworkers": 0}, ValueError, "nworkers must be >= 1"),
        ({"sigma_column": "SIGMA"}, NotImplementedError, "does not expose SIGMA"),
        ({"natural_grad": True, "psf_relative_size": 1.0}, ValueError, "natural-grad needs a PSF twice"),
        ({"beam_model": "meerkat-beams"}, ValueError, "meerkat-beams band"),
        ({"spatial_chunk": 0}, ValueError, "spatial_chunk must be >= 1"),
    ],
)
def test_hci_refusals_before_ray(ms_name, tmp_path, monkeypatch, over, err, match):
    def no_ray(*args, **kwargs):
        raise AssertionError("Ray was started")

    monkeypatch.setattr(hci_module, "init_ray", no_ray)
    with pytest.raises(err, match=match):
        hci_core([ms_name], str(tmp_path / "x.zarr"), **{**COMMON, **over})


def test_hci_overwrite_keeps_the_cube_when_validation_fails(ms_name, tmp_path):
    """A typo with --overwrite must not cost the previous cube."""
    out = tmp_path / "x.zarr"
    out.mkdir()
    (out / "keep").write_text("previous cube")
    with pytest.raises(ValueError, match="NOPE"):
        hci_core([ms_name], str(out), data_column="DATA-NOPE", **COMMON)
    assert (out / "keep").read_text() == "previous cube"


def test_hci_refuses_rephasing_without_casacore_before_ray(ms_name, tmp_path, monkeypatch):
    real = hci_module.importlib.util.find_spec
    monkeypatch.setattr(
        hci_module.importlib.util, "find_spec", lambda name, *a: None if name == "pyrap" else real(name, *a)
    )
    monkeypatch.setattr(hci_module, "init_ray", lambda *a, **k: pytest.fail("Ray was started"))
    with pytest.raises(ImportError, match="python-casacore"):
        hci_core([ms_name], str(tmp_path / "x.zarr"), phase_dir="00:20:00,-30:00:00", **COMMON)


# ---- ported from the retired dask hci's tests ----

# Field-of-view sizing, one worker, natural weighting: the geometry these tests predict pixels from.
FOV_RUN = dict(
    product="I",
    channels_per_image=-1,
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
)


def _lm_to_radec(lcoord, mcoord, ra0, dec0):
    """Invert the SIN (orthographic) projection about ``(ra0, dec0)``, all in radians."""
    n = np.sqrt(1.0 - lcoord**2 - mcoord**2)
    dec = np.arcsin(mcoord * np.cos(dec0) + n * np.sin(dec0))
    ra = ra0 + np.arctan2(lcoord, n * np.cos(dec0) - mcoord * np.sin(dec0))
    return ra, dec


def _field_centre(ms):
    """The test MS's single phase centre ``(ra, dec)`` in radians."""
    with arcae.table(f"{ms}::FIELD") as tab:
        return (float(v) for v in np.asarray(tab.getcol("PHASE_DIR")).squeeze())


def _clear_flags(ms):
    """Unflag every sample, so a per-sample analytic expectation holds."""
    with arcae.table(ms, readonly=False) as tab:
        flag = np.asarray(tab.getcol("FLAG"))
        tab.putcol("FLAG", np.zeros_like(flag))
        tab.putcol("FLAG_ROW", np.zeros(flag.shape[0], dtype=bool))


def _write_transients(path, transients):
    with open(path, "w") as f:
        yaml.safe_dump({"transients": transients}, f)
    return str(path)


def _steady(name, ra, dec):
    """A flat-spectrum source bright across the whole observation."""
    return {
        "name": name,
        "time": {"peak_time": 1800.0, "duration": 3000.0, "shape": "gaussian"},
        "frequency": {"peak_flux": 2.0, "reference_freq": 1.4e9, "spectral_index": 0.0},
        "position": {"ra": float(np.rad2deg(ra)), "dec": float(np.rad2deg(dec))},
    }


def _peak_near(img, ix, iy, half):
    """Pixel of the maximum in a ``(2*half+1)``-wide window about ``(ix, iy)``, and that maximum."""
    lo_y, lo_x = max(0, iy - half), max(0, ix - half)
    win = img[lo_y : iy + half + 1, lo_x : ix + half + 1]
    ly, lx = np.unravel_index(np.argmax(win), win.shape)
    return lo_x + lx, lo_y + ly, win.max()


@pytest.mark.slow
def test_hci_channels_per_bin_invariance_no_beam(ms_name, sky_truth, tmp_path):
    """Without a beam, sub-binning an image's channels telescopes back to one bin.

    The cube is a weight-normalised sum over bins of linear wgridder outputs, so
    cube, cube_mean and weight must not depend on channels_per_bin.
    """
    opts = {**FOV_RUN, "integrations_per_image": 10, "images_per_chunk": 6, "robustness": 0.0}
    full, bin4 = str(tmp_path / "full.zarr"), str(tmp_path / "bin4.zarr")
    hci_core([ms_name], full, channels_per_bin=-1, **opts)
    hci_core([ms_name], bin4, channels_per_bin=4, **opts)
    a, b = xr.open_zarr(full), xr.open_zarr(bin4)
    assert np.any(a.weight.values > 0)
    assert np.abs(np.nan_to_num(a.cube.values)).max() > 0  # weights do not depend on the data; the cube does
    assert_allclose(b.weight.values, a.weight.values, rtol=1e-6, atol=1e-6)
    for v in ("cube", "cube_mean"):
        scale = float(max(np.abs(a[v].values).max(), 1.0))
        assert_allclose(b[v].values, a[v].values, rtol=1e-5, atol=1e-5 * scale, err_msg=v)


@pytest.mark.slow
@pytest.mark.parametrize("wgt_mode", ("l2", "minvar"))
def test_hci_weighting_modes_image_the_sky(wgt_mode, ms_name, sky_truth, tmp_path):
    """minvar is the mode that exercised the weight_data recompile bug in production (#273)."""
    out = str(tmp_path / "cube.zarr")
    hci_core(
        [ms_name],
        out,
        **{**COMMON, "integrations_per_image": 8, "images_per_chunk": 4, "precision": "single", "epsilon": 1e-4},
        wgt_mode=wgt_mode,
    )
    ds = xr.open_zarr(out, chunks=None)
    imaged = ds.cube.values[ds.weight.values > 0]
    assert imaged.size, "no data was imaged"
    assert np.isfinite(imaged).all()
    assert np.abs(imaged).max() > 0


def test_hci_refuses_an_existing_output_without_overwrite(ms_name, tmp_path):
    out = tmp_path / "x.zarr"
    out.mkdir()
    (out / "marker").write_text("do not delete")
    with pytest.raises(RuntimeError, match="exists"):
        hci_core([ms_name], str(out), **{**COMMON, "overwrite": False})
    assert (out / "marker").exists()


@pytest.mark.slow
def test_hci_injected_transient_has_the_analytic_dynamic_spectrum(writable_ms, ms_meta, tmp_path):
    """An injected transient lands on its pixel with the injected (freq, time) profile.

    The base visibilities are zeroed (``DATA-DATA``) so the cube holds only the
    transient, which sits on an exact pixel centre: under natural weighting the
    PSF-peak normalisation then recovers the binned dynamic spectrum directly.
    """
    _clear_flags(writable_ms)
    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, 0.5, 2.0)
    ra0, dec0 = _field_centre(writable_ms)
    di, dj = -20, 16  # X (l) and Y (m) pixel offsets from the central pixel
    ra, dec = _lm_to_radec(di * cell_rad, dj * cell_rad, ra0, dec0)
    assert_allclose(radec_to_lm((ra, dec), (ra0, dec0)), [di * cell_rad, dj * cell_rad], atol=1e-12)
    transient = {
        "name": "test_transient",
        "time": {"peak_time": 1200.0, "duration": 600.0, "shape": "gaussian"},
        "frequency": {"peak_flux": 2.0, "reference_freq": 1.4e9, "spectral_index": -1.5},
        "position": {"ra": float(np.rad2deg(ra)), "dec": float(np.rad2deg(dec))},
    }
    ipi, cpi = 4, 2  # 60 integrations -> 15 images; 8 channels -> 4 bands
    out = str(tmp_path / "cube.zarr")
    hci_core(
        [writable_ms],
        out,
        **{**FOV_RUN, "channels_per_image": cpi},
        data_column="DATA-DATA",
        inject_transients=_write_transients(tmp_path / "transient.yaml", [transient]),
        integrations_per_image=ipi,
        images_per_chunk=15,
    )
    ds = xr.open_zarr(out)
    n_tbin, n_fbin = ms_meta.ntime // ipi, ms_meta.nchan // cpi
    assert (ds.sizes["FREQ"], ds.sizes["TIME"]) == (n_fbin, n_tbin)
    assert_allclose(ds.TIME.values, ms_meta.utime.reshape(n_tbin, ipi).mean(axis=1), atol=1e-6)
    assert_allclose(ds.FREQ.values, ms_meta.freq.reshape(n_fbin, cpi).mean(axis=1), atol=1e-3)

    # predicted from (l, m), not ds.X/ds.Y: the cube's RA labels omit the cos(dec) factor
    ix, iy = nx // 2 - di, ny // 2 + dj
    tprofile, fprofile = generate_transient_spectra(ms_meta.utime, ms_meta.freq, transient)
    expected_t = tprofile.reshape(n_tbin, ipi).mean(axis=1)
    expected_f = fprofile.reshape(n_fbin, cpi).mean(axis=1)
    img = ds.cube.values[0, int(np.argmax(expected_f)), int(np.argmax(expected_t))]
    assert np.unravel_index(np.argmax(img), img.shape) == (iy, ix)
    expected_ft = expected_f[:, None] * expected_t[None, :]
    # observed agreement is < 0.1%; 1% leaves margin for gridder numerics across platforms
    assert_allclose(ds.cube.values[0, :, :, iy, ix], expected_ft, rtol=1e-2, atol=1e-2 * expected_ft.max())


@pytest.mark.slow
def test_hci_injected_sources_keep_their_sky_position_out_to_the_field_edge(writable_ms, ms_meta, tmp_path):
    """Sources at increasing distance land on their pixel, and the cube's WCS maps it back to them.

    The controlled experiment of ratt-ru/breifast#263 on the test MS: both a
    pixel-placement and a coordinate-labelling error would grow with distance.
    """
    _clear_flags(writable_ms)
    fov = 1.0
    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, fov, 2.0)
    ra0, dec0 = _field_centre(writable_ms)
    offsets = [(20, 15), (60, 45), (100, 75), (140, 105), (170, 128)]
    radecs = [_lm_to_radec(di * cell_rad, dj * cell_rad, ra0, dec0) for di, dj in offsets]
    out = str(tmp_path / "cube.zarr")
    hci_core(
        [writable_ms],
        out,
        **{**FOV_RUN, "field_of_view": fov, "epsilon": 1e-8},
        data_column="DATA-DATA",
        inject_transients=_write_transients(
            tmp_path / "t.yaml", [_steady(f"src{k}", ra, dec) for k, (ra, dec) in enumerate(radecs)]
        ),
        integrations_per_image=ms_meta.ntime,  # one image
        images_per_chunk=1,
    )
    ds = xr.open_zarr(out)
    img = ds.cube.values[0, 0, 0]
    hdr = fits.Header()
    for key, val in ds.attrs["fits_header"]:
        hdr[key] = val
    wcs = WCS(hdr).celestial
    cell_arcsec = np.rad2deg(cell_rad) * 3600.0
    for (di, dj), (ra, dec) in zip(offsets, radecs):
        ix, iy, _ = _peak_near(img, nx // 2 - di, ny // 2 + dj, 6)
        assert np.hypot(ix - (nx // 2 - di), iy - (ny // 2 + dj)) <= 1.0, f"source ({di},{dj}) misplaced"
        sep = wcs.pixel_to_world(ix, iy).separation(SkyCoord(ra * units.rad, dec * units.rad, frame="fk5"))
        assert sep.to_value("arcsec") <= 0.2 * cell_arcsec, (
            f"source ({di},{dj}): WCS off by {sep.to_value('arcsec'):.2f} arcsec"
        )


def _as_parsed_phase_dir(ra, dec):
    """The ``hh:mm:ss,dd:mm:ss`` string for ``(ra, dec)`` radians and the radians it parses back to."""
    cc = SkyCoord(ra * units.rad, dec * units.rad, frame="fk5")
    text = f"{cc.ra.to_string(unit=units.hourangle, sep=':')},{cc.dec.to_string(unit=units.deg, sep=':')}"
    back = SkyCoord(*text.split(","), frame="fk5", unit=(units.hourangle, units.deg))
    return text, back


@pytest.mark.slow
def test_hci_injected_sources_land_on_their_pixels_when_rephased(writable_ms, ms_meta, tmp_path, needs_rephasing):
    """A chgcentre w-difference sign error displaced every source by -2*(field -> tangent) (breifast#263).

    Sources are injected about the rephasing centre, on which the image is centred.
    """
    _clear_flags(writable_ms)
    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, 0.5, 2.0)
    ra0, dec0 = _field_centre(writable_ms)
    phase_dir, cc = _as_parsed_phase_dir(ra0 + np.deg2rad(0.1) / np.cos(dec0), dec0 + np.deg2rad(0.05))
    ra_c, dec_c = cc.ra.rad, cc.dec.rad
    offsets = [(0, 0), (-25, -18), (25, 18), (-30, 25)]
    transients = [
        _steady(f"src{k}", *_lm_to_radec(di * cell_rad, dj * cell_rad, ra_c, dec_c))
        for k, (di, dj) in enumerate(offsets)
    ]
    out = str(tmp_path / "cube.zarr")
    hci_core(
        [writable_ms],
        out,
        **{**FOV_RUN, "epsilon": 1e-8},
        data_column="DATA-DATA",
        inject_transients=_write_transients(tmp_path / "t.yaml", transients),
        phase_dir=phase_dir,
        integrations_per_image=ms_meta.ntime,
        images_per_chunk=1,
    )
    ds = xr.open_zarr(out)
    assert (ds.sizes["FREQ"], ds.sizes["TIME"]) == (1, 1)
    img = ds.cube.values[0, 0, 0]
    for di, dj in offsets:
        ix, iy, peak = _peak_near(img, nx // 2 - di, ny // 2 + dj, 5)
        assert peak > 0.0, f"no flux recovered at offset ({di},{dj})"
        assert np.hypot(ix - (nx // 2 - di), iy - (ny // 2 + dj)) <= 1.0, f"rephased source ({di},{dj}) misplaced"


@pytest.mark.slow
def test_hci_target_recentres_the_grid(writable_ms, ms_meta, tmp_path, needs_rephasing):
    """``--target`` centres the image, and labels its coords, on the target rather than the phase centre."""
    _clear_flags(writable_ms)
    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, 0.5, 2.0)
    ra0, dec0 = _field_centre(writable_ms)
    dit, djt = 30, -20
    target, tc = _as_parsed_phase_dir(*_lm_to_radec(dit * cell_rad, djt * cell_rad, ra0, dec0))
    # offsets from the phase centre; the first is the target, so it lands dead centre
    offsets = [(dit, djt), (dit - 25, djt - 18), (dit + 25, djt + 18)]
    transients = [
        _steady(f"src{k}", *_lm_to_radec(di * cell_rad, dj * cell_rad, ra0, dec0)) for k, (di, dj) in enumerate(offsets)
    ]
    out = str(tmp_path / "cube.zarr")
    hci_core(
        [writable_ms],
        out,
        **{**FOV_RUN, "epsilon": 1e-8},
        data_column="DATA-DATA",
        inject_transients=_write_transients(tmp_path / "t.yaml", transients),
        target=target,
        integrations_per_image=ms_meta.ntime,
        images_per_chunk=1,
    )
    ds = xr.open_zarr(out)
    img = ds.cube.values[0, 0, 0]
    for di, dj in offsets:
        ix_pred, iy_pred = nx // 2 - (di - dit), ny // 2 + (dj - djt)
        ix, iy, peak = _peak_near(img, ix_pred, iy_pred, 5)
        assert peak > 0.0, f"no flux recovered at offset ({di},{dj})"
        assert np.hypot(ix - ix_pred, iy - iy_pred) <= 1.0, f"source ({di},{dj}) misplaced"
    assert_allclose(ds.X.values[nx // 2], tc.ra.deg, atol=1e-9)
    assert_allclose(ds.Y.values[ny // 2], tc.dec.deg, atol=1e-9)


def test_hci_rejects_a_moving_target_across_several_images(ms_name, tmp_path, needs_rephasing):
    """``--target Sun`` moves between images, so one shared X/Y axis cannot label them all."""
    with pytest.raises(ValueError, match="solar-system body"):
        hci_core([ms_name], str(tmp_path / "x.zarr"), **FOV_RUN, target="Sun", integrations_per_image=20)


@pytest.mark.slow
def test_hci_accepts_a_moving_target_in_a_single_image(ms_name, ms_meta, tmp_path, needs_rephasing):
    """One image has one target position: coords and FITS CRVAL are the body's at that time."""
    from pfb_imaging.utils.astrometry import resolve_target_radec  # deferred: pyrap ([casacore]) at module scope

    out = str(tmp_path / "cube.zarr")
    hci_core(
        [ms_name],
        out,
        **{**FOV_RUN, "robustness": 0.0},
        target="Sun",
        integrations_per_image=ms_meta.ntime,
        images_per_chunk=1,
    )
    ds = xr.open_zarr(out)
    assert ds.sizes["TIME"] == 1
    ra_sun, dec_sun, moving = resolve_target_radec("Sun", float(ds.TIME.values[0]))
    assert moving
    nx, ny = ds.sizes["X"], ds.sizes["Y"]
    assert_allclose(ds.X.values[nx // 2], np.rad2deg(ra_sun), atol=1e-9)
    assert_allclose(ds.Y.values[ny // 2], np.rad2deg(dec_sun), atol=1e-9)
    hdr = dict(ds.attrs["fits_header"])
    assert_allclose(hdr["CRVAL1"], np.rad2deg(ra_sun), atol=1e-9)
    assert_allclose(hdr["CRVAL2"], np.rad2deg(dec_sun), atol=1e-9)
