"""pfb hci-msv4 end to end: output layout, agreement with hci, gains, errors.

Every run of the command starts fresh worker actors that compile their kernels,
so even a four-image run costs ~4 s warm: the end-to-end tests are slow (D45).
"""

import shutil

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_imaging.core import hci_msv4 as hci_msv4_module
from pfb_imaging.core.hci_msv4 import hci_msv4 as hci_msv4_core
from pfb_imaging.utils.misc import radec_barycentre, to_mjd_time
from pfb_imaging.utils.msv4 import get_engine

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
def test_hci_msv4_output_layout(ms_name, tmp_path):
    out = str(tmp_path / "cube.zarr")
    hci_msv4_core([ms_name], out, images_per_chunk=4, **COMMON)
    ds = xr.open_zarr(out)
    assert ds.cube.dims == ("STOKES", "FREQ", "TIME", "Y", "X")
    assert ds.cube.encoding["chunks"] == (1, 1, 4, 64, 64)
    assert ds.rms.encoding["chunks"] == (1, 1, 4)
    for v in ("cube_mean", "psf", "psf2", "flag", "weight", "nonzero", "channel_width"):
        assert v in ds
    assert np.isfinite(ds.cube.values).all()
    assert ds.TIME.values[0] > 4e9  # MJD seconds, as hci


@pytest.mark.slow
@pytest.mark.timeout(900)  # two full runs (hci and hci-msv4) can exceed the 300 s default
@pytest.mark.parametrize("phase_dir", [None, "00:20:00,-30:00:00"])
def test_hci_msv4_matches_hci(ms_name, tmp_path, phase_dir):
    pytest.importorskip("daskms")
    from pfb_imaging.core.hci import hci as hci_core  # deferred: legacy reference needs the [casacore] extra

    old, new = str(tmp_path / "old.zarr"), str(tmp_path / "new.zarr")
    hci_core([ms_name], old, images_per_chunk=4, phase_dir=phase_dir, **COMMON)
    hci_msv4_core([ms_name], new, images_per_chunk=8, phase_dir=phase_dir, **COMMON)
    a, b = xr.open_zarr(old), xr.open_zarr(new)
    assert set(a.data_vars) == set(b.data_vars)
    assert a.attrs == b.attrs
    for c in ("STOKES", "X", "Y", "X_PSF", "Y_PSF"):
        np.testing.assert_array_equal(a[c].values, b[c].values)
    assert_allclose(a.TIME.values, b.TIME.values, rtol=0, atol=1e-6)
    assert_allclose(a.FREQ.values, b.FREQ.values, rtol=1e-12)
    # hci leaves cube_mean uninitialised where the flag-masked weight sum is zero;
    # hci-msv4 writes 0 there (a deliberate deviation), so those (stokes, band) are excluded
    masked_wsum = (a.weight.values * ~a.flag.values).sum(axis=-1)
    for v in a.data_vars:
        av, bv = a[v].values, b[v].values
        assert av.dtype == bv.dtype, v
        if v == "cube_mean":
            keep = masked_wsum > 0
            av, bv = av[keep], bv[keep]
        if av.dtype == bool:
            np.testing.assert_array_equal(av, bv, err_msg=v)
        else:
            finite = np.abs(av[np.isfinite(av)])
            scale = max(finite.max() if finite.size else 0.0, 1e-30)
            assert_allclose(bv, av, rtol=0, atol=1e-5 * scale, err_msg=v)


@pytest.mark.slow
def test_hci_msv4_constant_gain_scales_cube(ms_name, tmp_path):
    """Applying a constant diagonal gain g divides the normalised image by |g|^2."""
    node = _vis_node(ms_name)
    _gain_term(f"{tmp_path}/gains.qc/NET/NET_0", node, to_mjd_time(node.ds.time.values[:1]), 2.0 + 0j)
    plain, gained = str(tmp_path / "plain.zarr"), str(tmp_path / "gained.zarr")
    hci_msv4_core([ms_name], plain, **COMMON)
    hci_msv4_core([ms_name], gained, gain_table=[f"{tmp_path}/gains.qc::NET"], **COMMON)
    a, b = xr.open_zarr(plain).cube.values, xr.open_zarr(gained).cube.values
    assert_allclose(b, a / 4.0, rtol=1e-6, atol=1e-9 * np.abs(a).max())


@pytest.mark.slow
def test_hci_msv4_data_resolution_gains_with_several_bands(ms_name, tmp_path):
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
    hci_msv4_core([ms_name], plain, **opts)
    hci_msv4_core([ms_name], gained, gain_table=[f"{tmp_path}/gains.qc::NET"], **opts)
    a, b = xr.open_zarr(plain).cube.values, xr.open_zarr(gained).cube.values
    nband = -(-nchan // cpi)
    assert a.shape[1:3] == (nband, -(-ntime // ipi))
    scale = (t_amp[::ipi][None, :] * b_amp[::cpi][:, None]) ** 2  # (band, time)
    assert_allclose(b, a / scale[None, :, :, None, None], rtol=1e-6, atol=1e-9 * np.abs(a).max())


def test_hci_msv4_gain_table_count_mismatch(ms_name, tmp_path):
    with pytest.raises(ValueError, match="one gain table per"):
        hci_msv4_core([ms_name], str(tmp_path / "x.zarr"), gain_table=["a::NET", "b::NET"], **COMMON)


def test_hci_msv4_refuses_off_grid_gains_before_ray(ms_name, tmp_path, monkeypatch):
    """An off-grid gain term is refused by the driver from coordinates alone, before Ray starts.

    Worker-error propagation itself is covered by tests/test_hci_dispatch.py.
    """
    node = _vis_node(ms_name)
    t = to_mjd_time(node.ds.time.values) + 3.0  # 3 s off the data grid
    _gain_term(f"{tmp_path}/bad.qc/NET/NET_0", node, t, 1.0 + 0j)

    def no_ray(*args, **kwargs):
        raise AssertionError("Ray was started")

    monkeypatch.setattr(hci_msv4_module, "init_ray", no_ray)
    with pytest.raises(ValueError, match=r"bad\.qc/NET does not apply to .* node .*on-grid only"):
        hci_msv4_core([ms_name], str(tmp_path / "x.zarr"), gain_table=[f"{tmp_path}/bad.qc::NET"], **COMMON)


def test_hci_msv4_refuses_weight_grid_out(ms_name, tmp_path):
    with pytest.raises(NotImplementedError, match="weight-grid-out"):
        hci_msv4_core([ms_name], str(tmp_path / "x.zarr"), weight_grid_out=True, **COMMON)


@pytest.mark.slow
def test_hci_msv4_two_ms_centre_on_the_barycentre(ms_name, tmp_path, pctable):
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
    hci_msv4_core([ms_name, second], out, **COMMON)
    ds = xr.open_zarr(out)
    ra0, dec0 = np.rad2deg(radec_barycentre(np.array([pd0, pd0 + [0, np.deg2rad(0.2)]])))
    nx, ny = ds.sizes["X"], ds.sizes["Y"]
    assert_allclose(ds.X.values[nx // 2], ra0, atol=1e-9)
    assert_allclose(ds.Y.values[ny // 2], dec0, atol=1e-9)


def test_hci_msv4_refuses_non_positive_max_simul_chunks(ms_name, tmp_path):
    with pytest.raises(ValueError, match="max_simul_chunks must be >= 1"):
        hci_msv4_core([ms_name], str(tmp_path / "x.zarr"), max_simul_chunks=0, **COMMON)


def test_hci_msv4_plans_one_ms_at_a_time(ms_name, tmp_path, monkeypatch):
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
        hci_msv4_core([ms_name, ms_name], str(tmp_path / "x.zarr"), **COMMON)
    assert events == ["open", "close", "open", "close"]


def test_hci_msv4_refuses_unknown_column(ms_name, tmp_path):
    with pytest.raises(ValueError, match="NOPE"):
        hci_msv4_core([ms_name], str(tmp_path / "x.zarr"), data_column="DATA-NOPE", **COMMON)


@pytest.mark.parametrize(
    "over, err, match",
    [
        ({"nworkers": 0}, ValueError, "nworkers must be >= 1"),
        ({"sigma_column": "SIGMA"}, NotImplementedError, "MSv4 has no SIGMA"),
        ({"natural_grad": True, "psf_relative_size": 1.0}, ValueError, "natural-grad needs a PSF twice"),
    ],
)
def test_hci_msv4_refusals_before_ray(ms_name, tmp_path, monkeypatch, over, err, match):
    def no_ray(*args, **kwargs):
        raise AssertionError("Ray was started")

    monkeypatch.setattr(hci_msv4_module, "init_ray", no_ray)
    with pytest.raises(err, match=match):
        hci_msv4_core([ms_name], str(tmp_path / "x.zarr"), **{**COMMON, **over})


def test_hci_msv4_overwrite_keeps_the_cube_when_validation_fails(ms_name, tmp_path):
    """A typo with --overwrite must not cost the previous cube."""
    out = tmp_path / "x.zarr"
    out.mkdir()
    (out / "keep").write_text("previous cube")
    with pytest.raises(ValueError, match="NOPE"):
        hci_msv4_core([ms_name], str(out), data_column="DATA-NOPE", **COMMON)
    assert (out / "keep").read_text() == "previous cube"


def test_hci_msv4_refuses_rephasing_without_casacore_before_ray(ms_name, tmp_path, monkeypatch):
    real = hci_msv4_module.importlib.util.find_spec
    monkeypatch.setattr(
        hci_msv4_module.importlib.util, "find_spec", lambda name, *a: None if name == "pyrap" else real(name, *a)
    )
    monkeypatch.setattr(hci_msv4_module, "init_ray", lambda *a, **k: pytest.fail("Ray was started"))
    with pytest.raises(ImportError, match="python-casacore"):
        hci_msv4_core([ms_name], str(tmp_path / "x.zarr"), phase_dir="00:20:00,-30:00:00", **COMMON)
