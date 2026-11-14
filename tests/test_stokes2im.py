"""stokes2im reproduces stokes_image on the same image."""

import dataclasses

import arcae
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_imaging.utils import stokes2im
from pfb_imaging.utils.gains import NodeGains
from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes
from pfb_imaging.utils.stokes2im import HciConfig, image_item, read_item


def _cfg(**over):
    base = dict(
        dc1="VISIBILITY",
        dc2=None,
        operator=None,
        model_column=None,
        weight_column="WEIGHT",
        sigma_column=None,
        flag_column="FLAG",
        product="I",
        precision="double",
        nx=128,
        ny=128,
        cell_rad=np.deg2rad(5e-3),
        psf_relative_size=None,
        phase_dir=None,
        target=None,
        robustness=0.0,
        min_padding=2.0,
        filter_counts_level=10.0,
        npix_super=0,
        channels_per_bin=-1,
        epsilon=1e-7,
        do_wgridding=True,
        double_accum=True,
        natural_grad=False,
        eta=1e-5,
        cg_tol=1e-3,
        cg_maxit=150,
        psf_out=True,
        l2_reweight_dof=None,
        wgt_mode="l2",
        inject_transients=None,
        nthreads=2,
    )
    base.update(over)
    return HciConfig(**base)


@pytest.fixture
def node(ms_name):
    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    sel = select_vis_nodes(dt)[0]
    return dt[sel.path], sel


def test_read_item_shapes_and_dtypes(node):
    tree, sel = node
    inp = read_item(tree, {"time": slice(0, 2), "frequency": slice(0, sel.nchan)}, _cfg())
    nt, nbl, nchan, ncorr = inp.data.shape
    assert (nt, nchan) == (2, sel.nchan)
    assert inp.flag.dtype == bool and inp.flag.shape == inp.data.shape
    assert inp.uvw.shape == (nt, nbl, 3)
    assert inp.ant1.shape == (nbl,) and inp.ant1.dtype == np.int32
    assert inp.jones.shape == (1, 1, inp.antpos.shape[0], 2)
    assert np.isfinite(inp.data).all() and np.isfinite(inp.uvw).all()
    assert inp.time_mjd[0] > 4e9  # MJD seconds, not unix


# filter_extreme_counts takes the mean of an all-zero counts grid (robust weighting,
# unchanged from the legacy path, which warns identically)
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
def test_image_item_fully_flagged_returns_zero_weight(node):
    tree, sel = node
    inp = read_item(tree, {"time": slice(0, 1), "frequency": slice(0, sel.nchan)}, _cfg())
    inp.flag[:] = True
    out = image_item(inp, _cfg(), None, 128, 128)
    assert out["weight"][0] == 0
    assert not out["nonzero"][0]
    assert out["cube"].shape == (1, 128, 128) and out["cube"].dtype == np.float32
    assert np.all(out["cube"] == 0)
    # a zero-weight image stores NaN beam parameters, as the retired dask hci did
    for key in ("psf_maj", "psf_min", "psf_pa"):
        assert out[key].shape == (1,) and np.isnan(out[key]).all()


def test_worker_telemetry_times_the_read_and_each_imaging_section(ms_name, node, tmp_path, monkeypatch):
    """The worker's per-image clock: a prefetched read is timed apart from the wait on it.

    With PFB_CPROFILE_DIR set, close writes the worker's one capture of its calls.
    """
    import pstats

    from pfb_imaging.core.hci import WorkItem
    from pfb_imaging.utils.hci_actors import HciWorkerImpl
    from pfb_imaging.utils.memprof import CPROFILE_DIR_ENV, MEMRAY_DIR_ENV
    from pfb_imaging.utils.stokes2im import IMAGE_SECTIONS

    monkeypatch.delenv(MEMRAY_DIR_ENV, raising=False)
    monkeypatch.setenv(CPROFILE_DIR_ENV, str(tmp_path))
    _, sel = node
    items = [WorkItem(0, sel.path, slice(t, t + 1), slice(0, sel.nchan), t, 0, 0) for t in range(2)]
    worker = HciWorkerImpl([ms_name], _cfg(robustness=None), 128, 128)
    try:
        _, first = worker.image(items[0], items[1])
        _, second = worker.image(items[1])
    finally:
        worker.close()
    for mem in (first, second):
        assert {"wall_s", "cpu_s", "read_s", "wait_s"} | {f"{k}_s" for k in IMAGE_SECTIONS} <= set(mem)
        assert mem["dirty_s"] > 0 and mem["psf_s"] > 0
        sections = sum(mem[f"{k}_s"] for k in IMAGE_SECTIONS)
        assert mem["wait_s"] + sections <= mem["wall_s"]
    assert first["wait_s"] >= first["read_s"]  # nothing prefetched: the call read it itself
    assert second["read_s"] > 0  # read on the prefetch thread, still reported
    (f,) = tmp_path.glob("hci_worker-*.pstats")
    assert "image_item" in {k[2] for k in pstats.Stats(str(f)).stats}


def test_read_item_gain_flags_follow_antennas(node, monkeypatch):
    """A gain-flagged antenna flags every baseline it is on, at the flagged channels only."""
    tree, sel = node
    seen = {}

    def fake_gains(spec, scan_name, times_unix, freqs, antenna_names, field_name=None):
        seen.update(
            spec=spec,
            scan=scan_name,
            field=field_name,
            ntime=times_unix.size,
            nchan=freqs.size,
            ants=list(antenna_names),
        )
        nant = len(antenna_names)
        flags = np.zeros((1, freqs.size, nant), dtype=bool)
        flags[0, 1, 3] = True  # antenna 3, second channel of the region
        return NodeGains(np.full((1, freqs.size, nant, 2), 2.0 + 0j), flags)

    region = {"time": slice(0, 2), "frequency": slice(4, 7)}
    base = read_item(tree, region, _cfg()).flag
    monkeypatch.setattr(stokes2im, "load_node_gains", fake_gains)
    inp = read_item(tree, region, _cfg(), gain_spec="g.qc/NET")
    assert seen == {
        "spec": "g.qc/NET",
        "scan": sel.scan_name,
        "field": sel.field_name,
        "ntime": 2,
        "nchan": 3,
        "ants": list(tree["antenna_xds"].antenna_name.values),
    }
    assert inp.jones.shape == (1, 3, inp.antpos.shape[0], 2) and inp.jones.dtype == np.complex128
    on3 = (inp.ant1 == 3) | (inp.ant2 == 3)
    assert on3.any()
    assert not base[:, on3, 1].all() and inp.flag[:, on3, 1].all()
    assert (inp.flag[:, ~on3, 1] == base[:, ~on3, 1]).all()
    assert (inp.flag[:, :, [0, 2]] == base[:, :, [0, 2]]).all()


@pytest.mark.slow  # numba compiles the CORRECTED_DATA path: ~30 s cold (D45)
def test_model_column_is_subtracted_like_the_column_difference(writable_ms):
    """--model-column never worked in hci (a broadcast error); pin it against dc1 - dc2."""
    # The downloaded MS ships CORRECTED_DATA and MODEL_DATA all zero, so the
    # comparison is vacuous (0 - 0) unless both are filled, and differently.
    rng = np.random.default_rng(7)
    with arcae.table(writable_ms, readonly=False) as tab:
        shape = np.asarray(tab.getcol("DATA")).shape
        for col in ("CORRECTED_DATA", "MODEL_DATA"):
            vis = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            tab.putcol(col, vis.astype(np.complex64))
    dt = xr.open_datatree(writable_ms, **get_engine(writable_ms))
    sel = select_vis_nodes(dt)[0]
    tree = dt[sel.path]
    region = {"time": slice(0, 1), "frequency": slice(0, sel.nchan)}
    cfg_m = _cfg(dc1="CORRECTED_DATA", model_column="MODEL_DATA", robustness=None, psf_out=False)
    cfg_d = _cfg(dc1="CORRECTED_DATA", dc2="MODEL_DATA", operator="-", robustness=None, psf_out=False)
    with_model = image_item(read_item(tree, region, cfg_m), cfg_m, None, 128, 128)
    difference = image_item(read_item(tree, region, cfg_d), cfg_d, None, 128, 128)
    assert np.abs(with_model["cube"]).max() > 0
    assert_allclose(with_model["cube"], difference["cube"], rtol=0, atol=1e-6 * np.abs(difference["cube"]).max())


@pytest.mark.slow  # numba compiles the IQUV kernel: ~30 s cold (D45)
def test_l2_reweight_is_per_stokes(sky_truth, node):
    """--l2-reweight-dof never worked in hci; each Stokes is reweighted by its own variance."""
    tree, sel = node
    region = {"time": slice(0, 1), "frequency": slice(0, sel.nchan)}
    base = read_item(tree, region, _cfg())
    # the test MS is unpolarised (Q, U, V exactly zero); noise gives each Stokes its own variance
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(base.data.shape) + 1j * rng.standard_normal(base.data.shape)
    inp = dataclasses.replace(base, data=base.data + 0.1 * np.abs(base.data).max() * noise)
    out = {}
    for product in ("I", "IQUV"):
        cfg = _cfg(product=product, l2_reweight_dof=5, robustness=None, psf_out=False)
        out[product] = image_item(inp, cfg, None, 128, 128)
    plain = _cfg(robustness=None, psf_out=False)
    unweighted = image_item(inp, plain, None, 128, 128)
    assert np.isfinite(out["IQUV"]["cube"]).all() and (out["IQUV"]["weight"] > 0).all()
    # adding Q, U and V must not change how Stokes I is reweighted
    assert_allclose(out["IQUV"]["cube"][0], out["I"]["cube"][0], rtol=1e-6, atol=1e-7)
    assert_allclose(out["IQUV"]["weight"][0], out["I"]["weight"][0], rtol=1e-6)
    assert not np.allclose(out["I"]["weight"], unweighted["weight"])  # the reweighting did something
