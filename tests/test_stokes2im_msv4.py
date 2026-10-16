"""stokes2im_msv4 reproduces stokes_image on the same image."""

import dataclasses

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_imaging.utils import stokes2im_msv4
from pfb_imaging.utils.gains import NodeGains
from pfb_imaging.utils.misc import chunkify_rows
from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes
from pfb_imaging.utils.stokes2im_msv4 import HciConfig, image_item, read_item


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
    # legacy hci stores NaN beam parameters for a zero-weight image
    for key in ("psf_maj", "psf_min", "psf_pa"):
        assert out[key].shape == (1,) and np.isnan(out[key]).all()


@pytest.mark.slow
@pytest.mark.parametrize("phase_dir", [None, "00:20:00,-30:00:00"])
def test_matches_stokes_image(ms_name, node, phase_dir):
    """Same single-integration image through the legacy dask-ms path.

    The legacy side gets its phase centre and antenna positions from the MSv2 FIELD and
    ANTENNA tables, as hci reads them, so the port's own MSv4 reads are checked too.
    """
    pytest.importorskip("daskms")
    if phase_dir is not None:
        pytest.importorskip("pyrap.measures")
    from daskms import xds_from_storage_ms, xds_from_table  # deferred: [casacore] extra, legacy reference only

    from pfb_imaging.utils.stokes2im import stokes_image  # deferred: imports pyrap at module scope ([casacore])

    tree, sel = node
    cfg = _cfg(phase_dir=phase_dir)
    inp = read_item(tree, {"time": slice(0, 1), "frequency": slice(0, sel.nchan)}, cfg)
    new = image_item(inp, cfg, None, 128, 128)

    # the test MS carries a per-row WEIGHT and no WEIGHT_SPECTRUM; xarray-ms
    # broadcasts WEIGHT over channels, so give the legacy path the same
    columns = ("DATA", "FLAG", "UVW", "ANTENNA1", "ANTENNA2", "TIME", "INTERVAL", "FLAG_ROW", "WEIGHT")
    schema = {c: {"dims": ("chan", "corr")} for c in ("DATA", "FLAG")}
    ds = xds_from_storage_ms(
        ms_name, columns=columns, table_schema=schema, group_cols=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]
    )[0]
    utime = np.unique(ds.TIME.values)
    _, ridx, rcnt = chunkify_rows(ds.TIME.values, 1)
    dsi = ds[{"row": slice(0, int(rcnt[0]))}].load(scheduler="sync")
    nchan = dsi.DATA.shape[1]
    wspec = np.broadcast_to(dsi.WEIGHT.values[:, None, :], dsi.DATA.shape).copy()
    dsi = dsi.assign(WEIGHT_SPECTRUM=(("row", "chan", "corr"), wspec))
    assert nchan == inp.freq.size
    antpos = xds_from_table(f"{ms_name}::ANTENNA")[0].POSITION.values
    radec = xds_from_table(f"{ms_name}::FIELD")[0].PHASE_DIR.values[int(ds.FIELD_ID), 0].copy()
    radec[0] = radec[0] % (2 * np.pi)  # as hci's construct_mappings
    old = stokes_image._function(
        dc1="DATA",
        ds=dsi,
        nx=128,
        ny=128,
        freq=inp.freq,
        cell_rad=cfg.cell_rad,
        utime=utime[:1],
        tbin_idx=ridx[:1],
        tbin_counts=rcnt[:1],
        radec=radec,
        antpos=antpos,
        poltype=inp.poltype,
        bandid=0,
        timeid=0,
        msid=0,
        nthreads=2,
        precision="double",
        weight_column="WEIGHT_SPECTRUM",
        product="I",
        phase_dir=phase_dir,
        robustness=0.0,
        epsilon=1e-7,
        psf_out=True,
        wgt_mode="l2",
    )
    scale = np.abs(old.cube.values).max()
    assert_allclose(new["cube"], old.cube.values[:, 0, 0], atol=1e-6 * scale)
    assert_allclose(new["psf"], old.psf.values[:, 0, 0], atol=1e-6)
    assert_allclose(new["weight"], old.weight.values[:, 0, 0], rtol=1e-6)
    assert_allclose(new["rms"], old.rms.values[:, 0, 0], rtol=1e-5)
    assert_allclose(new["time_mjd"], old.TIME.values[0], atol=1e-6)


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
    monkeypatch.setattr(stokes2im_msv4, "load_node_gains", fake_gains)
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
def test_model_column_is_subtracted_like_the_column_difference(node):
    """--model-column never worked in hci (a broadcast error); pin it against dc1 - dc2."""
    tree, sel = node
    region = {"time": slice(0, 1), "frequency": slice(0, sel.nchan)}
    cfg_m = _cfg(dc1="CORRECTED_DATA", model_column="MODEL_DATA", robustness=None, psf_out=False)
    cfg_d = _cfg(dc1="CORRECTED_DATA", dc2="MODEL_DATA", operator="-", robustness=None, psf_out=False)
    with_model = image_item(read_item(tree, region, cfg_m), cfg_m, None, 128, 128)
    difference = image_item(read_item(tree, region, cfg_d), cfg_d, None, 128, 128)
    assert np.abs(with_model["cube"]).max() > 0
    assert_allclose(with_model["cube"], difference["cube"], rtol=0, atol=1e-6 * np.abs(difference["cube"]).max())


@pytest.mark.slow  # numba compiles the IQUV kernel: ~30 s cold (D45)
def test_l2_reweight_is_per_stokes(node):
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
