"""CubeWriterImpl: one write per slab, fill kept for missing images, hci's reductions."""

import random
from collections import Counter

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_imaging.utils.hci_actors import CubeWriterImpl
from pfb_imaging.utils.hci_common import build_scaffold

NY = NX = 16


def _store(tmp_path, ntime=5, nband=1, ipc=2, psf_out=True):
    ds, _ = build_scaffold(
        5.0e9 + 8.0 * np.arange(ntime),
        1e9 + 1e7 * np.arange(nband),
        np.array([10.0]),
        np.array([-30.0]),
        product="I",
        beam=False,
        psf_out=psf_out,
        nx=NX,
        ny=NY,
        nx_psf=NX,
        ny_psf=NY,
        cell_deg=1e-3,
        spatial_chunk=128,
        images_per_chunk=ipc,
        obs_label=None,
    )
    path = str(tmp_path / "cube.zarr")
    ds.to_zarr(path, mode="w", compute=False)
    return path


def _result(t, b, value, weight=1.0, rms=1.0, psf_out=True):
    r = {
        "time_index": t,
        "band_index": b,
        "cube": np.full((1, NY, NX), value, np.float32),
        "weight": np.array([weight], np.float32),
        "rms": np.array([rms], np.float32),
        "nonzero": np.array([weight > 0]),
        "psf_maj": np.ones(1, np.float32),
        "psf_min": np.ones(1, np.float32),
        "psf_pa": np.zeros(1, np.float32),
        "orig_phase_dir_ra": 0.1,
        "orig_phase_dir_dec": -0.5,
    }
    if psf_out:
        r["psf"] = np.full((1, NY, NX), 1.0, np.float32)
    return r


def _expected(ntime, ipc, band=0):
    return {(band, blk): min(ipc, ntime - blk * ipc) for blk in range(-(-ntime // ipc))}


def test_slab_written_once_complete_out_of_order(tmp_path):
    store = _store(tmp_path)
    w = CubeWriterImpl(store, _expected(5, 2), 2, np.array([1e6]), 1.5)
    assert w.add(_result(1, 0, 2.0)) is None
    assert w.open_blocks() == 1
    key, mem = w.add(_result(0, 0, 1.0))
    assert key == (0, 0)
    assert {"pid", "rss_gb", "peak_gb"} <= set(mem)  # writer telemetry on flush
    assert w.open_blocks() == 0
    cube = xr.open_zarr(store).cube.values
    assert_allclose(cube[0, 0, 0], 1.0)
    assert_allclose(cube[0, 0, 1], 2.0)
    assert np.isnan(cube[0, 0, 2]).all()  # block 1 not yet written: fill


def test_short_last_block(tmp_path):
    store = _store(tmp_path)
    w = CubeWriterImpl(store, _expected(5, 2), 2, np.array([1e6]), 1.5)
    assert w.add(_result(4, 0, 9.0))[0] == (0, 2)
    assert_allclose(xr.open_zarr(store).cube.values[0, 0, 4], 9.0)


def test_writer_partial_slab_keeps_fill_and_means_only_received(tmp_path):
    store = _store(tmp_path, ntime=4, ipc=4)
    # time 2 has no data for this band: the plan expects 3 images in the slab
    w = CubeWriterImpl(store, {(0, 0): 3}, 4, np.array([1e6]), 10.0)
    for t, v in [(0, 1.0), (1, 2.0)]:
        assert w.add(_result(t, 0, v)) is None
    assert w.add(_result(3, 0, 4.0))[0] == (0, 0)
    w.finalise()
    ds = xr.open_zarr(store)
    assert np.isnan(ds.cube.values[0, 0, 2]).all()
    assert_allclose(ds.cube_mean.values[0, 0], (1.0 + 2.0 + 4.0) / 3)
    assert bool(ds.flag.values[0, 0, 2])  # missing image is flagged (~nonzero), as in hci


def test_finalise_flags_excess_rms_and_excludes_it_from_means(tmp_path):
    store = _store(tmp_path, ntime=5, ipc=2)
    w = CubeWriterImpl(store, _expected(5, 2), 2, np.array([3e6]), 1.5)
    rms = [1.0, 1.1, 0.9, 5.0, 1.0]  # time 3 is an outlier
    vals = [1.0, 2.0, 3.0, 100.0, 5.0]
    wts = [1.0, 2.0, 1.0, 1.0, 0.5]
    for t in range(5):
        w.add(_result(t, 0, vals[t], weight=wts[t], rms=rms[t]))
    out = w.finalise()
    ds = xr.open_zarr(store)
    flag = ds.flag.values[0, 0]
    assert flag.tolist() == [False, False, False, True, False]
    assert out["nflagged"] == 1
    keep = ~flag
    ref = np.sum(np.array(vals)[keep] * np.array(wts)[keep]) / np.sum(np.array(wts)[keep])
    assert_allclose(ds.cube_mean.values[0, 0], ref, rtol=1e-6)
    # psf2 = sum((w psf)^2) / sum(w^2) over unflagged images; psf == 1
    assert_allclose(ds.psf2.values[0], 1.0, rtol=1e-6)
    assert_allclose(ds.channel_width.values, [3e6])
    assert_allclose(ds.rms.values[0, 0], rms, rtol=1e-6)
    assert_allclose(ds.orig_phase_dir_ra.values, 0.1)


def test_unknown_slab_is_an_error(tmp_path):
    store = _store(tmp_path)
    w = CubeWriterImpl(store, _expected(5, 2), 2, np.array([1e6]), 1.5)
    with pytest.raises(KeyError, match="not in the plan"):
        w.add(_result(0, 3, 1.0))


def test_finalise_drops_two_flagged_images_from_one_block(tmp_path):
    """Two outliers in the same time block are both removed from the means (one block read)."""
    store = _store(tmp_path, ntime=6, ipc=3)
    w = CubeWriterImpl(store, _expected(6, 3), 3, np.array([1e6]), 1.5)
    rms = [1.0, 6.0, 7.0, 1.0, 1.1, 0.9]  # times 1 and 2 share block 0
    vals = [1.0, 50.0, 80.0, 2.0, 3.0, 4.0]
    wts = [1.0, 2.0, 3.0, 1.0, 0.5, 2.0]
    for t in range(6):
        w.add(_result(t, 0, vals[t], weight=wts[t], rms=rms[t]))
    out = w.finalise()
    ds = xr.open_zarr(store)
    flag = ds.flag.values[0, 0]
    assert flag.tolist() == [False, True, True, False, False, False]
    assert out["nflagged"] == 2
    keep = ~flag
    ref = np.sum(np.array(vals)[keep] * np.array(wts)[keep]) / np.sum(np.array(wts)[keep])
    assert_allclose(ds.cube_mean.values[0, 0], ref, rtol=1e-6)
    assert_allclose(ds.psf2.values[0], 1.0, rtol=1e-6)


def test_multiband_shuffled_with_missing_zero_weight_and_flagged_images(tmp_path):
    """Three bands, arrivals shuffled across bands and blocks, against hand-built references.

    Band 1 has no image at time 4 (the plan expects one fewer), band 2's time 5 has zero
    weight, and bands 0 and 2 each have an rms outlier.
    """
    ntime, nband, ipc = 7, 3, 3
    store = _store(tmp_path, ntime=ntime, nband=nband, ipc=ipc)
    present = {(b, t) for b in range(nband) for t in range(ntime)} - {(1, 4)}
    expected = Counter((b, t // ipc) for b, t in present)
    rms = {(b, t): 1.0 + 0.01 * t for b, t in present}
    rms[(0, 2)] = 9.0
    rms[(2, 6)] = 8.0
    weight = {(b, t): 1.0 + 0.5 * b + 0.1 * t for b, t in present}
    weight[(2, 5)] = 0.0
    psf = {b: 1.0 + 0.5 * b for b in range(nband)}

    w = CubeWriterImpl(store, dict(expected), ipc, np.array([1e6, 2e6, 3e6]), 1.5)
    order = sorted(present)
    random.Random(0).shuffle(order)
    flushed = []
    for b, t in order:
        r = _result(t, b, 10.0 * b + t, weight=weight[(b, t)], rms=rms[(b, t)])
        r["psf"][:] = psf[b]
        out = w.add(r)
        if out is not None:
            flushed.append(out[0])
    assert sorted(flushed) == sorted(expected)  # every slab written exactly once
    assert w.finalise()["nflagged"] == 2
    ds = xr.open_zarr(store)

    flag = ds.flag.values[0]
    want_flag = np.zeros((nband, ntime), bool)
    want_flag[0, 2] = want_flag[2, 6] = True  # rms outliers
    want_flag[1, 4] = True  # missing: not nonzero, as in hci
    want_flag[2, 5] = True  # zero weight: not nonzero
    assert (flag == want_flag).all()

    cube = ds.cube.values[0]
    for b in range(nband):
        for t in range(ntime):
            if (b, t) in present:
                assert_allclose(cube[b, t], 10.0 * b + t)
            else:
                assert np.isnan(cube[b, t]).all()
        keep = [t for t in range(ntime) if (b, t) in present and not want_flag[b, t]]
        wts = np.array([weight[(b, t)] for t in keep])
        assert_allclose(ds.cube_mean.values[0, b], np.sum(wts * (10.0 * b + np.array(keep))) / wts.sum(), rtol=1e-6)
    kept = [(b, t) for b, t in present if not want_flag[b, t]]
    num = sum((weight[k] * psf[k[0]]) ** 2 for k in kept)
    den = sum(weight[k] ** 2 for k in kept)
    assert_allclose(ds.psf2.values[0], num / den, rtol=1e-6)
    assert_allclose(ds.channel_width.values, [1e6, 2e6, 3e6])
