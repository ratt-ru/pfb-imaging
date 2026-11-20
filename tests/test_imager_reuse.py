"""imager reuses a .scratch whose pass-1 inputs match the run (#354)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from numpy.testing import assert_allclose

import pfb_imaging.core.imager as imager_mod
from pfb_imaging.core.imager import (
    SCRATCH_SCHEMA,
    image_geometry,
    pass1_inputs,
    read_scratch_record,
    record_diffs,
)
from pfb_imaging.core.imager import imager as imager_core


def _record(inputs, counts=None):
    return {"schema": SCRATCH_SCHEMA, "inputs": inputs, "counts": counts or {"nx": 2, "ny": 2, "cell": 1.0}}


def test_record_diffs_names_each_changed_input_and_the_counts_grid():
    old = pass1_inputs(ms=["/a.ms"], chan_average=1, precision="double", phase_dir=None)
    assert record_diffs(_record(old), old, {"nx": 2, "ny": 2, "cell": 1.0}) == []
    new = pass1_inputs(ms=["/a.ms"], chan_average=2, precision="double", phase_dir=None)
    assert record_diffs(_record(old), new) == ["chan_average: 1 -> 2"]
    # inputs first: a counts change is only reported once they match
    diffs = record_diffs(_record(old), old, {"nx": 4, "ny": 4, "cell": 0.5})
    assert len(diffs) == 1 and diffs[0].startswith("counts grid:") and "--cell-size" in diffs[0]
    assert "no validity record" in record_diffs(None, old)[0]
    stale = dict(_record(old), schema=SCRATCH_SCHEMA - 1)
    assert "schema" in record_diffs(stale, old)[0]


def test_pass1_inputs_are_json_canonical_and_a_beam_object_never_matches():
    a = pass1_inputs(ms=(Path("/x.ms"),), nband=np.int64(3), beam_model="L")
    assert a == {"ms": ["/x.ms"], "nband": 3, "beam_model": "L"}
    assert json.loads(json.dumps(a)) == a
    obj = object()
    assert pass1_inputs(beam_model=obj) != pass1_inputs(beam_model=object())


def test_counts_grid_follows_the_image_only_under_weight_on_image_grid():
    kw = dict(
        field_of_view=1.0, super_resolution_factor=2.0, cell_size=None, nx=None, ny=None,
        psf_oversize=1.4, min_padding=1.7, max_fov_rad=np.deg2rad(3.0),
    )  # fmt: skip
    a = image_geometry(3000.0, 3100.0, 1.4e9, weight_on_image_grid=True, **kw)
    b = image_geometry(3000.0, 3100.0, 1.4e9, weight_on_image_grid=True, **{**kw, "field_of_view": 1.5})
    assert a["counts"] != b["counts"]
    c = image_geometry(3000.0, 3100.0, 1.4e9, weight_on_image_grid=False, **kw)
    d = image_geometry(3000.0, 3100.0, 1.4e9, weight_on_image_grid=False, **{**kw, "field_of_view": 1.5})
    assert c["counts"] == d["counts"]


def _no_ray(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)


def test_a_scratch_without_a_record_is_refused_before_the_ms_is_read(ms_name, tmp_path, monkeypatch):
    """An older or interrupted scratch: refused with the --overwrite hint, before any MS read."""
    _no_ray(monkeypatch)
    monkeypatch.setattr(imager_mod, "scan_ms", lambda *a, **k: (_ for _ in ()).throw(AssertionError("MS read")))
    out = tmp_path / "old"
    zarr.open_group(str(out) + "_I.scratch", mode="w")
    assert read_scratch_record(str(out) + "_I.scratch") is None
    with pytest.raises(RuntimeError, match="no validity record(.|\n)*--overwrite"):
        imager_core([Path(ms_name)], str(out), product="I", field_of_view=1.0, keep_ray_alive=True)


def test_an_existing_dt_is_refused_before_pass_1(ms_name, tmp_path, monkeypatch):
    _no_ray(monkeypatch)
    out = tmp_path / "done"
    (tmp_path / "done_I.dt").mkdir()
    with pytest.raises(RuntimeError, match="done_I.dt exists"):
        imager_core([Path(ms_name)], str(out), product="I", field_of_view=1.0, keep_ray_alive=True)


def _no_pass1(monkeypatch):
    class _Refuse:
        @staticmethod
        def remote(*a, **kw):
            raise AssertionError("pass 1 ran")

    monkeypatch.setattr(imager_mod, "safe_stokes_vis", _Refuse)


def _bands(path):
    dt = xr.open_datatree(path, engine="zarr", chunks=None)
    return {n: dt[n].ds for n in dt.children if n.startswith("band")}


@pytest.mark.slow
def test_a_robustness_change_reuses_the_scratch_and_matches_a_fresh_run(ms_name, sky_truth, tmp_path, monkeypatch):
    """The issue's acceptance loop: re-image at a new robustness without re-reading the MS."""
    kw = dict(
        integrations_per_image=-1,
        channels_per_image=4,
        product="I",
        field_of_view=1.0,
        fits_mfs=False,
        fits_cubes=False,
        keep_ray_alive=True,
    )
    out = str(tmp_path / "loop")
    imager_core([Path(ms_name)], out, robustness=0.0, **kw)
    record = read_scratch_record(out + "_I.scratch")
    assert record["schema"] == SCRATCH_SCHEMA and record["inputs"]["chan_average"] == 1

    # without --overwrite the .dt is refused; with it pass 1 is skipped
    with pytest.raises(RuntimeError, match="_I.dt exists"):
        imager_core([Path(ms_name)], out, robustness=-1.0, **kw)
    with monkeypatch.context() as m:
        _no_pass1(m)
        m.setattr(imager_mod, "scan_ms", lambda *a, **k: (_ for _ in ()).throw(AssertionError("MS read")))
        imager_core([Path(ms_name)], out, robustness=-1.0, overwrite=True, **kw)
    reused = _bands(out + "_I.dt")

    fresh_out = str(tmp_path / "fresh")
    imager_core([Path(ms_name)], fresh_out, robustness=-1.0, **kw)
    fresh = _bands(fresh_out + "_I.dt")
    assert reused.keys() == fresh.keys()
    for b in fresh:
        assert np.abs(fresh[b].DIRTY.values).max() > 0
        assert_allclose(reused[b].WSUM.values, fresh[b].WSUM.values, rtol=1e-10)
        assert_allclose(reused[b].DIRTY.values, fresh[b].DIRTY.values, rtol=0, atol=1e-8 * np.abs(fresh[b].DIRTY).max())
        assert reused[b].attrs["freq_out"] == pytest.approx(fresh[b].attrs["freq_out"], rel=1e-12)

    # a pass-1 input changed: refused naming it, rebuilt under --overwrite
    # (without the .dt, which is refused first)
    shutil.rmtree(out + "_I.dt")
    with pytest.raises(RuntimeError, match="chan_average: 1 -> 2"):
        imager_core([Path(ms_name)], out, robustness=-1.0, chan_average=2, overwrite=False, **kw)
    # the image grid changed under --weight-on-image-grid: refused naming the counts grid
    with pytest.raises(RuntimeError, match="counts grid"):
        imager_core([Path(ms_name)], out, robustness=-1.0, **{**kw, "field_of_view": 0.8}, overwrite=False)
    imager_core([Path(ms_name)], out, robustness=-1.0, chan_average=2, overwrite=True, **kw)
    assert read_scratch_record(out + "_I.scratch")["inputs"]["chan_average"] == 2
