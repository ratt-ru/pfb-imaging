"""HciWorkerImpl: the prefetched read is the one that gets imaged."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from pfb_imaging.core.hci_msv4 import plan_images
from pfb_imaging.utils import hci_actors
from pfb_imaging.utils.hci_actors import HciWorkerImpl
from pfb_imaging.utils.msv4 import get_engine
from pfb_imaging.utils.stokes2im_msv4 import HciConfig, image_item, read_item


def _cfg():
    return HciConfig(
        dc1="VISIBILITY", dc2=None, operator=None, model_column=None, weight_column="WEIGHT",
        sigma_column=None, flag_column="FLAG", product="I", precision="double", nx=64, ny=64,
        cell_rad=np.deg2rad(5e-3), psf_relative_size=None, phase_dir=None, target=None, robustness=None,
        min_padding=2.0, filter_counts_level=10.0, npix_super=0, channels_per_bin=-1, epsilon=1e-7,
        do_wgridding=True, double_accum=True, natural_grad=False, eta=1e-5, cg_tol=1e-3, cg_maxit=150,
        psf_out=False, l2_reweight_dof=None, wgt_mode="l2", inject_transients=None, nthreads=1,
    )  # fmt: skip


@pytest.fixture
def items(ms_name):
    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    return plan_images([dt], integrations_per_image=1, channels_per_image=-1, images_per_chunk=4).items, dt


@pytest.mark.slow  # two worker inits and three images: ~5 s (D45)
def test_prefetched_item_matches_a_synchronous_read(ms_name, items, monkeypatch):
    its, dt = items
    calls = []
    real = hci_actors.read_item

    def counting(node, region, *args, **kwargs):
        calls.append(region["time"])
        return real(node, region, *args, **kwargs)

    monkeypatch.setattr(hci_actors, "read_item", counting)
    w = HciWorkerImpl([ms_name], _cfg(), 128, 128)
    r0, mem0 = w.image(its[0], its[1])
    r1, _ = w.image(its[1], None)
    assert calls == [its[0].time, its[1].time]  # item 1 read once, by the prefetch
    assert {"pid", "rss_gb", "peak_gb"} <= set(mem0)
    assert (r1["time_index"], r1["band_index"]) == (its[1].time_index, its[1].band_index)
    ref = image_item(read_item(dt[its[1].node_path], its[1].region, _cfg()), _cfg(), None, 128, 128)
    assert_array_equal(r1["cube"], ref["cube"])
    w.close()


def test_errors_keep_their_type_and_name_the_item(ms_name, items):
    its, _ = items
    w = HciWorkerImpl([ms_name], _cfg(), 128, 128, gain_specs=["/no/such/gains.qc::NET"])
    with pytest.raises(FileNotFoundError) as excinfo:
        w.image(its[0], None)
    assert any("while imaging" in n for n in getattr(excinfo.value, "__notes__", []))
    w.close()


def test_mismatched_prefetch_is_a_driver_bug(ms_name, items):
    its, _ = items
    w = HciWorkerImpl([ms_name], _cfg(), 128, 128)
    w.image(its[0], its[1])
    with pytest.raises(RuntimeError, match="prefetch"):
        w.image(its[2], its[3])
    # without a next item too: the stale prefetch must not be silently kept
    with pytest.raises(RuntimeError, match="prefetch"):
        w.image(its[2], None)
    w.close()
