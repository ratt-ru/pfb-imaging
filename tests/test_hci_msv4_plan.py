"""plan_images: one WorkItem per output image, hci's time/band semantics."""

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.core.hci_msv4 import plan_images
from pfb_imaging.utils.misc import to_mjd_time
from pfb_imaging.utils.msv4 import get_engine


@pytest.fixture
def dts(ms_name):
    return [xr.open_datatree(ms_name, **get_engine(ms_name))]


def _node(dts):
    return next(n for n in dts[0].subtree if n.attrs.get("type") == "visibility")


def test_one_item_per_integration(dts):
    node = _node(dts)
    ntime, nchan = node.ds.sizes["time"], node.ds.sizes["frequency"]
    plan = plan_images(dts, integrations_per_image=1, channels_per_image=-1, images_per_chunk=4)
    assert len(plan.items) == ntime
    assert plan.out_times.size == ntime and plan.out_freqs.size == 1
    np.testing.assert_allclose(plan.out_times, to_mjd_time(node.ds.time.values))
    assert all(it.frequency == slice(0, nchan) for it in plan.items)
    assert [it.time_index for it in plan.items] == list(range(ntime))
    assert sum(plan.expected.values()) == ntime
    assert plan.expected[(0, 0)] == 4
    assert plan.expected[(0, (ntime - 1) // 4)] == ntime - 4 * ((ntime - 1) // 4)


def test_integrations_and_channels_per_image(dts):
    node = _node(dts)
    ntime, nchan = node.ds.sizes["time"], node.ds.sizes["frequency"]
    plan = plan_images(dts, integrations_per_image=7, channels_per_image=3, images_per_chunk=2)
    nimg_t = -(-ntime // 7)
    nband = -(-nchan // 3)
    assert len(plan.items) == nimg_t * nband
    assert plan.out_freqs.size == nband
    last = max(plan.items, key=lambda it: (it.time_index, it.band_index))
    assert last.time == slice(7 * (nimg_t - 1), ntime)
    assert last.frequency == slice(3 * (nband - 1), nchan)
    # sorted by (block, band, time)
    keys = [(it.block, it.band_index, it.time_index) for it in plan.items]
    assert keys == sorted(keys)


def test_whole_scan_is_one_image(dts):
    plan = plan_images(dts, integrations_per_image=-1, channels_per_image=-1, images_per_chunk=128)
    assert len(plan.items) == 1 and plan.out_times.size == 1


def test_plan_freq_range_offsets_regions_by_chan0(dts):
    node = _node(dts)
    freqs = node.ds.frequency.values
    plan = plan_images(
        dts, integrations_per_image=1, channels_per_image=-1, images_per_chunk=4, freq_min=freqs[2], freq_max=freqs[-2]
    )
    assert all(it.frequency == slice(2, freqs.size - 1) for it in plan.items)
    np.testing.assert_allclose(plan.out_freqs, [freqs[2:-1].mean()])
    np.testing.assert_allclose(plan.channel_width, [freqs[-2] - freqs[2]])


def test_duplicate_output_images_are_refused(dts):
    with pytest.raises(ValueError, match="same output image"):
        plan_images(dts + dts, integrations_per_image=1, channels_per_image=-1, images_per_chunk=4)


def test_empty_selection_is_refused(dts):
    with pytest.raises(ValueError, match="matched no data"):
        plan_images(dts, integrations_per_image=1, channels_per_image=-1, images_per_chunk=4, scan_names=["nope"])


def test_integrations_per_image_must_be_positive_or_sentinel(dts):
    with pytest.raises(ValueError, match="integrations_per_image"):
        plan_images(dts, integrations_per_image=-3, channels_per_image=-1, images_per_chunk=4)


def test_channels_per_image_must_be_positive_or_sentinel(dts):
    with pytest.raises(ValueError, match="channels_per_image"):
        plan_images(dts, integrations_per_image=1, channels_per_image=-3, images_per_chunk=4)


def test_images_per_chunk_must_be_positive(dts):
    with pytest.raises(ValueError, match="images_per_chunk"):
        plan_images(dts, integrations_per_image=1, channels_per_image=-1, images_per_chunk=0)


def test_plan_wraps_ra_and_keeps_full_spw_for_transients(dts):
    node = _node(dts)
    freqs = node.ds.frequency.values
    plan = plan_images(
        dts, integrations_per_image=1, channels_per_image=-1, images_per_chunk=4, freq_min=freqs[2], freq_max=freqs[-2]
    )
    assert ((plan.field_radecs[:, 0] >= 0) & (plan.field_radecs[:, 0] < 2 * np.pi)).all()
    np.testing.assert_array_equal(plan.all_freqs, np.unique(freqs))
