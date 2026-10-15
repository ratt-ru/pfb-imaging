"""QuartiCal gain reader for MSv4 nodes."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from pfb_imaging.utils.gains import load_node_gains, term_path
from pfb_imaging.utils.misc import to_mjd_time

ANTS = ["m000", "m001", "m002"]


def _write_term(root, groups, term="NET", ncorr=2):
    """Write QuartiCal's layout: <root>/<term>/<term>_<n>, one group per partition."""
    for n, (attrs, times_unix, freqs, gains, gflags) in enumerate(groups):
        ds = xr.Dataset(
            {
                "gains": (("gain_time", "gain_freq", "antenna", "direction", "correlation"), gains),
                "gain_flags": (("gain_time", "gain_freq", "antenna", "direction"), gflags),
            },
            coords={
                "gain_time": to_mjd_time(np.asarray(times_unix, dtype=float)),
                "gain_freq": freqs,
                "antenna": ANTS,
                "direction": [0],
                "correlation": ["XX", "XY", "YX", "YY"][:ncorr] if ncorr == 4 else ["XX", "YY"],
            },
            attrs=attrs,
        )
        ds.to_zarr(f"{root}/{term}/{term}_{n}", mode="w")
    return f"{root}/{term}"


def _grid(ntime=3, nchan=4, t0=1.6e9):
    return t0 + 8.0 * np.arange(ntime), 1e9 + 1e6 * np.arange(nchan)


def _diag(ntime, nchan, value=2.0 + 0j):
    g = np.full((ntime, nchan, len(ANTS), 1, 2), value, dtype=np.complex128)
    return g, np.zeros(g.shape[:-1], np.int8)


def test_term_path_accepts_both_spellings():
    assert term_path("gains.qc::NET") == "gains.qc/NET"
    assert term_path("gains.qc/NET/") == "gains.qc/NET"


def test_selects_by_scan_number_and_converts_mjd(tmp_path):
    times, freqs = _grid()
    g1, f1 = _diag(3, 4, 2.0)
    g2, f2 = _diag(3, 4, 3.0)
    spec = _write_term(
        tmp_path,
        [
            ({"FIELD_ID": 0, "DATA_DESC_ID": 0, "SCAN_NUMBER": 1}, times, freqs, g1, f1),
            ({"FIELD_ID": 0, "DATA_DESC_ID": 0, "SCAN_NUMBER": 2}, times + 100, freqs, g2, f2),
        ],
    )
    out = load_node_gains(spec, "2", times + 100, freqs, ANTS)
    assert out.jones.shape == (3, 4, 3, 2)
    assert np.all(out.jones == 3.0)
    assert not out.flags.any()


def test_selects_by_time_overlap_without_scan_attr(tmp_path):
    times, freqs = _grid()
    g1, f1 = _diag(3, 4, 2.0)
    g2, f2 = _diag(3, 4, 5.0)
    spec = _write_term(
        tmp_path,
        [
            ({"FIELD_ID": 0, "DATA_DESC_ID": 0}, times, freqs, g1, f1),
            ({"FIELD_ID": 1, "DATA_DESC_ID": 0}, times + 1000, freqs, g2, f2),
        ],
    )
    out = load_node_gains(spec, "7", times + 1000, freqs, ANTS)
    assert np.all(out.jones == 5.0)


def test_frequency_grid_disambiguates_spws(tmp_path):
    times, freqs = _grid()
    g1, f1 = _diag(3, 4, 2.0)
    g2, f2 = _diag(3, 4, 4.0)
    spec = _write_term(
        tmp_path,
        [
            ({"FIELD_ID": 0, "DATA_DESC_ID": 0, "SCAN_NUMBER": 1}, times, freqs, g1, f1),
            ({"FIELD_ID": 0, "DATA_DESC_ID": 1, "SCAN_NUMBER": 1}, times, freqs + 1e8, g2, f2),
        ],
    )
    out = load_node_gains(spec, "1", times, freqs + 1e8, ANTS)
    assert np.all(out.jones == 4.0)


def test_missing_antenna_is_identity_and_flagged(tmp_path):
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    out = load_node_gains(spec, "1", times, freqs, ANTS + ["m099"])
    assert np.all(out.jones[:, :, 3] == 1.0)
    assert out.flags[:, :, 3].all()
    assert not out.flags[:, :, :3].any()


def test_flags_fold_gain_flags_and_non_finite(tmp_path):
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    f[0, 0, 1, 0] = 1
    g[1, 2, 2, 0, 0] = np.nan
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    out = load_node_gains(spec, "1", times, freqs, ANTS)
    expected = np.zeros((3, 4, 3), bool)
    expected[0, 0, 1] = True
    expected[1, 2, 2] = True
    assert_array_equal(out.flags, expected)


def test_full_jones_layout(tmp_path):
    times, freqs = _grid()
    g = np.zeros((3, 4, 3, 1, 4), np.complex128)
    g[..., 0] = 1.0
    g[..., 1] = 0.5
    g[..., 3] = 1.0
    f = np.zeros(g.shape[:-1], np.int8)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)], ncorr=4)
    out = load_node_gains(spec, "1", times, freqs, ANTS)
    assert out.jones.shape == (3, 4, 3, 2, 2)
    assert out.jones[0, 0, 0, 0, 1] == 0.5


def test_broadcast_axes_and_subset(tmp_path):
    times, freqs = _grid()
    g, f = _diag(1, 1, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times[:1], freqs[:1], g, f)])
    out = load_node_gains(spec, "1", times, freqs, ANTS)
    assert out.jones.shape == (1, 1, 3, 2)
    sub = out.subset(slice(1, 3), slice(0, 2))
    assert sub.jones.shape == (1, 1, 3, 2)


def test_off_grid_gains_are_refused(tmp_path):
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    with pytest.raises(ValueError, match="on-grid only"):
        load_node_gains(spec, "1", times + 3.0, freqs, ANTS)


def test_no_matching_partition(tmp_path):
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    with pytest.raises(ValueError, match="No gain partitions"):
        load_node_gains(spec, "9", times, freqs, ANTS)
