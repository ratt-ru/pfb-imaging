"""QuartiCal gain reader for MSv4 nodes."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from rarg_python_patterns.multiton import Multiton

from pfb_imaging.utils.gains import check_node_grid, load_node_gains, open_term, term_path
from pfb_imaging.utils.misc import to_mjd_time

ANTS = ["m000", "m001", "m002"]


def _write_term(root, groups, term="NET", ncorr=2, ants=ANTS):
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
                "antenna": ants,
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


def test_broadcast_axes(tmp_path):
    times, freqs = _grid()
    g, f = _diag(1, 1, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times[:1], freqs[:1], g, f)])
    out = load_node_gains(spec, "1", times, freqs, ANTS)
    assert out.jones.shape == (1, 1, 3, 2)


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


def _ramp(ntime, nchan):
    """Diagonal gains whose value encodes (time, channel): 1 + t + 0.1 c."""
    g = np.empty((ntime, nchan, len(ANTS), 1, 2), dtype=np.complex128)
    g[...] = (1.0 + np.arange(ntime)[:, None] + 0.1 * np.arange(nchan)[None, :])[:, :, None, None, None]
    return g, np.zeros(g.shape[:-1], np.int8)


def test_channel_and_time_subset_of_the_term(tmp_path):
    """A region (channel subset, time subset) of a data-resolution term selects and aligns."""
    times, freqs = _grid(ntime=4, nchan=8)
    g, f = _ramp(4, 8)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    out = load_node_gains(spec, "1", times[1:3], freqs[2:5], ANTS)
    assert out.jones.shape == (2, 3, 3, 2)
    assert_array_equal(out.jones[..., 0, 0], g[1:3, 2:5, 0, 0, 0])
    check_node_grid(spec, "1", times[1:3], freqs[2:5])


def test_descending_gain_frequencies(tmp_path):
    times, freqs = _grid(ntime=2, nchan=5)
    g, f = _ramp(2, 5)
    # the term is written high-to-low; the data asks low-to-high, and for a subset
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs[::-1], g, f)])
    out = load_node_gains(spec, "1", times, freqs[1:4], ANTS)
    assert_array_equal(out.jones[..., 0, 0], g[:, ::-1][:, 1:4, 0, 0, 0])


def test_single_solution_interval_broadcasts_without_scan_attr(tmp_path):
    """No SCAN_NUMBER: a length-1 gain_time counts as overlapping the node; the spw decides."""
    times, freqs = _grid()
    g1, f1 = _diag(1, 4, 2.0)
    g2, f2 = _diag(1, 4, 3.0)
    spec = _write_term(
        tmp_path,
        [
            ({"FIELD_ID": 0, "DATA_DESC_ID": 0}, times[:1] - 500.0, freqs, g1, f1),
            ({"FIELD_ID": 0, "DATA_DESC_ID": 1}, times[:1] - 500.0, freqs + 1e8, g2, f2),
        ],
    )
    out = load_node_gains(spec, "3", times, freqs[1:3] + 1e8, ANTS)
    assert out.jones.shape == (1, 2, 3, 2)
    assert np.all(out.jones == 3.0)


def test_check_node_grid_refuses_off_grid_channels(tmp_path):
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    with pytest.raises(ValueError, match="has the channels of"):
        check_node_grid(spec, "1", times, freqs + 0.5e6)
    with pytest.raises(ValueError, match="on-grid only"):
        check_node_grid(spec, "1", times + 3.0, freqs)


def test_field_name_disambiguates_a_term_solved_across_scans(tmp_path):
    """Interleaved target and calibrator, solved per field across scans: both overlap in time."""
    times, freqs = _grid(ntime=4)
    g1, f1 = _diag(2, 4, 2.0)
    g2, f2 = _diag(2, 4, 3.0)
    spec = _write_term(
        tmp_path,
        [
            ({"FIELD_ID": 0, "FIELD_NAME": "cal", "DATA_DESC_ID": 0}, times[[0, 3]], freqs, g1, f1),
            ({"FIELD_ID": 1, "FIELD_NAME": "target", "DATA_DESC_ID": 0}, times[[1, 2]], freqs, g2, f2),
        ],
    )
    out = load_node_gains(spec, "5", times[1:3], freqs, ANTS, field_name="target")
    assert np.all(out.jones == 3.0)
    with pytest.raises(ValueError, match="2 gain partitions"):
        check_node_grid(spec, "5", times[1:3], freqs)  # without the field name it is ambiguous


def test_another_fields_term_is_refused(tmp_path):
    times, freqs = _grid()
    g, f = _diag(1, 4, 2.0)
    spec = _write_term(tmp_path, [({"FIELD_NAME": "cal", "DATA_DESC_ID": 0}, times[:1] - 1e6, freqs, g, f)])
    with pytest.raises(ValueError, match="No gain partitions .* for field target"):
        load_node_gains(spec, "1", times, freqs, ANTS, field_name="target")


def test_no_shared_antenna_name_is_refused(tmp_path):
    """A term whose antenna coordinate is integer ids would otherwise flag every visibility."""
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)], ants=[0, 1, 2])
    with pytest.raises(ValueError, match="matched to antennas by name"):
        load_node_gains(spec, "1", times, freqs, ANTS)
    with pytest.raises(ValueError, match="matched to antennas by name"):
        check_node_grid(spec, "1", times, freqs, antenna_names=ANTS)


def test_check_node_grid_reports_missing_antennas(tmp_path):
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    assert check_node_grid(spec, "1", times, freqs, antenna_names=ANTS + ["m099"]) == ["m099"]
    assert check_node_grid(spec, "1", times, freqs) == []


def test_only_the_region_is_read_and_nothing_is_cached(tmp_path):
    """The term stays in a process-lifetime Multiton, so a full .values would stay resident."""
    times, freqs = _grid(ntime=4, nchan=8)
    g, f = _ramp(4, 8)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    # unsorted and repeated wanted points still come back in the order asked for
    want_t, want_f = times[[2, 0, 2]], freqs[[5, 1]]
    out = load_node_gains(spec, "1", want_t, want_f, ANTS)
    assert_array_equal(out.jones[..., 0, 0], g[[2, 0, 2]][:, [5, 1], 0, 0, 0])
    (ds,) = Multiton(open_term, str(tmp_path / "NET")).instance
    assert not ds.gains.variable._in_memory
    assert not ds.gain_flags.variable._in_memory


def test_a_table_without_array_dimensions_is_refused_clearly(tmp_path):
    times, freqs = _grid()
    g, f = _diag(3, 4, 2.0)
    spec = _write_term(tmp_path, [({"SCAN_NUMBER": 1}, times, freqs, g, f)])
    zattrs = tmp_path / "NET" / "NET_0" / "gains" / ".zattrs"
    zattrs.write_text("{}")
    with pytest.raises(ValueError, match="older"):
        open_term(spec)
