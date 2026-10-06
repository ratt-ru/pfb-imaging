"""Unit tests for antenna classification and baseline-group masks.

Inputs are built in memory rather than read from an MS, so these run on every
CI leg including the casacore-free ones (see .claude/rules/testing-and-ci.md §1).
"""

import numpy as np
import pytest

from pfb_imaging.utils.baselines import (
    GROUP_LABELS,
    baseline_group_masks,
    check_telescope_is_meerkat,
    classify_antennas,
    parse_antenna_groups,
)

MK = ["m000", "m001", "m002"]
EXT = ["e001", "e002"]
NAMES = np.array(MK + EXT)
DIAMS = np.array([13.5, 13.5, 13.5, 15.0, 15.0])


def _all_baselines(names):
    """Every antenna pair including autos, as (ant1_names, ant2_names)."""
    a1, a2 = [], []
    for i in range(len(names)):
        for j in range(i, len(names)):
            a1.append(names[i])
            a2.append(names[j])
    return np.array(a1), np.array(a2)


def test_classifies_on_dish_diameter():
    is_ext = classify_antennas(NAMES, DIAMS)
    assert is_ext.tolist() == [False, False, False, True, True]


def test_larger_dish_is_meerkat_plus_when_names_carry_no_convention():
    """Diameter leads; the m/e prefix only corroborates where it is present."""
    names = np.array(["ant0", "ant1"])
    assert classify_antennas(names, np.array([15.0, 13.5])).tolist() == [True, False]


def test_refuses_uniform_diameters():
    with pytest.raises(ValueError, match="single dish diameter"):
        classify_antennas(NAMES, np.full(5, 13.5))


def test_refuses_more_than_two_diameters():
    with pytest.raises(ValueError, match="3 distinct dish diameters"):
        classify_antennas(NAMES, np.array([13.5, 13.5, 15.0, 15.0, 18.0]))


def test_refuses_when_diameter_and_prefix_disagree():
    """m002 has the large dish: the convention and the hardware contradict."""
    diams = np.array([13.5, 13.5, 15.0, 15.0, 15.0])
    with pytest.raises(ValueError, match="disagree"):
        classify_antennas(NAMES, diams)


def test_error_names_the_offending_antennas():
    diams = np.array([13.5, 13.5, 15.0, 15.0, 15.0])
    with pytest.raises(ValueError, match="m002"):
        classify_antennas(NAMES, diams)


def test_override_bypasses_diameter_entirely():
    """Uniform diameters are fine when the user names the groups."""
    is_ext = classify_antennas(NAMES, np.full(5, 13.5), override=("m*", "e*"))
    assert is_ext.tolist() == [False, False, False, True, True]


def test_override_works_without_diameters():
    is_ext = classify_antennas(NAMES, None, override=("m*", "e*"))
    assert is_ext.tolist() == [False, False, False, True, True]


def test_refuses_without_diameters_and_without_override():
    with pytest.raises(ValueError, match="ANTENNA_DISH_DIAMETER"):
        classify_antennas(NAMES, None)


def test_override_refuses_uncovered_antennas():
    with pytest.raises(ValueError, match="m002"):
        classify_antennas(NAMES, DIAMS, override=("m00[01]", "e*"))


def test_override_refuses_overlapping_patterns():
    with pytest.raises(ValueError, match="both"):
        classify_antennas(NAMES, DIAMS, override=("*", "e*"))


def test_override_matches_glob_metacharacters_literally_enough():
    """A name containing a bracket must still be classifiable via an exact pattern."""
    names = np.array(["m[0]", "e000"])
    is_ext = classify_antennas(names, None, override=("m[[]0]", "e*"))
    assert is_ext.tolist() == [False, True]


def test_parse_antenna_groups_splits_on_comma():
    assert parse_antenna_groups("m*,e*") == ("m*", "e*")


def test_parse_antenna_groups_strips_whitespace():
    assert parse_antenna_groups(" m* , e* ") == ("m*", "e*")


@pytest.mark.parametrize("spec", ["m*", "m*,e*,x*", "", ","])
def test_parse_antenna_groups_refuses_wrong_count(spec):
    with pytest.raises(ValueError, match="exactly two"):
        parse_antenna_groups(spec)


def test_masks_cover_every_baseline_exactly_once():
    a1, a2 = _all_baselines(NAMES)
    is_ext = classify_antennas(NAMES, DIAMS)
    masks = baseline_group_masks(is_ext, NAMES, a1, a2)
    assert set(masks) <= set(GROUP_LABELS)
    joined = np.concatenate([masks[k] for k in masks])
    assert sorted(joined.tolist()) == list(range(len(a1)))


def test_mask_group_membership_is_correct():
    a1, a2 = _all_baselines(NAMES)
    is_ext = classify_antennas(NAMES, DIAMS)
    masks = baseline_group_masks(is_ext, NAMES, a1, a2)
    # 3 MeerKAT, 2 MeerKAT+, autos included: 3*4/2=6 MM, 3*2=6 MPM, 2*3/2=3 MPMP
    assert len(masks["MM"]) == 6
    assert len(masks["MPM"]) == 6
    assert len(masks["MPMP"]) == 3


def test_cross_group_ignores_antenna_ordering():
    """An e-first baseline is still MPM."""
    is_ext = classify_antennas(NAMES, DIAMS)
    masks = baseline_group_masks(is_ext, NAMES, np.array(["e001"]), np.array(["m000"]))
    assert list(masks) == ["MPM"]


def test_empty_groups_are_omitted():
    """A pure-MeerKAT selection yields MM only -- not three keys, two of them empty."""
    names = np.array(MK)
    a1, a2 = _all_baselines(names)
    is_ext = np.zeros(len(names), dtype=bool)
    masks = baseline_group_masks(is_ext, names, a1, a2)
    assert list(masks) == ["MM"]


def test_masks_index_the_baseline_axis_in_order():
    a1, a2 = _all_baselines(NAMES)
    is_ext = classify_antennas(NAMES, DIAMS)
    masks = baseline_group_masks(is_ext, NAMES, a1, a2)
    for idx in masks.values():
        assert idx.dtype.kind == "i"
        assert (np.diff(idx) > 0).all()


def test_telescope_check_accepts_meerkat_case_insensitively():
    for name in ("MeerKAT", "MEERKAT", "meerkat", " MeerKAT "):
        check_telescope_is_meerkat(name, "/path/to.ms")


def test_telescope_check_refuses_other_arrays_naming_what_it_found():
    with pytest.raises(ValueError, match="vla"):
        check_telescope_is_meerkat("vla", "/path/to.ms")


def test_telescope_check_names_the_ms():
    with pytest.raises(ValueError, match="to.ms"):
        check_telescope_is_meerkat("vla", "/path/to.ms")
