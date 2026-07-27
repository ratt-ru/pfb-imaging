"""The single-field guard in pfb_imaging.core.init.

The check must fire on an offset in either direction (the original form compared
the signed difference, so half of all mismatches went undetected) without firing
on a pair that merely straddles RA = 0, since construct_mappings normalises ra
into [0, 2pi).
"""

import numpy as np
import pytest

from pfb_imaging.core.init import _phase_dirs_agree

pmp = pytest.mark.parametrize

TOL = 1e-8


def test_identical_phase_dirs_agree():
    radec = np.array([1.234, -0.567])
    assert _phase_dirs_agree(radec, radec.copy(), TOL)


@pmp("sign", (1, -1))
@pmp("axis", (0, 1))
def test_offset_within_tolerance_agrees(sign, axis):
    radec = np.array([1.234, -0.567])
    other = radec.copy()
    other[axis] += sign * TOL / 2
    assert _phase_dirs_agree(radec, other, TOL)


@pmp("sign", (1, -1))
@pmp("axis", (0, 1))
def test_offset_beyond_tolerance_disagrees(sign, axis):
    """Both signs must be caught - the negative direction is the regression."""
    radec = np.array([1.234, -0.567])
    other = radec.copy()
    other[axis] += sign * 10 * TOL
    assert not _phase_dirs_agree(radec, other, TOL)


def test_ra_wrap_is_not_a_mismatch():
    """Centres either side of RA = 0 are 2e-9 rad apart, not 2*pi."""
    dec = -0.567
    assert _phase_dirs_agree(np.array([1e-9, dec]), np.array([2 * np.pi - 1e-9, dec]), TOL)


def test_distinct_fields_disagree():
    assert not _phase_dirs_agree(np.array([1.234, -0.567]), np.array([2.345, 0.123]), TOL)
