"""Normalised frequency-precision matrix for the GP preconditioner prior (issue #307).

The prior generalises ``--eta`` from a scalar to an ``nband x nband`` matrix. The
invariants that make it a safe generalisation are the ones tested here:

* the maximum precision is exactly 1, so ``lambda_max(M)`` -- hence ``hess_norm`` and
  the primal-dual step sizes -- is unchanged;
* ``length_scale -> 0`` gives exactly the identity, so a vanishing correlation length
  degrades to today's uniform behaviour rather than to a uniform ``eta/cap``;
* the spectrum never dips below ``1/cap``, bounding the erosion of the stable gamma;
* the matrix stays symmetric positive definite, so CG still applies.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.operators.hessian import freq_correlation, freq_precision

FREQ = np.linspace(0.86e9, 1.71e9, 8)  # MeerKAT L-band, 8 bands


def test_disabled_when_length_scale_is_none():
    assert freq_precision(FREQ, None) is None


def test_disabled_for_a_single_band():
    assert freq_precision(np.array([1.0e9]), 0.5) is None


def test_disabled_when_every_band_shares_one_frequency():
    """Zero span would divide by zero; the prior has nothing to correlate over."""
    assert freq_precision(np.full(4, 1.0e9), 0.5) is None


def test_correlation_is_unit_diagonal_and_symmetric():
    corr = freq_correlation(FREQ, 0.5)
    assert corr.shape == (FREQ.size, FREQ.size)
    assert_allclose(np.diag(corr), 1.0, rtol=0, atol=1e-15)
    assert_allclose(corr, corr.T, rtol=0, atol=1e-15)


def test_correlation_decays_monotonically_with_frequency_separation():
    corr = freq_correlation(FREQ, 0.5)
    row = corr[0]
    assert np.all(np.diff(row) < 0), f"row 0 is not monotone decreasing: {row}"


@pytest.mark.parametrize("length_scale", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("cap", [3.0, 10.0, 100.0])
def test_spectrum_is_bounded_with_the_maximum_attained_exactly(length_scale, cap):
    """Max exactly 1 keeps lambda_max(M) fixed; the floor bounds the gamma erosion.

    The minimum is 1/min(spread, cap) and only reaches 1/cap when the correlation
    spectrum spans at least cap, so assert containment -- not an exact floor.
    """
    kinv = freq_precision(FREQ, length_scale, cap=cap)
    lam = np.linalg.eigvalsh(kinv)
    assert_allclose(lam.max(), 1.0, rtol=1e-12)
    assert lam.min() >= 1.0 / cap - 1e-12, f"floor {lam.min():.3e} below 1/cap"
    assert lam.min() > 0.0, "precision must stay positive definite"


def test_short_length_scale_degrades_to_the_identity():
    """The ell -> 0 limit must recover today's uniform eta, not a uniform eta/cap."""
    kinv = freq_precision(FREQ, 1e-4, cap=10.0)
    assert_allclose(kinv, np.eye(FREQ.size), rtol=0, atol=1e-9)


def test_long_length_scale_relaxes_the_smoothest_mode_to_the_cap():
    cap = 10.0
    kinv = freq_precision(FREQ, 5.0, cap=cap)
    lam = np.linalg.eigvalsh(kinv)
    assert_allclose(lam.min(), 1.0 / cap, rtol=1e-9)


def test_precision_is_symmetric():
    kinv = freq_precision(FREQ, 0.5, cap=10.0)
    assert_allclose(kinv, kinv.T, rtol=0, atol=1e-14)


def test_uneven_and_unsorted_frequencies_are_handled_by_value():
    """freq_out is data-dependent (wiki D28); nothing may assume even spacing."""
    freq = np.array([1.0e9, 1.02e9, 1.5e9])
    corr = freq_correlation(freq, 0.5)
    assert corr[0, 1] > corr[0, 2], "near bands must correlate more than far ones"


def test_bad_length_scale_and_cap_are_rejected():
    with pytest.raises(ValueError, match="gp_length_scale"):
        freq_precision(FREQ, 0.0)
    with pytest.raises(ValueError, match="gp_length_scale"):
        freq_precision(FREQ, -1.0)
    with pytest.raises(ValueError, match="gp_cap"):
        freq_precision(FREQ, 0.5, cap=0.5)
