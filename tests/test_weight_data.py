"""weight_data on the MSv4 (time, baseline, chan, corr) grid, against the measurement equation.

Visibilities are simulated as ``V_pq = J_p B J_q^H`` from a known brightness matrix ``B`` and
random per-(time, chan, antenna) gains, so a wrong time, channel or antenna index into the jones
cannot return the input Stokes parameters.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.utils.weighting import weight_data

pmp = pytest.mark.parametrize

NTIME, NBL, NCHAN, NANT = 3, 6, 5, 4
STOKES = {"I": 1.0, "Q": 0.3, "U": -0.2, "V": 0.1}
PAIRS = [(p, q) for p in range(NANT) for q in range(p + 1, NANT)][:NBL]
ANT1 = np.array([p for p, _ in PAIRS], np.int32)
ANT2 = np.array([q for _, q in PAIRS], np.int32)


def _jones(rng, kind, gtime, gchan):
    """Gains in QuartiCal order: (time, chan, ant, dir, corr) or (time, chan, ant, dir, 2, 2)."""
    tail = {"scalar": (1,), "diag": (2,), "full": (2, 2)}[kind]
    shape = (gtime, gchan, NANT, 1) + tail
    jones = 0.1 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    if kind == "full":
        jones += np.eye(2)
    else:
        jones += 1.0
    return jones


def _as_matrices(jones, kind):
    """Broadcast any jones to (NTIME, NCHAN, NANT, 2, 2), direction 0."""
    j = jones[:, :, :, 0]
    if kind != "full":
        j = np.broadcast_to(j, j.shape[:-1] + (2,))[..., None] * np.eye(2)
    return np.broadcast_to(j, (NTIME, NCHAN, NANT, 2, 2))


def _corrupt(jones, kind, ncorr):
    """(time, baseline, chan, corr) visibilities of the STOKES sky seen through ``jones``."""
    i, q, u, v = (STOKES[s] for s in "IQUV")
    bright = np.array([[i + q, u + 1j * v], [u - 1j * v, i - q]])  # linear feeds
    jm = _as_matrices(jones, kind)
    jp = jm[:, :, ANT1]  # (time, chan, baseline, 2, 2)
    jq = jm[:, :, ANT2]
    vis = np.einsum("tcbij,jk,tcblk->tbcil", jp, bright, jq.conj())
    vis = vis.reshape(NTIME, NBL, NCHAN, 4)
    return vis if ncorr == 4 else vis[..., [0, 3]]


# full Jones needs four correlations, and so do U and V; minvar is not implemented for full Jones
_CASES = [
    (kind, ncorr, product, gtime, gchan)
    for kind in ("scalar", "diag", "full")
    for ncorr in (2, 4)
    for product in ("I", "IQ", "IQUV")
    for gtime, gchan in ((NTIME, NCHAN), (1, NCHAN), (NTIME, 1), (1, 1))
    if not (kind == "full" and ncorr == 2) and not (product == "IQUV" and ncorr == 2)
]


@pmp("kind,ncorr,product,gtime,gchan", _CASES)
def test_recovers_the_sky_through_gains(kind, ncorr, product, gtime, gchan):
    rng = np.random.default_rng(42)
    jones = _jones(rng, kind, gtime, gchan)
    data = _corrupt(jones, kind, ncorr)
    weight = rng.uniform(0.5, 2.0, data.shape)
    flag = np.zeros(data.shape, bool)
    vis, wgt = weight_data(data, weight, flag, jones, ANT1, ANT2, "linear", product, str(ncorr), "l2")
    assert vis.shape == wgt.shape == (len(product), NTIME, NBL, NCHAN)
    for k, s in enumerate(product):
        assert_allclose(vis[k], STOKES[s], rtol=1e-12, atol=1e-12)
    assert (wgt > 0).all()


@pmp("kind", ("scalar", "diag"))
@pmp("wgt_mode", ("l2", "minvar"))
@pmp("ncorr", (2, 4))
@pmp("dtype", (np.complex128, np.complex64))
def test_diagonal_weights(kind, wgt_mode, ncorr, dtype):
    """Stokes I weights of a diagonal jones: the gain-scaled parallel-hand weights, summed or 4 * min."""
    rng = np.random.default_rng(3)
    jones = _jones(rng, kind, NTIME, NCHAN).astype(dtype)
    data = _corrupt(jones, kind, ncorr).astype(dtype)
    weight = rng.uniform(0.5, 2.0, data.shape).astype(data.real.dtype)
    flag = np.zeros(data.shape, bool)
    vis, wgt = weight_data(data, weight, flag, jones, ANT1, ANT2, "linear", "I", str(ncorr), wgt_mode)
    assert vis.dtype == dtype and wgt.dtype == weight.dtype
    g2 = np.abs(np.broadcast_to(jones[:, :, :, 0], (NTIME, NCHAN, NANT, 2))).astype(np.float64) ** 2
    scaled = np.transpose(g2[:, :, ANT1] * g2[:, :, ANT2], (0, 2, 1, 3)) * weight[..., [0, -1]]
    expected = scaled.sum(axis=-1) if wgt_mode == "l2" else 4 * scaled.min(axis=-1)
    assert_allclose(wgt[0], expected, rtol=1e-12 if dtype == np.complex128 else 1e-5)


def test_full_jones_weights_reduce_to_the_diagonal_ones():
    """A full Jones that happens to be diagonal weights every Stokes parameter as the diagonal kernel does."""
    rng = np.random.default_rng(5)
    diag = _jones(rng, "diag", NTIME, NCHAN)
    full = np.zeros(diag.shape + (2,), diag.dtype)
    full[..., 0, 0] = diag[..., 0]
    full[..., 1, 1] = diag[..., 1]
    data = _corrupt(diag, "diag", 4)
    weight = rng.uniform(0.5, 2.0, data.shape)
    flag = rng.uniform(size=data.shape) < 0.1
    dvis, dwgt = weight_data(data, weight, flag, diag, ANT1, ANT2, "linear", "IQUV", "4", "l2")
    fvis, fwgt = weight_data(data, weight, flag, full, ANT1, ANT2, "linear", "IQUV", "4", "l2")
    assert np.count_nonzero(dwgt) not in (0, dwgt.size)  # some samples flagged, not all
    assert_allclose(fvis, dvis, rtol=1e-12, atol=1e-12)
    assert_allclose(fwgt, dwgt, rtol=1e-12)


def test_flagged_samples_are_zero():
    rng = np.random.default_rng(1)
    jones = _jones(rng, "diag", 1, 1)
    data = _corrupt(jones, "diag", 2)
    weight = np.ones(data.shape)
    flag = np.zeros(data.shape, bool)
    flag[1, 2, 3, 0] = True  # one correlation flags the sample
    vis, wgt = weight_data(data, weight, flag, jones, ANT1, ANT2, "linear", "I", "2", "l2")
    assert vis[0, 1, 2, 3] == 0 and wgt[0, 1, 2, 3] == 0
    assert np.count_nonzero(wgt) == wgt.size - 1


@pmp(
    "shape",
    (
        (2, 1, NANT, 1, 2),  # time axis off the data's
        (1, 3, NANT, 1, 2),  # channel axis off the data's
        (1, 1, NANT, 1, 3),  # three diagonal correlations
        (1, 1, NANT, 2, 2),  # two directions, or a full Jones missing its direction axis
        (1, 1, NANT, 0, 2),  # no direction
        (1, 1, NANT, 1, 2, 1),  # full Jones that is not 2x2
        (1, 1, NANT, 1, 3, 3),
    ),
)
def test_refuses_a_jones_it_would_misread(shape):
    ncorr = 2 if len(shape) == 5 else 4  # a full Jones needs four correlations
    data = np.zeros((NTIME, NBL, NCHAN, ncorr), np.complex128)
    weight = np.ones(data.shape)
    flag = np.zeros(data.shape, bool)
    with pytest.raises(ValueError):
        weight_data(data, weight, flag, np.ones(shape, np.complex128), ANT1, ANT2, "linear", "I", str(ncorr), "l2")
