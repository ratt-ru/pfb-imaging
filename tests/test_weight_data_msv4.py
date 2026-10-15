"""weight_data_msv4 must equal the row-ordered weight_data on reshaped inputs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.utils.weighting import weight_data, weight_data_msv4

pmp = pytest.mark.parametrize


def _inputs(rng, ntime, nbl, nchan, ncorr, full, gtime, gchan, nant=4):
    shape = (ntime, nbl, nchan, ncorr)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    weight = rng.uniform(0.5, 2.0, shape)
    flag = rng.uniform(size=shape) < 0.1
    pairs = [(p, q) for p in range(nant) for q in range(p + 1, nant)][:nbl]
    ant1 = np.array([p for p, _ in pairs], np.int32)
    ant2 = np.array([q for _, q in pairs], np.int32)
    gshape = (gtime, gchan, nant, 2, 2) if full else (gtime, gchan, nant, 2)
    jones = (1 + 0.1 * rng.standard_normal(gshape) + 0.1j * rng.standard_normal(gshape)).astype(np.complex128)
    return data, weight, flag, jones, ant1, ant2


def _legacy(data, weight, flag, jones, ant1, ant2, pol, product, wgt_mode):
    ntime, nbl, nchan, ncorr = data.shape
    nrow = ntime * nbl
    # legacy jones: (time, ant, chan, dir, 2[, 2]), broadcast to the full grid
    j = np.broadcast_to(jones, (ntime, nchan) + jones.shape[2:])
    j = np.swapaxes(j, 1, 2)[:, :, :, None]
    vis, wgt = weight_data(
        data.reshape(nrow, nchan, ncorr),
        weight.reshape(nrow, nchan, ncorr),
        flag.reshape(nrow, nchan, ncorr),
        np.ascontiguousarray(j),
        np.arange(ntime, dtype=np.int32) * nbl,
        np.full(ntime, nbl, np.int32),
        np.tile(ant1, ntime),
        np.tile(ant2, ntime),
        pol,
        product,
        str(ncorr),
        wgt_mode,
    )
    ns = vis.shape[-1]
    return (
        vis.transpose(2, 0, 1).reshape(ns, ntime, nbl, nchan),
        wgt.transpose(2, 0, 1).reshape(ns, ntime, nbl, nchan),
    )


@pmp("full", [False, True])
@pmp("ncorr", [2, 4])
@pmp("product", ["I", "IQUV"])
@pmp("wgt_mode", ["l2", "minvar"])
@pmp("gtime,gchan", [(3, 5), (1, 1)])
def test_matches_legacy_kernel(full, ncorr, product, wgt_mode, gtime, gchan):
    if full and ncorr == 2:
        pytest.skip("full Jones needs four correlations")
    if full and wgt_mode == "minvar":
        pytest.skip("minvar weighting is not implemented for full 2x2 Jones (stokes_expr_funcs)")
    if product == "IQUV" and ncorr == 2:
        pytest.skip("Q/U need the cross hands")
    rng = np.random.default_rng(42)
    data, weight, flag, jones, ant1, ant2 = _inputs(rng, 3, 6, 5, ncorr, full, gtime, gchan)
    vis, wgt = weight_data_msv4(data, weight, flag, jones, ant1, ant2, "linear", product, str(ncorr), wgt_mode)
    lvis, lwgt = _legacy(data, weight, flag, jones, ant1, ant2, "linear", product, wgt_mode)
    assert vis.shape == (len(product), 3, 6, 5)
    assert_allclose(vis, lvis, rtol=1e-12, atol=1e-12)
    assert_allclose(wgt, lwgt, rtol=1e-12, atol=1e-12)


def test_flagged_samples_are_zero():
    rng = np.random.default_rng(1)
    data, weight, flag, jones, ant1, ant2 = _inputs(rng, 2, 3, 4, 2, False, 1, 1)
    flag[:] = False
    flag[1, 2, 3, 0] = True
    vis, wgt = weight_data_msv4(data, weight, flag, jones, ant1, ant2, "linear", "I", "2", "l2")
    assert vis[0, 1, 2, 3] == 0 and wgt[0, 1, 2, 3] == 0
    assert np.count_nonzero(wgt) == wgt.size - 1
