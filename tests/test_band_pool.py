"""The shared BandWorkerPool fixture: reuse must be numerically invisible.

A session-scoped pool is only sound because `init_hess` rebuilds every
per-band HessianTree from scratch. These tests are what pin that claim --
without them the optimisation is a silent correctness risk, not a speedup.
"""

import numpy as np
import pytest
from ducc0.fft import r2c
from numpy.testing import assert_allclose

from pfb_imaging.operators.band_worker import BandWorkerPool
from pfb_imaging.operators.hessian import HessTreeRay


def _part(rng, nx, ny, nx_psf, ny_psf, wsum=1.0):
    """Partition dict with a positive real psfhat, as HessianTree expects."""
    psf = rng.uniform(0.0, 1.0, size=(nx_psf, ny_psf))
    psfhat = np.abs(r2c(psf, axes=(0, 1), forward=True, inorm=0))[None]
    return {"psfhat": psfhat, "beam": np.ones((1, nx, ny)), "wsum": np.array([wsum])}


def test_band_pool_reuse_is_equivalent_to_a_private_pool(band_pool):
    """A pool used twice, at two geometries, gives bitwise what a fresh one does.

    The second construction deliberately changes nx/ny AND eta, so a pool that
    cached either would disagree.
    """
    rng = np.random.default_rng(90)
    shared = band_pool(2)

    # first use at 8x8, eta=0.5 -- dirties the pool
    warm_parts = [[_part(rng, 8, 8, 16, 16)] for _ in range(2)]
    HessTreeRay(warm_parts, 8, 8, 16, 16, etas=0.5, workers=shared).dot(rng.standard_normal((2, 8, 8)))

    # second use at 16x16, eta=0.01 -- the measurement
    parts = [[_part(rng, 16, 16, 32, 32)] for _ in range(2)]
    x = rng.standard_normal((2, 16, 16))

    reused = HessTreeRay(parts, 16, 16, 32, 32, etas=0.01, workers=shared).dot(x)

    private = BandWorkerPool(2, 1)
    try:
        fresh = HessTreeRay(parts, 16, 16, 32, 32, etas=0.01, workers=private).dot(x)
    finally:
        private.shutdown()

    assert_allclose(reused, fresh, rtol=0, atol=0)


def test_band_pool_hands_back_the_same_pool_for_one_nband(band_pool):
    """The whole point: one actor set per nband per session."""
    assert band_pool(2) is band_pool(2)
    assert band_pool(3) is not band_pool(2)


def test_band_pool_shutdown_is_a_noop_for_a_single_band():
    """nband == 1 runs in-process; there are no actors to kill."""
    pool = BandWorkerPool(1, 1)
    assert pool.actors is None
    pool.shutdown()  # must not raise
    assert pool.actors is None


def test_band_pool_shutdown_is_idempotent():
    """Teardown may run twice (fixture finalisation plus an explicit call)."""
    pool = BandWorkerPool(2, 1)
    pool.shutdown()
    pool.shutdown()  # must not raise
    assert pool.actors == []


def test_shut_down_pool_refuses_to_dispatch():
    """A killed multi-band pool must fail loudly, not silently run one band.

    `actors = []` rather than None is what makes this true: None is the
    sentinel for the in-process path, and a dead pool is not that.
    """
    rng = np.random.default_rng(91)
    pool = BandWorkerPool(2, 1)
    pool.shutdown()
    parts = [[_part(rng, 8, 8, 16, 16)] for _ in range(2)]
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        HessTreeRay(parts, 8, 8, 16, 16, etas=0.01, workers=pool).dot(rng.standard_normal((2, 8, 8)))
