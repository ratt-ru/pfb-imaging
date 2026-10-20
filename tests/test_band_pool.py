"""The shared BandWorkerPool fixture: reuse must be numerically invisible.

A session-scoped pool is only sound because `init_hess` rebuilds every
per-band HessianTree from scratch. These tests are what pin that claim --
without them the optimisation is a silent correctness risk, not a speedup.
"""

import numpy as np
import pytest
import ray
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


def test_band_pool_shutdown_kills_the_actors():
    """The point of shutdown(): the actor processes are gone, not just forgotten."""
    pool = BandWorkerPool(2, 1)
    handles = list(pool.actors)
    pool.shutdown()
    for h in handles:
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(h.get_mem.remote())


def test_shut_down_pool_refuses_to_dispatch():
    """A killed multi-band pool must fail loudly, not silently run one band.

    `actors = []` rather than None is what makes this true: None is the
    sentinel for the in-process path, and a dead pool is not that.
    """
    rng = np.random.default_rng(91)
    pool = BandWorkerPool(2, 1)
    pool.shutdown()
    parts = [[_part(rng, 8, 8, 16, 16)] for _ in range(2)]
    with pytest.raises(ValueError, match="band results"):
        HessTreeRay(parts, 8, 8, 16, 16, etas=0.01, workers=pool).dot(rng.standard_normal((2, 8, 8)))


@pytest.mark.parametrize("nband", [1, 2])
def test_call_stats_pair_worker_time_with_the_round_trip(band_pool, nband):
    """call_stats reports each role method's calls since the last report, then resets.

    nband == 1 is the in-process path; nband == 2 goes through Ray actors.
    """
    rng = np.random.default_rng(7)
    pool = band_pool(nband)
    pool.call_stats()  # flush whatever earlier users of the shared pool left
    parts = [[_part(rng, 8, 8, 16, 16)] for _ in range(nband)]
    hess = HessTreeRay(parts, 8, 8, 16, 16, etas=0.1, workers=pool)
    x = rng.standard_normal((nband, 8, 8))
    hess.dot(x)
    hess.dot(x)

    stats = pool.call_stats()
    assert stats["init_hess"]["calls"] == 1
    hd = stats["hess_dot"]
    assert hd["calls"] == 2
    assert hd["roundtrip_s"] >= hd["exec_max_s"] >= hd["exec_mean_s"] > 0
    assert hd["cores"] >= 0
    assert pool.call_stats() == {}  # reset


def test_log_telemetry_reports_calls_and_worker_memory(band_pool):
    lines = []

    class Log:
        def info(self, msg):
            lines.append(msg)

    rng = np.random.default_rng(8)
    pool = band_pool(2)
    pool.call_stats()
    HessTreeRay([[_part(rng, 8, 8, 16, 16)] for _ in range(2)], 8, 8, 16, 16, workers=pool).dot(
        rng.standard_normal((2, 8, 8))
    )
    pool.log_telemetry(Log())
    assert any(line.startswith("band workers hess_dot: 1 calls") and "Ray overhead" in line for line in lines)
    assert sum(line.startswith("band worker pid") and "shm" in line for line in lines) == 2


def test_driver_work_overlapping_dual_begin_is_not_its_round_trip(band_pool):
    """The driver's band-summed analysis runs between dispatch and ray.get; keep it apart."""
    import time

    lines = []

    class Log:
        def info(self, msg):
            lines.append(msg)

    pool = band_pool(2)
    nxmax, nymax = pool.init_psi(16, 16, ("self", "db1"), 1)
    pool.init_dual()
    pool.call_stats()
    t0 = time.perf_counter()
    pending = pool.dual_begin(np.ones((2, 16, 16)), 1.0)
    time.sleep(0.5)  # stands in for the driver's own analysis
    pool.dual_finish(pending, np.ones((2, nxmax, nymax)), np.empty((2, 16, 16)))
    wall = time.perf_counter() - t0
    db = pool.call_stats()["dual_begin"]
    assert db["overlap_s"] >= 0.5
    assert db["roundtrip_s"] + db["overlap_s"] <= wall
    pending = pool.dual_begin(np.ones((2, 16, 16)), 1.0)
    pool.dual_finish(pending, np.ones((2, nxmax, nymax)), np.empty((2, 16, 16)))
    pool.log_telemetry(Log())
    (line,) = [line for line in lines if line.startswith("band workers dual_begin:")]
    assert "driver work overlapped" in line and "Ray overhead" not in line
