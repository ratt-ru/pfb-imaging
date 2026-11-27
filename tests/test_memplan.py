"""Sharing the core and memory budgets between the imager's passes (#399)."""

import pytest

from pfb_imaging.utils.memplan import GIB, PassPlan, pass1_task_bytes, pass2_image_bytes, plan_pass


def test_pass1_runs_one_task_per_pair_of_cores():
    # com08: 96 logical cores, pass 1 at 2 threads per task (the measured optimum)
    plan = plan_pass(1152, ncpu=96, node_cpus=96, task_bytes=[3 * GIB] * 1152, mem_limit=600 * GIB, nthreads=2)
    assert plan == PassPlan(nconcurrent=48, nthreads=2, task_cpus=2)


def test_pass2_shares_the_cores_between_the_images():
    # 15 images, capped at nband = 16: all run at once with 96 // 15 threads each
    plan = plan_pass(15, ncpu=96, node_cpus=96, task_bytes=[25 * GIB] * 15, mem_limit=600 * GIB, max_concurrent=16)
    assert plan == PassPlan(nconcurrent=15, nthreads=6, task_cpus=6)


def test_pass2_concurrency_never_exceeds_nband():
    plan = plan_pass(64, ncpu=96, node_cpus=96, task_bytes=[GIB] * 64, mem_limit=600 * GIB, max_concurrent=16)
    assert plan.nconcurrent == 16
    assert plan.nthreads == 6


def test_memory_caps_concurrency_and_the_threads_go_up():
    # 15 images of 30 GiB in 200 GiB: 6 fit, and they share all the cores
    plan = plan_pass(15, ncpu=96, node_cpus=96, task_bytes=[30 * GIB] * 15, mem_limit=200 * GIB, max_concurrent=16)
    assert plan.nconcurrent == 6
    assert plan.nthreads == 16


def test_the_memory_cap_counts_the_largest_tasks():
    # three 40 GiB images and many small ones: 100 GiB fits two of the large ones
    sizes = [40 * GIB] * 3 + [GIB] * 12
    plan = plan_pass(15, ncpu=96, node_cpus=96, task_bytes=sizes, mem_limit=100 * GIB, max_concurrent=16)
    assert plan.nconcurrent == 2


def test_a_task_never_claims_more_cpus_than_a_node_has():
    # a 2 x 48-core cluster: one image may not claim all 96, or Ray never schedules it
    plan = plan_pass(1, ncpu=96, node_cpus=48, task_bytes=[GIB], mem_limit=600 * GIB, max_concurrent=16)
    assert plan.nthreads == 48
    plan = plan_pass(100, ncpu=96, node_cpus=1, task_bytes=[GIB] * 100, mem_limit=600 * GIB, nthreads=2)
    assert plan.nthreads == 1


def test_the_test_session_cluster_is_never_oversubscribed():
    # conftest.manage_ray starts num_cpus=2: a claim above 2 would hang the session
    p1 = plan_pass(8, ncpu=2, node_cpus=2, task_bytes=[GIB] * 8, mem_limit=8 * GIB, nthreads=2)
    p2 = plan_pass(4, ncpu=2, node_cpus=2, task_bytes=[GIB] * 4, mem_limit=8 * GIB, max_concurrent=4)
    assert p1.task_cpus <= 2 and p2.task_cpus <= 2
    assert (p2.nconcurrent, p2.nthreads) == (2, 1)


def test_a_task_larger_than_the_budget_is_refused():
    with pytest.raises(ValueError, match="more than the memory budget"):
        plan_pass(4, ncpu=8, node_cpus=8, task_bytes=[GIB, 9 * GIB], mem_limit=8 * GIB, max_concurrent=4)


# Measured with memray on the ESO137 production command (com08, 8748^2, PSF x1.4,
# --baseline-groups, --transfer-model-from, 24 x 4): the largest pass-1 task and
# the largest pass-2 image. The estimates must cover them (D68).
COUNTS_CELLS = 14872 * 14872  # the padded image grid, 1.7 x 8748


def test_pass1_estimate_covers_the_measured_production_peak():
    est = pass1_task_bytes(loaded_bytes=546_157_080, counts_cells=COUNTS_CELLS, accum_itemsize=8)
    assert est >= 4_354_015_232


def test_pass2_estimate_covers_the_measured_production_peak():
    est = pass2_image_bytes(
        partition_bytes=[5_401_337_248],
        npix=76_527_504,
        npsf=150_062_500,
        ncorr=1,
        itemsize=4,
        counts_cells=COUNTS_CELLS,
        model_bytes=4_482_864_996,
    )
    measured = 26_315_812_864
    # covered, and not so loose that it would throttle pass 2 for nothing
    assert measured <= est <= 1.25 * measured


def test_pass2_estimate_follows_the_largest_partition():
    small = pass2_image_bytes([GIB, GIB], 4096**2, 0, 1, 4, 0)
    large = pass2_image_bytes([GIB, 8 * GIB], 4096**2, 0, 1, 4, 0)
    # concatenating the largest partition holds two copies of it
    assert large - small >= 2 * 7 * GIB


def test_ray_init_still_sizes_the_memory_resource():
    # init_ray passes the memory budget as ray.init's underscored _memory, which
    # is not in its signature: a Ray upgrade that drops it must fail here, not
    # leave the budget silently unenforced
    import inspect

    import ray

    assert "_memory" in inspect.getsource(ray.init)
