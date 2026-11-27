"""Share one core budget and one memory budget between the imager's passes (#399).

The imager runs two Ray passes with very different tasks. Pass 1 runs many small
tasks that spend most of their time reading the MS. Pass 2 runs one task per
output image, and gridding dominates it. One ``--ncpu`` and one ``--mem-limit``
are shared out afresh by each pass (wiki glossary: core budget, concurrent tasks):

* every task claims its threads as Ray CPUs and its estimated peak as Ray
  ``memory``, so Ray itself holds each pass inside both budgets, per node;
* :func:`plan_pass` chooses each pass's threads per task, from how many tasks
  can run at once under both budgets.

The per-task estimates below are computed from array shapes and dtypes before a
task runs. Their terms follow the allocations alive together at each task's
peak, and their constants were measured with memray on the ESO137 production
command and five configurations of a 2-scan MS (wiki memory-and-ray.md,
"Pass 2 holds one partition at a time"; D68).
"""

from dataclasses import dataclass

GIB = 2**30

# Measured constants (D68). Every estimate is multiplied by MARGIN: across the
# ESO137 production run and five 2-scan configurations, the worst pass-2 image
# needed 1/0.90 of its unmargined estimate, so 1.12 covers every task measured.
MARGIN = 1.12
# pass 1: RSS per byte the task loads from the MS (the loaded columns, their
# Stokes products and BDA's outputs), and a worker's resident baseline
P1_PER_LOADED_BYTE = 3.49
P1_WORKER_BASELINE = 1.6 * GIB
# pass 2: ducc's grid, per pixel of the PSF it grids, at the measured visibility
# densities (26-50 B, rising with them; 45 B on production data)
DUCC_BYTES_PER_PIXEL = 45.0
# the transferred model, cached per worker, plus its render
MODEL_FACTOR = 1.07
P2_WORKER_BASELINE = 1.9 * GIB
# threads per pass-1 task: its tasks mostly read the MS, and 2 beat 1 and 3
P1_THREADS = 2


def pass1_task_bytes(loaded_bytes: int, counts_cells: int, accum_itemsize: int) -> int:
    """Estimated peak RSS of one pass-1 task, margin included.

    The dense counts accumulator (``compute_sparse_counts``) is counted whole,
    with its uint8 touched-mask. Its pages are touched only where the task's
    samples land, but a task covering much of the uv plane touches most of them.

    Args:
        loaded_bytes: bytes of the MS variables ``stokes_vis`` loads (the lazy
            node subset's ``nbytes``).
        counts_cells: cells of the counts grid (``counts_nx * counts_ny``).
        accum_itemsize: itemsize the counts accumulate in (8 under ``--double-accum``).

    Returns:
        Bytes.
    """
    peak = P1_PER_LOADED_BYTE * loaded_bytes + (accum_itemsize + 1) * counts_cells + P1_WORKER_BASELINE
    return int(MARGIN * peak)


def pass2_image_bytes(
    partition_bytes: list[int],
    npix: int,
    npsf: int,
    ncorr: int,
    itemsize: int,
    counts_cells: int,
    model_bytes: int = 0,
) -> int:
    """Estimated peak RSS of one pass-2 task (one output image), margin included.

    Pass 2 holds one partition at a time, and peaks in the larger of two phases
    for its largest partition: concatenating it (its pieces and their concat,
    two copies) or gridding its PSF (one copy, the partition's products and
    ducc's grid). Both phases also hold the image-space sums, the dense applied
    counts grid, the cached transferred model and the worker's baseline.

    Args:
        partition_bytes: in-memory bytes of each partition's scratch pieces.
        npix: image pixels (``nx * ny``).
        npsf: PSF pixels, or 0 without ``--psf``.
        ncorr: output correlations.
        itemsize: the tree's real itemsize (4 single, 8 double).
        counts_cells: cells of the applied counts grid, or 0 under natural weighting.
        model_bytes: in-memory bytes of the transferred model, or 0 without one.

    Returns:
        Bytes.
    """
    csize = 2 * itemsize
    part = max(partition_bytes)
    model = model_bytes > 0
    # dirty, beam and bdirty sums (+ model, residual, bresidual), and the psf sum
    sums = ncorr * (itemsize * npix * (6 if model else 3) + itemsize * npsf)
    # the partition's DIRTY and BEAM (+ RESIDUAL), PSF and PSFHAT
    prod = ncorr * (itemsize * npix * (3 if model else 2) + itemsize * npsf + csize * npsf // 2)
    fixed = ncorr * itemsize * counts_cells + MODEL_FACTOR * model_bytes + P2_WORKER_BASELINE
    concat = 2 * part + sums + fixed
    grid = part + sums + prod + DUCC_BYTES_PER_PIXEL * ncorr * (npsf or npix) + fixed
    return int(MARGIN * max(concat, grid))


@dataclass(frozen=True)
class PassPlan:
    """How one pass shares the core and memory budgets.

    Attributes:
        nconcurrent: tasks that can run at once under both budgets.
        nthreads: threads each task is given.
        task_cpus: Ray CPUs each task claims (its threads).
    """

    nconcurrent: int
    nthreads: int
    task_cpus: int


def plan_pass(
    ntasks: int,
    ncpu: int,
    node_cpus: int,
    task_bytes: list[int],
    mem_limit: int,
    max_concurrent: int | None = None,
    nthreads: int | None = None,
) -> PassPlan:
    """Concurrency and threads for one pass.

    With ``nthreads`` given (pass 1), each task gets that many threads and the
    pass runs ``ncpu // nthreads`` tasks at once. Without it (pass 2), the pass
    runs as many tasks at once as the budgets and ``max_concurrent`` allow, and
    shares ``ncpu`` out between them. Either way the memory budget caps the
    concurrency at what the largest tasks fit in, and no task claims more
    threads than one node has, or Ray could never schedule it.

    Args:
        ntasks: tasks in the pass.
        ncpu: the core budget.
        node_cpus: CPUs of the largest node in the cluster.
        task_bytes: each task's estimated peak, margin included.
        mem_limit: the memory budget, in bytes.
        max_concurrent: a cap on concurrency beyond the budgets (pass 2: nband).
        nthreads: fixed threads per task, or None to derive them.

    Returns:
        The pass's plan.

    Raises:
        ValueError: if the largest task alone does not fit in ``mem_limit``.
    """
    if ntasks < 1:
        raise ValueError(f"a pass needs at least one task, got {ntasks}")
    largest = max(task_bytes)
    if largest > mem_limit:
        raise ValueError(
            f"one task needs an estimated {largest / GIB:.1f} GiB, more than the "
            f"memory budget of {mem_limit / GIB:.1f} GiB"
        )
    # the largest tasks, run together, must fit: a conservative cap that holds
    # whatever order Ray starts them in
    by_mem = 0
    used = 0
    for nbytes in sorted(task_bytes, reverse=True):
        if used + nbytes > mem_limit:
            break
        used += nbytes
        by_mem += 1
    cap = min(ntasks, by_mem, max_concurrent or ntasks)
    if nthreads is not None:
        nthreads = max(1, min(nthreads, node_cpus, ncpu))
        nconcurrent = max(1, min(cap, ncpu // nthreads))
    else:
        nconcurrent = max(1, min(cap, ncpu))
        nthreads = max(1, min(ncpu // nconcurrent, node_cpus))
    return PassPlan(nconcurrent=nconcurrent, nthreads=nthreads, task_cpus=nthreads)
