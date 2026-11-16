"""Shared set-up and reporting for the kernel profiling scripts.

Import this before numba or ray: it turns off Ray's ``uv run`` runtime-env hook,
which is read when ray is imported.
"""

import gc
import os
import tracemalloc
from time import perf_counter

# Under `uv run` Ray relaunches its workers through `uv run --frozen python`,
# rebuilding a venv without the [full] extra, and refuses a working directory
# that does not hold pyproject.toml. Same workaround as tests/conftest.py.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def setup_threads(nthreads: int | None = None) -> int:
    """Load numba's threading layer the way a `pfb` command does and size its pool.

    `import pfb_imaging` requests the TBB layer where the tbb wheel is installed,
    but libtbb.so is only loaded by `set_envs`; a script that skips it dies at
    its first parallel kernel with "No threading layer could be loaded".

    Args:
        nthreads: numba threads; all cores when None.

    Returns:
        The number of threads in force.
    """
    # deferred: these read the environment set above when they are imported
    import numba

    import pfb_imaging

    ncpu = os.cpu_count() or 1
    nthreads = min(nthreads or ncpu, numba.config.NUMBA_NUM_THREADS)
    pfb_imaging.set_envs(nthreads, ncpu)
    numba.set_num_threads(nthreads)
    return nthreads


def parse_shape(text: str) -> tuple[int, ...]:
    """``"8,8000,8000"`` -> ``(8, 8000, 8000)``."""
    return tuple(int(s) for s in text.split(","))


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def rdiff(a, b) -> float:
    """``||a - b|| / ||a||``."""
    # deferred: keep this module importable before numpy's BLAS reads its thread env
    import numpy as np

    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


class ClockedOp:
    """Wrap a callable and accumulate the wall time spent inside it.

    The solvers under test are a loop around an operator, so total time minus
    operator time is the solver's own overhead -- the number these scripts exist
    to measure.
    """

    def __init__(self, fn):
        self.fn = fn
        self.seconds = 0.0
        self.calls = 0

    def reset(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        t0 = perf_counter()
        out = self.fn(*args, **kwargs)
        self.seconds += perf_counter() - t0
        self.calls += 1
        return out


def timed(run, name: str, niter: int, warmup, ops: dict[str, ClockedOp] | None = None):
    """Warm up, then time one run with numpy allocations traced.

    Args:
        run: ``run(niter) -> result``; must not reuse state from an earlier call.
        name: label for the report.
        niter: iterations of the timed run.
        warmup: number of 3-iteration warm-up runs (JIT compilation).
        ops: label -> :class:`ClockedOp` called inside ``run``; their time is
            reported separately from the solver's own.

    Returns:
        ``(result, seconds, peak_bytes)``. ``peak_bytes`` is tracemalloc's peak,
        which sees numpy's allocations but nothing allocated inside numba.
    """
    ops = ops or {}
    section(f"Profiling: {name}")
    for i in range(warmup):
        t0 = perf_counter()
        run(3)
        print(f"  warmup {i + 1}: {perf_counter() - t0:.3f} s")
    for op in ops.values():
        op.reset()
    gc.collect()
    tracemalloc.start()
    t0 = perf_counter()
    result = run(niter)
    seconds = perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  total:     {seconds:.3f} s, {1e3 * seconds / niter:.1f} ms/iter over {niter} iterations")
    inside = 0.0
    for label, op in ops.items():
        inside += op.seconds
        print(f"  {label + ':':<10} {op.seconds:.3f} s in {op.calls} calls ({100 * op.seconds / seconds:.0f}%)")
    if ops:
        own = seconds - inside
        print(f"  solver:    {own:.3f} s, {1e3 * own / niter:.1f} ms/iter ({100 * own / seconds:.0f}%)")
    print(f"  peak numpy allocation: {peak / 1e6:.1f} MB")
    return result, seconds, peak


def compare(label_a: str, a: tuple, label_b: str, b: tuple, niter: int) -> None:
    """Print the time and peak-memory ratio of two :func:`timed` results."""
    section("Summary")
    for label, (_, seconds, peak) in ((label_a, a), (label_b, b)):
        print(f"  {label + ':':<14} {seconds:.3f} s ({1e3 * seconds / niter:.1f} ms/iter), peak {peak / 1e6:.1f} MB")
    fast, slow = (label_a, label_b) if a[1] <= b[1] else (label_b, label_a)
    print(f"  {fast} is {max(a[1], b[1]) / min(a[1], b[1]):.2f}x faster than {slow}")
    if min(a[2], b[2]) > 0 and max(a[2], b[2]) / min(a[2], b[2]) > 1.02:
        lean, fat = (label_a, label_b) if a[2] <= b[2] else (label_b, label_a)
        print(f"  {lean} peaks {max(a[2], b[2]) / min(a[2], b[2]):.2f}x lower than {fat}")
