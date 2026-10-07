"""Per-task memory telemetry and opt-in memray tracking for Ray workers (#339).

``task_memory`` is the post-gc telemetry the imager prints in its progress
lines. RSS is split into anonymous and shared-memory pages because a Ray worker
maps the plasma object store: every task argument it reads zero-copy and every
large value it returns lands in ``RssShmem``, which ratchets per task without
being a leak in the worker (docs/wiki/memory-and-ray.md).

``memray_task`` wraps a task body in a ``memray.Tracker`` when
``PFB_MEMRAY_DIR`` is set (one capture file per task), and is a no-op otherwise.
``scripts/memray_report.py`` turns the captures into a text report.
"""

import contextlib
import itertools
import os
import resource

MEMRAY_DIR_ENV = "PFB_MEMRAY_DIR"
MEMRAY_NATIVE_ENV = "PFB_MEMRAY_NATIVE"

_task_seq = itertools.count()


def task_memory() -> dict:
    """Return this process's memory split, in GiB, plus its lifetime peak.

    Returns:
        dict with ``pid``, ``rss_gb``, ``anon_gb``, ``shmem_gb`` (Linux
        ``/proc/self/status``; zero elsewhere) and ``peak_gb`` (``ru_maxrss``,
        which counts shared pages too).
    """
    vals = {"VmRSS": 0, "RssAnon": 0, "RssShmem": 0}
    with contextlib.suppress(OSError):
        with open("/proc/self/status") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in vals:
                    vals[key] = int(rest.split()[0])  # kB
    return {
        "pid": os.getpid(),
        "rss_gb": vals["VmRSS"] / 2**20,
        "anon_gb": vals["RssAnon"] / 2**20,
        "shmem_gb": vals["RssShmem"] / 2**20,
        "peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }


def format_memory(mem: dict) -> str:
    """Render ``task_memory`` output for a progress line."""
    return (
        f"pid {mem['pid']} rss {mem['rss_gb']:.2f} GB "
        f"(anon {mem.get('anon_gb', 0.0):.2f} shm {mem.get('shmem_gb', 0.0):.2f}) "
        f"peak {mem['peak_gb']:.2f} GB"
    )


def memray_env() -> dict:
    """The memray switches to forward to Ray workers, if set in this process."""
    return {k: os.environ[k] for k in (MEMRAY_DIR_ENV, MEMRAY_NATIVE_ENV) if k in os.environ}


@contextlib.contextmanager
def memray_task(name: str):
    """Track allocations made inside the block when ``PFB_MEMRAY_DIR`` is set.

    Writes ``<dir>/<name>-<pid>-<seq>.bin`` in memray's aggregated format,
    which keeps the high-water mark, the allocations still live at exit and
    the RSS/heap snapshots. Native stacks are recorded unless
    ``PFB_MEMRAY_NATIVE=0``; report on the machine that ran the capture, since
    native symbols resolve against its shared libraries.

    Args:
        name: capture-file prefix, normally the task name.
    """
    out_dir = os.environ.get(MEMRAY_DIR_ENV)
    if not out_dir:
        yield
        return
    # deferred: optional diagnostic dependency, imported only when profiling
    import memray

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}-{os.getpid()}-{next(_task_seq):04d}.bin")
    native = os.environ.get(MEMRAY_NATIVE_ENV, "1") != "0"
    with memray.Tracker(path, native_traces=native, file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS):
        yield
