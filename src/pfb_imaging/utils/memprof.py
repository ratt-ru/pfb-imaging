"""Per-task telemetry and opt-in profiling for Ray workers (#339).

``task_memory`` is the post-gc telemetry the imager prints in its progress
lines. RSS is split into anonymous and shared-memory pages because a Ray worker
maps the plasma object store: every task argument it reads zero-copy and every
large value it returns lands in ``RssShmem``, which ratchets per task without
being a leak in the worker (docs/wiki/memory-and-ray.md).

``memray_task`` wraps a task body in a ``memray.Tracker`` when
``PFB_MEMRAY_DIR`` is set (one capture file per task), and is a no-op otherwise.
``scripts/memray_report.py`` turns the captures into a text report.

``profile_task`` is what the task bodies use. It adds a cProfile capture per
task when ``PFB_CPROFILE_DIR`` is set, and always measures the task's wall and
CPU time. ``dump_timeline`` writes Ray's task timeline to ``PFB_RAY_TIMELINE``
from the driver. ``scripts/profile_report.py`` reads both.
"""

import contextlib
import cProfile
import itertools
import os
import resource
import socket
import time
import uuid
import warnings

MEMRAY_DIR_ENV = "PFB_MEMRAY_DIR"
MEMRAY_NATIVE_ENV = "PFB_MEMRAY_NATIVE"
CPROFILE_DIR_ENV = "PFB_CPROFILE_DIR"
TIMELINE_ENV = "PFB_RAY_TIMELINE"

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
    """Render ``task_memory`` output for a progress line, with the task's times if present."""
    line = (
        f"pid {mem['pid']} rss {mem['rss_gb']:.2f} GB "
        f"(anon {mem.get('anon_gb', 0.0):.2f} shm {mem.get('shmem_gb', 0.0):.2f}) "
        f"peak {mem['peak_gb']:.2f} GB"
    )
    if "wall_s" in mem:
        # cpu/wall is the number of cores the task kept busy on average
        busy = mem["cpu_s"] / mem["wall_s"] if mem["wall_s"] > 0 else 0.0
        line += f" | wall {mem['wall_s']:.1f} s cpu {mem['cpu_s']:.1f} s ({busy:.1f} cores)"
    return line


def _cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def _capture_stem(name: str) -> str:
    # host and a random token because pids repeat across container nodes and
    # reruns into one directory; seq orders a worker's tasks
    return f"{name}-{socket.gethostname()}-{os.getpid()}-{next(_task_seq):04d}-{uuid.uuid4().hex[:8]}"


def memray_env() -> dict:
    """The memray switches to forward to Ray workers, if set in this process.

    Raises:
        ImportError: ``PFB_MEMRAY_DIR`` is set but memray is not importable --
            raised here, in the driver, rather than in every task.
    """
    env = {k: os.environ[k] for k in (MEMRAY_DIR_ENV, MEMRAY_NATIVE_ENV) if k in os.environ}
    if env.get(MEMRAY_DIR_ENV):
        try:
            # deferred: optional diagnostic dependency, imported only when profiling
            import memray  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{MEMRAY_DIR_ENV} is set but memray is not installed in this environment "
                "(it is in the dev group; in a container, pip install memray)"
            ) from e
    return env


@contextlib.contextmanager
def memray_task(name: str):
    """Track allocations made inside the block when ``PFB_MEMRAY_DIR`` is set.

    Writes ``<dir>/<name>-<host>-<pid>-<seq>-<token>.bin`` in memray's
    aggregated format, which keeps the high-water mark, the allocations still
    live at exit and the RSS/heap snapshots. Host and a random token are in the
    name because pids repeat across container nodes and reruns into one
    directory, and memray refuses to overwrite a capture -- an error that would
    kill the run. Native stacks are recorded unless
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
    path = os.path.join(out_dir, f"{_capture_stem(name)}.bin")
    native = os.environ.get(MEMRAY_NATIVE_ENV, "1") != "0"
    with memray.Tracker(path, native_traces=native, file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS):
        yield


def profile_env() -> dict:
    """The profiling switches to forward to Ray workers, if set in this process.

    Raises:
        ImportError: see ``memray_env``.
    """
    env = memray_env()
    if os.environ.get(CPROFILE_DIR_ENV):
        env[CPROFILE_DIR_ENV] = os.environ[CPROFILE_DIR_ENV]
    return env


@contextlib.contextmanager
def cprofile_task(name: str):
    """cProfile the block when ``PFB_CPROFILE_DIR`` is set, else do nothing.

    Writes ``<dir>/<name>-<host>-<pid>-<seq>-<token>.pstats``. Time inside
    compiled code (ducc0, numba, arcae) is charged to the Python call that
    entered it. Which Python threads are seen depends on the interpreter: on
    3.11 only the thread that enters the block, so work a task hands to a
    Python thread pool (meerkat-beams' ``nthreads``) shows up as a wait; from
    3.12 cProfile is built on ``sys.monitoring`` and records every thread, so
    the self times of concurrent threads add up and can exceed the wall time.

    Args:
        name: capture-file prefix, normally the task name.
    """
    out_dir = os.environ.get(CPROFILE_DIR_ENV)
    if not out_dir:
        yield
        return
    os.makedirs(out_dir, exist_ok=True)
    prof = cProfile.Profile()
    prof.enable()
    try:
        yield
    finally:
        prof.disable()
        prof.dump_stats(os.path.join(out_dir, f"{_capture_stem(name)}.pstats"))


@contextlib.contextmanager
def profile_task(name: str):
    """Wrap a Ray task body: opt-in memray and cProfile captures, always a clock.

    Yields a dict that holds ``wall_s`` and ``cpu_s`` once the block exits;
    merge it into the task's ``task_memory()`` for the progress line. CPU time
    is the whole process's, so it counts the ducc0/numba threads the task
    started, and ``cpu_s / wall_s`` is the cores the task kept busy.

    Args:
        name: capture-file prefix, normally the task name.
    """
    clock = {}
    wall0, cpu0 = time.perf_counter(), _cpu_seconds()
    try:
        with memray_task(name), cprofile_task(name):
            yield clock
    finally:
        clock["wall_s"] = time.perf_counter() - wall0
        clock["cpu_s"] = _cpu_seconds() - cpu0


class PhaseClock:
    """Driver-side phase timer: each ``lap`` logs the time since the previous one.

    The driver's own work (planning, task submission, which pickles every task's
    arguments, and the reductions between passes) is invisible to the task
    timeline and to per-task profiles; this is what accounts for it.
    """

    def __init__(self, log):
        self.log = log
        self.start = self.last = time.perf_counter()
        self.laps = []

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps.append((name, now - self.last))
        self.last = now
        self.log.info(f"{name} took {self.laps[-1][1]:.1f} s")

    def summary(self) -> None:
        total = self.last - self.start
        parts = ", ".join(f"{n} {dt:.1f} s ({100 * dt / total if total > 0 else 0:.0f}%)" for n, dt in self.laps)
        self.log.info(f"Time by phase: {parts}")


def dump_timeline(log=None) -> None:
    """Write Ray's task timeline to ``PFB_RAY_TIMELINE`` if it is set.

    Call from the driver while Ray is still up. The file is Chrome trace JSON:
    open it in https://ui.perfetto.dev or summarise it with
    ``scripts/profile_report.py``. It covers every task the cluster still
    remembers, so on a shared ``--ray-address`` cluster it includes other jobs.

    A failure to write it is reported, not raised: the run's work is done by
    the time this is called, and a diagnostic must not fail it or skip the
    caller's ``ray.shutdown()``.
    """
    path = os.environ.get(TIMELINE_ENV)
    if not path:
        return
    # deferred: optional heavy runtime (ray)
    import ray

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ray.timeline(filename=path)
    except Exception as e:
        msg = f"Could not write the Ray task timeline to {path}: {e!r}"
        if log is not None:
            log.warning(msg)
        else:
            warnings.warn(msg, stacklevel=2)
        return
    if log is not None:
        log.info(f"Wrote the Ray task timeline to {path}")
