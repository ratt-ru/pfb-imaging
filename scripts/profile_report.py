"""Summarise where a run spends its time: per-task cProfile captures and the Ray timeline.

Three sources, each switched on by an environment variable on the ``pfb`` command
line, and all three combine in one run. ``pfb imager``, ``pfb degrid`` and
``pfb deconv`` honour them.

1. The driver's phases, always on. The log ends with a ``Time by phase:`` line --
   set-up, pass-1 submission, pass 1, the counts reduction, pass 2, PSF fit, FITS.
   Submission is timed separately because ``.remote`` pickles every task's
   datatree node in the driver. This is the first thing to read: it says whether
   the time is in the remote calls or in the driver between them.

2. The Ray timeline, ``PFB_RAY_TIMELINE=<file.json>``. The driver dumps
   ``ray.timeline()`` before it shuts Ray down: every task's start and duration on
   every worker, split into deserialising its arguments, executing and storing its
   outputs. Open the file in https://ui.perfetto.dev to see it, or pass it here
   for, per task name, count / total / mean / p95 / max and the argument and
   output overheads, plus how busy the workers were kept. "Idle" is time inside
   the run with no task executing anywhere: the driver was planning, submitting
   or reducing.

3. Per-task profiles, ``PFB_CPROFILE_DIR=<dir>``. Every task body (``stokes_vis``,
   ``grid_image``, ``degrid``) runs under cProfile and writes
   ``<task>-<host>-<pid>-<seq>-<token>.pstats``. ``pfb deconv``'s band workers
   are long-lived actors, so each writes one ``band_worker-...`` capture covering
   all its role-method calls (``psi_dot``, ``hess_dot``, ``cg``, ...) when the run
   ends; the timeline shows those calls by method name. Here they are merged per task
   name and printed as the functions with the most cumulative time (where the
   time goes below the task) and the most self time (what is actually hot, each
   with the callers that account for it). The cumulative table lists only
   pfb_imaging's own functions unless ``--all-cumulative``; numba JIT compilation
   gets a line of its own because its compiler frames would otherwise fill both
   tables. The directory must be writable from
   every node running workers. To profile the driver as well::

       python -m cProfile -o <dir>/driver.pstats $(which pfb) imager ...

   cProfile sees Python frames only: time inside ducc0, numba or arcae is charged
   to the Python call that entered it, so ``vis2dirty`` or an arcae read is one
   line. That is usually the right resolution for choosing what to optimise. It
   also adds overhead per Python call, so xarray-heavy code looks somewhat more
   expensive than it is; check against the wall times of a run without it. On
   Python 3.11 only the thread that runs the task is profiled, so work handed to
   a Python thread pool (the beam's ``nthreads``) shows as a wait; from 3.12
   every thread is, and the self times of concurrent threads add up.

Every progress line also carries the task's wall and CPU time; CPU/wall is the
cores it kept busy, so a task that should use ``--nthreads`` threads but shows
~1.0 is running single-threaded.

Usage::

    PFB_RAY_TIMELINE=prof/timeline.json PFB_CPROFILE_DIR=prof/pstats pfb imager ...
    python scripts/profile_report.py prof/timeline.json prof/pstats > profile.txt

Run it with the environment's own python (``.venv/bin/python``), not ``uv run``:
under ``uv run`` Ray relaunches its workers without your ``--extra`` flags.
"""

import argparse
import json
import pstats
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _short_task(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -np.inf
    for lo, hi in sorted(intervals):
        if hi <= end:
            continue
        total += hi - max(lo, end)
        end = hi
    return total


def timeline_report(path: Path) -> None:
    events = [e for e in json.loads(path.read_text()) if e.get("ph") == "X"]
    tasks = [e for e in events if e.get("cat", "").startswith("task::")]
    if not tasks:
        print(f"== {path.name}: no task events")
        return
    # sub-phases carry no task name: attribute each to the task span on the
    # same worker that contains it (the first one, if overlapping spans do)
    spans = defaultdict(list)
    for e in tasks:
        spans[e["tid"]].append((e["ts"], e["ts"] + e["dur"], _short_task(e["name"])))
    for v in spans.values():
        v.sort()
    over = defaultdict(lambda: defaultdict(float))
    for e in events:
        phase = e.get("cat", "")
        if phase not in ("task:deserialize_arguments", "task:store_outputs"):
            continue
        for lo, hi, name in spans.get(e["tid"], ()):
            if lo <= e["ts"] <= hi:
                over[name][phase] += e["dur"]
                break

    by_name = defaultdict(list)
    for e in tasks:
        by_name[_short_task(e["name"])].append(e["dur"] / 1e6)
    t0 = min(e["ts"] for e in tasks)
    t1 = max(e["ts"] + e["dur"] for e in tasks)
    span = (t1 - t0) / 1e6
    # per worker the union of its task spans: an async actor (degrid's Serve
    # replica) runs overlapping requests on one tid, and a plain sum would
    # count that time twice
    busy = sum(_union_length([(lo, hi) for lo, hi, _ in v]) for v in spans.values()) / 1e6
    # every worker process that ran any task, FITS tasks and replaced workers included
    nworkers = len(spans)
    covered = _union_length([(e["ts"], e["ts"] + e["dur"]) for e in tasks]) / 1e6

    print(f"== timeline {path.name}")
    print(
        f"  first task start to last task end {span:.1f} s on {nworkers} workers: busy {busy:.1f} worker-s "
        f"({100 * busy / (nworkers * span):.0f}% of capacity); idle (no task running) {span - covered:.1f} s"
    )
    cols = ("total s", "mean", "p50", "p95", "max", "args s", "out s")
    print(f"  {'task':28s} {'n':>6s} " + " ".join(f"{c:>8s}" for c in cols))
    for name, durs in sorted(by_name.items(), key=lambda kv: -sum(kv[1])):
        d = np.array(durs)
        o = over[name]
        print(
            f"  {name[:28]:28s} {d.size:6d} {d.sum():8.1f} {d.mean():8.2f} {np.median(d):8.2f} "
            f"{np.percentile(d, 95):8.2f} {d.max():8.2f} {o['task:deserialize_arguments'] / 1e6:8.2f} "
            f"{o['task:store_outputs'] / 1e6:8.2f}"
        )
    print()


def _func(f: tuple) -> str:
    file, line, name = f
    if file == "~":  # builtins and C methods
        return name
    for marker in ("site-packages/", "src/", "lib/python3"):
        if marker in file:
            file = file.split(marker, 1)[1]
            break
    return f"{name} ({file}:{line})"


def pstats_report(name: str, files: list[Path], top: int, ncallers: int, everything: bool) -> None:
    st = pstats.Stats(str(files[0]))
    for f in files[1:]:
        st.add(str(f))
    total = st.total_tt
    print(f"== cProfile {name}: {len(files)} capture(s), {total:.1f} s profiled, {total / len(files):.2f} s each")
    rows = st.stats  # func -> (prim calls, calls, self, cumulative, callers)
    # cProfile counts a recursive function's cumulative time once, so the
    # dispatcher entry point is the whole of the JIT compilation below the task
    jit = max((v[3] for f, v in rows.items() if f[2] == "_compile_for_args" and "numba" in f[0]), default=0.0)
    if jit:
        print(f"  numba JIT compilation: {jit:.2f} s ({100 * jit / total:.1f}%); first call per worker process")

    scope = "" if everything else " in pfb_imaging"
    print(f"  cumulative (time in a function and everything it calls){scope}, top {top}:")
    cum = [kv for kv in rows.items() if everything or "pfb_imaging" in kv[0][0]]
    for func, (_, nc, tt, ct, _) in sorted(cum, key=lambda kv: -kv[1][3])[:top]:
        print(f"    {ct:9.2f} s {100 * ct / total:5.1f}% {nc:9d}x  {_func(func)}")

    print(f"  self (time in the function's own code, compiled callees included), top {top}:")
    for func, (_, nc, tt, ct, callers) in sorted(rows.items(), key=lambda kv: -kv[1][2])[:top]:
        print(f"    {tt:9.2f} s {100 * tt / total:5.1f}% {nc:9d}x  {_func(func)}")
        # callers maps caller -> (prim calls, calls, self, cumulative) of this
        # function when called from there
        for caller, cstat in sorted(callers.items(), key=lambda kv: -kv[1][2])[:ncallers]:
            print(f"    {'':27s}<- {cstat[2]:.2f} s from {_func(caller)}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="timeline .json files, .pstats files or directories of them"
    )
    parser.add_argument("--top", type=int, default=20, help="functions per table")
    parser.add_argument("--callers", type=int, default=2, help="callers shown under each self-time row")
    parser.add_argument(
        "--all-cumulative", action="store_true", help="cumulative table over every function, not just pfb_imaging's"
    )
    args = parser.parse_args()

    timelines, captures = [], defaultdict(list)
    for p in args.paths:
        files = sorted(p.iterdir()) if p.is_dir() else [p]
        for f in files:
            if f.suffix == ".json":
                timelines.append(f)
            elif f.suffix == ".pstats":
                captures[f.stem.split("-", 1)[0]].append(f)
    if not timelines and not captures:
        sys.exit("no timeline (.json) or capture (.pstats) files found")

    for f in timelines:
        timeline_report(f)
    for name, files in sorted(captures.items()):
        pstats_report(name, files, args.top, args.callers, args.all_cumulative)


if __name__ == "__main__":
    main()
