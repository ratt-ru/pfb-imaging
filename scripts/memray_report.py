"""Summarise memray captures as plain text (issue #339).

A ``pfb imager`` run is several processes, and memray tracks one process per
capture file, so there are two ways to capture -- use either or both.

1. Ray tasks (pass 1 ``stokes_vis``, pass 2 ``grid_image``). These run in Ray
   worker processes that the raylet starts, not ones your shell starts, so
   memray cannot be wrapped around them from the command line. Instead set
   ``PFB_MEMRAY_DIR``: ``pfb_imaging.utils.memprof.memray_task`` then opens a
   ``memray.Tracker`` around each task body, writing one file per task named
   ``<task>-<host>-<pid>-<seq>-<token>.bin`` (several files share a host and
   pid when a worker runs several tasks; seq orders them; the random token keeps
   reruns into one directory from colliding). Native C/C++ frames are recorded
   unless ``PFB_MEMRAY_NATIVE=0`` (cheaper, Python frames only). memray must be
   installed wherever the workers run -- it is in the dev group, so a container
   image needs ``pip install memray``; the driver refuses to start without it::

       PFB_MEMRAY_DIR=/scratch/mem pfb imager --ms ... --output-filename ...

   The directory must be writable from every node running workers.

2. The driver -- the process you launch. It is not a Ray task, so (1) does
   not cover it, yet it does the uv-counts reduction between the passes, holds
   the task results and runs the MFS PSF fit. To profile it, start the driver
   itself under ``memray run``, which executes a Python script with tracking
   on, exactly like ``python script.py`` would. ``pfb`` is a console-script
   entry point -- a small Python file on your PATH -- so ``$(which pfb)``
   expands to that script's path and the line below runs the identical
   command ``pfb imager ...`` with the driver tracked::

       python -m memray run --native --aggregate -o /scratch/mem/driver.bin \
           $(which pfb) imager --ms ... --output-filename ...

   ``--native`` records C/C++ frames; ``--aggregate`` writes the compact
   format this script needs (it cannot drive ``memray flamegraph
   --temporal``); ``-o`` names the capture. Only the driver is tracked --
   Ray workers are not children of the driver, so ``--follow-fork`` does not
   reach them -- which is why it combines with (1): set ``PFB_MEMRAY_DIR`` on
   the same command line to capture both in one run. Run it with the
   environment's own python (``.venv/bin/python``, or inside the container);
   under ``uv run`` Ray relaunches its workers with ``uv run`` and without
   your ``--extra`` flags, and they die with "No module named ray".

Then, in the same environment (native frames resolve against that machine's
shared libraries)::

    python scripts/memray_report.py /scratch/mem > memray_report.txt

For each capture it prints:

* the heap high-water mark against the peak RSS memray sampled. A large
  RSS-minus-heap gap is memory memray cannot attribute: Ray object-store
  (shared-memory) pages a task read or wrote, or allocator retention;
* the allocations live at the heap high-water mark, grouped by the innermost
  pfb_imaging source line, with the Python or C++ frame that allocated when
  that is outside pfb_imaging;
* the allocations still live when the task (or the driver) ended -- the
  cross-task retention candidates;

and a one-line-per-capture summary at the end.

Use memray captures to attribute memory, never to time a run or to judge a
peak on their own: native tracking slows every allocation, and in one local
MK+ run a pass-1 task under it reached 26.7 GiB RSS with 2.8 GiB of tracked
heap -- memory allocated where memray has no hooks (most likely Arrow's
mimalloc pool inside arcae, of which memray sees only the reservation) --
which the same code without memray never approached (3.5 GB peak). Check
peaks against the progress-line telemetry of a run without memray.

Heap counts *reserved* address space, not only touched pages. In a driver
capture the row ``init_ray`` / ``plasma::ClientMmapTableEntry`` is the Ray
object store's mapping (sized to the store, so often the largest row) and is
not resident memory; likewise a one-off ``unix_mmap_prim_aligned`` arena
from Arrow's allocator in the first pass-1 task.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import memray

GIB = 2**30


def _natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _frames(rec, native):
    try:
        return rec.hybrid_stack_trace() if native else rec.stack_trace()
    except NotImplementedError:
        return rec.stack_trace()


def _fmt(frame):
    fn, fl, ln = frame
    return f"{fn[:60]} {Path(fl).name}:{ln}"


def _site(frames, depth):
    """(pfb call chain, innermost Python frame, innermost C++ frame) of one stack.

    The C++ frame is shown because it names the library doing the allocating
    (ducc0, arrow, plasma, llvm); C-level numpy/CPython frames are skipped as
    they only say "an array was made".
    """
    ours = [f"{fn} {Path(fl).name}:{ln}" for fn, fl, ln in frames if "pfb_imaging" in fl]
    chain = " <- ".join(ours[:depth]) if ours else "(outside pfb_imaging)"
    py = next((_fmt(f) for f in frames if f[1].endswith(".py")), "-")
    # only frames below the innermost Python frame did the allocating
    npy = next((i for i, f in enumerate(frames) if f[1].endswith(".py")), len(frames))
    cxx = next((f[0][:70] for f in frames[:npy] if "::" in f[0] and not f[0].startswith("operator new")), "")
    return chain, py, cxx


def _table(records, native, depth, top, total, title):
    agg = defaultdict(lambda: [0, 0])
    for rec in records:
        key = _site(_frames(rec, native), depth)
        agg[key][0] += rec.size
        agg[key][1] += rec.n_allocations
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    shown = sum(v[0] for _, v in rows[:top])
    print(f"  {title}: {total / GIB:.3f} GiB in {len(rows)} sites (top {top} = {shown / GIB:.3f} GiB)")
    for (chain, py, nat), (size, nalloc) in rows[:top]:
        pct = 100 * size / total if total else 0.0
        print(f"    {size / GIB:8.3f} GiB {pct:5.1f}% {nalloc:7d}x  {chain}")
        if chain == "(outside pfb_imaging)":
            print(f"    {'':37s}  python: {py}")
        if nat:
            print(f"    {'':37s}  c++:    {nat}")


def report(path, top, depth):
    reader = memray.FileReader(str(path), report_progress=False)
    md = reader.metadata
    native = md.has_native_traces
    snaps = list(reader.get_memory_snapshots())
    dur = (md.end_time - md.start_time).total_seconds()
    print(f"== {path.name}  pid {md.pid}  {dur:.1f} s  native={native}")
    if snaps:
        rss = [s.rss for s in snaps]
        heap = [s.heap for s in snaps]
        print(
            f"  rss  start {rss[0] / GIB:.3f}  max {max(rss) / GIB:.3f}  end {rss[-1] / GIB:.3f} GiB"
            f"  | heap max {max(heap) / GIB:.3f}  end {heap[-1] / GIB:.3f} GiB"
            f"  | max(rss-heap) {max(r - h for r, h in zip(rss, heap)) / GIB:.3f} GiB"
        )
    hwm = list(reader.get_high_watermark_allocation_records(merge_threads=True))
    _table(hwm, native, depth, top, md.peak_memory, "live at heap peak")
    leaks = list(reader.get_leaked_allocation_records(merge_threads=True))
    _table(leaks, native, depth, top, sum(r.size for r in leaks), "still live at task end")
    reader.close()
    return {
        "name": path.name,
        "pid": md.pid,
        "dur": dur,
        "peak": md.peak_memory,
        "rss_max": max((s.rss for s in snaps), default=0),
        "rss_end": snaps[-1].rss if snaps else 0,
        "leaked": sum(r.size for r in leaks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="capture files or directories of them")
    parser.add_argument("--top", type=int, default=12, help="sites per table")
    parser.add_argument("--depth", type=int, default=3, help="pfb_imaging frames per call chain")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        files += sorted(p.glob("*.bin"), key=_natural_key) if p.is_dir() else [p]
    if not files:
        sys.exit("no capture files found")

    rows = [report(f, args.top, args.depth) for f in files]
    print("\n== summary (GiB)")
    print(f"  {'capture':40s} {'pid':>8s} {'sec':>7s} {'heap pk':>8s} {'rss max':>8s} {'rss end':>8s} {'live end':>8s}")
    for r in rows:
        print(
            f"  {r['name']:40s} {r['pid']:8d} {r['dur']:7.1f} {r['peak'] / GIB:8.3f} "
            f"{r['rss_max'] / GIB:8.3f} {r['rss_end'] / GIB:8.3f} {r['leaked'] / GIB:8.3f}"
        )


if __name__ == "__main__":
    main()
