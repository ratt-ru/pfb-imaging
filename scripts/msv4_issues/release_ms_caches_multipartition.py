#!/usr/bin/env python
"""Does pass 1 still need `_release_ms_caches` on many partitions? (pfb-imaging#325)

Copies an MS, rewrites SCAN_NUMBER into blocks of `per_scan` integrations so the
imager's partition schema (`utils.msv4.get_engine`: FIELD_ID, DATA_DESC_ID,
SCAN_NUMBER) yields many partitions, then mimics pass-1 Ray tasks in one process.
As the imager's dispatch loop does, each partition is sliced into
(time, frequency) blocks with `isel` and each slice pickled once (what Ray does
to a task argument); every "task" unpickles one, loads only the variables pass 1
reads, releases them and collects. It reports post-gc RssAnon and the Multiton cache
size per task, with and without the wholesale eviction between tasks.

    uv run python scripts/msv4_issues/release_ms_caches_multipartition.py <any.ms> [per_scan] [passes]

Flat anon without the eviction means the helper can go; a slope is retention
the eviction is still buying.
"""

import gc
import pickle
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from casacore.tables import table
from rarg_python_patterns.multiton import Multiton

from pfb_imaging.utils.msv4 import get_engine

warnings.filterwarnings("ignore")
NEEDED = ["VISIBILITY", "FLAG", "WEIGHT", "UVW"]
CHANS_PER_TASK = 2


def anon_mb():
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def clear():
    with Multiton._INSTANCE_LOCK:
        Multiton._INSTANCE_CACHE.clear()
        Multiton._EXPIRY_HEAP.clear()


def make_ms(src, dest, per_scan):
    shutil.copytree(src, dest)
    with table(str(dest), readonly=False, ack=False) as t:
        times = t.getcol("TIME")
        _, tidx = np.unique(times, return_inverse=True)
        t.putcol("SCAN_NUMBER", (tidx // per_scan + 1).astype(np.int32))


def run(blobs, passes, evict):
    clear()
    gc.collect()
    rows = []
    for _ in range(passes):
        for blob in blobs:
            node = pickle.loads(blob)
            ds = node.ds[NEEDED].load()
            _ = [np.asarray(ds[v].values).sum() for v in NEEDED]
            del ds, node
            gc.collect()
            if evict:
                clear()
            rows.append((anon_mb(), len(Multiton._INSTANCE_CACHE)))
    return np.array(rows)


def main():
    src = Path(sys.argv[1])
    per_scan = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    tmp = Path(tempfile.mkdtemp(prefix="release-ms-caches-"))
    try:
        ms = tmp / "multi.ms"
        make_ms(src, ms, per_scan)
        dt = xr.open_datatree(str(ms), **get_engine(str(ms)))
        blobs = []
        for name in dt.children:
            node = dt[name]
            nchan = node.ds.sizes["frequency"]
            for f0 in range(0, nchan, CHANS_PER_TASK):
                blobs.append(pickle.dumps(node.isel(frequency=slice(f0, f0 + CHANS_PER_TASK))))
        npart = len(dt.children)
        dt.close()
        del dt
        print(f"{npart} partitions, {len(blobs)} task slices x {passes} passes")
        for evict in (True, False, True):
            a = run(blobs, passes, evict)
            tail = a[len(a) // 2 :, 0]
            slope = np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0]
            print(
                f"  evict={str(evict):5s}: anon {a[0, 0]:7.1f} -> {a[-1, 0]:7.1f} MB "
                f"| 2nd-half slope {slope:+.3f} MB/task | cache entries max {int(a[:, 1].max())}"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
