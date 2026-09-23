#!/usr/bin/env python
"""xarray-ms: each `MSv2Structure` build retains ~1.5 MB that nothing reclaims.

Filed as ratt-ru/xarray-ms#177.

    uv run python scripts/msv4_issues/structure_rebuild_rss.py <any.ms> [iters]

Builds the structure repeatedly, sampling RSS after `gc.collect()` and
`malloc_trim(0)`. Growth is linear with no plateau, and it is this that makes a
long-running process reopening measurement sets grow without bound: a DataTree
`close()` releases the structure factory, so the next open rebuilds it.

Measured on tests/data/test_ascii_1h60.0s.MS (xarray-ms 0.4.0a11, arcae
0.4.0a11, idle machine): +1.53 MB per build, identical over 2000 iterations
(338 MB -> 3.44 GB) whether or not any data is read.

What it is not -- each ruled out by measurement:

* not the allocator: `malloc_trim(0)` reclaims nothing, and the count of
  mappings in /proc/self/maps is flat
* not pyarrow: `bytes_allocated()` stays 0 and `mimalloc` and `jemalloc` give
  the same slope
* not file descriptors or threads: both flat (they were the arcae 0.4.0a8
  leak, fixed by ska-sa/arcae#235 -- on a8 this same loop takes fds 27 -> 10886
  and threads 80 -> 7011 at +4.5 MB/iter)
* not Python objects: `len(gc.get_objects())` grows by ~600 over 2000 builds
* not arcae: opening and closing MAIN (1 or 8 instances), reading MAIN columns
  with `getcol`/`to_arrow`, and reading all subtables with `to_arrow` are each
  flat to within 0.003 MB/iter
* not the thread pool: the rate is unchanged at `max_workers` 1, 4, 11 and 22
"""

import ctypes
import gc
import sys
import warnings

import numpy as np
import psutil
from rarg_python_patterns.multiton import Multiton
from xarray_ms.backend.msv2.entrypoint_utils import CommonStoreArgs
from xarray_ms.backend.msv2.structure import MSv2Structure

warnings.filterwarnings("ignore")

PARTITION_SCHEMA = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]


def main(ms: str, niter: int) -> int:
    libc = ctypes.CDLL("libc.so.6")
    proc = psutil.Process()
    rss = []
    for _ in range(niter):
        args = CommonStoreArgs(ms, partition_schema=PARTITION_SCHEMA)
        structure = MSv2Structure(
            args.ms_factory,
            args.subtable_factories,
            args.partition_schema,
            args.epoch,
            auto_corrs=args.auto_corrs,
        )
        del structure, args
        # a DataTree close() releases these factories; drop them the same way
        with Multiton._INSTANCE_LOCK:
            Multiton._INSTANCE_CACHE.clear()
            Multiton._EXPIRY_HEAP.clear()
        gc.collect()
        libc.malloc_trim(0)
        rss.append(proc.memory_info().rss / 2**20)

    values = np.array(rss)
    tail = values[-len(values) // 4 :]
    slope = np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0]
    print(
        f"{niter} structure builds: {values[0]:.1f} -> {values[-1]:.1f} MB  (last-quarter slope {slope:+.4f} MB/build)"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200))
