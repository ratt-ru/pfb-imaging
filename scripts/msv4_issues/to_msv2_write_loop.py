#!/usr/bin/env python
"""xarray-ms: does repeated `to_msv2()` still leak threads/fds and then deadlock?

Tests the claim in ratt-ru/tricolour#106 (`4872aa0c`), which says:

    xarray-ms's to_msv2() reopens the MS on every call and leaks OS threads and
    file descriptors as it goes (reads do not leak). After a few dozen writes
    the call deadlocks with every thread parked on a futex.

That was observed before ska-sa/arcae#235 landed. `table_open_close_rss.py`
shows #235 made fds and threads flat on 0.4.0a11 -- but only for the *read*
path, and tricolour's note says reads were never the problem. This is the
write-path equivalent, so the two can be compared directly.

It matters for `pfb degrid` because each work item ends in one
`to_msv2()`, so a real run does tens to hundreds of writes per replica --
squarely in the range tricolour reports deadlocking.

    uv run python scripts/msv4_issues/to_msv2_write_loop.py <any.ms> [iters]

The MS is copied first, so the original is never touched. Each iteration writes
one small region back and samples threads, fds and post-gc RSS. A deadlock
shows up as the script simply stopping -- print output is flushed per
iteration, so the last line tells you which write wedged.
"""

import gc
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import psutil
import xarray as xr
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

from pfb_imaging.utils.degrid import ensure_model_columns
from pfb_imaging.utils.msv4 import get_engine

warnings.filterwarnings("ignore")

COLUMN = "MODEL_DATA_WRITE_LOOP"


def main(ms: str, niter: int) -> int:
    proc = psutil.Process()
    tmp = Path(tempfile.mkdtemp(prefix="to_msv2_write_loop_"))
    work = tmp / Path(ms).name
    shutil.copytree(ms, work)
    print(f"working on a copy at {work}")

    # Create the column once, through a single-MAIN-instance handle, and close
    # it before anything else opens the MS (ska-sa/arcae#241).
    dt = xr.open_datatree(str(work), **get_engine(str(work), main_ninstances=1))
    try:
        ensure_model_columns(str(work), dt, [COLUMN])
    finally:
        dt.close()
        del dt
    gc.collect()

    rss, threads, fds = [], [], []
    for i in range(niter):
        dt = xr.open_datatree(str(work), **get_engine(str(work)))
        try:
            node = next(n for n in dt.subtree if n.attrs.get("type") in VISIBILITY_XDS_TYPES)
            region = {"time": slice(0, 1), "frequency": slice(0, 1)}
            ds = node.ds.isel(**region)
            vis = np.zeros(
                (ds.sizes["time"], ds.sizes["baseline_id"], ds.sizes["frequency"], ds.sizes["polarization"]),
                dtype=np.complex64,
            )
            write_ds = ds.drop_vars(set(ds.data_vars)).assign(
                {COLUMN: (("time", "baseline_id", "frequency", "polarization"), vis)}
            )
            write_ds.to_msv2(compute=True, region=region, write_map={COLUMN: COLUMN})
        finally:
            dt.close()
            del dt
        gc.collect()
        rss.append(proc.memory_info().rss / 2**20)
        threads.append(proc.num_threads())
        fds.append(proc.num_fds())
        print(
            f"  write {i + 1:4d}/{niter}  rss {rss[-1]:8.1f} MB  threads {threads[-1]:5d}  fds {fds[-1]:5d}",
            flush=True,
        )

    half = rss[len(rss) // 2 :]
    slope = np.polyfit(np.arange(len(half), dtype=float), half, 1)[0]
    print(
        f"\nwrite: first={rss[0]:8.1f} MB  last={rss[-1]:8.1f} MB  "
        f"second-half slope={slope:+.3f} MB/iter  over {niter} writes\n"
        f"       threads {threads[0]} -> {threads[-1]}   fds {fds[0]} -> {fds[-1]}"
    )
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    niter = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    raise SystemExit(main(sys.argv[1], niter))
