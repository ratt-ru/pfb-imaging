#!/usr/bin/env python
"""xarray-ms/arcae: post-gc RSS growth across repeated open/read/close cycles.

Filed as ratt-ru/xarray-ms#177.

    uv run python scripts/msv4_issues/table_open_close_rss.py <any.ms> [open|uvw|vis] [iters]

`open` opens and closes the DataTree without reading anything, so it isolates
open/close cost from read retention. Each iteration samples RSS after an
explicit `gc.collect()`, and reports the second-half slope -- a flat slope is a
bounded footprint, a positive one is retention gc cannot reach.

Measured on tests/data/test_ascii_1h60.0s.MS on an idle machine:

    0.4.0a8   800 iters  open +4.53 MB/iter  (fds 27 -> 10886, threads 80 -> 7011)
    0.4.0a11 2000 iters  open +1.53 MB/iter  (fds and threads flat)
    0.4.0a11 2000 iters  vis  +1.54 MB/iter

ska-sa/arcae#235 removed the descriptor and thread leak, which was most of the
0.4.0a8 figure. The remainder is a separate retention, linear with no plateau,
and `structure_rebuild_rss.py` localises it: reading contributes nothing, and
rebuilding `MSv2Structure` alone reproduces the whole rate.
"""

import gc
import sys
import warnings

import numpy as np
import psutil
import xarray as xr
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

from pfb_imaging.utils.msv4 import get_engine

warnings.filterwarnings("ignore")


def main(ms: str, what: str, niter: int) -> int:
    proc = psutil.Process()
    rss = []
    for _ in range(niter):
        dt = xr.open_datatree(ms, **get_engine(ms))
        try:
            if what != "open":
                for node in dt.subtree:
                    if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
                        continue
                    node.ds.UVW.values
                    if what == "vis":
                        node.ds.VISIBILITY.values
        finally:
            dt.close()
            del dt
        gc.collect()
        rss.append(proc.memory_info().rss / 2**20)

    half = rss[len(rss) // 2 :]
    slope = np.polyfit(np.arange(len(half), dtype=float), half, 1)[0]
    print(
        f"{what:4s}: first={rss[0]:8.1f} MB  last={rss[-1]:8.1f} MB  "
        f"second-half slope={slope:+.3f} MB/iter  over {niter} iterations"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    what = sys.argv[2] if len(sys.argv) > 2 else "vis"
    niter = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    raise SystemExit(main(sys.argv[1], what, niter))
