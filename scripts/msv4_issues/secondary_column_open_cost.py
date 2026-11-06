#!/usr/bin/env python
"""Reproducer: opening an MSv4 tree spends most of its time probing secondary-column shapes.

    uv run python scripts/msv4_issues/secondary_column_open_cost.py <ms> [<ms> ...]

Opens each MS with ``xr.open_datatree`` under cProfile and prints the open time per MS and
the share spent in xarray-ms's ``CorrelatedFactory.secondary_variables``, which asks arcae
for the row shapes of every non-standard MAIN column (``CORRECTED_DATA_DI``,
``DDF_PREDICT``, ``IMAGING_WEIGHT``, ...). On a MeerKLASS single-integration MS over NFS
(1891 rows, 1024 channels) the open took ~1.9 s, ~1 s of it there; a tile of 7594 such MSs
then needs ~4 h to plan serially. Read only.
"""

import cProfile
import pstats
import sys
import time
import warnings

import xarray as xr

warnings.filterwarnings("ignore")


def main(paths: list[str]) -> int:
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    for ms in paths:
        prof.enable()
        dt = xr.open_datatree(ms, engine="xarray-ms:msv2", partition_schema=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"])
        prof.disable()
        dt.close()
    total = time.perf_counter() - t0
    stats = pstats.Stats(prof).stats
    secondary = sum(v[3] for k, v in stats.items() if k[2] == "secondary_variables")
    print(
        f"{len(paths)} MS: {total / len(paths):.2f} s per open, {secondary / len(paths):.2f} s in secondary_variables"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
