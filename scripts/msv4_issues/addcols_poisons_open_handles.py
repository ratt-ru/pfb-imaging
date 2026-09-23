#!/usr/bin/env python
"""arcae/xarray-ms: adding a column poisons table handles already open elsewhere.

    uv run python scripts/msv4_issues/addcols_poisons_open_handles.py <any.ms>

Open a DataTree and read from it, leave it open, then create a column on the
same MS through a second tree. Reads through the first tree's cached handle now
fail with

    Table::lock cannot sync table <ms>; another process changed the number of columns

even though both handles are in this process. Evicting xarray-ms's process-wide
Multiton table cache recovers.

This is what breaks an in-process `imager -> degrid-msv4 -> imager` chain on
xarray-ms 0.4.0a8+, where `sync_msv2` creates canonical columns itself
(ratt-ru/xarray-ms#171) instead of leaving them to a separate handle.
Copies the MS first, so the original is untouched.
"""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import xarray as xr
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
from rarg_python_patterns.multiton import Multiton

from pfb_imaging.utils.degrid_msv4 import ensure_model_columns
from pfb_imaging.utils.msv4 import get_engine

warnings.filterwarnings("ignore")


def main(src: str) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="addcols-poison-"))
    ms = str(tmp / "x.ms")
    shutil.copytree(src, ms)
    held = []

    def read(tag: str, keep_open: bool) -> bool:
        try:
            dt = xr.open_datatree(ms, **get_engine(ms))
            node = next(n for n in dt.subtree if n.attrs.get("type") in VISIBILITY_XDS_TYPES)
            node.ds.UVW.values
        except Exception as e:  # noqa: BLE001 - reporting, not handling
            print(f"   {tag}: FAILED -- {e}")
            return False
        print(f"   {tag}: ok")
        held.append(dt) if keep_open else dt.close()
        return True

    print("1. read, and leave the tree OPEN (a pipeline stage that has not closed yet)")
    read("read A", keep_open=True)

    print("2. create MODEL_DATA on the same MS through a second tree")
    dt = xr.open_datatree(ms, **get_engine(ms, main_ninstances=1))
    try:
        ensure_model_columns(ms, dt, ["MODEL_DATA"])
    finally:
        dt.close()

    print("3. read again -- served by the cached handle from step 1")
    poisoned = not read("read B", keep_open=False)

    print("4. evict the process-wide table cache, then read again")
    with Multiton._INSTANCE_LOCK:
        Multiton._INSTANCE_CACHE.clear()
        Multiton._EXPIRY_HEAP.clear()
    recovered = read("read C", keep_open=False)

    for dt in held:
        try:
            dt.close()
        except Exception:  # noqa: BLE001
            pass
    shutil.rmtree(tmp, ignore_errors=True)

    if poisoned and recovered:
        print("\nREPRODUCED: the open handle was poisoned; evicting the cache recovered it.")
        return 0
    print(
        "\nNOT reproduced. Expected on xarray-ms < 0.4.0a8, where sync_msv2 does not "
        "create canonical columns and pfb-imaging makes MODEL_DATA through its own "
        "short-lived handle; otherwise upstream may have fixed it."
    )
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
