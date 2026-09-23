#!/usr/bin/env python
"""arcae: `columns()` after `addcols()` is answered by a stale table instance.

    uv run python scripts/msv4_issues/addcols_multi_instance_race.py <any.ms> 8
    uv run python scripts/msv4_issues/addcols_multi_instance_race.py <any.ms> 1

arcae runs `AddColumns` on instance 0 (`SpawnWriter`; its own comment calls
adding columns "non-syncable") but routes reads to the least busy instance.
Keep instance 0 busy and the follow-up read lands on an instance that has not
seen the new column, which either fails its lock resync or returns a stale
column list. `ninstances=1` is clean.

The table is opened exactly as xarray-ms opens MAIN:
`Table.from_filename(ms, ninstances, readonly=True, "auto", cache_size=256)`.
Copies the MS first, so the original is untouched.
"""

import shutil
import sys
import tempfile
import threading
from pathlib import Path

from arcae.lib.arrow_tables import Table

DESC = {
    "NEW_COL": {
        "valueType": "COMPLEX",
        "option": 4,
        "ndim": 2,
        "shape": [4, 8],
        "dataManagerGroup": "NEW_GROUP",
        "dataManagerType": "TiledColumnStMan",
    }
}
DMINFO = {
    "*1": {
        "NAME": "NEW_GROUP",
        "TYPE": "TiledColumnStMan",
        "SPEC": {"DEFAULTTILESHAPE": [4, 8, 64]},
        "COLUMNS": ["NEW_COL"],
    }
}


def main(src: str, ninstances: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arcae-addcols-race-"))
    ms = str(tmp / "race.ms")
    shutil.copytree(src, ms)

    table = Table.from_filename(ms, ninstances, True, "auto", 256)
    table.addcols(DESC, DMINFO)  # always runs on instance 0
    try:
        print(f"idle  columns() sees NEW_COL: {'NEW_COL' in table.columns()}")
    except Exception as e:  # noqa: BLE001 - it can already fail while idle
        print(f"idle  columns(): FAILED -- {e}")

    stop = threading.Event()

    def keep_instance_0_busy():
        while not stop.is_set():
            try:
                table.getcol("DATA")
            except Exception:
                pass

    threads = [threading.Thread(target=keep_instance_0_busy) for _ in range(2)]
    for t in threads:
        t.start()
    ok = failed = 0
    error = None
    try:
        for _ in range(200):
            try:
                table.columns()
                ok += 1
            except Exception as e:  # noqa: BLE001 - reporting, not handling
                failed += 1
                error = e
    finally:
        stop.set()
        for t in threads:
            t.join()

    print(f"busy  columns(): ok={ok} failed={failed}  (ninstances={ninstances})")
    if error:
        print(f"error: {error}")
    try:
        table.close()
        print("close(): ok")
    except Exception as e:  # noqa: BLE001
        print(f"close(): FAILED -- {e}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], int(sys.argv[2])))
