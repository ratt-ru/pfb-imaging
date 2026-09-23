#!/usr/bin/env python
"""arcae: adding a column leaves other already-open table handles unable to resync.

    uv run python scripts/msv4_issues/addcols_poisons_open_handles.py <any.ms> [ninstances]

Handle A opens and reads. Handle B, on the same table in the same process, adds
a column. Every subsequent read through A fails with

    Table::lock cannot sync table <ms>; another process changed the number of columns

A handle opened *after* the change is fine, so this is about handles that were
already open. Independent of `ninstances` -- it happens at 1 -- which
distinguishes it from the sibling-instance race in
`addcols_multi_instance_race.py`, where the two "handles" are instances of one
arcae Table.

`columns()` recovers on a retry (its second call re-reads the descriptor), but
`getcol` does not: the handle stays unusable for data reads.

In pfb-imaging this breaks an in-process `imager -> degrid-msv4 -> imager`
chain on xarray-ms >= 0.4.0a8, where `sync_msv2` creates canonical columns
itself (ratt-ru/xarray-ms#171) through a tree other than the one still open.
Copies the MS first, so the original is untouched.
"""

import shutil
import sys
import tempfile
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


def attempt(label: str, fn) -> bool:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - reporting, not handling
        print(f"  {label}: FAILED -- {e}")
        return False
    print(f"  {label}: ok")
    return True


def main(src: str, ninstances: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="addcols-poison-"))
    ms = str(tmp / "x.ms")
    shutil.copytree(src, ms)

    # `readonly`, `lockoptions` and `cache_size` here are what xarray-ms uses for MAIN
    reader = Table.from_filename(ms, ninstances, True, "auto", 256)
    attempt(f"A getcol, before the change (ninstances={ninstances})", lambda: reader.getcol("DATA"))

    writer = Table.from_filename(ms, ninstances, False, "auto", 256)
    attempt("B addcols, a second handle on the same table", lambda: writer.addcols(DESC, DMINFO))

    after = [
        attempt("A getcol, after the change", lambda: reader.getcol("DATA")),
        attempt("A getcol, retried", lambda: reader.getcol("DATA")),
    ]
    fresh = Table.from_filename(ms, ninstances, True, "auto", 256)
    fresh_ok = attempt("C getcol, on a handle opened after the change", lambda: fresh.getcol("DATA"))

    for table in (reader, writer, fresh):
        try:
            table.close()
        except Exception:  # noqa: BLE001
            pass
    shutil.rmtree(tmp, ignore_errors=True)

    if not any(after) and fresh_ok:
        print("\nREPRODUCED: the already-open handle is poisoned; a fresh handle is fine.")
        return 0
    print("\nNOT reproduced (upstream may have fixed it).")
    return 1


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1))
