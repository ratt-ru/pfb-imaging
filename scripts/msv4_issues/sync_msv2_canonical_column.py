#!/usr/bin/env python
"""Reproducer: `sync_msv2` does not create a canonical-but-absent MAIN column.

    uv run python scripts/msv4_issues/sync_msv2_canonical_column.py <any.ms>

FIXED upstream in xarray-ms 0.4.0a8 (ratt-ru/xarray-ms#171). Kept as the
regression check: on 0.4.0a7 MODEL_DATA is silently NOT created while the
non-canonical MODEL_DATA_CUSTOM beside it is; on 0.4.0a8+ both are created.

It copies the MS first, so the original is untouched.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import arcae
import numpy as np
import xarray as xr
import xarray_ms
from arcae.lib.arrow_tables import ms_descriptor
from casacore.tables import table as pctable
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

MODEL_DIMS = ("time", "baseline_id", "frequency", "polarization")


def main(src: str) -> int:
    assert xarray_ms.multithreaded_writes(), "this xarray-ms has no write support; install the 0.4.0-alpha line"

    tmp = Path(tempfile.mkdtemp(prefix="syncmsv2-repro-"))
    ms = str(tmp / "repro.ms")
    shutil.copytree(src, ms)

    # MODEL_DATA is in casacore's canonical MAIN descriptor; MODEL_DATA_CUSTOM is not.
    canonical = ms_descriptor("MAIN", complete=True)
    print(f"MODEL_DATA in canonical MAIN descriptor: {'MODEL_DATA' in canonical}")
    print(f"MODEL_DATA_CUSTOM in canonical MAIN descriptor: {'MODEL_DATA_CUSTOM' in canonical}")

    # start from an MS that has neither column
    with pctable(ms, readonly=False, ack=False) as tab:
        for col in ("MODEL_DATA", "MODEL_DATA_CUSTOM"):
            if col in tab.colnames():
                tab.removecols(col)
    with arcae.table(ms) as tab:
        before = set(tab.columns())
    print(
        f"before: MODEL_DATA present={('MODEL_DATA' in before)} "
        f"MODEL_DATA_CUSTOM present={('MODEL_DATA_CUSTOM' in before)}"
    )

    # declare both variables on every correlated node, then sync
    dt = xr.open_datatree(ms, engine="xarray-ms:msv2", partition_schema=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"])
    try:
        for node in dt.subtree:
            if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
                continue
            shape = tuple(node.sizes[d] for d in MODEL_DIMS)
            # zero-strided placeholder: sync_msv2 reads only dims/shape/dtype
            ph = np.broadcast_to(np.array(0, np.complex64), shape)
            ds = node.ds.assign({c: (MODEL_DIMS, ph) for c in ("MODEL_DATA", "MODEL_DATA_CUSTOM")})
            dt[node.path] = xr.DataTree(ds)
        dt.sync_msv2(write_map={"MODEL_DATA": "MODEL_DATA", "MODEL_DATA_CUSTOM": "MODEL_DATA_CUSTOM"})
    finally:
        dt.close()

    with arcae.table(ms) as tab:
        after = set(tab.columns())
    got_canonical = "MODEL_DATA" in after
    got_custom = "MODEL_DATA_CUSTOM" in after
    print(f"after:  MODEL_DATA created={got_canonical} MODEL_DATA_CUSTOM created={got_custom}")

    print(f"\nworkspace: {tmp}  (delete when done)")
    if not got_canonical and got_custom:
        print("\nREPRODUCED: the canonical name was silently skipped, the non-canonical one was created.")
        return 0
    print("\nNOT reproduced (upstream may have fixed it).")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
