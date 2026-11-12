#!/usr/bin/env python
"""Reproducer: xarray-ms ignores FLAG_ROW when FLAG is present.

    uv run python scripts/msv4_issues/flag_row_ignored.py <any.ms>

Sets FLAG_ROW on every row of the first timestamp while leaving FLAG False there,
then opens the MS through xarray-ms. MSv2 says a FLAG_ROW row is flagged whatever
FLAG holds; xarray-ms only reads FLAG_ROW when the FLAG column is absent, so the
MSv4 FLAG for that time comes back all False. Legacy ``pfb hci`` ORs FLAG_ROW in
itself (utils/stokes2im.py); ``pfb hci-msv4`` takes the MSv4 FLAG as is.

It copies the MS first, so the original is untouched.
"""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from casacore.tables import table as pctable

warnings.filterwarnings("ignore")


def main(src: str) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="flagrow-repro-"))
    try:
        return _run(src, str(tmp / "repro.ms"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(src: str, ms: str) -> int:
    shutil.copytree(src, ms)

    with pctable(ms, readonly=False, ack=False) as tab:
        time = tab.getcol("TIME")
        rows = np.nonzero(time == time.min())[0]
        flag = tab.getcol("FLAG")
        flag[rows] = False
        tab.putcol("FLAG", flag)
        frow = tab.getcol("FLAG_ROW")
        frow[rows] = True
        tab.putcol("FLAG_ROW", frow)
    print(f"MSv2: {rows.size} rows at TIME {time.min():.1f} have FLAG_ROW=True and FLAG=False")

    dt = xr.open_datatree(ms, engine="xarray-ms:msv2", partition_schema=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"])
    try:
        node = next(n for n in dt.subtree if n.attrs.get("type") == "visibility")
        it = int(np.argmin(node.ds.time.values))
        f0 = node.ds.FLAG.isel(time=it).values
        # xarray-ms pads missing (time, baseline) cells with flags set; keep cells that had a row
        real = np.isfinite(node.ds.UVW.isel(time=it).values).all(axis=-1)
        fraction = f0[real].mean()
    finally:
        dt.close()
    print(f"MSv4: FLAG at that time is {fraction:.0%} True over {real.sum()} baselines")
    if fraction < 1:
        print("REPRODUCED: FLAG_ROW is not reflected in the MSv4 FLAG")
        return 1
    print("NOT reproduced: the MSv4 FLAG honours FLAG_ROW")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
