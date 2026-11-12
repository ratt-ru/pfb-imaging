#!/usr/bin/env python
"""Reproducer: xarray-ms does not expose SIGMA or SIGMA_SPECTRUM.

    uv run python scripts/msv4_issues/sigma_not_exposed.py <any.ms>

Lists the MSv2 MAIN columns, then the data variables of the first MSv4 visibility node.
xarray-ms drops every *standard* MAIN column it does not map explicitly, keeping only
CORRECTED_DATA, CORRECTED_WEIGHT_SPECTRUM, MODEL_DATA and FLOAT_DATA
(``CorrelatedFactory``'s secondary-variable pass), so SIGMA and SIGMA_SPECTRUM never
appear, while a non-standard column would. ``pfb hci-msv4`` therefore refuses
``--sigma-column``. Read only: the MS is not modified.
"""

import sys
import warnings

import xarray as xr
from casacore.tables import table as pctable

warnings.filterwarnings("ignore")


def main(ms: str) -> int:
    with pctable(ms, ack=False) as tab:
        sigma_v2 = sorted(c for c in tab.colnames() if c.startswith("SIGMA"))
    print(f"MSv2 MAIN sigma columns: {sigma_v2}")

    dt = xr.open_datatree(ms, engine="xarray-ms:msv2", partition_schema=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"])
    try:
        node = next(n for n in dt.subtree if n.attrs.get("type") == "visibility")
        names = sorted(node.ds.data_vars)
    finally:
        dt.close()
    print(f"MSv4 data variables: {names}")
    missing = [c for c in sigma_v2 if c not in names]
    print(f"Not exposed: {missing}" if missing else "Every sigma column is exposed")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
