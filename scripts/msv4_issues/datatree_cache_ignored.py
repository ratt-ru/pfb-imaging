#!/usr/bin/env python
"""xarray-ms's open_datatree ignores `cache`, so every load is cached on the tree.

xarray wraps a backend variable in a MemoryCachedArray when `cache=True` (its
default without `chunks`). xarray-ms opens each MSv4 partition with that
default and does not forward the caller's `cache`, so `cache=False` removes
only the outer wrapper: the inner MemoryCachedArray stays, and the first load
of a variable -- through the node, a selection of it, or a shallow copy --
stores the full read inside the node for as long as the node lives.

    uv run python scripts/msv4_issues/datatree_cache_ignored.py <any.ms>

Expected with `cache` honoured: no MemoryCachedArray in the cache=False chain,
and ~0 x VISIBILITY held after the loaded copy is dropped.
"""

import gc
import sys
import tracemalloc
import warnings

import xarray as xr

warnings.filterwarnings("ignore")


def chain(data):
    out = []
    while data is not None and len(out) < 8:
        out.append(type(data).__name__)
        data = getattr(data, "array", None)
    return " > ".join(out)


def main():
    ms = sys.argv[1]
    schema = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]
    for cache in (True, False):
        dt = xr.open_datatree(ms, engine="xarray-ms:msv2", partition_schema=schema, cache=cache)
        node = next(iter(dt.children.values()))
        print(f"cache={cache}: {chain(node.ds.VISIBILITY.variable._data)}")
        tracemalloc.start()
        ds = node.ds[["VISIBILITY"]].copy(deep=False).load()
        nbytes = ds.VISIBILITY.nbytes
        del ds
        gc.collect()
        held = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        print(f"  after dropping a loaded shallow copy: {held / nbytes:.2f} x VISIBILITY still held by the node")
        dt.close()


if __name__ == "__main__":
    main()
