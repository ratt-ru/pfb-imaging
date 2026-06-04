# Imager DataTree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pfb imager` produce a single unified `xarray.DataTree` (one node per output image, one child node per data partition) that subsumes today's `.xds`+`.dds` split, with a two-pass MS→tree pipeline, a reducible uv-`COUNTS` weighting product, a sum-over-partitions `HessianTree` operator, and FITS export.

**Architecture:** Pass 1 reads raw MSv4 data finely (per time-chunk/scan), converts to Stokes, averages, writes fine pieces + per-piece uv-counts into a `.scratch` store. A reduction stage combines counts by a named strategy. Pass 2 groups fine pieces into partitions `(field, spw, baseline_group)`, applies imaging weights, grids per-partition image products, sums them into per-band-time output-image nodes, and writes the `.dt` DataTree. `init`/`grid` stay live and untouched.

**Tech Stack:** Python 3.10+, `xarray` (DataTree + zarr), `ducc0` wgridder/FFT, `numba`, `ray`, `numexpr`, `pytest`.

---

## File Structure

**New files:**
- `src/pfb_imaging/utils/treestore.py` — DataTree/zarr access helpers (node naming, group read/write, iteration). Replaces `xds_from_url`/`xds_from_list` inside the imager path.
- `src/pfb_imaging/utils/counts_reduce.py` — named uv-counts reduction strategy registry.
- `src/pfb_imaging/operators/hessian_tree.py` — `HessianTree` operator (PSF-approx + exact-gridder backends).
- `src/pfb_imaging/utils/fits_tree.py` — `rdt2fits`, a tree-aware FITS writer (separate from `rdds2fits`).
- `src/pfb_imaging/core/imager_pass1.py` — pass-1 worker (`stokes_pass1`): fine Stokes piece + per-piece counts.
- `src/pfb_imaging/core/imager_pass2.py` — pass-2 worker (`grid_partition_image`): group→weight→grid→sum→write tree.
- `tests/test_treestore.py`, `tests/test_counts_reduce.py`, `tests/test_hessian_tree.py`, `tests/test_imager_tree.py` — new tests.

**Modified files:**
- `src/pfb_imaging/core/imager.py` — orchestrate the two passes + reduction + FITS.
- `src/pfb_imaging/cli/imager.py` — add imaging/weighting/fits/control options.
- `tests/test_imager.py` — update smoke assertions to the `.dt` product.

**Untouched (must keep working):** `utils/naming.py` (`xds_from_*`), `utils/fits.py` (`rdds2fits`), `operators/gridder.py`, `core/grid.py`, `core/init.py`, and all current `.dds` consumers.

---

## Conventions used across tasks

- Output-image node name: `band{bandid:04d}_time{timeid:04d}` (e.g. `band0000_time0000`).
- Partition node name: `part{pid:04d}` (e.g. `part0000`).
- A partition identity is the tuple `(msid, field_name, spw_name, baseline_group)`. `baseline_group` defaults to the string `"all"` this iteration.
- Counts grids are `(ncorr, nx_pad, ny_pad)` float arrays; the reduction registry is keyed by `(bandid, timeid)`.
- Run linting after each implementation step: `uv run ruff format . && uv run ruff check . --fix`.

---

## Phase 0 — Tree store helpers

### Task 0.1: Node-name helpers

**Files:**
- Create: `src/pfb_imaging/utils/treestore.py`
- Test: `tests/test_treestore.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_treestore.py
from pfb_imaging.utils.treestore import image_node_name, partition_node_name


def test_image_node_name_zero_pads():
    assert image_node_name(0, 0) == "band0000_time0000"
    assert image_node_name(3, 12) == "band0003_time0012"


def test_partition_node_name_zero_pads():
    assert partition_node_name(0) == "part0000"
    assert partition_node_name(7) == "part0007"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_treestore.py -v`
Expected: FAIL — `ModuleNotFoundError` / cannot import names.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/utils/treestore.py
"""Helpers for reading and writing the unified imager DataTree (.dt store).

The store is a single zarr hierarchy with one group per output image
(``band{b:04d}_time{t:04d}``) and one child group per data partition
(``part{p:04d}``). These helpers replace the bespoke ``xds_from_url`` /
``xds_from_list`` access pattern inside the imager code path.
"""


def image_node_name(bandid: int, timeid: int) -> str:
    """Return the group name for an output-image node."""
    return f"band{bandid:04d}_time{timeid:04d}"


def partition_node_name(pid: int) -> str:
    """Return the group name for a partition child node."""
    return f"part{pid:04d}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_treestore.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/treestore.py tests/test_treestore.py
git commit -m "feat(imager): add DataTree node-name helpers"
```

### Task 0.2: Write and read tree groups round-trip

**Files:**
- Modify: `src/pfb_imaging/utils/treestore.py`
- Test: `tests/test_treestore.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_treestore.py  (append)
import numpy as np
import xarray as xr

from pfb_imaging.utils.treestore import (
    image_node_name,
    open_tree,
    partition_node_name,
    write_group,
)


def test_write_and_open_tree_roundtrip(tmp_path):
    store = str(tmp_path / "out.dt")
    img = image_node_name(0, 0)
    part = partition_node_name(0)

    # band node: image-space sum products
    band_ds = xr.Dataset(
        {"DIRTY": (("corr", "x", "y"), np.ones((1, 4, 4)))},
        coords={"corr": ["I"]},
        attrs={"bandid": 0, "timeid": 0, "freq_out": 1.0e9},
    )
    write_group(store, img, band_ds, mode="w")

    # partition child: ragged row vis-space
    part_ds = xr.Dataset(
        {"VIS": (("corr", "row", "chan"), np.zeros((1, 5, 2), dtype=complex))},
        attrs={"field_name": "0", "spw_name": "0", "baseline_group": "all"},
    )
    write_group(store, f"{img}/{part}", part_ds, mode="a")

    dt = open_tree(store)
    assert dt[img].DIRTY.shape == (1, 4, 4)
    assert dt[img].attrs["freq_out"] == 1.0e9
    assert dt[f"{img}/{part}"].VIS.shape == (1, 5, 2)
    assert dt[f"{img}/{part}"].attrs["baseline_group"] == "all"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_treestore.py::test_write_and_open_tree_roundtrip -v`
Expected: FAIL — cannot import `write_group` / `open_tree`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/utils/treestore.py  (append)
import xarray as xr


def write_group(store_url: str, group: str, ds: xr.Dataset, mode: str = "a") -> None:
    """Write a single dataset to a group path inside the .dt zarr store.

    Args:
        store_url: Path/URL of the .dt store.
        group: Group path, e.g. ``"band0000_time0000"`` or
            ``"band0000_time0000/part0000"``.
        ds: Dataset to write.
        mode: zarr write mode (``"w"`` to (re)create the store, ``"a"`` to add).
    """
    ds.to_zarr(store_url, group=group, mode=mode)


def open_tree(store_url: str) -> xr.DataTree:
    """Open the unified imager DataTree from a .dt zarr store (lazy)."""
    return xr.open_datatree(store_url, engine="zarr", chunks=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_treestore.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/treestore.py tests/test_treestore.py
git commit -m "feat(imager): add tree group write/open round-trip"
```

### Task 0.3: Iterate image nodes and partition children

**Files:**
- Modify: `src/pfb_imaging/utils/treestore.py`
- Test: `tests/test_treestore.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_treestore.py  (append)
from pfb_imaging.utils.treestore import iter_image_nodes, iter_partitions


def test_iter_image_nodes_and_partitions(tmp_path):
    store = str(tmp_path / "out2.dt")
    # two image nodes, first has two partitions
    write_group(store, "band0000_time0000", xr.Dataset(attrs={"bandid": 0}), mode="w")
    write_group(store, "band0000_time0000/part0000", xr.Dataset(attrs={"field_name": "0"}), mode="a")
    write_group(store, "band0000_time0000/part0001", xr.Dataset(attrs={"field_name": "1"}), mode="a")
    write_group(store, "band0001_time0000", xr.Dataset(attrs={"bandid": 1}), mode="a")

    dt = open_tree(store)
    names = [name for name, _ in iter_image_nodes(dt)]
    assert names == ["band0000_time0000", "band0001_time0000"]

    first = dict(iter_image_nodes(dt))["band0000_time0000"]
    part_fields = [node.attrs["field_name"] for _, node in iter_partitions(first)]
    assert part_fields == ["0", "1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_treestore.py::test_iter_image_nodes_and_partitions -v`
Expected: FAIL — cannot import `iter_image_nodes` / `iter_partitions`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/utils/treestore.py  (append)
from collections.abc import Iterator


def iter_image_nodes(dt: xr.DataTree) -> Iterator[tuple[str, xr.DataTree]]:
    """Yield ``(name, node)`` for each output-image node, sorted by name.

    Output-image nodes are the children of the root whose name starts with
    ``"band"``.
    """
    for name in sorted(dt.children):
        if name.startswith("band"):
            yield name, dt[name]


def iter_partitions(image_node: xr.DataTree) -> Iterator[tuple[str, xr.DataTree]]:
    """Yield ``(name, node)`` for each partition child, sorted by name."""
    for name in sorted(image_node.children):
        if name.startswith("part"):
            yield name, image_node[name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_treestore.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/treestore.py tests/test_treestore.py
git commit -m "feat(imager): add image-node and partition iterators"
```

---

## Phase 1 — Counts reduction registry

### Task 1.1: `per-band-time` (identity) reduction

**Files:**
- Create: `src/pfb_imaging/utils/counts_reduce.py`
- Test: `tests/test_counts_reduce.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_counts_reduce.py
import numpy as np

from pfb_imaging.utils.counts_reduce import reduce_counts


def _grid(val):
    return np.full((1, 4, 4), float(val))


def test_per_band_time_is_identity():
    counts = {(0, 0): _grid(1), (1, 0): _grid(2)}
    out = reduce_counts(counts, "per-band-time", nband=2, ntime=1)
    assert set(out) == {(0, 0), (1, 0)}
    np.testing.assert_array_equal(out[(0, 0)], _grid(1))
    np.testing.assert_array_equal(out[(1, 0)], _grid(2))


def test_unknown_strategy_raises():
    try:
        reduce_counts({}, "nonsense", nband=1, ntime=1)
    except ValueError as e:
        assert "nonsense" in str(e)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_counts_reduce.py -v`
Expected: FAIL — cannot import `reduce_counts`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/utils/counts_reduce.py
"""Named reduction strategies for the uv-cell COUNTS product.

Pass 1 produces one counts grid per output image, keyed ``(bandid, timeid)``.
A reduction strategy maps that mapping to the *applied* counts used to build
imaging weights in pass 2. Strategies are registered in ``COUNTS_REDUCTIONS``;
add a new entry to expose a new ``--weight-grouping`` value.
"""

import numpy as np

CountsMap = dict[tuple[int, int], np.ndarray]


def _reduce_per_band_time(counts: CountsMap, nband: int, ntime: int) -> CountsMap:
    """Each output image keeps its own counts."""
    return {k: v for k, v in counts.items()}


COUNTS_REDUCTIONS = {
    "per-band-time": _reduce_per_band_time,
}


def reduce_counts(counts: CountsMap, strategy: str, nband: int, ntime: int) -> CountsMap:
    """Apply a named counts-reduction strategy.

    Args:
        counts: Mapping ``(bandid, timeid) -> (ncorr, nx, ny)`` counts grid.
        strategy: Registered strategy name.
        nband: Total number of output bands.
        ntime: Total number of output time chunks.

    Returns:
        Mapping with the same keys as ``counts`` whose values are the reduced
        (possibly shared) counts grids.

    Raises:
        ValueError: If ``strategy`` is not registered.
    """
    try:
        fn = COUNTS_REDUCTIONS[strategy]
    except KeyError:
        raise ValueError(f"Unknown weight grouping strategy {strategy!r}; expected one of {sorted(COUNTS_REDUCTIONS)}")
    return fn(counts, nband, ntime)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_counts_reduce.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/counts_reduce.py tests/test_counts_reduce.py
git commit -m "feat(imager): add counts reduction registry with per-band-time"
```

### Task 1.2: `mfs`, `per-band`, `per-time` reductions

**Files:**
- Modify: `src/pfb_imaging/utils/counts_reduce.py`
- Test: `tests/test_counts_reduce.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_counts_reduce.py  (append)
def test_mfs_sums_over_bands_within_time():
    counts = {(0, 0): _grid(1), (1, 0): _grid(2), (0, 1): _grid(10), (1, 1): _grid(20)}
    out = reduce_counts(counts, "mfs", nband=2, ntime=2)
    # time 0: 1+2=3 shared across both bands; time 1: 10+20=30
    np.testing.assert_array_equal(out[(0, 0)], _grid(3))
    np.testing.assert_array_equal(out[(1, 0)], _grid(3))
    np.testing.assert_array_equal(out[(0, 1)], _grid(30))
    np.testing.assert_array_equal(out[(1, 1)], _grid(30))


def test_per_band_sums_over_time_within_band():
    counts = {(0, 0): _grid(1), (0, 1): _grid(4), (1, 0): _grid(2)}
    out = reduce_counts(counts, "per-band", nband=2, ntime=2)
    np.testing.assert_array_equal(out[(0, 0)], _grid(5))
    np.testing.assert_array_equal(out[(0, 1)], _grid(5))
    np.testing.assert_array_equal(out[(1, 0)], _grid(2))


def test_per_time_sums_over_bands_within_time():
    counts = {(0, 0): _grid(1), (1, 0): _grid(2)}
    out = reduce_counts(counts, "per-time", nband=2, ntime=1)
    np.testing.assert_array_equal(out[(0, 0)], _grid(3))
    np.testing.assert_array_equal(out[(1, 0)], _grid(3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_counts_reduce.py -v`
Expected: FAIL — `mfs`/`per-band`/`per-time` not registered (`ValueError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/utils/counts_reduce.py  (replace the registry section)
def _sum_over(counts: CountsMap, fixed_axis: str, n_fixed: int) -> CountsMap:
    """Sum counts grids that share a fixed index.

    ``fixed_axis`` is the axis held constant while summing over the other:
    ``"time"`` sums over bands within each time (MFS / per-time),
    ``"band"`` sums over time within each band (per-band).
    """
    sums: dict[int, np.ndarray] = {}
    for (b, t), grid in counts.items():
        key = t if fixed_axis == "time" else b
        sums[key] = grid.copy() if key not in sums else sums[key] + grid
    out: CountsMap = {}
    for (b, t) in counts:
        key = t if fixed_axis == "time" else b
        out[(b, t)] = sums[key]
    return out


def _reduce_mfs(counts: CountsMap, nband: int, ntime: int) -> CountsMap:
    return _sum_over(counts, "time", ntime)


def _reduce_per_time(counts: CountsMap, nband: int, ntime: int) -> CountsMap:
    return _sum_over(counts, "time", ntime)


def _reduce_per_band(counts: CountsMap, nband: int, ntime: int) -> CountsMap:
    return _sum_over(counts, "band", nband)


COUNTS_REDUCTIONS = {
    "per-band-time": _reduce_per_band_time,
    "mfs": _reduce_mfs,
    "per-band": _reduce_per_band,
    "per-time": _reduce_per_time,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_counts_reduce.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/counts_reduce.py tests/test_counts_reduce.py
git commit -m "feat(imager): add mfs/per-band/per-time counts reductions"
```

---

## Phase 2 — Pass 1: fine Stokes pieces + per-piece counts

The pass-1 worker is an adaptation of `src/pfb_imaging/utils/stokes2vis_msv4.py::stokes_vis`
(read → column arithmetic → Stokes via `weight_data` → average → small-grid beam). The new
behaviour is: (a) write the fine piece into the `.scratch` store via `treestore`, tagged with
the full partition identity including `baseline_group`; and (b) compute and write that piece's
uv-counts contribution on the padded imaging grid.

### Task 2.1: `compute_piece_counts` helper

**Files:**
- Create: `src/pfb_imaging/core/imager_pass1.py`
- Test: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py
import numpy as np

from pfb_imaging.core.imager_pass1 import compute_piece_counts


def test_compute_piece_counts_bins_to_centre():
    # single baseline at uvw=0 lands in the central uv cell; weight 1, mask on
    uvw = np.zeros((1, 3))
    freq = np.array([1.0e9])
    mask = np.ones((1, 1), dtype=np.uint8)
    weight = np.ones((1, 1, 1))  # (corr, row, chan)
    nx_pad, ny_pad = 8, 8
    cell = 1.0e-6  # rad
    counts = compute_piece_counts(uvw, freq, mask, weight, nx_pad, ny_pad, cell, cell)
    assert counts.shape == (1, nx_pad, ny_pad)
    assert counts.sum() == 1.0
    # u=0,v=0 maps to centre cell (nx//2, ny//2)
    assert counts[0, nx_pad // 2, ny_pad // 2] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py::test_compute_piece_counts_bins_to_centre -v`
Expected: FAIL — cannot import `compute_piece_counts`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/core/imager_pass1.py
"""Pass-1 worker for the DataTree imager.

Reads a finely-partitioned MSv4 node slice, converts to Stokes, averages, and
writes the result plus its uv-counts contribution into the .scratch store.
Adapted from ``pfb_imaging.utils.stokes2vis_msv4.stokes_vis``.
"""

import numpy as np

from pfb_imaging.utils.weighting import _compute_counts


def compute_piece_counts(uvw, freq, mask, weight, nx_pad, ny_pad, cellx, celly, nthreads=1):
    """Bin one fine piece's weights onto the padded uv grid.

    Args:
        uvw: ``(nrow, 3)`` baseline coordinates.
        freq: ``(nchan,)`` channel frequencies (Hz).
        mask: ``(nrow, nchan)`` uint8 sampling mask.
        weight: ``(ncorr, nrow, nchan)`` imaging-direction weights.
        nx_pad, ny_pad: padded uv-grid dimensions.
        cellx, celly: image cell sizes (rad).
        nthreads: passed through to the numba kernel as the grid count.

    Returns:
        ``(ncorr, nx_pad, ny_pad)`` counts grid (float).
    """
    return _compute_counts(
        uvw,
        freq,
        mask,
        weight,
        nx_pad,
        ny_pad,
        cellx,
        celly,
        weight.dtype,
        ngrid=int(np.maximum(nthreads, 1)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imager_tree.py::test_compute_piece_counts_bins_to_centre -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_pass1.py tests/test_imager_tree.py
git commit -m "feat(imager): add pass-1 per-piece uv counts helper"
```

### Task 2.2: `stokes_pass1` worker writes fine piece + counts to scratch

**Files:**
- Modify: `src/pfb_imaging/core/imager_pass1.py`
- Test: `tests/test_imager_tree.py`

This worker is the body of `stokes_vis` with the output redirected to the scratch tree and a
counts piece written alongside. Reuse the exact Stokes/averaging/beam logic from
`stokes2vis_msv4.py:65-316` (do not change its numerics).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py  (append)
from pathlib import Path

import xarray as xr

from pfb_imaging.core.imager import get_engine  # reused MSv4 engine resolver
from pfb_imaging.core.imager_pass1 import stokes_pass1
from pfb_imaging.utils.treestore import open_tree
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES


def _first_vis_node(ms_name):
    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    for node in dt.children.values():
        if node.attrs.get("type") in VISIBILITY_XDS_TYPES:
            return node
    raise AssertionError("no visibility node")


def test_stokes_pass1_writes_piece_and_counts(ms_name, tmp_path):
    scratch = str(tmp_path / "out_I.scratch")
    node = _first_vis_node(ms_name).isel(time=slice(0, 8), frequency=slice(0, 4))

    res = stokes_pass1(
        dc1="VISIBILITY",
        dc2=None,
        operator=None,
        node_dt=node,
        scratch_store=scratch,
        bandid=0,
        timeid=0,
        msid=0,
        freq_out=float(node.ds.frequency.values.mean()),
        product="I",
        nx_pad=64,
        ny_pad=64,
        cell_rad=1.0e-6,
        max_blength=1.0e4,
        max_freq=2.0e9,
    )
    assert res is not None
    bandid, timeid, piece_name = res

    dt = open_tree(scratch)
    piece = dt[f"band0000_time0000/{piece_name}"]
    for v in ("VIS", "WEIGHT", "MASK", "UVW", "FREQ", "BEAM", "COUNTS"):
        assert v in piece.ds, f"missing {v}"
    assert piece.attrs["baseline_group"] == "all"
    assert piece.ds.COUNTS.shape == (piece.ds.corr.size, 64, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py::test_stokes_pass1_writes_piece_and_counts -v`
Expected: FAIL — cannot import `stokes_pass1`.

- [ ] **Step 3: Write minimal implementation**

Copy the numerics of `stokes_vis` (`stokes2vis_msv4.py`) verbatim through the averaging and
beam blocks, then replace the final write. The full worker:

```python
# src/pfb_imaging/core/imager_pass1.py  (append)
from datetime import datetime, timezone

import numexpr as ne
import ray
from katbeam import JimBeam
from scipy import ndimage
from scipy.constants import c as lightspeed

from pfb_imaging import pfb_version
from pfb_imaging.utils.treestore import image_node_name, write_group
from pfb_imaging.utils.weighting import weight_data


@ray.remote
def safe_stokes_pass1(*args, **kwargs):
    return stokes_pass1(*args, **kwargs)


def stokes_pass1(
    dc1=None,
    dc2=None,
    operator=None,
    node_dt=None,
    scratch_store=None,
    bandid=None,
    timeid=None,
    msid=None,
    freq_out=None,
    precision="double",
    sigma_column=None,
    weight_column=None,
    product="I",
    chan_average=1,
    bda_decorr=1.0,
    max_field_of_view=3.0,
    beam_model=None,
    wgt_mode="minvar",
    baseline_group="all",
    nx_pad=None,
    ny_pad=None,
    cell_rad=None,
    max_blength=None,
    max_freq=None,
    nthreads=1,
):
    """Convert one fine MSv4 node slice to a Stokes piece and write it + counts.

    Mirrors ``stokes2vis_msv4.stokes_vis`` numerically; differs only in output:
    the averaged piece (plus its ``COUNTS`` grid) is written into the .scratch
    store under ``band{b}_time{t}/<piece_name>``. Returns
    ``(bandid, timeid, piece_name)`` or ``None`` if fully flagged.
    """
    node_dt.load()
    ds = node_dt.ds
    field_name = np.unique(ds.field_name.values).item()
    spw_name = ds.frequency.attrs["spectral_window_name"]
    scan_name = np.unique(ds.scan_name.values).item()
    piece_name = (
        f"ms{msid:04d}_fid{field_name}_spw{spw_name}_bg{baseline_group}_scan{scan_name}"
    )

    if precision.lower() == "single":
        real_type, complex_type = np.float32, np.complex64
    elif precision.lower() == "double":
        real_type, complex_type = np.float64, np.complex128
    else:
        raise ValueError(f"Unsupported precision {precision!r}; expected 'single' or 'double'")

    data = getattr(ds, dc1).values
    if dc2 is not None:
        if operator not in ("+", "-"):
            raise ValueError(f"Unsupported operator {operator!r}; expected '+' or '-'")
        data = ne.evaluate(
            f"data {operator} data2",
            local_dict={"data": data, "data2": getattr(ds, dc2).values},
            casting="same_kind",
        )

    freq = ds.frequency.values
    freq_min, freq_max = freq.min(), freq.max()
    ntime, nbl, nchan, ncorr = data.shape
    nrow = ntime * nbl
    utime = ds.time.values
    time_out = np.mean(utime)

    it_attr = ds.time.attrs.get("integration_time", {}).get("data")
    if "INTEGRATION_TIME" in ds.data_vars:
        interval = ds.INTEGRATION_TIME.values.ravel()
    else:
        interval = np.full(nrow, it_attr, dtype=np.float64)

    ant1_names = ds.baseline_antenna1_name.values
    ant2_names = ds.baseline_antenna2_name.values
    ant12 = np.concatenate([ant1_names, ant2_names])
    _, inv = np.unique(ant12, return_inverse=True)
    ant1_bl = inv[: len(inv) // 2]
    ant2_bl = inv[len(inv) // 2 :]
    time, ant1, ant2 = np.broadcast_arrays(utime[:, None], ant1_bl[None, :], ant2_bl[None, :])
    time = time.ravel()
    ant1 = ant1.ravel().astype(np.int32)
    ant2 = ant2.ravel().astype(np.int32)
    uvw = ds.UVW.values.reshape(nrow, 3)
    flag = ds.FLAG.values.reshape(nrow, nchan, ncorr)

    frow = ant1 == ant2
    flag = np.logical_or(flag, frow[:, None, None])
    if flag.all():
        return None

    if sigma_column is not None:
        weight = ne.evaluate("1.0/sigma**2", local_dict={"sigma": getattr(ds, sigma_column).values})
        weight = weight.reshape(nrow, nchan, ncorr)
    elif weight_column is not None:
        weight = getattr(ds, weight_column).values.reshape(nrow, nchan, ncorr)
    else:
        weight = np.ones((nrow, nchan, ncorr), dtype=real_type)

    antpos = node_dt["antenna_xds"].ANTENNA_POSITION.values
    nant = antpos.shape[0]

    if set(ds.polarization.values).issubset({"XX", "XY", "YX", "YY"}):
        poltype = "linear"
    elif set(ds.polarization.values).issubset({"RR", "RL", "LR", "LL"}):
        poltype = "circular"
    else:
        raise ValueError("Unknown polarization types")

    base = node_dt.ds.attrs["data_groups"]["base"]
    radec = node_dt[base["field_and_source"].rsplit("/", 1)[-1]].ds.FIELD_PHASE_CENTER_DIRECTION.values[0]

    if data.dtype != complex_type:
        data = data.astype(complex_type)
    if weight.dtype != real_type:
        weight = weight.astype(real_type)

    jones = np.ones((ntime, nant, nchan, 1, 2), dtype=complex_type)
    tbin_idx = np.arange(ntime) * nbl
    tbin_counts = np.full(ntime, nbl)

    data = data.reshape(nrow, nchan, ncorr)
    data, weight = weight_data(
        data, weight, flag, jones, tbin_idx, tbin_counts, ant1, ant2, poltype, product, str(ncorr), wgt_mode
    )

    mrow = ~frow
    data = data[mrow]
    time = time[mrow]
    interval = interval[mrow]
    ant1 = ant1[mrow]
    ant2 = ant2[mrow]
    uvw = uvw[mrow]
    flag = flag[mrow]
    weight = weight[mrow]

    ncorr = data.shape[-1]
    flag = np.tile(flag.any(axis=-1, keepdims=True), (1, 1, ncorr))
    corr = list("".join(dict.fromkeys(sorted(product))))
    ncorr = len(corr)

    if "CHANNEL_WIDTH" in ds.data_vars:
        chan_width = ds.CHANNEL_WIDTH.values
    else:
        cw = ds.frequency.attrs["channel_width"]["data"]
        chan_width = np.full(ds.frequency.size, cw, dtype=np.float64)

    if chan_average > 1:
        from africanus.averaging import time_and_channel

        res = time_and_channel(
            time, interval, ant1, ant2, uvw=uvw, flag=flag, weight_spectrum=weight,
            visibilities=data, chan_freq=freq, chan_width=chan_width,
            time_bin_secs=1e-15, chan_bin_size=chan_average,
        )
        data, weight, flag = res.visibilities, res.weight_spectrum, res.flag
        freq, chan_width, uvw = res.chan_freq, res.chan_width, res.uvw
        nchan = freq.size

    if bda_decorr < 1:
        from africanus.averaging import bda

        res = bda(
            time, interval, ant1, ant2, uvw=uvw, flag=flag, weight_spectrum=weight,
            visibilities=data, chan_freq=freq, chan_width=chan_width,
            decorrelation=bda_decorr, min_nchan=freq.size, max_fov=max_field_of_view,
        )
        offsets = res.offsets
        uvw = res.uvw[offsets[:-1], :]
        weight = res.weight_spectrum.reshape(-1, nchan, ncorr)
        data = res.visibilities.reshape(-1, nchan, ncorr)
        flag = res.flag.reshape(-1, nchan, ncorr)

    flag = flag.any(axis=-1)
    mask = (~flag).astype(np.uint8)

    beam, l_beam, m_beam = _eval_beam_grid(
        corr, freq_out, freq_min, freq_max, max_field_of_view, max_blength, max_freq, beam_model, real_type
    )

    # transpose to corr-first for downstream contiguity
    vis_cf = data.transpose(2, 0, 1)
    wgt_cf = weight.transpose(2, 0, 1)

    # per-piece counts on the padded imaging grid
    counts = compute_piece_counts(uvw, freq, mask, wgt_cf, nx_pad, ny_pad, cell_rad, cell_rad, nthreads=nthreads)

    data_vars = {
        "VIS": (("corr", "row", "chan"), vis_cf),
        "WEIGHT": (("corr", "row", "chan"), wgt_cf),
        "MASK": (("row", "chan"), mask),
        "UVW": (("row", "three"), uvw),
        "FREQ": (("chan",), freq),
        "BEAM": (("corr", "l_beam", "m_beam"), beam),
        "COUNTS": (("corr", "u", "v"), counts),
    }
    coords = {
        "chan": (("chan",), freq),
        "l_beam": (("l_beam",), l_beam),
        "m_beam": (("m_beam",), m_beam),
        "corr": (("corr",), corr),
    }
    utc = datetime.fromtimestamp(time_out, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    attrs = {
        "pfb-imaging-version": pfb_version,
        "ra": radec[0], "dec": radec[1],
        "msid": msid, "field_name": field_name, "spw_name": spw_name,
        "baseline_group": baseline_group, "scan_name": scan_name,
        "freq_out": freq_out, "freq_min": freq_min, "freq_max": freq_max, "bandid": bandid,
        "time_out": time_out, "time_min": utime.min(), "time_max": utime.max(), "timeid": timeid,
        "product": product, "utc": utc,
        "max_freq": max_freq, "max_blength": max_blength, "beam_model": beam_model,
    }

    out_ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    image_name = image_node_name(bandid, timeid)
    write_group(scratch_store, f"{image_name}/{piece_name}", out_ds, mode="a")
    return bandid, timeid, piece_name


def _eval_beam_grid(corr, freq_out, freq_min, freq_max, max_field_of_view, max_blength, max_freq, beam_model, real_type):
    """Evaluate the rotation-averaged katbeam (or unit beam) on a small grid.

    Verbatim port of the beam block in ``stokes2vis_msv4.stokes_vis``.
    """
    if max_blength is None or max_freq is None:
        raise ValueError("max_blength and max_freq must be provided to size the beam grid")
    ncorr = len(corr)
    fov = max_field_of_view
    cell_rad = 1.0 / (max_blength * max_freq / lightspeed)
    cell_deg = np.rad2deg(cell_rad)
    npix = int(fov / cell_deg)
    l_beam = (-(npix // 2) + np.arange(npix)) * cell_deg
    m_beam = (-(npix // 2) + np.arange(npix)) * cell_deg
    if beam_model is None:
        return np.ones((ncorr, npix, npix), dtype=real_type), l_beam, m_beam
    if beam_model.lower() != "katbeam":
        raise ValueError(f"Unknown beam model {beam_model}")
    if freq_min >= 8.5e8 and freq_max <= 1.8e9:
        beamo = JimBeam("MKAT-AA-L-JIM-2020")
    elif freq_min >= 5.4e8 and freq_max <= 1.1e9:
        beamo = JimBeam("MKAT-AA-UHF-JIM-2020")
    else:
        raise ValueError("Freq range not covered by katbeam")
    xx, yy = np.meshgrid(l_beam, m_beam, indexing="ij")
    fmhz = freq_out / 1e6
    beam = np.zeros((ncorr, npix, npix), dtype=np.float64)
    angles = np.linspace(0, 359, 25)
    for i, product in enumerate(corr):
        beam0 = getattr(beamo, product)(xx, yy, fmhz)
        for angle in angles:
            beam[i] += ndimage.rotate(beam0, angle, reshape=False, mode="nearest")
        beam[i] /= angles.size
    return beam, l_beam, m_beam
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imager_tree.py::test_stokes_pass1_writes_piece_and_counts -v`
Expected: PASS (downloads test MS on first run).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_pass1.py tests/test_imager_tree.py
git commit -m "feat(imager): add pass-1 stokes worker writing scratch pieces + counts"
```

---

## Phase 3 — Pass 2: partition assembly, weighting, per-partition gridding

Pass 2 groups scratch pieces into partitions, applies the reduced imaging weights, and grids
per-partition image products. Numerics are adapted from
`src/pfb_imaging/operators/gridder.py::image_data_products` and `vis2im`. Model/residual major
cycles are out of scope here (deconv owns them); a fresh image has `RESIDUAL == DIRTY`.

### Task 3.1: `concat_partition` — concatenate same-partition pieces along `row`

**Files:**
- Create: `src/pfb_imaging/core/imager_pass2.py`
- Test: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py  (append)
from pfb_imaging.core.imager_pass2 import concat_partition


def _piece(nrow, ra=1.0, beamval=1.0):
    return xr.Dataset(
        {
            "VIS": (("corr", "row", "chan"), np.ones((1, nrow, 2), dtype=complex)),
            "WEIGHT": (("corr", "row", "chan"), np.full((1, nrow, 2), 2.0)),
            "MASK": (("row", "chan"), np.ones((nrow, 2), dtype=np.uint8)),
            "UVW": (("row", "three"), np.zeros((nrow, 3))),
            "FREQ": (("chan",), np.array([1.0e9, 1.1e9])),
            "BEAM": (("corr", "l_beam", "m_beam"), np.full((1, 3, 3), beamval)),
        },
        coords={"corr": ["I"], "l_beam": [-1.0, 0.0, 1.0], "m_beam": [-1.0, 0.0, 1.0]},
        attrs={"field_name": "0", "spw_name": "0", "baseline_group": "all", "ra": ra, "dec": 0.5, "freq_out": 1.05e9},
    )


def test_concat_partition_stacks_rows_keeps_beam():
    out = concat_partition([_piece(5), _piece(3)])
    assert out.VIS.shape == (1, 8, 2)
    assert out.MASK.shape == (8, 2)
    assert out.BEAM.shape == (1, 3, 3)  # beam not concatenated
    assert out.attrs["ra"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py::test_concat_partition_stacks_rows_keeps_beam -v`
Expected: FAIL — cannot import `concat_partition`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/core/imager_pass2.py
"""Pass-2 workers for the DataTree imager.

Assemble scratch pieces into partitions, apply reduced imaging weights, and
grid per-partition image-space products. Numerics adapted from
``pfb_imaging.operators.gridder.image_data_products``.
"""

import numpy as np
import xarray as xr


def concat_partition(pieces: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate same-partition scratch pieces along ``row``.

    All pieces share field/spw/baseline_group (hence phase centre and beam), so
    row-space arrays are concatenated while the (identical) ``BEAM`` and ``FREQ``
    are taken from the first piece. ``attrs`` are inherited from the first piece.

    Args:
        pieces: Datasets produced by pass 1 for one partition.

    Returns:
        A single dataset with concatenated ``VIS``/``WEIGHT``/``MASK``/``UVW``.
    """
    if len(pieces) == 1:
        return pieces[0]
    row_vars = ["VIS", "WEIGHT", "MASK", "UVW"]
    concatenated = xr.concat([p[row_vars] for p in pieces], dim="row", coords="minimal", compat="override")
    out = pieces[0].drop_vars(row_vars)
    for v in row_vars:
        out[v] = concatenated[v]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imager_tree.py::test_concat_partition_stacks_rows_keeps_beam -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_pass2.py tests/test_imager_tree.py
git commit -m "feat(imager): add pass-2 partition concatenation"
```

### Task 3.2: `apply_imaging_weights` — reduced counts → Briggs weights

**Files:**
- Modify: `src/pfb_imaging/core/imager_pass2.py`
- Test: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py  (append)
from pfb_imaging.core.imager_pass2 import apply_imaging_weights


def test_apply_imaging_weights_natural_passthrough_when_robustness_high():
    # robustness > 2 disables Briggs reweighting -> weights unchanged
    part = _piece(4)
    counts = np.ones((1, 8, 8))
    out = apply_imaging_weights(part, counts, robustness=2.5, cell_rad=1.0e-6, nx_pad=8, ny_pad=8)
    np.testing.assert_allclose(out.WEIGHT.values, part.WEIGHT.values)


def test_apply_imaging_weights_changes_weights_when_uniform():
    part = _piece(4)
    counts = np.full((1, 8, 8), 5.0)
    out = apply_imaging_weights(part, counts, robustness=-2.0, cell_rad=1.0e-6, nx_pad=8, ny_pad=8)
    assert out.WEIGHT.shape == part.WEIGHT.shape
    # uvw=0 lands in centre cell with counts=5 -> downweighted
    assert out.WEIGHT.values.max() <= part.WEIGHT.values.max()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py -k apply_imaging_weights -v`
Expected: FAIL — cannot import `apply_imaging_weights`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/core/imager_pass2.py  (append)
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.weighting import box_sum_counts, counts_to_weights, filter_extreme_counts


def apply_imaging_weights(
    part: xr.Dataset,
    counts: np.ndarray,
    robustness: float | None,
    cell_rad: float,
    nx_pad: int,
    ny_pad: int,
    l0: float = 0.0,
    m0: float = 0.0,
    filter_counts_level: float = 5.0,
    npix_super: int = 0,
) -> xr.Dataset:
    """Return ``part`` with ``WEIGHT`` scaled to Briggs weights from ``counts``.

    When ``robustness`` is ``None`` or ``> 2`` the natural weights are returned
    unchanged (``counts_to_weights`` is a no-op for ``robust > 2``).

    Args:
        part: Partition dataset (corr-first ``WEIGHT``/``UVW``/``MASK``/``FREQ``).
        counts: Reduced counts grid ``(ncorr, nx_pad, ny_pad)``.
        robustness: Briggs robustness; ``None`` => natural.
        cell_rad: Image cell size (rad), same used to build ``counts``.
        nx_pad, ny_pad: Padded uv-grid dims matching ``counts``.
        l0, m0: Phase-centre offsets for the wgridder sign convention.
    """
    if robustness is None:
        return part
    flip_u, flip_v, _, _, _ = wgridder_conventions(l0, m0)
    usign = 1.0 if flip_u else -1.0
    vsign = 1.0 if flip_v else -1.0

    counts = filter_extreme_counts(counts.copy(), level=filter_counts_level)
    counts = box_sum_counts(counts, npix_super)

    wgt = part.WEIGHT.values.copy()
    uvw = part.UVW.values
    freq = part.FREQ.values
    mask = part.MASK.values
    wgt = counts_to_weights(
        counts, uvw, freq, wgt, mask, nx_pad, ny_pad, cell_rad, cell_rad, robustness, usign=usign, vsign=vsign
    )
    out = part.copy()
    out["WEIGHT"] = (("corr", "row", "chan"), wgt)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imager_tree.py -k apply_imaging_weights -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_pass2.py tests/test_imager_tree.py
git commit -m "feat(imager): apply reduced counts as imaging weights per partition"
```

### Task 3.3: `grid_partition` — per-partition DIRTY/PSF/PSFHAT/BEAM/WSUM

**Files:**
- Modify: `src/pfb_imaging/core/imager_pass2.py`
- Test: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py  (append)
from pfb_imaging.core.imager_pass2 import grid_partition


def test_grid_partition_shapes_and_wsum():
    part = _piece(6)
    # spread uvw so gridding is well-defined
    rng = np.random.default_rng(0)
    part = part.copy()
    part["UVW"] = (("row", "three"), rng.standard_normal((6, 3)) * 100.0)
    nx = ny = 16
    nx_psf = ny_psf = 32
    out = grid_partition(part, nx, ny, nx_psf, ny_psf, cellx=1.0e-6, celly=1.0e-6)
    assert out["DIRTY"].shape == (1, nx, ny)
    assert out["PSF"].shape == (1, nx_psf, ny_psf)
    assert out["PSFHAT"].shape == (1, nx_psf, ny_psf // 2 + 1)
    assert out["BEAM"].shape == (1, nx, ny)
    assert out["WSUM"].shape == (1,)
    # wsum equals summed masked weights
    expected = (part.WEIGHT.values[0] * part.MASK.values).sum()
    np.testing.assert_allclose(out["WSUM"].values[0], expected, rtol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py::test_grid_partition_shapes_and_wsum -v`
Expected: FAIL — cannot import `grid_partition`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/core/imager_pass2.py  (append)
from ducc0.fft import r2c
from ducc0.wgridder.experimental import vis2dirty

from pfb_imaging.utils.beam import eval_beam
from pfb_imaging.utils.misc import fitcleanbeam

ifftshift = np.fft.ifftshift


def grid_partition(
    part: xr.Dataset,
    nx: int,
    ny: int,
    nx_psf: int,
    ny_psf: int,
    cellx: float,
    celly: float,
    l0: float = 0.0,
    m0: float = 0.0,
    nthreads: int = 1,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    do_dirty: bool = True,
    do_psf: bool = True,
    do_beam: bool = True,
) -> dict:
    """Grid one partition's image-space products.

    Returns a dict of numpy arrays (``DIRTY``/``PSF``/``PSFHAT``/``BEAM``/
    ``PSFPARSN``/``WSUM``), each correlation gridded independently. Per-partition
    phase offsets come from ``l0,m0``. PSF phase-shift handling for off-centre
    fields mirrors ``image_data_products``.
    """
    from scipy.constants import c as lightspeed

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(l0, m0)
    signu = -1.0 if flip_u else 1.0
    signv = -1.0 if flip_v else 1.0
    signx = -1.0 if flip_u else 1.0
    signy = -1.0 if flip_v else 1.0
    n = np.sqrt(1 - x0**2 - y0**2)

    uvw = part.UVW.values
    vis = part.VIS.values
    wgt = part.WEIGHT.values
    mask = part.MASK.values
    freq = part.FREQ.values
    ncorr = part.corr.size

    out: dict = {}
    wsum = wgt[:, mask.astype(bool)].sum(axis=-1)
    out["WSUM"] = wsum

    if do_beam:
        x = (-nx / 2 + np.arange(nx)) * cellx + x0
        y = (-ny / 2 + np.arange(ny)) * celly + y0
        xx, yy = np.meshgrid(np.rad2deg(x), np.rad2deg(y), indexing="ij")
        beam = np.zeros((ncorr, nx, ny), dtype=float)
        for c in range(ncorr):
            beam[c] = eval_beam(part.BEAM.values[c], part.l_beam.values, part.m_beam.values, xx, yy)
        out["BEAM"] = beam

    if do_dirty:
        dirty = np.zeros((ncorr, nx, ny), dtype=float)
        for c in range(ncorr):
            vis2dirty(
                uvw=uvw, freq=freq, vis=vis[c], wgt=wgt[c], mask=mask, npix_x=nx, npix_y=ny,
                pixsize_x=cellx, pixsize_y=celly, center_x=x0, center_y=y0, epsilon=epsilon,
                flip_u=flip_u, flip_v=flip_v, flip_w=flip_w, do_wgridding=do_wgridding,
                divide_by_n=False, nthreads=nthreads, sigma_min=1.1, sigma_max=3.0,
                double_precision_accumulation=double_accum, dirty=dirty[c],
            )
        out["DIRTY"] = dirty

    if do_psf:
        if x0 or y0:
            freqfactor = 2j * np.pi * freq[None, :] / lightspeed
            psf_vis = np.exp(
                freqfactor * (signu * uvw[:, 0:1] * x0 * signx + signv * uvw[:, 1:2] * y0 * signy - uvw[:, 2:] * (n - 1))
            )
        else:
            psf_vis = np.broadcast_to(np.ones((1,), dtype=vis.dtype), (uvw.shape[0], freq.size))
        psf = np.zeros((ncorr, nx_psf, ny_psf), dtype=float)
        for c in range(ncorr):
            vis2dirty(
                uvw=uvw, freq=freq, vis=psf_vis, wgt=wgt[c], mask=mask, npix_x=nx_psf, npix_y=ny_psf,
                pixsize_x=cellx, pixsize_y=celly, center_x=x0, center_y=y0, flip_u=flip_u, flip_v=flip_v,
                flip_w=flip_w, epsilon=epsilon, do_wgridding=do_wgridding, divide_by_n=False,
                nthreads=nthreads, sigma_min=1.1, sigma_max=3.0, double_precision_accumulation=double_accum,
                dirty=psf[c],
            )
        out["PSF"] = psf
        out["PSFHAT"] = r2c(ifftshift(psf, axes=(1, 2)), axes=(1, 2), nthreads=nthreads, forward=True, inorm=0)
        out["PSFPARSN"] = np.array(fitcleanbeam(psf, level=0.5, pixsize=1.0))

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imager_tree.py::test_grid_partition_shapes_and_wsum -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_pass2.py tests/test_imager_tree.py
git commit -m "feat(imager): grid per-partition dirty/psf/psfhat/beam/wsum"
```

---

## Phase 4 — `HessianTree` operator (sum over partitions)

`HessianTree` realises `H x = (1/Σ_p wsum_p) Σ_p B_pᵀ G_pᵀ W_p G_p B_p x + η x`. It carries
**both** backends sharing the stored data: a fast PSF-convolution approximation (per-partition
`PSFHAT`+`BEAM`) and an exact per-partition gridder round-trip (`hessian_slice`). Summing the
un-normalised per-partition convolutions then dividing by the total `wsum` matches how `grid.py`
forms `residual_mfs`/`psf_mfs`.

### Task 4.1: PSF-approx backend

**Files:**
- Create: `src/pfb_imaging/operators/hessian_tree.py`
- Test: `tests/test_hessian_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hessian_tree.py
import numpy as np
from ducc0.fft import r2c

from pfb_imaging.operators.hessian_tree import HessianTree

ifftshift = np.fft.ifftshift


def _psf_partition(nx, ny, nx_psf, ny_psf, wsum, seed):
    rng = np.random.default_rng(seed)
    psf = np.zeros((1, nx_psf, ny_psf))
    psf[0, nx_psf // 2, ny_psf // 2] = wsum  # delta PSF scaled by wsum
    psfhat = r2c(ifftshift(psf, axes=(1, 2)), axes=(1, 2), forward=True, inorm=0)
    beam = np.ones((1, nx, ny))
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([float(wsum)])}


def test_psf_backend_delta_psf_is_identity_up_to_eta():
    nx = ny = 8
    nx_psf = ny_psf = 16
    parts = [_psf_partition(nx, ny, nx_psf, ny_psf, wsum=3.0, seed=0)]
    H = HessianTree(parts, nx, ny, nx_psf, ny_psf, eta=0.0, mode="psf")
    x = np.zeros((1, nx, ny))
    x[0, 4, 4] = 1.0
    out = H.dot(x)
    # normalised delta PSF convolution is the identity
    np.testing.assert_allclose(out[0, 4, 4], 1.0, atol=1e-6)
    assert abs(out.sum() - 1.0) < 1e-6


def test_psf_backend_two_identical_partitions_equal_one():
    nx = ny = 8
    nx_psf = ny_psf = 16
    one = [_psf_partition(nx, ny, nx_psf, ny_psf, 3.0, 0)]
    two = [_psf_partition(nx, ny, nx_psf, ny_psf, 3.0, 0), _psf_partition(nx, ny, nx_psf, ny_psf, 3.0, 0)]
    x = np.zeros((1, nx, ny))
    x[0, 5, 2] = 1.7
    o1 = HessianTree(one, nx, ny, nx_psf, ny_psf, eta=0.1, mode="psf").dot(x)
    o2 = HessianTree(two, nx, ny, nx_psf, ny_psf, eta=0.1, mode="psf").dot(x)
    np.testing.assert_allclose(o1, o2, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hessian_tree.py -v`
Expected: FAIL — cannot import `HessianTree`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/operators/hessian_tree.py
"""Sum-over-partitions Hessian operator over the unified imager DataTree.

H x = (1/Σ_p wsum_p) Σ_p B_pᵀ (PSF_p ⊛ (B_p x)) + η x   [mode="psf"]
or the exact per-partition gridder round-trip                [mode="exact"].
"""

import numpy as np
from ducc0.fft import c2r, r2c

from pfb_imaging.operators.hessian import hessian_slice


class HessianTree:
    def __init__(self, partitions, nx, ny, nx_psf=None, ny_psf=None, eta=0.0, nthreads=1, mode="psf"):
        """Args:
        partitions: list of per-partition dicts. For ``mode="psf"`` each holds
            ``psfhat`` ``(ncorr, nx_psf, nyo2)``, ``beam`` ``(ncorr, nx, ny)``,
            ``wsum`` ``(ncorr,)``. For ``mode="exact"`` each holds ``uvw``,
            ``freq``, ``weight`` ``(ncorr,row,chan)``, ``mask``, ``beam``,
            ``wsum``, and convention floats ``cell``/``x0``/``y0``/``flip_*``.
        nx, ny: image dimensions.
        nx_psf, ny_psf: PSF dimensions (required for ``mode="psf"``).
        eta: Tikhonov parameter.
        mode: ``"psf"`` or ``"exact"``.
        """
        self.parts = partitions
        self.nx, self.ny = nx, ny
        self.nx_psf, self.ny_psf = nx_psf, ny_psf
        self.eta = eta
        self.nthreads = nthreads
        self.mode = mode
        self.ncorr = partitions[0]["wsum"].size
        self.wsum = np.zeros(self.ncorr)
        for p in partitions:
            self.wsum += p["wsum"]

    def _dot_psf(self, x):
        out = np.zeros_like(x)
        for p in self.parts:
            beam = p["beam"]
            psfhat = p["psfhat"]
            for c in range(self.ncorr):
                xpad = np.zeros((self.nx_psf, self.ny_psf))
                xpad[: self.nx, : self.ny] = x[c] * beam[c]
                xhat = r2c(xpad, axes=(0, 1), forward=True, inorm=0, nthreads=self.nthreads)
                xhat *= psfhat[c]
                conv = c2r(xhat, axes=(0, 1), forward=False, lastsize=self.ny_psf, inorm=2, nthreads=self.nthreads)
                out[c] += beam[c] * conv[: self.nx, : self.ny]
        out /= self.wsum[:, None, None]
        out += self.eta * x
        return out

    def _dot_exact(self, x):
        out = np.zeros_like(x)
        for p in self.parts:
            beam = p["beam"]
            for c in range(self.ncorr):
                out[c] += hessian_slice(
                    x[c],
                    uvw=p["uvw"],
                    weight=p["weight"][c],
                    vis_mask=p["mask"],
                    freq=p["freq"],
                    beam=beam[c],
                    cell=p["cell"],
                    x0=p["x0"],
                    y0=p["y0"],
                    flip_u=p["flip_u"],
                    flip_v=p["flip_v"],
                    flip_w=p["flip_w"],
                    do_wgridding=p.get("do_wgridding", True),
                    epsilon=p.get("epsilon", 1e-7),
                    double_accum=p.get("double_accum", True),
                    nthreads=self.nthreads,
                )
        out /= self.wsum[:, None, None]
        out += self.eta * x
        return out

    def dot(self, x):
        x = x if x.ndim == 3 else x[None]
        if self.mode == "psf":
            return self._dot_psf(x)
        if self.mode == "exact":
            return self._dot_exact(x)
        raise ValueError(f"Unknown mode {self.mode!r}")

    def hdot(self, x):
        return self.dot(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hessian_tree.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/operators/hessian_tree.py tests/test_hessian_tree.py
git commit -m "feat(imager): add HessianTree PSF-approx backend"
```

### Task 4.2: Exact gridder backend agrees with single `hessian_slice`

**Files:**
- Test: `tests/test_hessian_tree.py` (implementation already present from Task 4.1)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hessian_tree.py  (append)
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.operators.hessian import hessian_slice


def test_exact_backend_matches_single_hessian_slice():
    rng = np.random.default_rng(1)
    nx = ny = 12
    nrow, nchan = 50, 1
    uvw = rng.standard_normal((nrow, 3)) * 50.0
    freq = np.array([1.0e9])
    weight = np.abs(rng.standard_normal((1, nrow, nchan)))
    mask = np.ones((nrow, nchan), dtype=np.uint8)
    beam = np.ones((1, nx, ny))
    wsum = np.array([weight[0].sum()])
    cell = 1.0e-6
    fu, fv, fw, x0, y0 = wgridder_conventions(0.0, 0.0)
    part = {
        "uvw": uvw, "freq": freq, "weight": weight, "mask": mask, "beam": beam, "wsum": wsum,
        "cell": cell, "x0": x0, "y0": y0, "flip_u": fu, "flip_v": fv, "flip_w": fw,
    }
    x = rng.standard_normal((1, nx, ny))
    H = HessianTree([part], nx, ny, eta=0.0, mode="exact")
    out = H.dot(x)

    ref = hessian_slice(
        x[0], uvw=uvw, weight=weight[0], vis_mask=mask, freq=freq, beam=beam[0], cell=cell,
        x0=x0, y0=y0, flip_u=fu, flip_v=fv, flip_w=fw, do_wgridding=True, epsilon=1e-7,
        double_accum=True, nthreads=1,
    ) / wsum[0]
    np.testing.assert_allclose(out[0], ref, rtol=1e-5, atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `uv run pytest tests/test_hessian_tree.py::test_exact_backend_matches_single_hessian_slice -v`
Expected: PASS (implementation from Task 4.1 already supports `mode="exact"`). If it fails,
fix `_dot_exact` until it matches.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add tests/test_hessian_tree.py
git commit -m "test(imager): exact HessianTree backend matches hessian_slice"
```

---

## Phase 5 — Tree-aware FITS export (`rdt2fits`)

`rdds2fits` (`utils/fits.py`) must stay untouched for the live `.dds` consumers. We add a
separate `rdt2fits` that reads band nodes from the `.dt` store and reuses the access-agnostic
FITS helpers `set_wcs`, `save_fits`, `create_beams_table` from `utils/fits.py`.

### Task 5.1: `rdt2fits` writes MFS + cube from band nodes

**Files:**
- Create: `src/pfb_imaging/utils/fits_tree.py`
- Test: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py  (append)
from astropy.io import fits as afits

from pfb_imaging.utils.fits_tree import rdt2fits
from pfb_imaging.utils.treestore import write_group


def _band_node(bandid, freq_out, val):
    return xr.Dataset(
        {
            "DIRTY": (("corr", "x", "y"), np.full((1, 8, 8), float(val))),
            "WSUM": (("corr",), np.array([10.0])),
        },
        coords={"corr": ["I"]},
        attrs={
            "bandid": bandid, "timeid": 0, "freq_out": freq_out, "time_out": 0.0,
            "ra": 0.1, "dec": -0.2, "cell_rad": 1.0e-6,
        },
    )


def test_rdt2fits_writes_mfs(tmp_path):
    store = str(tmp_path / "out_I.dt")
    write_group(store, "band0000_time0000", _band_node(0, 1.0e9, 1.0), mode="w")
    write_group(store, "band0001_time0000", _band_node(1, 1.1e9, 3.0), mode="a")

    outname = str(tmp_path / "img")
    rdt2fits(store, "DIRTY", outname, norm_wsum=True, do_mfs=True, do_cube=False)

    mfs = outname + "_dirty_time0_mfs.fits"
    data = afits.getdata(mfs)
    assert data.shape[-2:] == (8, 8)
    # norm_wsum MFS = sum(cube)/sum(wsum) = (1+3)/(10+10) per pixel
    np.testing.assert_allclose(np.squeeze(data)[0, 0], (1.0 + 3.0) / 20.0, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py::test_rdt2fits_writes_mfs -v`
Expected: FAIL — cannot import `rdt2fits`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/utils/fits_tree.py
"""Tree-aware FITS export for the unified imager DataTree (.dt store).

Separate from ``utils.fits.rdds2fits`` (which reads a flat .dds list and must
keep working). Reuses the access-agnostic helpers ``set_wcs`` / ``save_fits`` /
``create_beams_table`` from ``utils.fits``.
"""

import numpy as np
import ray
import xarray as xr

from pfb_imaging.utils.fits import create_beams_table, save_fits, set_wcs
from pfb_imaging.utils.treestore import iter_image_nodes, open_tree


@ray.remote
def rrdt2fits(*args, **kwargs):
    return rdt2fits(*args, **kwargs)


def rdt2fits(
    store_url,
    column,
    outname,
    norm_wsum=True,
    nthreads=1,
    do_mfs=True,
    do_cube=True,
    psfpars_mfs=None,
    otype=np.float32,
    force_unit=None,
):
    """Render a band-node variable from the .dt store to FITS.

    Mirrors ``rdds2fits`` semantics but sources band-stacked data from the tree.
    Band nodes must carry ``WSUM`` and attrs ``freq_out``/``time_out``/``timeid``/
    ``ra``/``dec``/``cell_rad``. ``PSFPARSN`` is used for cube beam tables if present.
    """
    basename = outname + "_" + column.lower()
    unit = force_unit or ("Jy/beam" if norm_wsum else "Jy/pixel")

    dt = open_tree(store_url)
    nodes = [node.ds for _, node in iter_image_nodes(dt) if column in node.ds]
    if not nodes:
        return column

    timeids = np.unique([int(ds.attrs["timeid"]) for ds in nodes])
    freqs = np.unique([ds.attrs["freq_out"] for ds in nodes])
    nband = freqs.size

    for timeid in timeids:
        dst = sorted(
            [ds for ds in nodes if int(ds.attrs["timeid"]) == timeid],
            key=lambda d: d.attrs["freq_out"],
        )
        # stack along a new band axis, carrying attrs explicitly
        dsb = xr.concat(dst, dim="band").assign_coords({"band": np.arange(nband)})
        ref = dst[0]
        wsums = dsb.WSUM.values  # (band, corr)
        wsum = wsums.sum(axis=0)  # (corr,)
        cube = dsb.get(column).values  # (band, corr, nx, ny)
        _, _, nx, ny = cube.shape
        radec = (ref.attrs["ra"], ref.attrs["dec"])
        cell_deg = np.rad2deg(ref.attrs["cell_rad"])
        ncorr = ref.corr.size

        beams_hdu = None
        psfpars_mfs_timeid = None
        if psfpars_mfs is not None:
            pp = np.asarray(psfpars_mfs[timeid])
            assert pp.shape == (ncorr, 3)
            da = xr.DataArray(pp[None], coords={"band": [0], "corr": ref.corr.values, "bpar": ["BMAJ", "BMIN", "BPA"]})
            beams_hdu = create_beams_table(da, cell2deg=cell_deg)
            psfpars_mfs_timeid = pp[0]

        if do_mfs:
            freq_mfs = np.sum(freqs[:, None] * wsums) / wsum.sum()
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_mfs, unit=unit,
                          ms_time=ref.attrs["time_out"], gausspar=psfpars_mfs_timeid)
            hdr["WSUM"] = float(wsum[0])
            if norm_wsum:
                cube_mfs = np.sum(cube, axis=0) / wsum[:, None, None]
            else:
                cube_mfs = np.sum(cube * wsums[:, :, None, None], axis=0) / wsum[:, None, None]
            save_fits(cube_mfs, basename + f"_time{timeid}_mfs.fits", hdr, overwrite=True, dtype=otype, beams_hdu=beams_hdu)

        if do_cube:
            cube_out = cube / wsums[:, :, None, None] if norm_wsum else cube
            cube_beams = None
            gausspars = None
            if "PSFPARSN" in dsb:
                da = dsb.PSFPARSN
                gausspars = da.values[:, 0]
                cube_beams = create_beams_table(da, cell2deg=cell_deg)
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freqs, unit=unit,
                          ms_time=ref.attrs["time_out"], gausspar=psfpars_mfs_timeid, gausspars=gausspars)
            # FITS cube wants (band, corr, nx, ny) -> (corr, band, nx, ny)
            save_fits(np.moveaxis(cube_out, 0, 1), basename + f"_time{timeid}.fits", hdr, overwrite=True, dtype=otype, beams_hdu=cube_beams)

    return column
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imager_tree.py::test_rdt2fits_writes_mfs -v`
Expected: PASS.

> If `save_fits`/`set_wcs` expect a specific axis order that mismatches, adjust the
> `np.moveaxis`/`np.squeeze` only — do **not** modify `utils/fits.py`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/fits_tree.py tests/test_imager_tree.py
git commit -m "feat(imager): add tree-aware rdt2fits export"
```

---

## Phase 6 — Orchestration and CLI wiring

Bring the pieces together in `core/imager.py`: compute imaging geometry, run pass 1 into
`.scratch`, reduce counts, run pass 2 into `.dt`, export FITS. The existing `core/imager.py`
already resolves MSs, scans frequencies/baselines, and builds the per-slice task loop — reuse
that scaffold and extend it.

### Task 6.1: `compute_geometry` helper (extract + extend)

**Files:**
- Create: `src/pfb_imaging/core/imager_plan.py`
- Test: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py  (append)
from pathlib import Path

from pfb_imaging.core.imager_plan import compute_geometry


def test_compute_geometry_on_test_ms(ms_name):
    geo = compute_geometry([Path(ms_name)], channels_per_image=2, field_of_view=1.0)
    assert geo["nband"] >= 1
    assert geo["nx"] > 0 and geo["ny"] > 0
    assert geo["nx_psf"] >= geo["nx"] and geo["ny_psf"] >= geo["ny"]
    assert geo["nx_pad"] >= geo["nx"]
    assert geo["cell_rad"] > 0
    assert geo["freq_out"].size == geo["nband"]
    assert geo["max_blength"] > 0 and geo["max_freq"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py::test_compute_geometry_on_test_ms -v`
Expected: FAIL — cannot import `compute_geometry`.

- [ ] **Step 3: Write minimal implementation**

Lift the frequency/baseline scan (`core/imager.py:186-245`) and image-size computation
(`core/grid.py:166-179`) into one helper, adding the padded uv-grid sizing from
`gridder.image_data_products:540-545`.

```python
# src/pfb_imaging/core/imager_plan.py
"""Imaging-geometry planning for the DataTree imager."""

from pathlib import Path

import numpy as np
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

from pfb_imaging.core.imager import get_engine
from pfb_imaging.utils.misc import set_image_size


def _resolve_ms(ms: list[Path]) -> list[str]:
    names = []
    for ms_path in ms:
        store = DaskMSStore(str(ms_path).rstrip("/"))
        matches = store.fs.glob(str(ms_path).rstrip("/"))
        if not matches:
            raise ValueError(f"No MS at {ms_path}")
        names += list(map(store.fs.unstrip_protocol, matches))
    return names


def compute_geometry(
    ms: list[Path],
    field_names=None,
    spw_names=None,
    scan_names=None,
    freq_min=-np.inf,
    freq_max=np.inf,
    channels_per_image=-1,
    field_of_view=None,
    super_resolution_factor=2.0,
    cell_size=None,
    nx=None,
    ny=None,
    psf_oversize=1.4,
    min_padding=1.7,
) -> dict:
    """Scan the MSs and return the shared imaging geometry.

    Returns a dict with ``nband``, ``freq_out`` ``(nband,)``, ``band_edges``,
    ``nx``/``ny``/``nx_psf``/``ny_psf``/``nx_pad``/``ny_pad``, ``cell_rad``,
    ``max_blength``, ``max_freq`` and the resolved ``msnames``.
    """
    msnames = _resolve_ms(ms)
    all_freqs, all_chan_widths = [], []
    max_blength = 0.0
    for ms_name in msnames:
        kwargs = get_engine(ms_name)
        clean = ms_name.replace("file://", "")
        dt = xr.open_datatree(clean, **kwargs)
        for node in dt.children.values():
            if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
                continue
            ds = node.ds.sel(frequency=slice(freq_min, freq_max))
            if ds.frequency.size == 0:
                continue
            field_name = np.unique(ds.field_name.load().values).item()
            scan_name = np.unique(ds.scan_name.load().values).item()
            spw_name = ds.frequency.attrs["spectral_window_name"]
            if field_names is not None and field_name not in field_names:
                continue
            if spw_names is not None and spw_name not in spw_names:
                continue
            if scan_names is not None and scan_name not in scan_names:
                continue
            all_freqs.append(ds.frequency.values)
            all_chan_widths.append(ds.frequency.attrs["channel_width"]["data"])
            uvw = ds.UVW.load().values
            uvw = uvw[~np.isnan(uvw).all(axis=-1)]
            if uvw.size:
                max_blength = max(max_blength, np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max())

    cw = np.asarray(all_chan_widths)
    cw = cw[np.isfinite(cw)]
    if cw.size == 0:
        raise ValueError("No SPW has a usable channel_width")
    min_chan_width = np.min(cw)
    if channels_per_image in (0, None, -1):
        nband = len(np.unique([f.tobytes() for f in all_freqs]))
    else:
        flat = np.concatenate(all_freqs)
        nband = max(int(np.ceil((flat.max() - flat.min()) / (min_chan_width * channels_per_image))), 1)
    uniq_freqs = np.unique(np.concatenate(all_freqs))
    band_edges = np.linspace(uniq_freqs.min() - min_chan_width / 2, uniq_freqs.max() + min_chan_width / 2, nband + 1)
    half = (band_edges[1] - band_edges[0]) / 2
    freq_out = band_edges[:-1] + half
    max_freq = float(uniq_freqs.max())

    nxv, nyv, nx_psf, ny_psf, _, cell_rad, _ = set_image_size(
        max_blength, max_freq, field_of_view, super_resolution_factor, cell_size, nx, ny, psf_oversize
    )

    def _pad(v):
        p = int(np.ceil(min_padding * v))
        return p + 1 if p % 2 else p

    return {
        "msnames": msnames,
        "nband": nband,
        "freq_out": freq_out,
        "band_edges": band_edges,
        "nx": nxv,
        "ny": nyv,
        "nx_psf": nx_psf,
        "ny_psf": ny_psf,
        "nx_pad": _pad(nxv),
        "ny_pad": _pad(nyv),
        "cell_rad": cell_rad,
        "max_blength": float(max_blength),
        "max_freq": max_freq,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imager_tree.py::test_compute_geometry_on_test_ms -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_plan.py tests/test_imager_tree.py
git commit -m "feat(imager): add imaging-geometry planning helper"
```

### Task 6.2: `reduce_scratch_counts` — accumulate per-(band,time) counts then reduce

**Files:**
- Modify: `src/pfb_imaging/core/imager_pass2.py`
- Test: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imager_tree.py  (append)
from pfb_imaging.core.imager_pass2 import reduce_scratch_counts
from pfb_imaging.utils.treestore import write_group


def _counts_piece(val, nx=8):
    return xr.Dataset(
        {"COUNTS": (("corr", "u", "v"), np.full((1, nx, nx), float(val)))},
        coords={"corr": ["I"]},
        attrs={"bandid": 0, "timeid": 0},
    )


def test_reduce_scratch_counts_sums_partitions_then_strategy(tmp_path):
    store = str(tmp_path / "s.scratch")
    write_group(store, "band0000_time0000/p0", _counts_piece(1.0), mode="w")
    write_group(store, "band0000_time0000/p1", _counts_piece(2.0), mode="a")
    reduced = reduce_scratch_counts(store, nband=1, ntime=1, strategy="per-band-time")
    # two partitions summed: 1+2 = 3
    np.testing.assert_array_equal(reduced[(0, 0)], np.full((1, 8, 8), 3.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imager_tree.py::test_reduce_scratch_counts_sums_partitions_then_strategy -v`
Expected: FAIL — cannot import `reduce_scratch_counts`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pfb_imaging/core/imager_pass2.py  (append)
from pfb_imaging.utils.counts_reduce import reduce_counts
from pfb_imaging.utils.treestore import iter_image_nodes, iter_partitions, open_tree


def reduce_scratch_counts(scratch_store: str, nband: int, ntime: int, strategy: str) -> dict:
    """Sum per-piece COUNTS into per-(band,time) grids, then apply ``strategy``.

    Returns a mapping ``(bandid, timeid) -> (ncorr, nx_pad, ny_pad)``.
    """
    dt = open_tree(scratch_store)
    raw: dict[tuple[int, int], np.ndarray] = {}
    for _, image_node in iter_image_nodes(dt):
        for _, piece in iter_partitions(image_node):
            key = (int(piece.attrs["bandid"]), int(piece.attrs["timeid"]))
            grid = piece.ds.COUNTS.values
            raw[key] = grid.copy() if key not in raw else raw[key] + grid
    return reduce_counts(raw, strategy, nband, ntime)
```

> Note: pass-1 piece group names need not start with `"part"`; widen `iter_partitions` to also
> match `"ms"`-prefixed scratch piece names, OR (preferred) have `stokes_pass1` write pieces
> under `part`-style names. To keep iterators simple, change `stokes_pass1` (Task 2.2) to write
> each piece group as `partition_node_name`-style is **not** possible pre-grouping; instead
> generalise `iter_partitions` to yield all non-empty child groups:
> change its guard to `if image_node[name].ds.data_vars:`. Update `tests/test_treestore.py`
> accordingly (the existing test groups have attrs only, so also give them a dummy var, or
> assert on names starting with `part`). Implement the generalisation now:

```python
# src/pfb_imaging/utils/treestore.py  (replace iter_partitions)
def iter_partitions(image_node: xr.DataTree) -> Iterator[tuple[str, xr.DataTree]]:
    """Yield ``(name, node)`` for each child group of an image node, sorted."""
    for name in sorted(image_node.children):
        yield name, image_node[name]
```

Update `tests/test_treestore.py::test_iter_image_nodes_and_partitions` to give partition
groups a data var (e.g. `xr.Dataset({"X": ("a", [1])}, attrs={"field_name": "0"})`) so they are
distinguishable, and keep asserting the two field names in order.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_imager_tree.py::test_reduce_scratch_counts_sums_partitions_then_strategy tests/test_treestore.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_pass2.py src/pfb_imaging/utils/treestore.py tests/test_imager_tree.py tests/test_treestore.py
git commit -m "feat(imager): accumulate and reduce scratch counts per band-time"
```

### Task 6.3: `build_images` — pass-2 driver writing the `.dt` tree

**Files:**
- Modify: `src/pfb_imaging/core/imager_pass2.py`
- Test: covered by the Phase 7 integration test.

- [ ] **Step 1: Write the implementation**

```python
# src/pfb_imaging/core/imager_pass2.py  (append)
from pfb_imaging.utils.treestore import image_node_name, partition_node_name, write_group


def build_images(scratch_store, dt_store, reduced_counts, geo, robustness, nthreads=1, mode="w"):
    """Pass-2 driver: assemble partitions, weight, grid, sum, write the .dt tree.

    Args:
        scratch_store: pass-1 scratch store URL.
        dt_store: output .dt store URL.
        reduced_counts: mapping ``(bandid, timeid) -> counts grid``.
        geo: geometry dict from ``compute_geometry``.
        robustness: Briggs robustness (``None`` => natural).
        mode: zarr mode for the first write (``"w"`` to create the store).

    Returns:
        ``{timeid: psfparsn (ncorr,3)}`` MFS beam params for FITS, plus writes the tree.
    """
    dt = open_tree(scratch_store)
    nx, ny, nx_psf, ny_psf = geo["nx"], geo["ny"], geo["nx_psf"], geo["ny_psf"]
    cell = geo["cell_rad"]
    first = True
    psfparsn_mfs = {}
    psf_mfs_acc = {}
    wsum_acc = {}

    for image_name, image_node in iter_image_nodes(dt):
        pieces = [p.ds for _, p in iter_partitions(image_node)]
        if not pieces:
            continue
        bandid = int(pieces[0].attrs["bandid"])
        timeid = int(pieces[0].attrs["timeid"])
        counts = reduced_counts[(bandid, timeid)]

        # group pieces by partition identity (field, spw, baseline_group)
        groups: dict[tuple, list] = {}
        for p in pieces:
            key = (p.attrs["field_name"], p.attrs["spw_name"], p.attrs["baseline_group"])
            groups.setdefault(key, []).append(p)

        ncorr = pieces[0].corr.size
        corr = pieces[0].corr.values
        dirty_sum = np.zeros((ncorr, nx, ny))
        psf_sum = np.zeros((ncorr, nx_psf, ny_psf))
        wsum_sum = np.zeros(ncorr)

        out_image = image_name  # band{b}_time{t}
        for pid, (key, plist) in enumerate(sorted(groups.items())):
            part = concat_partition(plist)
            part = apply_imaging_weights(part, counts, robustness, cell, geo["nx_pad"], geo["ny_pad"])
            prod = grid_partition(part, nx, ny, nx_psf, ny_psf, cell, cell, nthreads=nthreads)

            part_ds = xr.Dataset(
                {
                    "VIS": part.VIS,
                    "WEIGHT": part.WEIGHT,
                    "MASK": part.MASK,
                    "UVW": part.UVW,
                    "FREQ": part.FREQ,
                    "PSF": (("corr", "x_psf", "y_psf"), prod["PSF"]),
                    "PSFHAT": (("corr", "x_psf", "yo2"), prod["PSFHAT"]),
                    "BEAM": (("corr", "x", "y"), prod["BEAM"]),
                    "PSFPARSN": (("corr", "bpar"), prod["PSFPARSN"]),
                },
                coords={"corr": corr, "bpar": ["BMAJ", "BMIN", "BPA"]},
                attrs={**{k: part.attrs[k] for k in ("field_name", "spw_name", "baseline_group", "ra", "dec")},
                       "wsum": prod["WSUM"].tolist(), "msid": int(plist[0].attrs["msid"])},
            )
            write_group(dt_store, f"{out_image}/{partition_node_name(pid)}", part_ds, mode="a" if not first else "w")
            first = False

            dirty_sum += prod["DIRTY"]
            psf_sum += prod["PSF"]
            wsum_sum += prod["WSUM"]

        ref = pieces[0]
        band_ds = xr.Dataset(
            {
                "DIRTY": (("corr", "x", "y"), dirty_sum),
                "RESIDUAL": (("corr", "x", "y"), dirty_sum.copy()),  # no model yet
                "WSUM": (("corr",), wsum_sum),
            },
            coords={"corr": corr, "bpar": ["BMAJ", "BMIN", "BPA"]},
            attrs={
                "bandid": bandid, "timeid": timeid,
                "freq_out": float(ref.attrs["freq_out"]), "time_out": float(ref.attrs["time_out"]),
                "ra": float(ref.attrs["ra"]), "dec": float(ref.attrs["dec"]),
                "cell_rad": cell, "robustness": robustness, "niters": 0,
            },
        )
        write_group(dt_store, out_image, band_ds, mode="a")

        from pfb_imaging.utils.misc import fitcleanbeam

        psf_norm = psf_sum / wsum_sum[:, None, None]
        psf_mfs_acc.setdefault(timeid, np.zeros_like(psf_sum))
        psf_mfs_acc[timeid] += psf_sum
        wsum_acc.setdefault(timeid, np.zeros(ncorr))
        wsum_acc[timeid] += wsum_sum

    for timeid in psf_mfs_acc:
        psf_mfs = psf_mfs_acc[timeid] / wsum_acc[timeid][:, None, None]
        from pfb_imaging.utils.misc import fitcleanbeam

        psfparsn_mfs[timeid] = np.array(fitcleanbeam(psf_mfs))
    return psfparsn_mfs
```

- [ ] **Step 2: Lint and commit (driver lands; exercised by Phase 7)**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager_pass2.py
git commit -m "feat(imager): add pass-2 build_images driver writing .dt tree"
```

### Task 6.4: Rewire `core/imager.py` and add CLI options

**Files:**
- Modify: `src/pfb_imaging/core/imager.py`
- Modify: `src/pfb_imaging/cli/imager.py`

- [ ] **Step 1: Replace the pass-1 launch + add stages in `core/imager.py`**

In `core/imager.py`:
1. Add params to the `imager(...)` signature (after `wgt_mode`): `weight_grouping: str = "per-band-time"`, `robustness: float | None = None`, `field_of_view: float | None = None`, `super_resolution_factor: float = 2.0`, `cell_size: float | None = None`, `nx: int | None = None`, `ny: int | None = None`, `psf_oversize: float = 1.4`, `epsilon: float = 1e-7`, `do_wgridding: bool = True`, `double_accum: bool = True`, `keep_scratch: bool = True`, `fits_output_folder: str | None = None`, `fits_mfs: bool = True`, `fits_cubes: bool = True`.
2. Replace the geometry block (`core/imager.py:186-245`) with a call to `compute_geometry(...)` and unpack `geo`.
3. Point the output stores at the new suffixes:

```python
from pfb_imaging.core.imager_pass1 import safe_stokes_pass1
from pfb_imaging.core.imager_pass2 import build_images, reduce_scratch_counts
from pfb_imaging.core.imager_plan import compute_geometry
from pfb_imaging.utils.fits_tree import rrdt2fits

scratch_store = DaskMSStore(f"{basename}.scratch")
dt_store = DaskMSStore(f"{basename}.dt")
for store in (scratch_store, dt_store):
    if store.exists() and overwrite:
        store.rm(recursive=True)
```

4. In the per-slice task loop (`core/imager.py:247-326`), replace `safe_stokes_vis.remote(...)`
   with `safe_stokes_pass1.remote(...)`, passing `scratch_store=scratch_store.url`,
   `nx_pad=geo["nx_pad"]`, `ny_pad=geo["ny_pad"]`, `cell_rad=geo["cell_rad"]`,
   `max_blength=geo["max_blength"]`, `max_freq=geo["max_freq"]`, `baseline_group="all"`, and the
   same `bandid`/`timeid` assignment already computed there. Collect results as
   `(bandid, timeid, piece)` and track `ntime = next_tid`.
5. After pass 1 completes, run the reduction + pass 2 + FITS:

```python
reduced = reduce_scratch_counts(scratch_store.url, geo["nband"], ntime, weight_grouping)
psfparsn = build_images(
    scratch_store.url, dt_store.url, reduced, geo, robustness, nthreads=nthreads, mode="w"
)
if fits_mfs or fits_cubes:
    fits_tasks = []
    for col, norm in (("DIRTY", True), ("PSF", True), ("RESIDUAL", True)):
        fits_tasks.append(
            rrdt2fits.remote(dt_store.url, col, f"{fits_output_folder or basename}", norm_wsum=norm,
                             nthreads=nthreads, do_mfs=fits_mfs, do_cube=fits_cubes, psfpars_mfs=psfparsn)
        )
    for t in fits_tasks:
        ray.get(t)
if not keep_scratch:
    scratch_store.rm(recursive=True)
```

   (Add a `"PSF"` band-node variable to `build_images`' `band_ds` if FITS for PSF is required:
   write `"PSF": (("corr","x_psf","y_psf"), psf_sum)` and `"PSFPARSN"` on the band node.)

- [ ] **Step 2: Add the matching CLI options in `cli/imager.py`**

Add Typer options (Annotated form per project rules) for each new core param and forward them
in both the `imager_core(...)` call and the `run_in_container(...)` dict:

```python
    weight_grouping: Annotated[
        Literal["per-band-time", "mfs", "per-band", "per-time"],
        typer.Option(help="How uv counts are grouped to form imaging weights.", rich_help_panel="Weighting"),
    ] = "per-band-time",
    robustness: Annotated[
        float | None,
        typer.Option(help="Briggs robustness in [-2, 2]; None for natural weighting.", rich_help_panel="Weighting"),
    ] = None,
    field_of_view: Annotated[
        float | None, typer.Option(help="Field of view in degrees.", rich_help_panel="Imaging")
    ] = None,
    super_resolution_factor: Annotated[
        float, typer.Option(help="Super-resolution factor.", rich_help_panel="Imaging")
    ] = 2.0,
    cell_size: Annotated[
        float | None, typer.Option(help="Cell size in arcsec (overrides super-resolution).", rich_help_panel="Imaging")
    ] = None,
    nx: Annotated[int | None, typer.Option(help="Image npix in x.", rich_help_panel="Imaging")] = None,
    ny: Annotated[int | None, typer.Option(help="Image npix in y.", rich_help_panel="Imaging")] = None,
    psf_oversize: Annotated[float, typer.Option(help="PSF oversize factor.", rich_help_panel="Imaging")] = 1.4,
    epsilon: Annotated[float, typer.Option(help="Gridder accuracy.", rich_help_panel="Imaging")] = 1e-7,
    do_wgridding: Annotated[bool, typer.Option(help="Enable w-gridding.", rich_help_panel="Imaging")] = True,
    double_accum: Annotated[bool, typer.Option(help="Double-precision gridding accumulation.", rich_help_panel="Imaging")] = True,
    keep_scratch: Annotated[bool, typer.Option(help="Keep the .scratch cache after imaging.", rich_help_panel="Control")] = True,
    fits_output_folder: Annotated[Directory | None, typer.Option(parser=parse_upath, help="FITS output folder.", rich_help_panel="Output")] = None,
    fits_mfs: Annotated[bool, typer.Option(help="Write MFS FITS.", rich_help_panel="Output")] = True,
    fits_cubes: Annotated[bool, typer.Option(help="Write cube FITS.", rich_help_panel="Output")] = True,
```

Also update the `@stimela_output` `implicit` for `xds-out` to the new product
(`"{current.output-filename}_{current.product}.dt"`) and rename the output to `dt-out`.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/imager.py src/pfb_imaging/cli/imager.py
git commit -m "feat(imager): wire two-pass DataTree pipeline + CLI options"
```

### Task 6.5: Regenerate cab definitions

**Files:**
- Modify: `src/pfb_imaging/cabs/imager.yml` (generated — do not hand-edit)

- [ ] **Step 1: Regenerate**

Run:
```bash
uv run hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs
```

- [ ] **Step 2: Verify CLI help renders (no Typer errors)**

Run: `uv run pfb imager --help`
Expected: help text lists the new options; no `AttributeError`.

- [ ] **Step 3: Commit**

```bash
git add src/pfb_imaging/cabs/imager.yml
git commit -m "chore(imager): regenerate cab definition"
```

---

## Phase 7 — Integration tests

### Task 7.1: End-to-end smoke test of the `.dt` product

**Files:**
- Modify: `tests/test_imager.py`

- [ ] **Step 1: Replace the smoke test to assert the new product**

```python
# tests/test_imager.py
"""Smoke test for the MSv4 DataTree imager core function."""

from pathlib import Path

import numpy as np

from pfb_imaging.core.imager import imager as imager_core
from pfb_imaging.utils.treestore import iter_image_nodes, iter_partitions, open_tree


def test_imager_runs(ms_name):
    """imager() runs through with ipi=15 and cpi=2 on the test MS and writes a .dt tree."""
    test_dir = Path(ms_name).resolve().parent
    outname = str(test_dir / "test_imager")

    imager_core(
        [Path(ms_name)],
        outname,
        integrations_per_image=15,
        channels_per_image=2,
        product="I",
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = open_tree(outname + "_I.dt")
    images = list(iter_image_nodes(dt))
    assert images, "no output-image nodes produced"
    name, node = images[0]
    assert "DIRTY" in node.ds
    assert "WSUM" in node.ds
    parts = list(iter_partitions(node))
    assert parts, "image node has no partition children"
    _, part = parts[0]
    for v in ("VIS", "WEIGHT", "MASK", "UVW", "FREQ", "PSF", "PSFHAT", "BEAM"):
        assert v in part.ds, f"partition missing {v}"
    assert np.isfinite(node.ds.DIRTY.values).all()
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_imager.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_imager.py
git commit -m "test(imager): smoke-test the .dt DataTree product"
```

### Task 7.2: Single-field equivalence vs `init`+`grid`

**Files:**
- Modify: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the equivalence test**

Run `init`+`grid` and `imager` on the same MS with matching geometry, then compare the
wsum-normalised MFS dirty images. They use the same gridder, so they should agree closely;
allow a modest tolerance for weighting/averaging path differences.

```python
# tests/test_imager_tree.py  (append)
import numpy as np

from pfb_imaging.core.grid import grid as grid_core
from pfb_imaging.core.imager import imager as imager_core
from pfb_imaging.core.init import init as init_core
from pfb_imaging.utils.naming import xds_from_url
from pfb_imaging.utils.treestore import iter_image_nodes, open_tree


def _mfs_dirty_from_dt(store):
    dt = open_tree(store)
    nodes = [n.ds for _, n in iter_image_nodes(dt)]
    num = sum(n.DIRTY.values[0] for n in nodes)
    den = sum(n.WSUM.values[0] for n in nodes)
    return num / den


def _mfs_dirty_from_dds(store):
    dds, _ = xds_from_url(store)
    num = sum(ds.DIRTY.values[0] for ds in dds)
    den = sum(ds.WSUM.values[0] for ds in dds)
    return num / den


def test_imager_matches_init_grid_single_field(ms_name, tmp_path):
    base = str(tmp_path / "eq")
    common = dict(channels_per_image=2, integrations_per_image=-1, product="I", overwrite=True)
    fov = 0.5

    # legacy path
    init_core([__import__("pathlib").Path(ms_name)], base + "_legacy", **common, keep_ray_alive=True)
    grid_core(base + "_legacy", field_of_view=fov, robustness=None, psf=True, residual=False,
              nthreads=1, nworkers=1, fits_mfs=False, fits_cubes=False, keep_ray_alive=True)
    dds_dirty = _mfs_dirty_from_dds(base + "_legacy_main.dds")

    # new path
    imager_core([__import__("pathlib").Path(ms_name)], base + "_new", field_of_view=fov,
                robustness=None, fits_mfs=False, fits_cubes=False, keep_ray_alive=True, **common)
    dt_dirty = _mfs_dirty_from_dt(base + "_new_I.dt")

    assert dt_dirty.shape == dds_dirty.shape
    # peak-normalised comparison is robust to small absolute-scale path differences
    a = dt_dirty / np.abs(dt_dirty).max()
    b = dds_dirty / np.abs(dds_dirty).max()
    assert np.abs(a - b).max() < 5e-2
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_imager_tree.py::test_imager_matches_init_grid_single_field -v`
Expected: PASS. If the tolerance is exceeded, investigate weighting/averaging parity (this is a
genuine numerical check, not a placeholder — debug with `superpowers:systematic-debugging`
before loosening the tolerance, and only loosen with a documented reason).

> Confirm the exact `init`/`grid` keyword names against `src/pfb_imaging/cli/init.py` and
> `src/pfb_imaging/cli/grid.py` before running; adjust the calls to match (the core signatures
> are the source of truth).

- [ ] **Step 3: Commit**

```bash
git add tests/test_imager_tree.py
git commit -m "test(imager): equivalence vs init+grid on single field"
```

### Task 7.3: Scratch-cache reuse and full suite

**Files:**
- Modify: `tests/test_imager_tree.py`

- [ ] **Step 1: Write the cache test**

```python
# tests/test_imager_tree.py  (append)
from pathlib import Path

from daskms.fsspec_store import DaskMSStore

from pfb_imaging.core.imager import imager as imager_core


def test_scratch_kept_by_default(ms_name, tmp_path):
    base = str(tmp_path / "cache")
    imager_core([Path(ms_name)], base, channels_per_image=2, product="I", overwrite=True,
                keep_scratch=True, fits_mfs=False, fits_cubes=False, keep_ray_alive=True)
    assert DaskMSStore(base + "_I.scratch").exists()
    assert DaskMSStore(base + "_I.dt").exists()
```

- [ ] **Step 2: Run the full new suite**

Run:
```bash
uv run pytest tests/test_treestore.py tests/test_counts_reduce.py tests/test_hessian_tree.py tests/test_imager_tree.py tests/test_imager.py -v
```
Expected: all PASS.

- [ ] **Step 3: Final lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add tests/test_imager_tree.py
git commit -m "test(imager): scratch cache retained by default"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §3 stores (`.dt`/`.scratch`) → Tasks 0.2, 2.2, 6.4.
- §4 tree layout (band node + partition children, `baseline_group`) → Tasks 0.1–0.3, 2.2, 6.3.
- §5 counts product + named reductions (`per-band-time`/`mfs`/`per-band`/`per-time`, no `natural`) → Tasks 1.1, 1.2, 2.1, 6.2.
- §6 pass 1 (per-scan/ipi fine granularity) → Phase 2; granularity governed by `integrations_per_image` in the Task 6.4 loop.
- §7 pass 2 (group→weight→grid→sum→write) → Phase 3 + Task 6.3.
- §8 `HessianTree` both backends → Phase 4.
- §9 access layer (no `xds_from_*` in imager) → `treestore` used throughout Phases 0/2/3/6.
- §10 FITS, `rdds2fits` untouched, separate `rdt2fits` → Phase 5.
- §11 testing (single-field equivalence, counts reductions, scratch cache) → Tasks 1.2, 7.1–7.3. The two-partition mosaic summation invariant is covered by the `HessianTree` two-partition test (Task 4.1); a full two-field MS fixture is not available in the test data, so mosaic gridding is validated via the operator invariant rather than an end-to-end MS.

**Known follow-ups (out of scope, per spec §2):** rewiring `sara`/`kclean`/`deconv`/`restore`/`degrid` to consume `.dt`; baseline-group partitioning + Mueller beams (`BeamWizard`); model/residual major-cycle gridding in `build_images` (currently `RESIDUAL == DIRTY`).

**Placeholder scan:** no `TBD`/`TODO`/"handle edge cases" steps; every code step has complete code. The one numeric tolerance (Task 7.2) is flagged as a real check, not a placeholder.

**Type consistency:** node-name helpers (`image_node_name`/`partition_node_name`), `reduce_counts`/`reduce_scratch_counts`, `HessianTree(mode=...)`, and the `grid_partition` output keys (`DIRTY`/`PSF`/`PSFHAT`/`BEAM`/`PSFPARSN`/`WSUM`) are used consistently across Tasks 0–7.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-04-imager-datatree.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
