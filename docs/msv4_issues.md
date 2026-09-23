# MSv4 stack issues (arcae / xarray-ms)

Upstream issues we have hit on the MSv4 write path, what has already landed, and what
still needs filing. Keep this current: an issue we rediscover because it was never written
down costs a debugging session every time.

Reproducers live in [`scripts/msv4_issues/`](../scripts/msv4_issues) and take a path to any
MS, which they copy first, so they never touch the original:

```bash
uv run python scripts/msv4_issues/<script>.py tests/data/test_ascii_1h60.0s.MS [args]
```

## Versions

| | pinned in `pyproject.toml` | installed | latest alpha | latest stable |
|---|---|---|---|---|
| xarray-ms | `>=0.4.0a11,<0.5.0` | 0.4.0a11 | 0.4.0a11 | 0.5.11 |
| arcae | `>=0.4.0a11,<0.5.0` | 0.4.0a11 | 0.4.0a11 | 0.5.5 |

The `<0.5.0` ceiling is deliberate and must stay: write support ships only on the
`0.4.0-alpha` line, which is cut from *later* commits than the 0.5.x line. Version numbers
do not order by capability here (wiki D14, ratt-ru/xarray-ms#170).

The floors are load bearing, not just currency: 0.4.0a8 is what lets `sync_msv2` create
canonical MAIN columns, which is what allowed the `_create_missing_columns` fallback to go.

## Status

| # | Issue | Repo | Filed | Status |
|---|---|---|---|---|
| 1 | `sync_msv2` skips a canonical MAIN column that is absent | xarray-ms | [#171](https://github.com/ratt-ru/xarray-ms/issues/171) | **Fixed** in 0.4.0a8; verified on a11 |
| 2 | `addcols` then read is answered by a stale table instance | arcae | [ska-sa/arcae#241](https://github.com/ska-sa/arcae/issues/241) | Present a7 → a11 |
| 3 | `addcols` poisons table handles already open in the same process | arcae | [ska-sa/arcae#241](https://github.com/ska-sa/arcae/issues/241) | Same root cause as 2; blocks our a11 bump |
| 4 | Every `MSv2Structure` build retains ~1.5 MB | xarray-ms | [ratt-ru/xarray-ms#177](https://github.com/ratt-ru/xarray-ms/issues/177) | Filed with issue 5 |
| 5 | No public way to evict xarray-ms's own table cache | xarray-ms | [ratt-ru/xarray-ms#177](https://github.com/ratt-ru/xarray-ms/issues/177) | Filed with issue 4 |

---

### 1. `sync_msv2` silently skips a canonical MAIN column — FIXED

`MODEL_DATA`, `CORRECTED_DATA` and `DATA` are in casacore's canonical MAIN descriptor.
`generate_column_descriptor` validated such a name and then emitted nothing, so `addcols`
was never asked for it and the following `to_msv2` had nowhere to write. No error, no warning.

- Filed as [ratt-ru/xarray-ms#171](https://github.com/ratt-ru/xarray-ms/issues/171), fixed by
  [`d1a37e6`](https://github.com/ratt-ru/xarray-ms/commit/d1a37e68fdd27a8e09df43df7c9b8fde135e3b5a)
  in 0.4.0a8, which synthesises a creatable fixed-shape descriptor rather than reusing the
  canonical variable-shape one.
- Reproducer: [`sync_msv2_canonical_column.py`](../scripts/msv4_issues/sync_msv2_canonical_column.py).
  On a7 `MODEL_DATA created=False`; on a11 `created=True`.
- **Done:** `_create_missing_columns` is deleted and the pins now floor at a11, so
  `ensure_model_columns` relies on `sync_msv2` alone (and on its `ColumnCreationError` to
  verify). Dropping below 0.4.0a8 would make `degrid-msv4` silently write nothing.

### 2. `addcols` then read is answered by a stale table instance — FILED

Filed together with issue 3 as [ska-sa/arcae#241](https://github.com/ska-sa/arcae/issues/241), "Adding a column leaves other open table handles
unable to resync": they are the same behaviour, once between sibling instances of one
`Table` and once between separate handles.

xarray-ms opens MAIN with 8 casacore instances (`DEFAULT_MAIN_NINSTANCES`). arcae runs
`AddColumns` on instance 0 (`IsolatedTableProxy::SpawnWriter`, whose comment calls adding
columns "non-syncable") but routes reads to the least busy instance. A read issued after the
column is added therefore lands on an instance that has not seen it and either throws

```
Table::lock cannot sync table <ms>; another process changed the number of columns
```

or returns a stale column list. `sync_msv2` does exactly this: it calls `columns()` straight
after `addcols` to verify creation, so it can fail against its own write.

- Reproducer: [`addcols_multi_instance_race.py`](../scripts/msv4_issues/addcols_multi_instance_race.py).
  Measured on a11: **60–79 of 200** `columns()` calls fail with `ninstances=8`, **0 of 200**
  with `ninstances=1`. Same on a7/a8, so a11 does not fix it.
- Timing dependent, so it hides: 0/25 locally in isolation, but it failed CI on
  [pfb-imaging#329](https://github.com/ratt-ru/pfb-imaging/pull/329) (Python 3.12 only).
- **Our workaround:** `get_engine(..., main_ninstances=1)` for any tree that creates columns;
  the driver's guard tree uses it (`core/degrid_msv4.py`). Wiki: design-decisions gotchas.
- This matters more from 0.4.0a8 on, because issue 1's fix routes canonical columns
  (i.e. the default `MODEL_DATA`) through the same `addcols` path.

### 3. `addcols` poisons table handles already open in the same process — FILED ([ska-sa/arcae#241](https://github.com/ska-sa/arcae/issues/241))

The same underlying behaviour as issue 2, but across handles rather than instances: a
DataTree left open across a column creation cannot be read afterwards, even though both
handles are in one process and one of them made the change.

- Reproducer: [`addcols_poisons_open_handles.py`](../scripts/msv4_issues/addcols_poisons_open_handles.py),
  pure arcae. Handle A reads, handle B adds a column, every subsequent read through A fails;
  a handle opened afterwards is fine. Reproduces at `ninstances=1`, which is what separates it
  from issue 2. `getcol` never recovers on the poisoned handle; `columns()` fails once then
  succeeds, so the handle ends up half-recovered rather than resynced.
- **Blocks the a11 bump.** `tests/test_degrid_parity.py::test_imager_degrid_msv4_nulls_the_residual`
  is `3/3 pass` on a7/a8 and `3/3 fail` on a11, at `CorrelatedFactory.__init__`'s
  `ms.tabledesc()` in the second `imager` call of an in-process
  `imager → degrid-msv4 → imager` chain. It appears at a8 because that is when `sync_msv2`
  began creating canonical columns itself (issue 1) rather than leaving `MODEL_DATA` to
  pfb-imaging's own short-lived handle.
- **Upstream's recommended workaround** (sjperkins on arcae#241) is `table.close()` followed
  by reopening, rather than evicting the cache. That is already what we do for the handle we
  own — the column-creation tree is closed explicitly. It does not reach the handle that
  actually fails here, which belongs to a `DataTree` opened earlier by another stage and held
  in xarray-ms's process-wide cache: a consumer cannot close what it does not hold, which is
  the ask in ratt-ru/xarray-ms#177. Simon also floats making `AddColumns` reopen the other
  instances as the long-term fix.
- **Known gap on our side:** `core/imager.py` never closes the MS `DataTree` it opens (no
  `.close()` anywhere in that module), so it is our own unclosed handle that gets poisoned in
  an in-process chain. Closing it would remove the need for the eviction below — but it cannot
  simply be closed after selection, because the dispatch loop still reads `node.ds` and slices
  it for the workers. It would have to be closed after dispatch.
- **Fixed on our side:** `core/degrid_msv4.degrid_msv4` calls `_release_ms_caches()` once
  after the column-creation loop — *not* per work item, which is what made it reload the model
  every chunk (see wiki memory-and-ray). The test passes on a11 with it.
- Production impact is narrower than the test suggests: degrid's replicas are separate
  processes that open after the driver closes. It bites any in-process pipeline that reads
  the MS both before and after degridding.

### 4. Every `MSv2Structure` build retains ~1.5 MB — FILED ([ratt-ru/xarray-ms#177](https://github.com/ratt-ru/xarray-ms/issues/177))

Repeatedly opening, reading and closing a DataTree grows post-`gc.collect()` RSS linearly
with **no plateau**. Re-measured on an idle machine, 2000 iterations, xarray-ms 0.4.0a11 +
arcae 0.4.0a11: **+1.53 MB per iteration**, 338 MB → 3.44 GB, with the last-decile slope
(+1.535) indistinguishable from the overall slope (+1.537). It is not a slow-filling bounded
cache.

Note this is **xarray-ms, not arcae** — an earlier version of this page had it under arcae.

- Reproducers: [`table_open_close_rss.py`](../scripts/msv4_issues/table_open_close_rss.py)
  for the end-to-end figure, and
  [`structure_rebuild_rss.py`](../scripts/msv4_issues/structure_rebuild_rss.py), which
  localises it.

**Localisation.** Reading contributes nothing — `open` (no read at all) leaks +1.533 MB/iter
against `vis` (full read) at +1.537. Building `MSv2Structure` on its own reproduces the entire
rate (+1.537). Skipping the `close()` is flat (+0.0002), because the Multiton cache then
serves every open from one entry; it is the rebuild that costs, and `close()` releases the
structure factory.

**Ruled out, each by measurement:**

| candidate | evidence |
|---|---|
| allocator fragmentation | `malloc_trim(0)` reclaims nothing; `/proc/self/maps` flat at 590 |
| pyarrow buffers | pool `bytes_allocated()` stays 0; mimalloc and jemalloc give the same slope |
| fds / threads | both flat on a11 (6, and 73→78) |
| Python objects | `len(gc.get_objects())` grows ~600 over 2000 builds |
| arcae | open/close of MAIN at 1 and 8 instances, MAIN reads via `getcol` and `to_arrow`, and all 13 subtables via `to_arrow` are each flat to within 0.003 MB/iter |
| thread-pool stacks | rate unchanged at `max_workers` 1, 4, 11, 22 |

**What arcae#235 fixed, and what it did not.** On 0.4.0a8 the same loop runs at +4.53 MB/iter
and takes file descriptors 27 → 10,886 and threads 80 → 7,011 over 800 iterations — the leak
that [ska-sa/arcae#235](https://github.com/ska-sa/arcae/pull/235) fixed. On a11 both are flat
and the rate is a third of that. The residual is a different problem, and the earlier "~3×
better, maybe a bounded cache" reading on this page was taken under load and over too few
iterations to see that it never flattens.

**Why it matters to us.** `_release_ms_caches()` drops the structure factory, so every Ray
task that calls it pays a rebuild — and therefore ~1.5 MB — on its next open. Over a long
imager run that is the same shape as the pass-1 pathology this discipline was built to avoid
(wiki memory-and-ray).

### 5. No public way to evict xarray-ms's own table cache — FILED ([ratt-ru/xarray-ms#177](https://github.com/ratt-ru/xarray-ms/issues/177))

xarray-ms creates `Multiton`s internally for its arcae tables and exposes no way to release
them. `Multiton.release()` is a clean per-key eviction, but only for keys you hold, so the
only lever a consumer has is `Multiton._INSTANCE_CACHE.clear()` — a private, class-level
wipe that destroys *every* consumer's Multitons as collateral.

- This is not hypothetical: it is what made `degrid-msv4` reload a 633 MB `.mds` on every
  work item, because the deployment keys its model and masks on Multitons too
  ([pfb-imaging#331](https://github.com/ratt-ru/pfb-imaging/pull/331)).
- Issue 3 makes the ask sharper: consumers now *need* a targeted eviction after creating a
  column, and the wholesale wipe is the only thing available.
- [`1b4ca7d8`](https://github.com/ratt-ru/xarray-ms/commit/1b4ca7d8) (a9, "Only release
  derived factories from the store that owns them") is adjacent — it stops a write store
  evicting factories it borrowed — but adds no consumer-facing hook.
- Our helper is `utils/stokes2vis_msv4._release_ms_caches`, flagged for deletion once a hook
  exists. Filed together with issue 4 as [ratt-ru/xarray-ms#177](https://github.com/ratt-ru/xarray-ms/issues/177): the wholesale wipe is the only eviction
  available, and it also forces the structure rebuild that issue 4 makes expensive.

---

## Considered, not filed

- **`sync_msv2` drops a variable that is not on every correlated node.** It warns rather than
  raising. Defensible — a CASA column belongs to the whole MAIN table — and declaring
  tree-wide is the correct usage. Documented in the wiki gotchas instead.
- **`to_msv2(region="auto")` silently writes to the start of the array.** An `isel`'d chunk
  written with the default region lands at offset 0 rather than where it came from, with no
  error. We always pass `region` explicitly. Worth raising as an API-safety question — a
  default that silently corrupts is a poor default — but it is a design opinion rather than a
  bug, so it needs a decision before filing.

## Adjacent, not MSv4

- **Ray Serve's `autoscaling_config` silently swallows unknown keys.** `max_ongoing_requests`
  is a plausible thing to put there, is not a field of `AutoscalingConfig`, and pydantic drops
  it without complaint, so a deployment silently runs at the default 5 concurrent requests per
  replica. Tracked on [pfb-imaging#331](https://github.com/ratt-ru/pfb-imaging/pull/331); not
  filed against Ray.

## What landed upstream recently

Since our pins (xarray-ms a7, arcae a8), for context when bumping:

**xarray-ms** — `d1a37e6` + `318d675` synthesise creatable descriptors for canonical columns
and add columns one at a time (issue 1, a8); `#172` closes `arcae.Table` file-descriptor
leaks in `subtable_factory`; `1b4ca7d8` narrows factory release to the owning store (a9);
`#174`/`8a97872` upgrade to arcae 0.5.5 / 0.4.0-alpha.11 (a11); an `AGENTS.md` was added.

**arcae** — `#235` releases table resources when the last reference is dropped on an
isolation thread, fixing a destructor deadlock that leaked fds and threads (a9, see issue 4);
`#234` adds `PHASED_ARRAY` subtable support; `#233` adds `GetCellSlice` for slicing single
cells in array columns; `#239` rejects slices `Selection` cannot represent; `abdeebd` fixes a
Cython 3.2 build failure; `#238`/`#230` add design documentation.

## Open upstream work to watch

- [ratt-ru/xarray-ms#170](https://github.com/ratt-ru/xarray-ms/pull/170) — the write-support
  branch itself. Merges to `main` periodically; the plan is to land it once write support is
  sufficiently stress-tested.
- [ratt-ru/xarray-ms#18](https://github.com/ratt-ru/xarray-ms/issues/18) — the tracking issue
  for MAIN-table column writes.
- [ratt-ru/xarray-ms#173](https://github.com/ratt-ru/xarray-ms/issues/173) — a third-party
  request for full MSv2 *dataset* writing (not just columns).
- [ratt-ru/xarray-ms#176](https://github.com/ratt-ru/xarray-ms/pull/176) — `phased_array_xds`
  support, paired with arcae#234.
- [ska-sa/arcae#165](https://github.com/ska-sa/arcae/pull/165) — the 0.4.0 write-support
  development PR.
- [ska-sa/arcae#240](https://github.com/ska-sa/arcae/issues/240) — `Selection` cannot
  distinguish an empty selection from an absent one.

Neither issue 2 nor issue 3 appears anywhere upstream: searches for `addcols`, `ninstances`
and `cannot sync` across both repos returned nothing relevant, so both were ours to file.
They went up as a single arcae issue, [ska-sa/arcae#241](https://github.com/ska-sa/arcae/issues/241).
