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
| xarray-ms | `>=0.4.0a7,<0.5.0` | 0.4.0a7 | **0.4.0a11** | 0.5.11 |
| arcae | `>=0.4.0a8,<0.5.0` | 0.4.0a8 | **0.4.0a11** | 0.5.5 |

The `<0.5.0` ceiling is deliberate and must stay: write support ships only on the
`0.4.0-alpha` line, which is cut from *later* commits than the 0.5.x line. Version numbers
do not order by capability here (wiki D14, ratt-ru/xarray-ms#170).

**We are not yet on a11.** See issue 3 — the bump is currently blocked by a deterministic
test failure, with a known one-line fix that has not been applied.

## Status

| # | Issue | Repo | Filed | Status |
|---|---|---|---|---|
| 1 | `sync_msv2` skips a canonical MAIN column that is absent | xarray-ms | [#171](https://github.com/ratt-ru/xarray-ms/issues/171) | **Fixed** in 0.4.0a8; verified on a11 |
| 2 | `addcols` then read is answered by a stale table instance | arcae | drafted | Present a7 → a11 |
| 3 | `addcols` poisons table handles already open in the same process | arcae | drafted | Same root cause as 2; blocks our a11 bump |
| 4 | RSS ratchets across repeated open/read/close | arcae | **no** | Much improved by arcae#235; re-measure before filing |
| 5 | No public way to evict xarray-ms's own table cache | xarray-ms | **no** | Partly touched by a9; still no hook |

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
- **When we bump:** delete `_create_missing_columns` in `src/pfb_imaging/utils/degrid_msv4.py`
  and the `ensure_model_columns` docstring paragraph about it. That workaround exists only for
  this gap.

### 2. `addcols` then read is answered by a stale table instance — DRAFTED

Filed together with issue 3 as one arcae issue ("Adding a column leaves other open table
handles unable to resync"): they are the same behaviour, once between sibling instances of
one `Table` and once between separate handles.

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

### 3. `addcols` poisons table handles already open in the same process — DRAFTED

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
- **Fix on our side (verified, not yet applied):** evict the process-wide table cache after
  column creation — `_release_ms_caches()` once after `ensure_model_columns` in
  `core/degrid_msv4.degrid_msv4`, *not* per work item (see wiki memory-and-ray). With that
  eviction inserted the test passes on a11.
- Production impact is narrower than the test suggests: degrid's replicas are separate
  processes that open after the driver closes. It bites any in-process pipeline that reads
  the MS both before and after degridding.

### 4. RSS ratchets across repeated open/read/close — MEASURE, THEN DECIDE

Repeatedly opening, reading and closing a DataTree grows post-`gc.collect()` RSS with no
plateau over 120 iterations, and most of it is open/close rather than reading.

- Reproducer: [`table_open_close_rss.py`](../scripts/msv4_issues/table_open_close_rss.py).

| | `open` (no read) | `vis` (full read) |
|---|---|---|
| arcae 0.4.0a8 | +4.48 MB/iter | +7.45 MB/iter |
| arcae 0.4.0a11 | +1.66 MB/iter | +1.78 MB/iter |

- [ska-sa/arcae#235](https://github.com/ska-sa/arcae/pull/235) ("Release table resources when
  the last reference is dropped on an isolation thread", in a9) cut it by roughly 3×, but it
  is still not flat. Before filing, re-measure on a11 on an idle machine over more iterations
  and confirm it is genuinely unbounded rather than a slow-filling bounded cache — the
  numbers above were taken while the test suite was running.

### 5. No public way to evict xarray-ms's own table cache — NEEDS FILING

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
  exists.

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
and `cannot sync` across both repos return nothing relevant, so both are ours to file. They
are drafted as a single arcae issue, pending review before filing.
