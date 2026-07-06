# Memory retention patterns on the Ray + MSv4 imaging path

How `pfb imager`'s footprint went from a 932 GB OOM to 87 GB (and 23m36s to
2m24s) on an 8-worker, 80-task MeerKAT 1024-channel run, and the reusable
diagnostics that got it there. Kept for posterity: each of these mechanisms
will bite again in any Ray + xarray + arcae pipeline.

## The core fact

Ray workers are **long-lived processes that run many tasks sequentially**.
Anything a task leaves behind compounds across the run. Three distinct
retention layers were found, each invisible to the tools of the layer above.

### Layer 1 — Python reference cycles

xarray Datasets/DataTrees deserialised from the Ray object store sit in
reference cycles, so refcounting never frees them, and the generational GC
rarely triggers in numeric code (few container allocations). Each completed
task left its fully loaded node behind → +1 node per task → >100 GB/worker
and raylet OOM kills late in the run (at task 72/80 — lateness is the tell
for accumulation vs per-task footprint).

**Fix:** `gc.collect()` in a `try/finally` at every Ray-task boundary
(`safe_stokes_vis`, `_grid_image`).

### Layer 2 — Over-reading and pinned eager inputs

`node.load()` on an MSv4 node reads **every** data variable — `VISIBILITY`,
`CORRECTED_DATA`, `MODEL_DATA`, `TIME_CENTROID`, … — when only the selected
data/weight/flag columns are used (651 GB read vs the 176 GB actually
needed). And the eager Dataset local pins the raw c64/f32 inputs alongside
all derived copies until the function returns; the legacy daskms path never
had this problem because its datasets are dask-backed and hold nothing.

**Fix:** load only the needed variables (`ds = node.ds[needed].load()`),
extract everything into plain numpy, then `del` + collect *before* the heavy
conversions (`stokes_vis` is the template).

### Layer 3 — Below-Python caches (the xarray-ms Multiton)

xarray-ms keeps every opened arcae `Table` in a **class-level Multiton cache
with a 300 s inactivity TTL**, swept only on cache access. Each pass-1 task
opens its own partition under a distinct cache key; a busy worker never
idles long enough for anything to expire, and pass 2 never touches the
cache, so the retained tables (holding the casacore read buffers) sit to the
end of the run. Telemetry showed post-gc RSS ratcheting **+3.55 GB/task,
perfectly linearly, identically on all 8 workers** to ~39.5 GB each — the
349 GB machine peak. Strong references: `gc.collect()` cannot touch them.

**Fix:** `_release_ms_caches()` in `utils/stokes2vis_msv4.py` clears
`Multiton._INSTANCE_CACHE` between tasks (safe: Ray runs one task at a time
per worker; measured cost ~7 MB/task of subtable re-open churn vs ~3.5
GB/task retained). This is private API by necessity — worth an upstream
xarray-ms issue asking for a TTL/cache-off knob or public eviction hook, and
this helper should be deleted when one exists.

## The diagnostic that separates the layers

Pass-1/2 tasks return `{pid, rss_gb, peak_gb}` measured **after** the task's
`gc.collect()`; the driver prints them in the progress lines:

    Completed: 5 / 80 [pid 615627 rss 3.12 GB peak 11.40 GB]

Reading it:

- **post-gc `rss` flat per pid, `peak` high** → footprint is per-task
  transients (look at conversion copies, concat doubling, counts grids).
- **post-gc `rss` ratcheting linearly per pid** → retention *below Python*
  (C-level caches, allocator arenas); more gc changes nothing — find the
  cache holding strong references.
- **OOMs late in a run** → accumulation across tasks, not per-task size.

Local repro needs no cluster: pickle-roundtrip a datatree node (that is
exactly what Ray does to task args), `.load()` it, drop it, and watch
`psutil.Process().memory_info().rss` across iterations; then a manual
`gc.collect()`. Growth that gc releases is layer 1; growth that survives gc
is layer 3. `tests/data/test_ascii_1h60.0s.MS` works for this — but note
that a single-node MS reuses the same cache keys, so layer 3 only *grows*
when tasks span different partitions (as on real data).

## Baseline numbers (same data, same parameters throughout)

| state | wall | peak mem | reads |
|-------|------|----------|-------|
| initial | 23:36 | 932 GB (OOM kills) | 651 GB |
| + layers 1 & 2 | 6:33 | 373 GB | 284 GB |
| + driver single-open, pass-2 trims, unpinned inputs, streamed counts | 2:24 | 349 GB | ~0 (warm cache) |
| + layer 3 eviction | 2:24 | **87 GB** | ~0 |

Legacy `init`+`grid` reference on the same data: 5:22, 36 GB peak.

## Related conventions uncovered along the way

- The MSv4 `time` coordinate (and the `.dt`'s `time_out` attrs) are **unix
  seconds**; the legacy `.dds` carries MSv2 **MJD seconds**. Applying the
  MJD→unix shift twice put FITS `DATE-OBS` in 1909 — ERFA "dubious year"
  warnings are the symptom, `utils/fits.set_wcs(time_is_unix=...)` is the
  switch.
- Driver-side counts are accumulated **at the applied `weight_grouping`
  granularity** (`counts_key` in `core/imager`), never per `(band,time)`
  node, bounding driver memory at `ngroups` grids; natural weighting
  (`robustness=None`) skips counts entirely.
