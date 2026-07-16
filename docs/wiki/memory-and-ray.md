---
type: Engineering Notes
title: Memory retention and Ray discipline (MSv4 imager + deconv)
description: The three memory-retention layers on the Ray + MSv4 path, the telemetry that separates them, the scheduling/memory rules the imager and deconv band workers must not regress, and the cleanup runbook for interrupted runs.
tags: [ray, memory, xarray, arcae, imager, deconv, telemetry, runbook]
timestamp: 2026-07-16T13:40:00Z
last_verified_commit: 6bf4a32
---

# Memory retention and Ray discipline (MSv4 imager + deconv)

How `pfb imager`'s footprint went from a 932 GB OOM to 87 GB (and 23m36s to
2m24s) on an 8-worker, 80-task MeerKAT 1024-channel run, and the reusable
diagnostics that got it there. Kept for posterity: each of these mechanisms
will bite again in any Ray + xarray + arcae pipeline. The final section covers
the deconv band workers, which inherit this discipline.

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

**Fix:** `gc.collect()` at every Ray-task boundary — `try/finally` in
`safe_stokes_vis` and `_BandWorkerImpl.load_band`; end-of-task in
`_grid_image` (not exception-safe there; wrap it if you touch that code).

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

Pass-1/2 tasks (and the deconv band workers via `get_mem`) return
`{pid, rss_gb, peak_gb}` measured **after** the task's `gc.collect()`. The
imager prints them in its progress lines; the deconv driver logs them
per-worker once per major cycle at `verbosity > 1`:

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

## Baseline numbers (imager; same data, same parameters throughout)

| state | wall | peak mem | reads |
|-------|------|----------|-------|
| initial | 23:36 | 932 GB (OOM kills) | 651 GB |
| + layers 1 & 2 | 6:33 | 373 GB | 284 GB |
| + driver single-open, pass-2 trims, unpinned inputs, streamed counts | 2:24 | 349 GB | ~0 (warm cache) |
| + layer 3 eviction | 2:24 | **87 GB** | ~0 |

Legacy `init`+`grid` reference on the same data: 5:22, 36 GB peak.

## The deconv band workers

`pfb deconv` runs one long-lived `_BandWorkerImpl` actor per band
(`operators/band_worker.py`) co-locating the band's Hessian, wavelet jitclass
and exact-residual inputs. Its memory/scheduling rules:

- **Worker-side loading.** Each worker reads its own band's vis-scale inputs
  (`UVW`/`WEIGHT`/`MASK`/`FREQ`/`BEAM`/`PSFHAT`/`DIRTY`) straight from the
  `.dt` store (`load_bands`), layer-2 style (selective load → extract →
  release → gc). Vis-scale data never enters the driver or the Ray object
  store; the driver holds only `(nband, nx, ny)` cubes and attrs. (Exception:
  for `nband == 1` the pool runs in-process, so the one band's data lives in
  the driver by construction.)
- **Deliberate pinning.** The loaded inputs are held for the life of the run
  (no per-cycle re-fetch). This is an RSS-for-work trade; it shows up as a
  *flat* per-worker `rss_gb` in the telemetry — a flat plateau is by design,
  a ratchet is a bug.
- **Nominal CPU claims.** Workers claim `num_cpus=1e-2` because they are
  thread-pool-bound (their parallelism is `nthreads // nband` internal
  threads); real claims that scale with nband can exceed the cluster's CPUs
  and silently deadlock the init `ray.get`. The driver sizes its local
  cluster `num_cpus = max(nworkers, nband+1)` so the raylet's worker-startup
  throttle (`max(1, num_cpus)`) matches true process demand — undersizing is
  harmless but serialises worker startup and triggers the raylet's
  "startup concurrency" warning.
- **Fixed overhead floor.** A deconv session costs ~9 resident processes for
  4 bands (band actors + Ray prestarts) plus the plasma reservation; on small
  test images this dwarfs the data and dominates stimela's memory stats.
  Judge data-scale behaviour by the per-worker `rss_gb` telemetry, not the
  session total.

## Runbook: cleaning up after an interrupted run (Ctrl+C, swap thrash)

Verified on a real interrupt 2026-07-16. A single Ctrl+C on the driver
normally takes the whole local Ray cluster down with it (the raylet reaps its
workers when the driver's GCS goes away) — leftovers are the exception, not
the rule, and usually mean the driver died uncleanly (SIGKILL, OOM-killer).
Check before killing anything.

1. **Find survivors.** `pgrep -af 'raylet|ray::|gcs_server|plasma'` plus a
   broader `ps aux | grep -iE 'ray|pfb'`. Ray daemon names don't contain
   "pfb": the ones to know are `raylet`, `gcs_server`, `ray::IDLE` /
   `ray::<task-name>` workers, and the plasma store.
2. **Kill them in order of politeness.** `ray stop` (from the project venv:
   `uv run ray stop`) terminates every Ray process on the machine; add
   `--force` (SIGKILL) if any survive. Last resort:
   `pkill -9 -f raylet && pkill -9 -f 'ray::' && pkill -9 -f gcs_server`.
3. **Reclaim shared memory.** The plasma object store lives in `/dev/shm` and
   counts against RAM while the file exists. After all Ray processes are
   dead, `ls /dev/shm` should show no `plasma*`/`ray*` files; delete any that
   remain.
4. **Clear stale session dirs.** Ray leaves `/tmp/ray/session_*` (logs +
   sockets, a few MB each) behind forever — hundreds of them accumulate to
   ~1 GB. With nothing Ray-related running, `rm -rf /tmp/ray` is safe.
5. **Understand the lingering swap — it is not a leak.** During the thrash
   the kernel swapped out *idle pages of unrelated processes* (browsers, IDEs,
   JVMs), and Linux never proactively swaps pages back in: `free -h` keeps
   showing used swap long after the culprit is dead. Attribute it with
   `awk '/VmSwap/ {print $2, FILENAME}' /proc/[0-9]*/status | sort -rn | head`
   (or `smem -s swap`) — if the holders are desktop apps, the run's memory is
   already gone.
6. **Optionally force swap back to RAM.** Only worth it if the swapped-out
   apps feel sluggish. Check `free -h` first: `available` must comfortably
   exceed swap `used`, then `sudo swapoff -a && sudo swapon -a` (blocks for
   a minute or two while every swapped page is read back). If available RAM
   is tighter than that, leave it — pages fault back in on use.

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
