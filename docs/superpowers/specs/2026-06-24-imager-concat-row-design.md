# Imager `concat_row` — Design

**Date:** 2026-06-24
**Branch:** `imager`
**Status:** Approved, not yet implemented.

## 1. Motivation

`pfb grid` exposes a `concat_row` option that collapses all time chunks of a band into a
single output image (`time0000`), giving one wideband-in-time image per band. The MSv4
`pfb imager` front-end is missing this option: its pass-2 image granularity is `(band, time)`,
and because `timeid` is assigned per `(scan, time-block)` (`core/imager.py:473-477`), a
multi-scan MS produces **one output image — and one DIRTY+PSF gridding — per scan**, even at
`integrations_per_image=-1`. On a real single-field/single-spw MS with many scans this is the
dominant cost: N scans ⇒ N Ray tasks ⇒ N independent griddings (each `vis2dirty` for DIRTY and
PSF, per correlation).

The fix is to concatenate the rows of a band's pieces **before** gridding so each band is
gridded once, mirroring the legacy `image_data_products` (`operators/gridder.py:375`), which
weighted-averages the beam across a dataset list, `xr.concat(dsl, dim="row")`, then grids once.
The parameter has already been plumbed through the CLI/cabs and `core.imager.imager`
(`concat_row: bool = True`); only the pass-2 behaviour is unimplemented.

## 2. Scope

In scope:

- Implement `concat_row` in the imager pass-2 path (`core/imager.py`), default `True`.
- `concat_row=True`: one output image **per band** (`band{b:04d}_time0000`), collapsing the
  time/scan axis; rows concatenated before a single gridding per partition.
- `concat_row=False`: unchanged — one image per `(band, time)`.
- Auto-aggregate the uv `COUNTS` to per-band when `concat_row=True`, with a coercion warning
  for time-resolved `weight_grouping`.
- A regression test that `concat_row=True` produces one `time0000` node per band and agrees
  with `concat_row=False` on the MFS dirty image.

Out of scope:

- Any change to `pfb grid`, `pfb init`, or the legacy `.dds` consumers.
- Changing the partition identity `(msid, field, spw, baseline_group)` or the `.dt` tree
  schema (only the *number* of time nodes and the *row extent* of partition children change).
- Concatenating across genuinely-different partitions (different phase centre / freq / beam):
  these remain gridded-separately-and-summed — they cannot share one uv-grid.

## 3. Behaviour

`concat_row` controls only the **output-image time granularity** and the row extent fed to a
single gridding. The within-image structure is unchanged: pieces are grouped by partition key
`(msid, field, spw, baseline_group)`; pieces sharing a key are concatenated along `row`;
distinct keys are gridded separately and image-space-summed.

- **`concat_row=True` (default).** One image per band. Its pass-2 worker gathers every fine
  piece of the band across **all** `timeid`s (i.e. all scans/time-blocks), groups by partition
  key, and concatenates each key's pieces along `row` — across scans **and** times. Each key is
  gridded once. Output node: `band{b:04d}_time0000`.

- **`concat_row=False`.** One image per `(band, time)`, exactly as today. The worker gathers
  only the one scratch node's pieces.

Rows are only concatenatable when they share phase centre **and** freq axis. This holds by
construction within a partition key: same `field` ⇒ same phase centre; same `spw` + same band
channel-chunk ⇒ identical `FREQ`. A defensive `assert` (see §6) guards it. Different fields/spws
fall in different partition keys and are summed, never concatenated.

## 4. Data flow

Pass 1 and the counts-reduction structure are unchanged. The changes are confined to the
region between counts-reduction and the end of pass 2 in `core/imager.imager`, plus the
`_grid_image` worker signature.

### 4.1 Pass-2 dispatch (`core/imager.py`, the `image_meta` / dispatch region ~539-619)

- Build the per-task work list from the scratch tree's `band{b}_time{t}` nodes:
  - `concat_row=True`: group node names by **band**. One task per band; the task receives the
    **list** of that band's scratch node names (all `timeid`s).
  - `concat_row=False`: one task per node (current); the list has a single element.
- Per-band `meta`: `bandid` from the band; `timeid = 0`; `time_out = mean` of the collapsed
  nodes' `time_out` (mirrors grid's `times_out = np.mean(times_in)`); `ra`/`dec` from the
  pieces (unchanged source).

### 4.2 `_grid_image` worker (`core/imager.py:41`)

- Replace the single `image_name: str` parameter with two: `src_names: list[str]` (the band's
  scratch nodes to read) and `out_name: str` (the `.dt` node to write). This decouples input
  from output. Open the scratch datatree once and gather piece-children from **all** `src_names`
  into the existing `pieces` list. Everything downstream is unchanged: the group-by-partition-
  key dict (`core/imager.py:76-79`) and the `xr.concat([...], dim="row")` block
  (`core/imager.py:93-103`) now concatenate across times automatically.
- `out_name` is `band{b:04d}_time0000` when `concat_row=True`, else `band{b:04d}_time{t:04d}`
  (= the single `src_names` element). Partition children `part{pid:04d}` are written under
  `out_name` as today.
- The per-time MFS PSF bookkeeping (`psf_mfs`/`wsum_mfs` keyed by `timeid`,
  `core/imager.py:621-642`) is unchanged: with `concat_row=True` every band reports `timeid=0`,
  so it naturally becomes a single full-band MFS fit.

### 4.3 Weighting / counts aggregation

- `concat_row=True`: reduce counts with **`per-band`** so the band worker's Briggs weights come
  from the band's total uv-density (`reduce_counts` already returns the per-band sum under every
  `(b,t)` key — `utils/weighting.py:420-426`).
- Coerce a time-resolved `weight_grouping` with a warning, mapping each to its band-collapsed
  analogue:
  - `per-band-time → per-band`
  - `per-time → mfs`
  - `per-band` / `mfs` → used as-is.
- `concat_row=False`: `weight_grouping` is honoured unchanged (current behaviour).

## 5. Tree layout & downstream operators

- `.dt` schema is unchanged. With `concat_row=True` there is one `band{b:04d}_time0000` node per
  band instead of one per `(band, time)`, and each `part{pid:04d}` child's `row` dimension now
  spans all scans/times of the band.
- This *also* benefits the deconvolution operators that iterate partition children:
  `HessianTree` and `residual_from_partitions` (`operators/gridder.py:928`) see fewer, larger
  partitions ⇒ fewer FFTs per major/minor cycle. No code change required there; the win is a
  side effect of the larger row extent.

## 6. Edge cases & error handling

- **Defensive FREQ guard.** Before concatenating a partition key's pieces, assert all share the
  same `FREQ` (mirrors `image_data_products`'s `assert (ds.FREQ.values == freq).all()`). Safe by
  construction; the assert documents and protects the invariant.
- **Beam across times.** All pieces of a band share `freq_out` and the global beam grid
  (`max_blength`/`max_freq`), so their `BEAM` is identical; the existing rule of keeping the
  first piece's `BEAM`/`FREQ` during row-concat (`core/imager.py:99-103`) stays exact.
- **Single time / single scan.** `concat_row=True` with one timeid degenerates to today's
  single-node path (list of length 1) — no special-casing needed.
- **Default change.** `concat_row=True` is the default, so the imager's default output is now
  one node per band (matching `pfb grid`). Accepted.

## 7. Testing

- New test in `tests/test_imager.py`: run the imager twice on the shared MS,
  `concat_row=True` and `concat_row=False`, **both** with `integrations_per_image=15`,
  `robustness=0.0`, and `weight_grouping="per-band"`. The shared MS
  (`test_ascii_1h60.0s.MS`) is a single scan of 60 integrations, so `-1` would collapse to a
  single time node and make the granularity assertion vacuous; `integrations_per_image=15`
  splits it into 4 time blocks (= 4 `timeid`s) instead. Pinning both runs to per-band counts
  makes the per-row imaging weights identical, and `vis2dirty` is linear in the rows
  (`grid(A∪B) == grid(A) + grid(B)` up to floating-point accumulation), so the two MFS dirty
  images (sum `DIRTY` / sum `WSUM` over band nodes) must agree to gridder precision. Assert:
  - `concat_row=True` produces exactly one `band{b}_time0000` node per band and no other time
    index;
  - `concat_row=False` produces multiple time nodes per band (one per `integrations_per_image`
    block — 4 for the shared MS at `ipi=15`);
  - the two MFS dirty images agree via the divide-safe idiom
    `assert_allclose(1 + a, 1 + b, rtol=1e-4, atol=1e-4)` (consistent with the existing
    equivalence test; the true agreement is far tighter, ~1e-10).
- Existing tests are unaffected: they compare MFS and check `startswith("band")`, both of which
  hold under the new default.

## 8. Files touched

- `src/pfb_imaging/core/imager.py` — pass-2 dispatch grouping; `_grid_image` signature
  (`src_names: list[str]` + `out_name: str` replacing `image_name`); per-band counts reduction
  + coercion warning.
- `tests/test_imager.py` — new `concat_row` regression test.

No changes to `operators/gridder.py`, `utils/weighting.py`, `utils/stokes2vis_msv4.py`, the CLI,
or the cab definitions (the CLI parameter is already in place).
