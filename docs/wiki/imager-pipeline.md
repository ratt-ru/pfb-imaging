---
type: Subsystem Notes
title: MSv4 DataTree imager pipeline
description: Why the imager writes a DataTree, the two-pass data flow, the .dt layout, counts/weight-grouping and concat_row semantics, and the operator split that downstream deconvolution relies on.
tags: [imager, msv4, datatree, weighting, gridding, mosaic]
timestamp: 2026-07-17T20:00:00Z
last_verified_commit: 4b571e9
---

# MSv4 DataTree imager pipeline

`pfb imager` subsumes `init`+`grid` for MSv4 data: two Ray-distributed passes producing a
single unified `xarray.DataTree` (`<out>_<PRODUCT>.dt`) plus a `.scratch` cache, replacing
the legacy `.xds`+`.dds` split. The quick operational summary lives in
`.claude/rules/architecture.md §8`; this page holds the *why* and the semantics that are
expensive to re-derive. Memory discipline for both passes:
[memory-and-ray.md](memory-and-ray.md). Deferred work: `docs/look-ahead.md`.

## Why a DataTree

The central requirement is **bookkeeping for mosaicking**: one *output image* per Hessian
application, but **multiple data partitions can feed one output image**. Each partition
(different field, spw, or — future — baseline group) carries its own phase centre, primary
beam and PSF, and the Hessian is a sum over partitions:

```
H x = Σ_p  B_pᵀ G_pᵀ W_p G_p B_p x
```

Partitions have **different row counts** (BDA, different scans/fields), so they cannot
share a `row` dimension in one `Dataset`. The tree is a 1:1 map of the equation: the
**band node is the summation domain** (holds the image-space sums `DIRTY`/`RESIDUAL`/
`PSF`/`PSFPARSN`/`WSUM`), the **partition children are the terms** (ragged vis-space
arrays plus per-partition `PSF`/`PSFHAT`/`BEAM` and the phase offsets `l0, m0`).
`baseline_group` sits in the partition identity `(msid, field, spw, baseline_group)` —
only ever `"all"` today — so per-antenna-pair Mueller beams (MeerKAT+) become a storage
no-op later: just another `part{p}` child with its own `BEAM`.

## Tree layout (as built)

```
<out>_<PRODUCT>.dt/                   # opened via xr.open_datatree(...)
  (root attrs) pfb-imaging-version, product, nband, ntime,
               nx, ny, nx_psf, ny_psf, cell_rad, max_blength, max_freq
  band{b:04d}_time{t:04d}/            # ONE OUTPUT IMAGE / Hessian summation domain
      attrs:  bandid, timeid, freq_out, time_out, ra, dec, cell_rad,
              robustness (omitted when natural), niters
      vars:   DIRTY, RESIDUAL, PSF (corr, y[, _psf], x[, _psf]),   # (Y, X), D20
              PSFPARSN (corr, bpar), WSUM (corr,)
              # MODEL / NOISE added later by the deconv consumer
      part{p:04d}/                    # ONE DATA PARTITION
          attrs:  msid, field_name, spw_name, baseline_group,
                  ra, dec, l0, m0, wsum
          vis-space:   VIS, WEIGHT (corr, row, chan), MASK (row, chan),
                       UVW (row, three), FREQ (chan,)
          image-space: PSF, PSFHAT (corr, y_psf, xo2), BEAM, PSFPARSN
```

All partitions in a band share one output grid (cell, common phase centre); per-field
offsets are the `l0, m0` attrs, with `x0,y0`/`flip_*` derived on demand from
`wgridder_conventions(l0, m0)` rather than stored. `time_out` is **unix seconds** (the
legacy `.dds` is MJD seconds — `utils/fits.set_wcs(time_is_unix=…)`).

## Two passes and the counts product

1. **Pass 1** (`utils/stokes2vis_msv4.stokes_vis`, one Ray task per fine piece): load the
   node subset, column arithmetic (`dc1 [+|-] dc2`), Stokes conversion via `weight_data`,
   channel-average/BDA, beam on the small grid, write the piece keyed
   `(ms, field, spw, baseline_group, scan, band, time)` into `.scratch`, and return the
   piece's uv `COUNTS` contribution. Fine granularity (per scan /
   `integrations_per_image`) is needed because pass 1 reads raw unaveraged data.
2. **Counts reduction** (driver, between passes): `COUNTS` is a first-class reducible
   intermediate. The driver streams per-piece counts into **one grid per applied
   `weight_grouping` group** — `per-band-time` (default), `mfs` (sum over bands per
   time), `per-band` (sum over time), `per-time` — holding `ngroups` grids, never
   `nband*ntime` (`counts_key` in `core/imager.imager`; semantics in
   `utils/weighting.reduce_counts`). Summing counts across pieces is valid because the
   uv grid is commensurate across bands (fixed cell + padding via `set_image_size`).
   Natural weighting is `robustness None` or `> 2`: counts are skipped entirely.
3. **Pass 2** (`core/imager._grid_image`, one Ray task per output image): group scratch
   pieces by partition key, concat scans along `row` (keeping the first piece's
   `BEAM`/`FREQ` — exact because all of a band's pieces share `freq_out` and the global
   beam grid), apply `counts_to_weights` per partition, grid each partition with
   `operators/gridder.grid_partition`, sum image-space products into the band node,
   write the `.dt`. Each `(band, part)` is an independent zarr group path, so workers
   write concurrently without contention.

## `concat_row` semantics

`concat_row=True` (default) collapses the time axis: **one output image per band**
(`band{b:04d}_time0000`), the pass-2 task receives *all* of the band's scratch nodes,
and rows are concatenated across scans *and* times within each partition key before a
single gridding — mirroring legacy `image_data_products`. Without it, a multi-scan MS
pays one full DIRTY+PSF gridding per scan even at `integrations_per_image=-1`.

- Rows are only concatenatable when they share phase centre and `FREQ` axis; this holds
  by construction within a partition key and is guarded by an assert. Different
  fields/spws are separate partitions: **summed, never concatenated**.
- Counts are auto-coerced to band granularity, with a warning for time-resolved
  groupings: `per-band-time → per-band`, `per-time → mfs`.
- `time_out` of the collapsed node is the mean of the constituents' `time_out`.
- Side benefit downstream: `HessianTree`/`residual_from_partitions` see fewer, larger
  partitions ⇒ fewer FFTs per cycle.

## Operator split for the deconv consumer

The PSF is computed once in pass 2 and **never recomputed**:

- `operators/hessian.HessianTree` — PSF-convolution Hessian,
  `H x = (1/Σ_p wsum_p) Σ_p B_pᵀ (PSF_p ⊛ (B_p x)) + η x`, from stored `PSFHAT`+`BEAM`.
  Cheap minor-cycle operator; preallocated FFT scratch for actor reuse.
- `operators/gridder.residual_from_partitions` — the exact once-per-major-cycle
  gradient `dirty − Σ_p G_pᵀ W_p G_p (beam_p · model)` from the stored vis-space
  arrays; owns mosaic correctness (per-partition beam + offsets). An exact-gridder
  `HessianTree` mode was deliberately dropped as redundant with this function.

This mirrors the legacy `image_data_products`/`compute_residual` split.

## Access layer and testing strategy

Native DataTree API only (`xr.open_datatree`, `ds.to_zarr(group=…, mode="a")`,
`dt.children`) — no `xds_from_url`-style wrappers for the `.dt` (design-decisions.md
D12, including the `mode="a"` attrs-replacement gotcha). Correctness rests on the
legacy path as oracle: single-field imager↔`init`+`grid` MFS equivalence
(`tests/test_imager.py`, in-process since arcae 0.5.2), `grid_partition`
row-additivity and `residual_from_partitions` invariants (`tests/test_imager_pass2.py`),
`HessianTree` delta-PSF/two-partition identities (`tests/test_hessian_tree.py`), counts
reductions (`tests/test_weighting.py`), tree FITS (`tests/test_fits_tree.py`), and the
`concat_row` True/False MFS agreement test pinned to `per-band` counts (gridding is
linear in rows, so the two must agree to gridder precision).

## Known risks / accepted costs

- **Concurrent zarr group writes:** distinct group paths with `mode="a"` and no
  consolidated metadata are safe on a local filesystem; the e2e test runs `nworkers=1`,
  so higher worker counts are the less-exercised path.
- **`PSFHAT` footprint:** every partition stores its own `PSFHAT` + vis-space arrays —
  more disk than the legacy single-PSF-per-band layout, required by the operator split;
  no opt-out flag.
- **Band-drop indexing:** a fully-flagged band leaves a gap in imager band ids while
  legacy `init`+`grid` reindexes contiguously; align on `freq_out`, never on band index.
