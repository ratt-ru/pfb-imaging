# Imager DataTree — Design

**Date:** 2026-06-04
**Branch:** `imager`
**Status:** Design (approved-pending)

## 1. Motivation

`pfb imager` currently replicates `init` against an MSv4-compatible data format. The goal
of this feature is to make `imager` subsume what `init` + `grid` do today, but backed by a
single, deliberately designed intermediary data product instead of the two-store split
(`.xds` vis-space Stokes products → `.dds` image-space products) glued together by bespoke
globbing helpers (`xds_from_url`, `xds_from_list` in `utils/naming.py`).

The central new requirement is **bookkeeping for mosaicking**: there is one *output image*
per Hessian application (one per imaging band, optionally per time chunk), but **multiple
data partitions can feed a single output image**. When those partitions are different fields
(or, in future, different baseline groups), each carries its own phase centre, primary beam,
and PSF, and the Hessian becomes a sum over partitions:

```
H x = Σ_p  B_pᵀ G_pᵀ W_p G_p B_p x
```

A plain `xarray.Dataset` cannot host this because partitions have **different row counts**
(BDA, different scans/fields), so they cannot share a `row` dimension. The natural primitive
is an `xarray.DataTree` with **one node per output image**, each node owning a set of data
partitions as child nodes.

## 2. Scope

In scope this iteration:

- Two-pass `imager` producing a unified DataTree (`.dt`) plus FITS outputs.
- Full multi-partition storage layout, including a `baseline_group` axis in the partition
  identity (storage extensibility only — partitioning *by* baseline group is out of scope).
- The image-space sum-over-partitions Hessian (`HessianTree`), with **both** backends
  (PSF-approx and exact gridder), implemented and unit-tested directly against a tree.
- A first-class, reducible `COUNTS` product with named weighting-reduction strategies.
- Scratch store retained as an optional cache.

Out of scope this iteration (deliberate follow-ups):

- Rewiring `sara` / `kclean` / `deconv` / `restore` / `degrid` / `fluxtractor` /
  `model2comps` to consume the new tree. `init` and `grid` remain the live default pipeline
  and are untouched. The new tree is *designed to be consumable* downstream; the eventual
  primary consumer is the (currently non-operational) `deconv` sub-command, which will be
  validated against `sara`/`kclean` before the old tooling is deprecated in a major bump.
- Actual partitioning of data by baseline group and per-antenna-pair-type Mueller beams
  (MeerKAT+). Only the *storage layer* must accommodate them.

## 3. Pipeline and stores

`imager` runs two passes and manages two zarr stores:

| Store           | Suffix (relative to `output_filename_<PRODUCT>`) | Lifetime                    | Content |
|-----------------|---------------------------------------------------|-----------------------------|---------|
| Scratch         | `.scratch`                                        | optional cache (kept by default) | pass-1 fine averaged Stokes vis (per scan) + per-`(band,time)` `COUNTS` |
| Imaging product | `.dt`                                             | persistent                  | the unified DataTree (replaces `.xds` + `.dds`) |

Distinct suffixes ensure the new product never collides with `init`/`grid`'s `.xds`/`.dds`.

## 4. The unified DataTree (`.dt`)

One node per **output image** named `band{b:04d}_time{t:04d}`; one child node per **data
partition** named `part{p:04d}`. A partition is identified by the tuple
`(field, spw, baseline_group)` and scans within a partition are concatenated along `row`.
`baseline_group` defaults to a single group (e.g. `"all"`) until baseline-group partitioning
is implemented.

```
<output>_<P>.dt/                      # single zarr store, opened via xr.open_datatree(...)
  (root attrs)
      pfb-imaging-version, product, nband, ntime,
      freq_out[nband], time_out[ntime],
      nx, ny, nx_psf, ny_psf, cell_rad,
      max_blength, max_freq,
      flip_u, flip_v, flip_w (wgridder convention defaults)

  band{b:04d}_time{t:04d}/            # === ONE OUTPUT IMAGE / one Hessian summation domain
      coords: x, y, corr, x_psf, y_psf
      attrs:  bandid, timeid, freq_out, time_out, freq_min, freq_max,
              ra, dec,            # output image phase centre
              l0, m0, x0, y0,     # wgridder centre conventions for the image frame
              flip_u, flip_v, flip_w,
              robustness, weight_grouping,
              wsum,               # Σ over partitions, per corr
              niters
      vars (Σ over partitions, image-space):
              MODEL    (corr, x, y)
              DIRTY    (corr, x, y)
              RESIDUAL (corr, x, y)
              NOISE    (corr, x, y)

      part{p:04d}/                    # === ONE DATA PARTITION (field, spw, baseline_group)
          coords: row, chan, corr     # x, y, x_psf, y_psf inherited from parent
          attrs:  msid, field_name, spw_name, baseline_group,
                  scan_names[],
                  ra, dec,            # field phase centre
                  l0, m0, x0, y0,     # field offset wrt output image centre
                  wsum                # per corr
          vis-space:
                  VIS    (corr, row, chan)
                  WEIGHT (corr, row, chan)   # imaging weights applied (post reduction)
                  MASK   (row, chan)
                  UVW    (row, three)
                  FREQ   (chan,)
          image-space (per partition; required for "store both"):
                  PSF     (corr, x_psf, y_psf)
                  PSFHAT  (corr, x_psf, yo2)
                  BEAM    (corr, x, y)
                  PSFPARSN(corr, bpar)
```

Design rationale:

- The band node is the **summation domain**: image-space sums
  (`DIRTY`/`MODEL`/`RESIDUAL`/`NOISE`) and the summed `wsum` live here.
- Partition children hold everything partition-specific: vis-space arrays (ragged `row`),
  the per-field `BEAM`/`PSF`/`PSFHAT`, and the phase offsets `x0,y0`. This is a 1:1 map of
  `H x = Σ_p B_pᵀ G_pᵀ W_p G_p B_p x` — band node = the sum, children = the terms.
- All partitions in a band share **one** output image grid (`x,y`, cell, common phase
  centre). Per-field offsets are carried as `x0,y0`/`l0,m0` partition attrs.
- Heterogeneous `row` counts across partitions are exactly what DataTree allows; this is the
  reason a single `Dataset` is insufficient.
- `baseline_group` in the partition identity makes per-antenna-pair-type Mueller beams a
  storage no-op later: a new baseline group is simply another `part{p}` node with its own
  `BEAM`.

## 5. Counts product and named weighting reductions

Robust/uniform weighting needs a global (per output-image, or coarser) reduction over the uv
distribution. We make the uv-cell `COUNTS` grid a **first-class, reducible intermediate
product** sitting between the two passes.

- **Accumulation (pass 1):** each fine piece contributes to a `COUNTS` grid keyed by its
  `(band, time)` output image. Summing over partitions/scans is just binning more uv samples
  and is valid because the uv grid is commensurate across bands (fixed image cell + padding,
  via `set_image_size`). Stored in the scratch store at the `(band,time)` level.
- **Reduction (between passes):** a stage maps raw `COUNTS` → applied counts according to a
  `--weight-grouping` option, implemented as a registry of named strategies:
  - `per-band-time` (default): each output image uses its own counts.
  - `mfs`: sum counts over all bands (per time) → a truly uniform MFS image.
  - `per-band`: sum over time within a band.
  - `per-time`: sum over bands within a time.

  New strategies are added as new registry entries; nothing else changes.
- **Application (pass 2):** `counts_to_weights` is run per partition against the reduced
  grid; the result multiplies the natural weights and is written as `WEIGHT` into each
  partition node.

## 6. Pass 1 — fine, per scan

A refactor of today's `stokes2vis_msv4.stokes_vis`:

1. Load node subset; apply column arithmetic (`dc1 [+|-] dc2`).
2. Convert to Stokes via `weight_data` (gains, poltype, product, `wgt_mode`).
3. Channel-average and/or BDA (`chan_average`, `bda_decorr`).
4. Evaluate the beam on the small `(l_beam, m_beam)` grid (unchanged from today).
5. Write a fine averaged piece keyed `(ms, field, spw, baseline_group, scan, band, time)` to
   the scratch store.
6. Accumulate this piece's contribution into the `(band, time)` `COUNTS` grid.

Fine partitioning (per scan by default but governed by integrations-per-image in general) is required because pass 1 reads raw unaveraged data and needs
finer granularity than the imaging products.

## 7. Pass 2 — coarse

1. Group scratch pieces by output image `(band, time)`, then by partition
   `(field, spw, baseline_group)`; concat scans within a partition along `row`.
2. Apply reduced imaging weights (`WEIGHT` per partition).
3. Per partition: weighted-sum the small-grid beam onto the image grid; grid `DIRTY`, `PSF`,
   `PSFHAT`, `BEAM`, `PSFPARSN`.
4. Sum image-space products over partitions into the band node (`DIRTY`/`RESIDUAL`/`NOISE`,
   and `wsum`).
5. Write the tree. Each `(band, part)` is an independent zarr group path, so Ray workers
   write concurrently without contention.

## 8. Hessian over the tree (`HessianTree`, "store both")

A `HessianTree` operator walks a band node's partition children and sums their contributions.
Two interchangeable backends share the stored data:

- **PSF-approx:** `Σ_p` beam-weighted PSF convolution using per-partition `PSFHAT`/`BEAM`
  (fast; approximate across a wide mosaic). Mirrors today's `HessPSF` but generalised to a
  sum over partitions with per-partition beams.
- **Exact gridder:** `Σ_p hessian_slice` over each partition's vis-space arrays
  (accurate; slower; one degrid/grid round-trip per partition per major cycle).

Both are implemented and unit-tested directly against a tree this iteration. `HessianTree` is
the intended bridge for the future `deconv` consumer.

## 9. Access layer

Within the imager code path, all reads/writes go through `xr.open_datatree`, per-group
`Dataset.to_zarr`, and `DataTree.to_zarr`. No `xds_from_url`/`xds_from_list` inside imager.
Those bespoke helpers remain only for the still-live `grid`/`deconv`/`restore`/etc. consumers
and will be retired when those are migrated.

## 10. FITS output

`imager` emits FITS (`DIRTY`/`PSF`/`RESIDUAL`/`MODEL`/`NOISE`/`BEAM`) from the band nodes,
controlled by the same `fits_mfs`/`fits_cubes`/`fits_output_folder` options as `grid`.

The existing `rdds2fits` (`utils/fits.py`) must keep working unchanged for the live
`grid`/`restore`/etc. consumers — it reads a flat dds list via `xds_from_list` and that
contract is not to be broken. Rather than retrofit `rdds2fits` to walk a DataTree, add a
**separate** tree-aware FITS utility (e.g. `rdt2fits`) that reads band nodes from the `.dt`
store and reuses the lower-level FITS-writing helpers in `utils/fits.py` where they are
already independent of the dds-list access pattern.

## 11. Testing

- **Single-field equivalence:** `imager` `DIRTY`/`PSF` match `init`+`grid` within gridding
  tolerance on the existing test MS.
- **Two-partition mosaic:** synthetic two-field case exercising the summation Hessian; the
  PSF-approx and exact-gridder backends agree to tolerance.
- **Counts reductions:** `mfs` vs `per-band-time` produce the expected difference in applied
  weights / image weighting.
- **Scratch cache:** a re-run with changed imaging/weighting params reuses the scratch store
  and skips MS re-reads; cache invalidation works like `grid`'s `opts.pkl` validation.

## 12. Open questions / risks

- **Concurrent zarr group writes:** independent group paths in one store should be safe for
  parallel Ray writers; confirm with the chosen zarr backend during implementation.
- **`PSFHAT` footprint:** storing per-partition `PSFHAT` for every partition increases disk
  use vs the current single-PSF-per-band layout; accepted given the "store both" decision.
  No opt-out flag for now — both PSF-approx and exact-gridder backends are always supported.
- **Beam interpolation** currently uses a coarse rotation-averaged katbeam; unchanged here,
  but the per-partition layout is where improved/Mueller beams will slot in. The eventual plan (out of scope for now) is to use the BeamWizard class to do the beam interpolation (as is done in the hci sub-command in the stokes2im function). This functionality will live outside of pfb-imaging.
