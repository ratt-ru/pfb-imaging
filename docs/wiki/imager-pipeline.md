---
type: Subsystem Notes
title: MSv4 DataTree imager pipeline
description: Why the imager writes a DataTree, the two-pass data flow, the .dt layout, counts/weight-grouping and concat_row semantics, and the operator split that downstream deconvolution relies on.
tags: [imager, msv4, datatree, weighting, gridding, mosaic]
timestamp: 2026-11-20T16:00:00Z
last_verified_commit: c23a68f
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
**band node is the summation domain** (holds the image-space sums `DIRTY`/`WSUM`/`BEAM`,
plus `PSF`/`PSFPARSN` when `--psf` is on), the **partition children are the terms** (ragged
vis-space arrays plus per-partition `BEAM` (+ `PSF`/`PSFHAT` with `--psf`) and the phase
offsets `l0, m0`).
`baseline_group` sits in the partition identity `(msid, field, spw, baseline_group)`:
`"all"` by default, or `MM` / `MPM` / `MPMP` under `--baseline-groups`, which splits
each partition by dish-pair class so each carries its own MeerKAT+ Mueller beam
(#335). It was always a storage no-op — a split group is just another `part{p}`
child with its own `BEAM`.

## Tree layout (as built)

```
<out>_<PRODUCT>.dt/                   # opened via xr.open_datatree(...)
  (root attrs) pfb-imaging-version, product, nband, ntime,
               nx, ny, nx_psf, ny_psf, cell_rad, max_blength, max_freq
  band{b:04d}_time{t:04d}/            # ONE OUTPUT IMAGE / Hessian summation domain
      attrs:  bandid, timeid, freq_out, freq_nominal, time_out, ra, dec, cell_rad,
              robustness (omitted when natural), niters,
              freq_bounds (outer channel edges of the band's channels, Hz; D59), time_bounds (outermost integration
              edges, per-row intervals; unix s), transferred_from (--transfer-model-from only)
              # freq_out = effective (wsum-weighted) frequency of the channels
              # actually gridded; freq_nominal = band-edge midpoint, assignment
              # grid only (D28)
      vars:   DIRTY, BDIRTY, BEAM (corr, y, x),                    # (Y, X), D20
              PSF (corr, y_psf, x_psf), PSFPARSN (corr, bpar),     # only with --psf
              WSUM (corr,)
              # band BEAM = wsum-weighted mean of partition beams (linear-mosaic
              # response, still B/n per D22)
              # BDIRTY = sum_p B_p * dirty_p: model-free term of the exact deconv
              # gradient (D23) -- not derivable from the summed DIRTY
              # MODEL / RESIDUAL / BRESIDUAL are written by pass 2 under
              # --transfer-model-from, else added later by the deconv consumer
              # (with NOISE)
              # IMAGE / BIMAGE / KIMAGE (corr, y, x) + PSFPARSF (corr, bpar) and a
              # psfparsf_mfs attr added by the restore consumer -- the three flux
              # scales (intrinsic / apparent / mixed) of D29
      part{p:04d}/                    # ONE DATA PARTITION
          attrs:  msid, field_name, spw_name, baseline_group, freq_out,
                  ra, dec, l0, m0, wsum
          vis-space:   VIS, WEIGHT (corr, row, chan), MASK (row, chan),
                       UVW (row, three), FREQ (chan,)
          image-space: BEAM (corr, y, x; image grid, placed in pass 2);
                       PSF, PSFHAT (corr, y_psf, xo2), PSFPARSN only with --psf
```

**Every float array above carries the requested `--precision`** — f4/c4 for `single`,
f8/c8 for `double` — in both the `.dt` and the `.scratch`. The only exceptions are `UVW`
and `FREQ` (always f8: ducc takes those as double whatever the vis precision), `MASK`
(u1) and the scratch's `COUNTS_INDEX` (i8, cell indices rather than data). This is an invariant ducc enforces, not a memory optimisation: see design-decisions
D27. Pinned by `tests/test_imager_precision.py`.

## Product selection

`--psf` (default on) is a **compute** toggle: off skips the padded-grid PSF gridding, the
`PSFHAT` FFT and the clean-beam fits everywhere — a quicklook dirty-only tree that
`deconv` refuses with "re-run pfb imager with --psf" (guard fires before Ray init).
`--beam` (default on) gates only the beam FITS; the `.dt` `BEAM` is load-bearing (D22)
and always stored. Beam FITS are dimensionless (`BUNIT=""`), not wsum-normalised, and
carry a `BEAMINCN` card because the stored beam includes the folded n-term. Without `--transfer-model-from` the imager does not write `RESIDUAL`, so **`deconv` falls back to `DIRTY` when it is absent**
(`core/deconv.py`, where the driver loads the band cubes). A resumed run is different: a band with a `MODEL` but no `BRESIDUAL` is
refused outright, because the forward solver needs the beam-attenuated gradient (D23).
`--transfer-model-from <.mds>` (#309) seeds that resume: the product, path and flux scale (an
apparent model is refused, D54) are checked before the MS metadata is read, and the tangent point (D52) right after it; all of them before Ray starts. Frequency coverage is checked between the passes, against each image's effective `freq_out`, the frequency the model is rendered at (D52, D65).
Any cell and image size are accepted: the render remaps the model onto the run's grid
conserving flux, refining or coarsening (box kernel, D52). Each pass-2 task renders the model
**before** its gridding loop, at the `freq_out` the driver predicted from the counts (D65),
and `grid_partition` forms each partition's residual visibilities `r = V − G(B_p·x)` while
its VIS is in memory, gridding them into the partition's `RESIDUAL`. The band sums are
`RESIDUAL = Σ_p RESIDUAL_p` and `BRESIDUAL = Σ_p B_p·RESIDUAL_p` (D23), written with `MODEL`
and the `transferred_from` attr. `RESIDUAL` also goes to FITS (wsum-normalised, Jy/beam, like
`deconv`'s) under `--fits-mfs`/`--fits-cubes`; `MODEL` and `BRESIDUAL` stay tree-only.
This is the same quantity as deconv's image-space `DIRTY − Σ_p G_pᵀW_pG_p(B_p·x)`
(`residual_from_partitions`; the two share `gridder.degrid_apparent` and agree to 1e-8 of the
peak, `tests/test_imager_pass2.py::test_pass2_residual_matches_the_image_space_residual`), but
gridding the residual puts ducc's relative `epsilon` on the residual rather than the dirty
image, and no partition is read back from the `.dt` (until D65 a second sweep after the loop
re-read every partition's UVW/WEIGHT/MASK). Masked samples are zeroed in `r` outright: ducc's
mask does not stop a NaN on aarch64. The task holds three extra images through its
gridding loop (the rendered model and the `RESIDUAL`/`BRESIDUAL` sums: 1.1 GB at 6720² in
float64), and one vis-sized residual array per correlation, formed in the degridded
buffer. The `.mds` is loaded once per worker through a `Multiton` and stays resident there
(default 300 s inactivity TTL, swept lazily on cache access, as for the xarray-ms tables in
[memory-and-ray.md](memory-and-ray.md)). That residency is bounded by the model's size, which
is not negligible: `.mds` stores have reached 633 MB at 6720², so budget one per pass-2
worker.
`--fits-per-partition` (default off) makes the pass-2 workers write per-partition
sanity FITS while the products are in memory (per-partition DIRTY is never stored in the
tree): `<var>_band####_time####_part####_<field>.fits` under
`<fits_output_folder>/<oname>_partitions/`, dirty/psf wsum-normalised per partition,
beam as stored — the field name in the filename is what makes multi-pointing
orientation checks usable. With `--transfer-model-from` each partition's own residual is
written too (`residual_…`, wsum-normalised): every partition against the same band model, so
one that calibrated worse than its siblings stands out. It costs nothing extra, being the
term pass 2 sums into `RESIDUAL` (pinned in the slow
`test_transferred_truth_nulls_the_residual`).

All partitions in a band share one output grid: multi-field selections are rephased in
pass 1 to a common tangent point (D21; `--phase-dir`/barycentre default) and `--target`
offsets are the `l0, m0` attrs, with `x0,y0`/`flip_*` derived on demand from
`wgridder_conventions(l0, m0)` rather than stored. `time_out` is **unix seconds** (the
MSv2 epoch is MJD seconds — `utils/fits.set_wcs(time_is_unix=…)`).

## Two passes and the counts product

1. **Pass 1** (`utils/stokes2vis.stokes_vis`, one Ray task per fine piece): load the
   node subset, column arithmetic (`dc1 [+|-] dc2`), Stokes conversion via `weight_data`,
   channel-average/BDA, the piece's **effective frequency** (natural-weight-weighted mean
   over surviving channels — this is what the beam is evaluated at, D28), the beam
   evaluated on a small field-centred grid and stored there as `BEAM_SMALL`
   `(corr, m_beam, l_beam)` (no beam model: none at all), and write the piece keyed
   `(ms, field, spw, baseline_group, scan, band, time)` into `.scratch`, including its uv
   counts, which are gridded whatever the weighting so one scratch serves natural and robust
   imaging (#354, D66). They go on the padded image grid by default (`--weight-on-image-grid`);
   `--no-weight-on-image-grid` puts them on `core/imager.counts_grid`, whose uv cell is the
   padded cell of the largest square image within `--max-field-of-view` and which reaches the
   longest baseline. The beam is meerkat-beams' rotation average, run on the task's `--nthreads`
   with parallactic angles merged to `BEAM_PA_TOLERANCE_DEG` (0.1°); single-threaded and
   unmerged it was 90% of pass-1 task time (landmanbester/meerkat-beams#39). The driver
   builds each wizard once (staging, validation) but ships pass 1 a `Multiton` recipe,
   so each worker builds its own once and keeps it, spline prefilters included, across
   its tasks; a wizard shipped by value arrived with an empty prefilter cache and
   rebuilt it per correlation per task (~0.8 s each). The counts
   are stored **sparse**, as `COUNTS_INDEX`/`COUNTS_VALUE` (`utils/weighting.compute_sparse_counts`,
   which records each occupied cell as the scatter first touches it rather than scanning the
   grid with `np.flatnonzero`, #365):
   a piece fills a sliver of the padded uv plane, 0.05% of a 491 MB float64 grid at
   4608², so a dense grid cost a 491 MB encode per piece and a 491 MB decode and add per
   piece in the driver. Fine granularity (per scan /
   `integrations_per_image`) is needed because pass 1 reads raw unaveraged data.
   Under `--baseline-groups` the driver emits one pass-1 task per group per
   (node, time block, channel block), slicing the node with
   `isel(baseline_id=…)`. Masks are computed **per selected node** — different
   scans/SPWs can carry different antenna subsets, so `baseline_id` indices are
   only valid for the node they came from. Groups with no baselines are omitted
   rather than written as empty partitions.

   **The beam grid** (meerkat-beams wizards; katbeam keeps its own Nyquist-sized grid,
   because `ndimage.rotate` makes its beam depend on the grid) is sized once by the driver,
   in `core/imager.field_beam_axis`. It does not depend on the image: it reaches
   `--max-field-of-view` (a radius `R` about the tangent point) plus the farthest field's
   distance from the tangent point (`pointing_reach`, D66), at the cell `utils/beam.beam_grid` sets:
   `h = sqrt(tol) * lambda_min / D_max`. Here `D_max` is the largest dish any of the run's
   beams describes, and `BEAM_CELL_TOL = 1e-4`. Pass 2 places the beam bilinearly (meerkat-beams `beam_on_grid`,
   0 outside the grid), and its
   worst error in the main lobe was measured at `(h / (lambda_min/D))^2`
   (`docs/experiments/beam_time_gradient.md`). It replaced a fixed 1024² over the whole
   cube, whatever the image (#365). **Two refusals, both at set-up before Ray starts:** an
   image whose farthest corner lies beyond `R` (`check_image_extent`), and, when the cube
   caps the axis below what `R` asks for, an image reaching past the cube's largest centred
   disk less two cube pixels (`check_beam_reach`; 1.81° from the pointing for the MdV-2026
   group cubes, 3.81° for the ungrouped L cube). The rotation average turns the field with
   parallactic angle, so only that disk is covered at every angle, and beyond it the
   placement error does not converge with the cell. D64 has the measurements.
2. **Counts reduction** (driver, between passes): `COUNTS` is a first-class reducible
   intermediate. The driver streams per-piece counts into **one grid per applied
   `weight_grouping` group** — `per-band-time` (default), `mfs` (sum over bands per
   time), `per-band` (sum over time), `per-time` (`counts_key` in
   `core/imager.imager` defines the groups).
   `utils/weighting.write_group_counts` does it one group at a time — one dense
   accumulator, into which each piece's occupied cells are added — applies
   `filter_extreme_counts` and `box_sum_counts` once per group, and writes the result
   to `.scratch` as `counts/group####` (`COUNTS (corr, u, v)`). Pass-2 tasks receive
   that node path and read the grid themselves: a counts grid is `(ncorr, counts_nx,
   counts_ny)` — 0.32 GiB at the 3840² image's padded grid — and passing it as a Ray argument put a copy per task
   in the object store (#339). With the applied grid in hand, a second sparse read of
   the group's pieces gives each piece's exact imaging-weight sum
   (`weighting.piece_imaging_wsum`, with the off-grid natural weight pass 1 recorded as
   `wsum_offgrid`), from which the driver reduces every image's
   `freq_out` before pass 2 (D65). Before the pieces went sparse this step read every
   piece's dense grid back: 34.1 s of driver time for 120 pieces of an 8-band 4608²
   MeerKAT+ run, 10.8 s after. Summing counts across pieces is valid because the
   uv grid is commensurate across bands (one counts geometry per run).
   Natural weighting (`robustness None`, or `> 2`, which `effective_robustness` maps to
   `None` up front) does not reduce the counts, which pass 1 still grids (D66). Before
   that mapping a value above 2 was Briggs-weighted (close to natural, not equal to it).
   Counts are reduced per `weight_grouping` group, **not** per partition, so
   baseline groups share one counts grid exactly as multiple fields do. Robust
   weighting is therefore computed over the whole array rather than
   independently per group — which is what makes summing the groups
   algebraically identical to not splitting (pinned by
   `tests/test_imager.py::test_baseline_groups_sum_to_the_ungrouped_image`,
   which agrees to within the wgridder's own `epsilon=1e-7`).
3. **Pass 2** (`core/imager._grid_image`, one Ray task per output image): group scratch
   pieces by partition key, reduce each group with `_concat_pieces` (rows concatenated
   along `row`; `BEAM_SMALL` and `freq_out` combined as means weighted by each piece's
   predicted imaging wsum, `wsum_img` (D65) — pieces may
   differ in both, so piece 0's beam is *not* representative, D28), place each partition's
   mean `BEAM_SMALL` on the image grid once at the offset it is gridded with and fold in `1/n`
   (`stokes2vis.partition_beam`, #365; `1/n` alone without a beam model — placement is linear,
   so this equals averaging placed beams, and a wizard piece never writes an image-sized
   beam; katbeam pieces keep their own Nyquist grid, out to the same reach),
   read the image's `counts/group####` grid and apply `counts_to_weights` per partition, grid
   each partition with
   `operators/gridder.grid_partition`, sum image-space products into the band node
   (including the band's `freq_out` as the imaging-wsum-weighted mean of the partition
   frequencies, which must match the driver's prediction, D65), write the `.dt`. Each `(band, part)` is an independent zarr group path,
   so workers write concurrently without contention.

## `concat_row` semantics

`concat_row=True` (default) collapses the time axis: **one output image per band**
(`band{b:04d}_time0000`), the pass-2 task receives *all* of the band's scratch nodes,
and rows are concatenated across scans *and* times within each partition key before a
single gridding — mirroring legacy `image_data_products`. Without it, a multi-scan MS
pays one full DIRTY+PSF gridding per scan even at `integrations_per_image=-1`.

- Rows are only concatenatable when they share phase centre and `FREQ` axis; this holds
  by construction within a partition key and is guarded by an assert. Different
  fields/spws are separate partitions: **summed, never concatenated**.
- Pieces within a concat group may still differ in `BEAM_SMALL` and effective frequency
  (different times ⇒ different `BeamWizard` beams; different flagging ⇒ different
  effective frequency), so `_concat_pieces` reduces both as `wsum_img`-weighted means
  rather than taking the first piece's (D28).
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

This mirrors the split the retired MSv2 pipeline had between `image_data_products` and `compute_residual`.

## Access layer and testing strategy

Native DataTree API only (`xr.open_datatree`, `ds.to_zarr(group=…, mode="a")`,
`dt.children`) — no `xds_from_url`-style wrappers for the `.dt` (design-decisions.md
D12, including the `mode="a"` attrs-replacement gotcha). Correctness rests on ground
truth, since the legacy `init`+`grid` oracle was retired with those commands in 0.1.0
(#277): an injected point-source sky recovered at its known positions and fluxes
(`tests/test_imager.py::test_imager_groundtruth`) and the dirty image against a direct
Fourier transform (`test_imager_matches_dft`); then `grid_partition`
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
- **`--baseline-groups` triples the image-space cost, not the vis-space one.**
  Rows are partitioned rather than duplicated, so total vis-space storage and
  total gridding work stay roughly constant; what triples is the per-partition
  `PSF`/`PSFHAT`/`BEAM` set, the FFTs over them (including `HessianTree`'s, per
  minor cycle), and the beam evaluation and placement, which are independent of row
  count (one small grid per piece in pass 1, one image-sized placement per partition
  in pass 2). The driver also opens each node once per group; whether strided
  baseline reads cost more than one contiguous read is unmeasured.
- **Groups are very uneven.** On real MeerKAT+ data (37 antennas, 3 extension
  dishes) the split is `MM` 561 / `MPM` 102 / `MPMP` 3 baselines, so one pass-1
  task per block handles 84% of the baselines and another 0.45%. Keep the small
  groups regardless: the extension dishes carry the longest spacings (`MPMP` up
  to 8.6 km, `MPM` up to 11.8 km, against 6.3 km for the longest `MM`), so they
  hold the high-resolution uv coverage.
- **Band-drop indexing:** a fully-flagged band leaves a gap in imager band ids (the
  retired `init`+`grid` reindexed contiguously); align on `freq_out`, never on band index.

## Shared MSv4 access

`get_engine` — which maps an MS path to the backend kwargs `xr.open_datatree`
needs — no longer lives in `core/imager.py`. It moved to
`utils/msv4.py` (#278) so both MSv4 front ends resolve backends identically;
`imager` imports it from there. That module also carries `select_vis_nodes`
and `wrapped_angle_diff`, both shared with `degrid`: the imager's node loop
selects through `select_vis_nodes` (name-based selection, the frequency window
and `chan0`, so the two front ends cannot drift; commit `08a5697`) and adds its
own imaging geometry, channel widths and rephasing barycentre on top, and its
transfer tangent-point check (D52) uses `wrapped_angle_diff`.

`degrid` also shares the baseline-group machinery in `utils/baselines.py`
(`preflight_baseline_groups`, `check_group_telescope`, `node_baseline_groups`), but none of the `.dt`: it applies the
primary beam itself (#353). `--beam-model` names a meerkat-beams band (katbeam is refused),
`--baseline-groups` gives each `MM`/`MPM`/`MPMP` group its own beam, and `--antenna-groups`
overrides the dish classification exactly as in the imager. `partition_mueller` gets the
replica's `--nthreads` and the same `BEAM_PA_TOLERANCE_DEG` as pass 1. The beam is the bare `B`, a full
Stokes Mueller block, and every degrid prediction calls ducc with `divide_by_n=True`, beam or
not, which is the opposite of the imager's folded `B/n` (D22). The cross group is predicted
exactly rather than as `Re(B)`. D55 has the reasoning and the guards.
