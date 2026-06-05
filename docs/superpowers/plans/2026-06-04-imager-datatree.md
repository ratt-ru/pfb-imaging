# Imager DataTree Implementation Plan (revised)

> **For agentic workers:** implement task-by-task with TDD. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `pfb imager` produce a single unified `xarray.DataTree` (one node per output image, one child per data partition) that subsumes today's `.xds`+`.dds` split, via a two-pass MS→tree pipeline with a reducible uv-`COUNTS` weighting product, a sum-over-partitions `HessianTree`, and FITS export.

**Architecture:** Pass 1 reads raw MSv4 finely (per time-chunk/scan), converts to Stokes, averages, and writes fine pieces + per-piece uv-counts into a `.scratch` DataTree. A reduction combines counts by a named grouping. Pass 2 groups fine pieces into partitions `(field, spw, baseline_group)`, applies imaging weights, grids per-partition image products with ducc0, sums them into per-band-time output-image nodes, and writes the `.dt` DataTree. The whole path is casacore-free so arcae (MSv4 read) and ducc0 gridding coexist. `init`/`grid` stay live and untouched.

**Hard constraints (see spec §2a):** native DataTree API only (no wrapper shims); extend existing modules (no new `imager_*`/`*_tree` files); keep python-casacore off the imaging import path.

---

## Module map (where each piece lives)

| Concern | File | Form |
|---|---|---|
| casacore isolation | `utils/{misc,beam,fits}.py` | **DONE — commit 89b51cc** |
| counts reduction | `utils/weighting.py` | `reduce_counts(counts, grouping)` |
| pass-1 fine Stokes + counts | `utils/stokes2vis_msv4.py` | extend `stokes_vis` |
| pass-2 per-partition grid + sum | `operators/gridder.py` | extend `image_data_products` |
| `HessianTree` | `operators/hessian.py` | class beside `HessPSF` |
| tree FITS | `utils/fits.py` | `rdt2fits` beside `rdds2fits` |
| geometry + orchestration + CLI | `core/imager.py`, `cli/imager.py` | inline |

DataTree conventions: output-image group `band{b:04d}_time{t:04d}`; partition child `part{p:04d}`; partition identity `(msid, field_name, spw_name, baseline_group)` with `baseline_group="all"` for now.

---

## Phase 0 — casacore isolation — DONE (commit 89b51cc)

Deferred `africanus`/`daskms` imports in `misc.py`/`beam.py` to the functions that use them; switched `lightspeed` to `scipy.constants`; replaced `casacore.quanta` time conversion in `fits.py` with astropy. Verified: `operators.gridder`, `utils.fits`, `operators.hessian` import casacore-free and `xarray-ms:msv2` resolves; `test_beam.py`+`test_convolve2gaussres.py` (74 tests) pass; time conversion matches casacore to <1e-6 s.

---

## Phase 1 — `reduce_counts` in `utils/weighting.py`

**Function:** `reduce_counts(counts: dict[tuple[int,int], np.ndarray], grouping: str) -> dict[tuple[int,int], np.ndarray]`. `counts` maps `(bandid, timeid)` → `(ncorr, nx, ny)`. Returns the same keys with reduced (possibly shared, read-only) grids. `grouping`:
- `per-band-time` → identity (`dict(counts)`).
- `mfs`, `per-time` → sum over bands within each time.
- `per-band` → sum over time within each band.
- unknown → `ValueError` naming the value and the valid set.

- [ ] **Step 1 (test):** add `tests/test_weighting.py` cases (reuse the file) — identity preserves keys/values; mfs sums bands within a time and shares the grid across those keys; per-band sums time within a band; unknown raises `ValueError` (use `pytest.raises(..., match=...)`).
- [ ] **Step 2:** run, confirm fail (no `reduce_counts`).
- [ ] **Step 3:** implement `reduce_counts` (pure numpy `match` statement; document that returned grids may be shared and are read-only).
- [ ] **Step 4:** run, confirm pass.
- [ ] **Step 5:** `uv run ruff format . && uv run ruff check . --fix`; commit `feat(imager): add reduce_counts weighting grouping`.

---

## Phase 2 — extend `stokes_vis` (pass 1)

Modify `utils/stokes2vis_msv4.py::stokes_vis` (only used by `core/imager.py`):
- New args: `scratch_store`, `nx_pad`, `ny_pad`, `cell_rad`, `baseline_group="all"` (replace the old `xds_store`/`oname` flat write).
- After computing `mask`/`weight` (corr-first), compute the piece's counts with `_compute_counts(uvw, freq, mask, wgt_cf, nx_pad, ny_pad, cell_rad, cell_rad, wgt_cf.dtype, ngrid=...)` directly (no wrapper).
- Add `COUNTS (corr,u,v)` to the dataset and partition-identity attrs incl. `baseline_group`, `msid`, `field_name`, `spw_name`.
- Write via `out_ds.to_zarr(scratch_store, group=f"band{bandid:04d}_time{timeid:04d}/{piece_name}", mode="a")` where `piece_name` encodes `(ms,field,spw,bg,scan)`. Keep numerics identical otherwise. Return `(bandid, timeid, piece_name)` / `None`.

- [ ] **Test:** in `tests/test_imager.py` (or a new `tests/test_imager_tree.py`), slice the first visibility node of the session MS, call `stokes_vis` directly (no ray) writing to `tmp_path`, reopen with `xr.open_datatree`, assert the piece has `VIS/WEIGHT/MASK/UVW/FREQ/BEAM/COUNTS` and `baseline_group=="all"`, and `COUNTS.shape == (ncorr, nx_pad, ny_pad)`. **Run MSv4 reads in a process that never imports the casacore gridding path** (this test imports only stokes2vis_msv4 + xarray, not gridder).
- [ ] Confirm fail → implement → confirm pass → lint → commit `feat(imager): pass-1 stokes_vis writes scratch tree + counts`.

---

## Phase 3 — extend `image_data_products` (pass 2)

In `operators/gridder.py`, add the tree-writing, sum-over-partitions pass-2 path that reuses the existing dirty/psf/psfhat/beam gridding internals (factor the single-partition gridding into a reusable helper if needed, keeping `image_data_products` working for `grid`). Responsibilities:
- Read a `(band,time)` output image's scratch pieces (`xr.open_datatree`), group by partition `(field,spw,baseline_group)`, concat scans along `row`.
- Apply reduced counts → imaging `WEIGHT` per partition (reuse `filter_extreme_counts`/`box_sum_counts`/`counts_to_weights`).
- Grid per-partition `DIRTY`/`PSF`/`PSFHAT`/`BEAM`/`PSFPARSN`/`WSUM` (ducc0); interpolate the small-grid beam to the image grid with `eval_beam`.
- Write each partition child node; sum image-space products into the band node (`DIRTY`/`RESIDUAL`=DIRTY/`WSUM`). Return MFS psf params for FITS.

- [ ] **Tests:** synthetic single-partition dataset → shapes + `WSUM == (weight*mask).sum()`; two-partition synthetic → band-node `DIRTY` equals sum of per-partition dirties. Confirm fail → implement → confirm pass → lint → commit `feat(imager): pass-2 per-partition gridding into .dt tree`.

---

## Phase 4 — `HessianTree` in `operators/hessian.py`

Class beside `HessPSF`. `H x = (1/Σ_p wsum_p) Σ_p B_pᵀ G_pᵀ W_p G_p B_p x + η x`, two backends sharing stored data: `mode="psf"` (per-partition `PSFHAT`+`BEAM` convolution, reusing the `r2c/c2r` pattern already in `hessian.py`) and `mode="exact"` (`Σ_p hessian_slice`). Constructor takes a list of per-partition dicts + `nx,ny[,nx_psf,ny_psf]`, `eta`, `mode`; `dot`/`hdot`.

- [ ] **Tests** (`tests/test_hessian_tree.py` or extend `tests/test_hessian_approx.py`): delta-PSF psf-mode ≈ identity (η=0); two identical partitions ≡ one (psf mode); exact mode single partition matches `hessian_slice/wsum`. Confirm fail → implement → confirm pass → lint → commit `feat(imager): add HessianTree sum-over-partitions operator`.

---

## Phase 5 — `rdt2fits` in `utils/fits.py`

Add `rdt2fits(store_url, column, outname, norm_wsum=True, do_mfs=True, do_cube=True, psfpars_mfs=None, otype=np.float32, ...)` beside `rdds2fits` (leave `rdds2fits` untouched). Reads band nodes from the `.dt` store (`xr.open_datatree`, `dt.children`), stacks along band per timeid, and reuses `set_wcs`/`save_fits`/`create_beams_table`. Provide a `@ray.remote` wrapper `rrdt2fits` mirroring `rdds2fits`'s remote.

- [ ] **Test:** build a tiny 2-band `.dt` with `DIRTY`+`WSUM`+attrs, call `rdt2fits`, assert the MFS FITS exists with shape and `sum(cube)/sum(wsum)` value. Confirm fail → implement → confirm pass → lint → commit `feat(imager): add tree-aware rdt2fits export`.

---

## Phase 6 — orchestrate in `core/imager.py` + CLI

In `core/imager.py`: keep geometry inline (compute `nband`/`freq_out`/`nx,ny,nx_psf,ny_psf`/`cell_rad`/`nx_pad,ny_pad`/`max_blength,max_freq` via the existing scan + `set_image_size` + padded-grid sizing). Replace the old `safe_stokes_vis` flat write with the extended `stokes_vis` writing into `<out>_<P>.scratch`. After pass 1: accumulate per-`(band,time)` counts from the scratch pieces, `reduce_counts(..., grouping)`, then dispatch the Phase-3 pass-2 workers to write `<out>_<P>.dt`, then `rrdt2fits`. Keep scratch by default (`keep_scratch`). Add CLI options in `cli/imager.py` (Annotated form): `weight_grouping`, `robustness`, `field_of_view`, `super_resolution_factor`, `cell_size`, `nx`, `ny`, `psf_oversize`, `epsilon`, `do_wgridding`, `double_accum`, `keep_scratch`, `fits_output_folder`, `fits_mfs`, `fits_cubes`; rename the `@stimela_output` to `dt-out` with implicit `{...}_{product}.dt`. Regenerate cabs.

- [ ] Wire + smoke-run; commit `feat(imager): wire two-pass DataTree pipeline + CLI`; regenerate cabs → commit `chore(imager): regenerate cab definition`.

---

## Phase 7 — integration tests

- [ ] `tests/test_imager.py`: end-to-end `.dt` smoke (band nodes with `DIRTY`/`WSUM`; partition children with `VIS/WEIGHT/MASK/UVW/FREQ/PSF/PSFHAT/BEAM`).
- [ ] single-field equivalence vs `init`+`grid` (peak-normalised MFS dirty within tolerance). The imager keeps arcae in worker subprocesses, so the pytest process can run both paths.
- [ ] scratch retained by default.
- [ ] Commit `test(imager): integration tests for the DataTree imager`.

Two-field mosaic summation is validated by the Phase-4 two-partition `HessianTree` test (the session MS is single-field).
