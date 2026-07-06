# Architecture & Domain Logic

Read this when editing `src/pfb_imaging/**/*.py` files.

## 1. CLI Architecture (hip-cargo Format)

* CLI uses Typer with `@stimela_cab` / `@stimela_output` decorators from hip-cargo.
* Each command lives in a separate file under `src/pfb_imaging/cli/` and is registered in `cli/__init__.py`.
* CLI modules must stay lightweight — lazy-import core implementations so `pfb --help` and cab generation don't pull in the scientific stack.

## 2. Typer Option/Argument Syntax (CRITICAL)

**NEVER** use `None` as a positional argument to `typer.Option()` — it causes `AttributeError`.

* **Required:** `Annotated[Type, typer.Option(..., help="...")]` (no `= default`).
* **Optional with default:** `Annotated[Type, typer.Option(help="...")] = default`.
* **Optional None:** `Annotated[Type | None, typer.Option(help="...")] = None`.

## 3. Import Style

**Prefer top-level imports.** Lazy (in-function) imports are only acceptable in:

1. **CLI modules** (`src/pfb_imaging/cli/`): to keep them lightweight.
2. **Optional heavy runtimes** (e.g. `ray`) in library modules that are also usable without that runtime. Example: `import ray` is deferred to `PsiNocopytRay` methods only.
3. **python-casacore-pulling imports on the MSv4 imaging path.** `africanus`, `daskms` and `casacore` transitively import python-casacore. As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) this no longer clashes with the `arcae` `xarray-ms` engine in one process, but the imager/gridding/FITS path still keeps these imports *out of module scope* — deferred into the functions that need them — to keep the path lightweight (fast CLI startup, lightweight install). Existing examples: the daskms imports in `construct_mappings` (`utils/misc.py`) and the `africanus.rime` imports in `interp_beam` (`utils/beam.py`). Prefer `from scipy.constants import c as lightspeed` (not `africanus.constants`) and `utils/misc.to_unix_time` (not `casacore.quanta.quantity`). See §8.

## 4. Cab Generation & Container Workflow

* Cab definitions in `src/pfb_imaging/cabs/*.yml` are auto-generated. **Never edit these files manually.**
* Generation: pre-commit hook, `update-cabs.yml` workflow on merge to `main`, or manual `hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs`.

### Image Tag Lifecycle

The single source of truth is `CONTAINER_IMAGE` in `src/pfb_imaging/_container_image.py`, loaded via `importlib` (no CWD dependency, no `uv sync` needed).

1. **Feature branches (manual):** Edit `_container_image.py` tag to match your branch name.
2. **Merge to main (`update-cabs` workflow):** Resets tag to `latest`, regenerates cabs, commits with `[skip checks]`.
3. **Releases (`tbump`):** Updates tag to semver via before-commit hooks.

### Execution Backends

Every CLI command gets `--backend` (`auto`|`native`|`docker`|`podman`|`apptainer`|`singularity`) and `--always-pull-images` options from hip-cargo. Both are marked `{"stimela": {"skip": True}}`. Volume mounts are resolved from type hints: input paths read-only, output paths read-write.

## 5. Mathematical Operators

Implement as callable classes with `dot` (forward/analysis) and `hdot` (adjoint/synthesis) methods. The Psi wavelet operator family (`Psi`, `PsiNocopyt`, `PsiNocopytRay`) follows this convention.

## 6. Processing Pipeline

Two front-ends produce the intermediary products consumed by deconvolution:

**Legacy (MSv2, python-casacore):**
1. `pfb init` — MS → vis-space Stokes datasets (.xds)
2. `pfb grid` — dirty images, PSFs, weights (.dds)

**MSv4 (arcae):**
- `pfb imager` — combines init+grid in two passes over MSv4 data into a single `xarray.DataTree` (`.dt`) plus a `.scratch` cache. See §8.

Shared downstream consumers (currently read the legacy `.dds`):
3. `pfb kclean` — classical deconvolution (Hogbom/Clark)
4. `pfb sara` — sparsity-constrained deconvolution
5. `pfb restore` — restore clean components
6. `pfb degrid` — subtract model from visibilities

**Data flow:** MS → `.xds`/`.dds` or `.dt` (Zarr) → FITS. Dask for lazy evaluation, distributed execution.

## 7. Performance

* Numba JIT with TBB threading for critical loops.
* DUCC0 for gridding and FFT.
* Dask for parallel chunk processing (`--nworkers`), threads for FFTs/gridding (`--nthreads`).
* Ray actors for process-level parallelism in wavelet operators.
* See `scripts/profiling.md` for profiling guides.

## 8. MSv4 DataTree Imager (`pfb imager`)

`pfb imager` is the MSv4 front-end. It reads MSv4 data via the `arcae` `xarray-ms` engine and
writes a single unified `xarray.DataTree`, replacing the legacy `.xds`+`.dds` split for this
path. Design/plan: `docs/superpowers/specs/2026-06-04-imager-datatree-design.md` and
`docs/superpowers/plans/2026-06-04-imager-datatree.md`.

**Two passes (both Ray-distributed):**
1. **Pass 1** — `utils/stokes2vis_msv4.stokes_vis`: reads raw MSv4 finely (per scan /
   `integrations_per_image`), converts to Stokes, averages, and writes fine pieces plus a
   per-piece uv `COUNTS` grid into a `.scratch` DataTree.
2. **Reduction** — the driver streams per-piece `COUNTS` directly into one grid per applied
   `weight_grouping` group (`per-band-time` default, `mfs`, `per-band`, `per-time`), holding
   `ngroups` grids rather than `nband*ntime` (`counts_key` in `core/imager.imager`;
   `utils/weighting.reduce_counts` documents the grouping semantics). Natural weighting is
   `robustness` `None` or `> 2` (no `natural` grouping; counts are skipped entirely).
3. **Pass 2** — `core/imager._grid_image` (one Ray task per output image): groups fine pieces
   into partitions, concatenates scans along `row`, grids each partition with
   `operators/gridder.grid_partition`, sums the image-space products over partitions into the
   band node, and writes the `.dt` tree. FITS via `utils/fits.dt2fits`/`rdt2fits`.

**Tree layout** (`<output>_<PRODUCT>.dt`): one node per output image
`band{b:04d}_time{t:04d}`; one child `part{p:04d}` per data partition identified by
`(msid, field, spw, baseline_group)` (`baseline_group` is a single `"all"` group for now,
extensible for MeerKAT+ per-antenna-pair Mueller beams). Band nodes hold the summed image-space
products (`DIRTY`, `RESIDUAL`, `PSF`, `PSFPARSN`, `WSUM`); partition children hold the ragged
vis-space arrays (`VIS`, `WEIGHT`, `MASK`, `UVW`, `FREQ`) and per-partition `PSF`/`PSFHAT`/`BEAM`.
`MODEL`/`NOISE` are added later by the (future) `deconv` consumer.

**Access layer — native DataTree only.** Use `xr.open_datatree(store)`,
`ds.to_zarr(store, group="band…/part…", mode="a")`, and `dt.children` directly. Do **not** add
`xds_from_url`/`xds_from_list`-style wrappers for the `.dt` (those remain only for the legacy
`.dds` consumers), and do not add one-level-deep shims around the native API.

**Deconvolution operators** (per output image; the embarrassingly-parallel `(band,time)` axis is
distributed by Ray, the sum over a band's partitions is not):
* `operators/hessian.HessianTree` — PSF-convolution Hessian summed over a band's partitions,
  using the stored `PSFHAT`/`BEAM` (cheap minor-cycle operator; preallocated FFT scratch so a
  future Ray actor can reuse one instance across iterations).
* `operators/gridder.residual_from_partitions` — exact degrid/grid residual reusing the stored
  per-partition inputs, **never recomputing the PSF** (per-major-cycle gradient). This mirrors
  the legacy `image_data_products`/`compute_residual` split and owns the exact path, so
  `HessianTree` is PSF-convolution only.

**arcae + python-casacore.** As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and
python-casacore coexist in one process, so the suite runs as a single `pytest tests/` and the
imager↔`init`+`grid` equivalence test runs both paths in-process. The whole imaging path
(`stokes2vis_msv4`, `operators/gridder`, `operators/hessian`, `utils/fits`, and the `misc`/`beam`
helpers they touch) is nonetheless kept casacore-free *by choice* (lightweight install, fast
startup) — keep it that way (see §3 for the import discipline).
