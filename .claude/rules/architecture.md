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

**Always place imports at the top of the file when possible.** Lazy (in-function) imports are only acceptable for:

1. **CLI modules** (`src/pfb_imaging/cli/`): to keep them lightweight.
2. **Optional heavy runtimes** (`ray`, `dask`/`distributed`) in library modules that are also usable without that runtime. Examples: `import ray` deferred to `PsiNocopytRay` and `BandWorkerPool` methods; `dask`/`distributed` deferred to `set_client`.
3. **Import-cycle breakers** — name the cycle in the comment. Existing cycles: `utils/misc` ↔ `utils/fits` (`load_fits`), `opt/pcg` ↔ `operators/hessian`, `operators/band_worker` ↔ `operators/hessian`/`operators/psi`.
4. **Serialisation/runtime constraints** — for example objects that break Ray/pickle serialisation of the enclosing function when captured at module scope (existing example: the ducc0 imports in `stokes2im.stokes_image`).
5. **Heavy imports on rarely-taken paths** — when the common path shouldn't pay the import cost (existing example: the sympy-pulling `utils/modelspec` import in `core/grid.py`, needed only when transferring a model).

(A **python-casacore-pulling** exception previously applied to the MSv4 imaging path; it was retired once arcae ≥ 0.5.2 made arcae and python-casacore coexist in one process — see wiki design-decisions D14. `africanus`/`daskms` imports now live at module scope like any other.)

**Every in-function import must carry a short inline comment stating why it cannot live
at module scope** (e.g. `# deferred: import cycle with operators.hessian` or
`# deferred: optional heavy runtime (ray)`). An undocumented lazy import looks like an
accident and will eventually be "fixed" back to top-level, reintroducing the cost it
was avoiding.

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

Operators are callable classes with `dot` (forward/analysis) and `hdot`
(adjoint/synthesis) methods. The composable deconvolution framework
(`pfb deconv`, issue #185) formalises its seams as `typing.Protocol` classes —
`LinearOperator`/`PsiOperator` (`operators/__init__.py`), `ForwardSolver`/
`BackwardSolver` (`opt/__init__.py`), `Regulariser`/`DeconvSolver`
(`deconv/__init__.py`). **Never introduce ABCs for these seams**; implementations
are plain classes satisfying the Protocols structurally, composed by
`deconv/pfb.PFBSolver` and the `deconv/presets.py` registry (issue #185). Math→code map
and the load-bearing numerical conventions (nu, wsum, λ schedule): `docs/wiki/deconv-primer.md`;
rationale ledger: `docs/wiki/design-decisions.md`.

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
* DUCC0 for gridding anOne thing to note is that the raylets d FFT.
* Dask for parallel chunk processing (`--nworkers`), threads for FFTs/gridding (`--nthreads`).
* Ray actors for process-level parallelism in wavelet operators.
* See `scripts/profiling.md` for profiling guides.

## 8. MSv4 DataTree Imager (`pfb imager`)

`pfb imager` is the MSv4 front-end. It reads MSv4 data via the `arcae` `xarray-ms` engine and
writes a single unified `xarray.DataTree`, replacing the legacy `.xds`+`.dds` split for this
path. Design rationale, `concat_row` semantics and known risks: `docs/wiki/imager-pipeline.md`.

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

**Tree layout** (`<out>_<PRODUCT>.dt`): one node per output image
`band{b:04d}_time{t:04d}`; one child `part{p:04d}` per data partition identified by
`(msid, field, spw, baseline_group)` (`baseline_group` is a single `"all"` group for now,
extensible for MeerKAT+ per-antenna-pair Mueller beams). Band nodes hold the summed image-space
products (`DIRTY`, `RESIDUAL`, `PSF`, `PSFPARSN`, `WSUM`); partition children hold the ragged
vis-space arrays (`VIS`, `WEIGHT`, `MASK`, `UVW`, `FREQ`) and per-partition `PSF`/`PSFHAT`/`BEAM`.
**Image-space arrays are (Y, X)-ordered end to end** — `.dt` dims `("corr", "y", "x")` etc.,
scratch beam `("corr", "m_beam", "l_beam")`; ducc's x-major world exists only behind zero-copy
`.T` views at the wgridder call sites (wiki design-decisions D19/D20).
`MODEL`/`NOISE` are added later by the (future) `deconv` consumer.

**Access layer — native DataTree only.** Use `xr.open_datatree(store)`,
`ds.to_zarr(store, group="band…/part…", mode="a")`, and `dt.children` directly. Do **not** add
`xds_from_url`/`xds_from_list`-style wrappers for the `.dt` (those remain only for the legacy
`.dds` consumers), and do not add one-level-deep shims around the native API.

**Memory discipline (Ray + MSv4) — battle-tested, do not regress.** Ray workers are long-lived;
anything a task leaves behind compounds across the run. Full story, measured numbers and the
local repro harness: `docs/wiki/memory-and-ray.md`.
* Never blanket-`.load()` an MSv4 node — it reads *every* correlated-data column
  (`VISIBILITY`, `CORRECTED_DATA`, `MODEL_DATA`, …). Load only the needed variables, extract
  to plain numpy, then release the Dataset *before* heavy processing (`stokes_vis` is the
  template).
* `gc.collect()` in a `try/finally` at every Ray-task boundary: deserialised xarray objects
  sit in reference cycles that refcounting cannot free.
* Evict xarray-ms's process-level Multiton table cache between tasks
  (`stokes2vis_msv4._release_ms_caches`): its 300 s *inactivity* TTL + per-partition cache
  keys mean a busy worker otherwise retains ~a task's read footprint per task, below Python.
* Pass-1/2 tasks return post-gc `{pid, rss_gb, peak_gb}`, printed in the progress lines.
  Read this before theorising about memory: ratcheting post-gc rss per pid = below-Python
  retention; flat rss with high peak = per-task transients.

**Time epochs.** The MSv4 `time` coordinate and the `.dt`'s `time_out` attrs are **unix
seconds**; the legacy `.dds` carries MSv2 **MJD seconds**. `utils/fits.set_wcs` takes
`time_is_unix=` — applying the wrong convention shifts FITS `DATE-OBS` by ~111 years (ERFA
"dubious year" warnings are the symptom).

**Deconvolution operators** (per output image; the embarrassingly-parallel `(band,time)` axis is
distributed by Ray, the sum over a band's partitions is not):
* `operators/hessian.HessianTree` — PSF-convolution Hessian summed over a band's partitions,
  using the stored `PSFHAT`/`BEAM` (cheap minor-cycle operator; preallocated FFT scratch so a
  future Ray actor can reuse one instance across iterations).
* `operators/gridder.residual_from_partitions` — exact degrid/grid residual reusing the stored
  per-partition inputs, **never recomputing the PSF** (per-major-cycle gradient). This mirrors
  the legacy `image_data_products`/`compute_residual` split and owns the exact path, so
  `HessianTree` is PSF-convolution only.
* `operators/band_worker.BandWorkerPool` — one Ray worker process per band co-locating all
  per-band deconv state (the band's `HessianTree` with in-worker CG, the wavelet jitclass, and
  the pinned gridding inputs for the exact residual). The `HessTreeRay`/`PsiNocopytRay` facades
  and the `pfb deconv` driver's residual step share one pool, so a run needs exactly nband
  workers; workers claim nominal (1e-2) CPUs because they are thread-pool-bound (a real claim
  can deadlock scheduling), and the driver sizes the local cluster's `num_cpus` to
  `max(nworkers, nband+1)` so worker startup is not throttled. Each worker reads its own band's
  vis-scale inputs (`UVW`/`WEIGHT`/`MASK`/`FREQ`/`BEAM`/`PSFHAT`/`DIRTY`) straight from the
  `.dt` store (`load_bands`), so that data never enters the driver or the Ray object store —
  the driver reads only image-scale cubes (`RESIDUAL`/`MODEL`/`UPDATE`), `WSUM` and attrs.

**arcae + python-casacore.** As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and
python-casacore coexist in one process, so the suite runs as a single `pytest tests/` and the
imager↔`init`+`grid` equivalence test runs both paths in-process. The historical
casacore-free discipline on the imaging path was retired (wiki design-decisions D14):
`africanus`/`daskms`/`casacore` imports follow the ordinary §3 rules — top-level unless a
documented §3 exception applies.
