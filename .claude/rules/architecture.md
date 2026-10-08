# Architecture & Domain Logic

Read this when editing `src/pfb_imaging/**/*.py` files.

## 1. CLI Architecture (hip-cargo Format)

* CLI uses Typer with `@stimela_cab` / `@stimela_output` decorators from hip-cargo.
* Each command lives in a separate file under `src/pfb_imaging/cli/` and is registered in `cli/__init__.py`.
* CLI modules must stay lightweight — lazy-import core implementations so `pfb --help` and cab generation don't pull in the scientific stack.

## 2. Typer Option/Argument Syntax

Canonical copy lives in `python-standards.md` §2, which loads on every `**/*.py` edit — a
superset of the files this page loads on. Section number kept so existing cross-references
resolve.

## 3. Import Style

**Always place imports at the top of the file when possible.** Lazy (in-function) imports are only acceptable for:

1. **CLI modules** (`src/pfb_imaging/cli/`): to keep them lightweight.
2. **Optional heavy runtimes** (`ray`, `dask`) in library modules that are also usable without that runtime. Examples: `import ray` deferred to `PsiNocopytRay` and `BandWorkerPool` methods; `dask`/`daskms` deferred to `utils/misc.construct_mappings`. (`distributed` left the dependency set entirely in #330, with the MSv2 `degrid`.)
3. **Import-cycle breakers** — name the cycle in the comment. Existing cycles: `utils/misc` ↔ `utils/fits` (`load_fits`), `opt/pcg` ↔ `operators/hessian`, `operators/band_worker` ↔ `operators/hessian`/`operators/psi`.
4. **Serialisation/runtime constraints** — for example objects that break Ray/pickle serialisation of the enclosing function when captured at module scope (existing example: the ducc0 imports in `stokes2im.stokes_image`).
5. **Heavy imports on rarely-taken paths** — when the common path shouldn't pay the import cost (existing example: the debug-only `pdb` imports in `opt/primal_dual.py`'s `primal_dual_numba`).

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

**Data flow:** MS -> `.dt` (Zarr) -> FITS.

1. `pfb imager` — two passes over MSv4 data (via arcae) into a single `xarray.DataTree`
   (`.dt`) plus a `.scratch` cache. See §8.
   `--baseline-groups` splits each partition into the `MM`/`MPM`/`MPMP` dish-pair
   classes, each with its own beam (#335, D46/D47). Off by default and
   byte-identical to not using it.
2. `pfb deconv` — composable deconvolution of the `.dt` (see §5). Writes `MODEL`/`RESIDUAL`/
   `NOISE`, plus `MODEL_MOPPED`/`RESIDUAL_MOPPED` when `--mop` is on (the default, D35).
3. `pfb restore` — the three explicitly-scaled restored products, apparent/intrinsic/mixed
   (D29). No longer uses Ray.
4. `pfb degrid` — degrid a `.mds` component model into MSv4 measurement sets (#278). Guards,
   the Ray Serve `Degridder` deployment and the driver live in `core/degrid.py`; the pure
   MSv4<->kernel seam is `utils/degrid.py`. All numerics go through
   `pfb_model_spec.utils.degrid`. Load-bearing decisions: D38-D42.

`pfb hci` is the separate high-cadence-imaging front-end.

**`degrid`'s option contract is MSv4-native and is not the MSv2 command's.** Selection is by
**name** — `--scan-names`/`--spw-names`/`--field-names`, not integer ids; chunking is
`--integrations-per-chunk`/`--channels-per-chunk`; the cluster is `--ray-address`. The MSv2
`degrid` was retired outright in #330 (D43), so an old recipe fails with an unrecognised-option
error rather than silently doing something else.

**Removed commands.** The legacy MSv2 subcommands (`init`, `grid`, `kclean`, `sara`,
`fluxtractor`) were retired in 0.1.0 (#277); their correctness coverage lives in the
ground-truth imager tests. `pfb model2comps` was removed in #286 — its portable
WSClean-FITS -> `.mds` path is now `pfbspec model2comps` in
[pfb-model-spec](https://github.com/landmanbester/pfb-model-spec), and the component-model
spec library and `.mds` writer are imported from `pfb_model_spec.utils`.

## 7. Performance

* Numba JIT with TBB threading for critical loops.
* DUCC0 for gridding and FFT.
* Dask for parallel chunk processing on the **`hci`** path only (`core/hci.py`, `utils/misc.py`,
  `operators/fft.py`, `utils/spi.py`, `utils/correlations.py`); threads for FFTs/gridding (`--nthreads`). The imager/deconv/degrid path is
  Ray, and `degrid`'s `--nworkers` sizes Ray Serve replicas, not a dask pool (#330 removed the
  last `distributed` consumer).
* Ray actors for process-level parallelism in wavelet operators.
* See `scripts/profiling.md` for profiling guides.

## 8. MSv4 DataTree Imager (`pfb imager`)

Two Ray-distributed passes over MSv4 data into a single `xarray.DataTree` (`.dt`) plus a
`.scratch` cache. **Tree layout, every stored variable and its dims, product selection
(`--psf`/`--beam`/`--fits-per-partition`), the two passes and the counts reduction:
`docs/wiki/imager-pipeline.md`.** Do not restate them here.

* **Native DataTree API only.** `xr.open_datatree`, `ds.to_zarr(group=…)`, `dt.children`.
  Do not add `xds_from_url`/`xds_from_list`-style wrappers for the `.dt` (those remain only
  for the legacy `.dds` consumers), and do not add one-level-deep shims around the native API.
* **Image-space arrays are (Y, X)-ordered end to end** — `.dt` dims `("corr", "y", "x")`,
  scratch beam `("corr", "m_beam", "l_beam")`. ducc's x-major world exists only behind
  zero-copy `.T` views at the wgridder call sites (D19/D20).
* **`.dt` times are unix seconds; the legacy `.dds` is MJD seconds.** `utils/fits.set_wcs`
  takes `time_is_unix=` — the wrong convention shifts FITS `DATE-OBS` by ~111 years (D13).
* **The stored `BEAM` is the effective response `B/n`** and every ducc call on this path stays
  `divide_by_n=False` (D22). Do not "fix" this; consumers wanting the bare primary beam must
  use `B = BEAM·n` or check `beam_includes_n`.
* **Baseline groups are a partition axis, not a new concept.** `--baseline-groups`
  populates the `baseline_group` slot that already exists in the partition
  identity; pass 2, the `.dt` schema and every deconv operator are unchanged.
  Classify antennas on `ANTENNA_DISH_DIAMETER`, never on `telescope_name` (it is
  broadcast from `OBSERVATION::TELESCOPE_NAME` and is identical for every
  antenna). The cross group's beam is stored as `Re(B)` (D46).
* **The tree is uniformly at `--precision`** except `UVW`/`FREQ` (always f8) and `MASK` (u1).
  ducc enforces this; it is not a memory optimisation (D27).
* **Memory discipline — do not regress** (`docs/wiki/memory-and-ray.md` has the measured
  story; D44 is the test-harness equivalent):
  * Never blanket-`.load()` an MSv4 node — it reads *every* correlated-data column. Load only
    the needed variables, extract to plain numpy, and release the Dataset before heavy
    processing (`stokes_vis` is the template).
  * `gc.collect()` in a `try/finally` at every Ray-task boundary: deserialised xarray objects
    sit in reference cycles that refcounting cannot free.
  * Never wipe xarray-ms's class-level Multiton cache: it is bounded and keyed per MS
    upstream, and a wholesale clear evicts every other Multiton in the process (`degrid`'s
    model). If an eviction is ever needed, use the keyed `Multiton.clear_cache(where=...)`.
  * Large arrays never travel as Ray task arguments or return values -- write them to the
    store and pass the path (D10; #339). They land in the object store and show as `shm`.
  * Read the per-task post-gc telemetry in the progress lines before theorising:
    ratcheting post-gc `anon` per pid = below-Python retention; ratcheting `shm` = arrays
    shipped through the object store; flat with a high peak = per-task transients.
* **Deconvolution operators.** `HessianTree` is PSF-convolution only;
  `gridder.residual_from_partitions` owns the exact degrid/grid path and **never recomputes
  the PSF**; `BandWorkerPool` workers claim nominal (1e-2) CPUs because a real claim can
  deadlock scheduling (D8), and read their own vis-scale inputs straight from the store so
  that data never enters the driver or the Ray object store (D10). The driver sizes the local
  cluster to `max(nworkers, nband+1)`. Rationale: `docs/wiki/imager-pipeline.md`.
