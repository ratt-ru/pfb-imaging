---
type: Design Ledger
title: Design decisions, known debt and recurring gotchas
description: Context/Decision/Rationale/Consequences ledger for pfb-imaging's load-bearing choices, plus the debt list and the gotchas that have already cost real debugging sessions.
tags: [design, decisions, debt, gotchas, ray, deconvolution, imager]
timestamp: 2026-09-08T10:00:00Z
last_verified_commit: 3699b79
---

# Design decisions, known debt and recurring gotchas

Each entry: **Context / Decision / Rationale / Consequences / Source**. Sources are
code, specs, PRs or commits. When you change something that invalidates an entry,
update it (and this page's `last_verified_commit`) in the same session.

## Decisions

### D1 — Protocols, not ABCs, at every algorithmic seam

- **Context:** The first gendeconv draft used ABC template-method inheritance
  (`SARABase`, `L21PrimalDual`, …) — an M×N (algorithm × regulariser) class explosion
  with duplicated reweighting bodies.
- **Decision:** Seams are `typing.Protocol` classes (`LinearOperator`, `PsiOperator`,
  `Regulariser`, `ForwardSolver`, `BackwardSolver`, `DeconvSolver`); implementations
  are plain final classes satisfying them structurally; `PFBSolver` does all wiring.
  Conformance enforced by `operators.require_protocol` (TypeError naming missing
  members).
- **Rationale:** Every regulariser in scope decomposes as `R(x) = g(Ψᵀx)`; both
  backward algorithms derive what they need from `(Ψ, prox_g, ν)`, so pairings are
  wiring, not classes.
- **Consequences:** New algorithm = regulariser + preset factory. Optional fast paths
  (`dual_update`, reweighting trio) are `hasattr`-sniffed, not Protocol members.
- **Source:** issue #185; architecture.md §5; `deconv-primer.md`.

### D2 — (Partially retired 2026-07-17) Legacy code served as the test oracle

- **Context:** Rewrites of numerical code need ground truth.
- **Decision (original):** `core/sara.py`, `core/kclean.py`, the `.dds` consumers and
  the legacy `opt` functions (`primal_dual`, `primal_dual_numba`, `pcg*`, `fista`) were
  not modified (behaviour-wise); new implementations were validated against them in a
  three-tier pyramid (unit rdiff < 1e-10, operator equality, e2e on a real MS).
- **Status:** the e2e oracles (`core/sara.py`, `core/kclean.py`, `init`+`grid`) were
  deleted in 0.1.0 (#277) once ground-truth tests against an injected sky replaced the
  equivalence tests (which were vacuous in CI — the downloaded MS had zero DATA). The
  frozen unit-level `opt` oracles (`primal_dual{,_numba}`, `pcg_numba`, `fista`) remain
  and are still not to be modified.
- **Rationale:** Mirrors the `init`+`grid` → `imager` strategy, which caught real bugs
  at every tier; ground truth beats equivalence once the legacy side must go.
- **Consequences:** Known legacy warts stay (see Debt); e2e comparisons must pin a
  shared `hess_norm` to remove power-method nondeterminism. Docstring-only additions
  to legacy code are fine.
- **Source:** spec above; `tests/test_deconv.py::test_deconv_matches_legacy_sara`.

### D3 — `nu = nbasis` for the SARA dictionary

- **Context:** PD step sizes divide by `nu = ‖ΨΨᵀ‖`; for a concatenation of `nbasis`
  orthonormal bases that is `nbasis`, but `L21`'s constructor default is the
  tight-frame 1.0.
- **Decision:** `make_sara` passes `nu=len(bases)` explicitly.
- **Rationale/History:** The omission survived a 13-task subagent build with review
  gates AND a tier-3 oracle test, because the unit test passed `nu=1.0` to both sides
  and the e2e ran single-band, which survives on stability margin. On a 4-band tree the
  backward solve diverged at the first reweighted iteration (peak residual ×3.7 per
  major cycle). Fixed in `7879817`.
- **Consequences:** Any new dictionary-style regulariser must set `nu` from its actual
  frame bound. Guard: `tests/test_pfb_solver.py::test_make_sara_sets_dictionary_nu`.
- **Source:** `deconv/presets.py`; legacy `core/sara.py` (`nu=nbasis` to `primal_dual`).

### D4 — Total-wsum normalisation convention

- **Context:** Image-space products are stored raw in the `.dt`; something must divide
  by the weight sum, consistently, across residual, Hessian and eta.
- **Decision:** Normalise by the TOTAL wsum across all bands at point of use:
  `residual/wsum_tot`; each band's `HessianTree` gets `wsum=wsum_tot` override;
  `HessianTree` consumes `abs(PSFHAT)`. **`--eta` is a fraction of the total wsum**:
  a uniform `+eta·x` on every normalised band operator (`eta·wsum_tot` raw), so its
  meaning is invariant to data volume and band count.
- **Rationale:** The wsum normalisation matches legacy sara (`wsums /= wsum;
  abspsf /= wsum`), keeping `rmsfactor` meaningful across paths. The eta scaling
  originally also matched legacy (`eta·wsum_b/wsum_tot` per band) but that made the
  damping shrink as bands were added (same data, more bands → ~eta/nband); with
  legacy sara retired, eta was redefined as a fraction of the total wsum (maintainer
  request, session pfb010). Single-band runs are numerically unchanged. Guard:
  `tests/test_pfb_solver.py::test_build_hess_eta_is_fraction_of_total_wsum`.
- **Consequences:** Feeding raw complex `PSFHAT` (or per-band wsums) into the Hessian
  produces garbage-scale or non-Hermitian operators — both happened during bring-up
  (e2e rdiff 8e7 before `daf94ab`).
- **Source:** `deconv/presets._build_hess`; `operators/band_worker.load_band`;
  `deconv-primer.md`.

### D5 — λ schedule: `init_factor` applies only to the very first iteration

- **Context:** Legacy sara computed `lam = init_factor·rmsfactor·rms` under
  `if iter0 == 0`, i.e. for EVERY iteration of a fresh run (and never on resume) — a
  bug masquerading as a schedule.
- **Decision:** Both drivers now use `iter0 == 0 and k == 0`.
- **Consequences:** Results before/after `52d5fb1` are not comparable at matched
  iteration count (pre-fix fresh runs effectively ran λ halved throughout).
- **Source:** `52d5fb1` (legacy fix); `core/deconv.py` lam line; found by diffing the
  old/new recipe logs (identical Iter 0–1, divergence from Iter 2).

### D6 — `reweight_active` means "stop at convergence"

- **Context:** The driver needs to know whether outer convergence should terminate the
  run or trigger reweighting instead.
- **Decision:** `PFBSolver.reweight_active` returns True when there is nothing left to
  trigger (no reweighting support, `l1_reweight_from < 0`, or already armed); the
  driver does `if not reweight_active: trigger_reweight() else: break`.
- **Consequences:** The name reads inverted ("active" ⇒ stop); do not "fix" the
  polarity without changing both sides.
- **Source:** `deconv/pfb.py::reweight_active`; `core/deconv.py` convergence block.

### D7 — `first()` is the preprocessing hook; `forward()` consumes its cache

- **Context:** The legacy driver applied the beam to the residual in `first()`.
- **Decision:** `DeconvSolver.first(residual)` caches the residual;
  `forward(residual)`'s argument is Protocol-shape only and is NOT read —
  calling `forward` before `first` raises RuntimeError. Since D23 the driver
  passes the **beam-attenuated gradient** (`BRESIDUAL/wsum`) to both.
- **Rationale:** Keeps a seam for cube-level residual preprocessing without
  smuggling it into the forward solver.
- **Source:** `deconv/pfb.py::first/forward`;
  `tests/test_pfb_solver.py::test_forward_requires_first`; Copilot thread on PR #269.

### D8 — Thread-pool-bound Ray workers claim nominal (1e-2) CPUs

- **Context:** Band workers are internally threaded (numba/FFT/gridder pools); Ray
  never preempts a scheduled actor. Real CPU claims that scale with nband exceeded the
  cluster's `num_cpus` and **silently deadlocked** the init `ray.get` (e.g. default
  `nworkers=1` with nband > 1).
- **Decision:** Flat `num_cpus=1e-2` per band worker; the deconv driver sizes its
  local cluster `num_cpus = max(nworkers, nband+1)` so the raylet's worker-startup
  throttle (`max(1, num_cpus)`) matches real demand.
- **Consequences:** Ray's CPU resource is bookkeeping on this path — parallelism is
  the per-worker thread budget (`nthreads // nband`). Guard:
  `tests/test_deconv.py::test_deconv_two_band_smoke` (timeout = deadlock regression).
- **Source:** `afab68e` (nominal claims); `4bbbbbe` (num_cpus sizing);
  `operators/band_worker.BandWorkerPool`; `core/deconv.py` init_ray comment.

### D9 — One band worker co-locates Hessian + Psi + residual

- **Context:** Separate `HessTreeRay`/`PsiNocopytRay` actor pools plus per-cycle
  residual tasks spawned ~3×nband processes (34 observed for nband=4), each with its
  own JIT warm-up, FFT plans and mostly-idle thread pool.
- **Decision:** `operators/band_worker.BandWorkerPool`: one `_BandWorkerImpl` actor per
  band owning all per-band state; `HessTreeRay`/`PsiNocopytRay` are thin facades over a
  shared pool; roles initialise on demand.
- **Rationale:** Within a band the three roles never run concurrently (the PD iteration
  and major-cycle phases are sequential per band), so co-location loses no parallelism
  and removes duplicated per-process state. Bands still parallelise fully.
- **Consequences:** 34 → 9 worker processes measured (4 actors + 5 idle prestarts),
  identical numerics. Per-PD-iteration driver↔worker traffic is unchanged (bands couple
  through the prox; that exchange is algorithmic).
- **Source:** `4bbbbbe`; architecture.md §8 band-worker bullet.

### D10 — Band workers read their own vis-scale inputs from the store

- **Context:** The driver (a port of the legacy shape) loaded every band's
  UVW/WEIGHT/MASK/FREQ/BEAM/PSFHAT/DIRTY into driver RSS, then copied it all into the
  Ray object store for the workers — two copies of data the driver never uses.
- **Decision:** `BandWorkerPool.load_bands`: each worker opens the `.dt` and
  selectively loads its own node (load → extract → release → gc). The driver reads only
  image-scale cubes (`RESIDUAL`/`MODEL`/`UPDATE`), `WSUM` and attrs. The in-memory
  construction path survives for tests/standalone (facades accept partitions directly).
- **Consequences:** Driver peak goes from O(vis data) to O(nband·nx·ny) at scale; on
  multi-node clusters workers read shared storage instead of the head node fanning data
  out. `wsums` must be passed to the presets by the driver (partition dicts are no
  longer driver-side).
- **Source:** `f6c8a80`; `tests/test_deconv.py::test_band_workers_load_matches_driver_side`.

### D11 — Copy Ray task arguments before in-place mutation

- **Context:** Ray deserialises task args as read-only zero-copy views;
  `pcg_numba` updates its `x0` in place. Warm-started forward solves crashed with a
  numba readonly-setitem TypingError (only from major cycle 2 — tests warm-start from
  zero and missed it).
- **Decision:** The band worker's `cg` copies `x0` before calling `pcg_numba`.
- **Consequences:** Any new worker method that passes a Ray argument into numba/in-place
  code needs the same treatment.
- **Source:** `5d955b7` (fixed the then-separate hess actor; the code has
  since moved); `operators/band_worker._BandWorkerImpl.cg`.

### D12 — Native DataTree API only for the `.dt`

- **Decision:** Consumers of the imager tree use `xr.open_datatree`,
  `ds.to_zarr(group=…, mode="a")` and `dt.children` directly — no
  `xds_from_url`/`xds_from_list`-style wrappers (those remain for the legacy `.dds`),
  and no one-level shims over the native API.
- **Gotcha bundled with it:** `to_zarr(mode="a")` replaces a group's **attrs
  wholesale** (variables merge, attrs don't) — write-back must start from the full
  original band attrs (`75d55d1` fixed silent attr loss).
- **Source:** architecture.md §8; `core/deconv.py` write-back comment.

### D13 — Time epochs: `.dt` is unix seconds, `.dds` is MJD seconds

- **Decision:** `utils/fits.set_wcs(time_is_unix=…)` selects the convention.
- **Consequences:** Applying the MJD→unix shift twice puts FITS `DATE-OBS` ~111 years
  off; ERFA "dubious year" warnings are the symptom.
- **Source:** architecture.md §8; `memory-and-ray.md` related conventions.

### D14 — (Retired 2026-07-15) The MSv4 imaging path stayed casacore-free by choice

- **Context:** Before arcae 0.5.2, arcae and python-casacore could not coexist in one
  process (hard segfault constraint), so the MSv4 imaging path deferred every
  `africanus`/`daskms`/`casacore` import into functions. After the coexistence fix
  (ratt-ru/arcae#211, #212) the deferrals were kept for a while as a
  lightweight-startup preference.
- **Decision (retired):** The preference was dropped once coexistence had soaked: the
  deferred casacore-pulling imports moved to module scope (`construct_mappings`'s
  daskms imports in `utils/misc.py`, `interp_beam`'s `africanus.rime` imports in
  `utils/beam.py`, `africanus.averaging` in both `stokes2vis` modules). In-function
  imports now need one of the documented reasons in architecture.md §3 (cycle,
  optional runtime, serialisation, rare heavy path), each stated in an inline comment.
- **Consequences:** No import-placement restriction remains on the imaging path. The
  lightweight CLI install is unaffected (CLI modules still lazy-import the core).
- **Source:** ratt-ru/arcae#211/#212; architecture.md §3/§8; branch `issue270`.

### D15 — Imager driver accumulates counts at `weight_grouping` granularity

- **Decision:** One counts grid per applied weighting group (`per-band-time` default),
  never per `(band,time)` node; natural weighting skips counts entirely.
- **Rationale:** Bounds driver memory at `ngroups` grids on wide-band runs.
- **Source:** `core/imager.py` (`counts_key`); `utils/weighting.reduce_counts`.

### D16 — Super-uniform weighting is a box-filter preprocessing of counts

- **Context:** Super-uniform (Briggs 1995) normalises each visibility by the counts
  summed over a `(2·npix_super+1)²` uv-box instead of its own cell.
- **Decision:** One function, `utils/weighting.box_sum_counts`, applied between
  `filter_extreme_counts` and `counts_to_weights`; `npix_super=0` is a no-op,
  bit-for-bit identical to standard uniform for any `robustness`.
- **Rationale:** The existing Briggs normalisation inside `counts_to_weights` then
  operates on the smoothed counts, yielding "super-robust" for free when `robustness`
  is set alongside `npix_super`. No changes to the counting or weighting kernels.
- **Source:** `utils/weighting.box_sum_counts`; formerly `core/grid.py`, now
  `utils/weighting` + `core/imager.py`; `tests/test_weighting.py`.

### D17 — weight_data closures hold only plain functions (numba cache safety)

- **Context:** `weight_data`'s `@overload` impl closed over sympy-lambdified njit
  dispatchers built at every overload resolution. A `Dispatcher`'s pickled bytes embed a
  per-process UUID, so the numba disk cache never hit: every fresh Ray worker paid a
  full compile (~3.5 s) *and* appended a new `.nbc`, growing `/tmp/numba` without bound
  (issue #273; #183 wanted the sympy machinery gone).
- **Decision:** Per-Stokes expression functions are pre-generated into
  `radiomesh.generated._stokes_expr` (radiomesh ≥ 0.1.2; diag-jones/minvar variants
  included) and `register_jitable`'d; the overload closure holds **only plain
  module-level functions and ints**; the outer njit is `cache=True`; every front-end
  call site (`stokes2im`, `stokes2vis`, `stokes2vis_msv4`) feeds `weight_data`
  C-contiguous readonly views via `utils/weighting.as_contiguous_readonly_view` so
  Ray's per-task readonly/writable mix doesn't multiply compiled signatures. The sympy
  derivation and its oracle test live in radiomesh
  (`radiomesh/tests/test_stokes_expr.py`), not here.
- **Consequences:** Never capture an njit `Dispatcher` in an `@overload` impl closure —
  it silently poisons the cache key. Measured: fresh-process first call 3.5 s → 0.15 s.
  Guard: `tests/test_weight_data_cache.py` (cross-process load-not-recompile).
- **Source:** PR #274; ratt-ru/radiomesh#81; issues #273, #183.

### D18 — Cache dirs (numba, meerkat-beams) default to per-user directories under /tmp

- **Context:** `NUMBA_CACHE_DIR` was hard-coded to `/tmp/numba` (Dockerfile ENV plus
  implicit cab outputs as mount hints). On shared hosts the first user to create it
  owned it; everyone else got cryptic `PermissionError`s (issue #270). Bare `/tmp` is
  no fix: numba nests `<srcdirname>_<sha1(source dir)>` subdirs under the cache root,
  and identical install paths (guaranteed inside containers) collide one level down.
  The meerkat-beams cache (`MBEAMS_CACHE_DIR`) later hit the same shared-ownership
  problem and follows the same pattern.
- **Decision:** `pfb_imaging/__init__.py` sets
  `NUMBA_CACHE_DIR=/tmp/numba-cache-<uid>` and `MBEAMS_CACHE_DIR=/tmp/mbeams-cache-<uid>`
  via `os.environ.setdefault`, and `set_envs` forwards both to child processes
  (including raylets). The Dockerfile ENV is gone. The implicit `numba-cache-dir` and
  `beam-cache-dir` cab outputs remain solely as mount hints (`write_parent` mounts
  `/tmp` read-write).
- **Rationale:** The package `__init__` runs before any submodule import, so the value
  is set before numba can be imported from any entry point — CLI, stimela-called core
  functions, Ray workers, tests — with no numba import deferrals. The cache root is
  hard-coded `/tmp`, **not** `gettempdir()`: the cab mount hints are static
  `/tmp/...` strings, and apptainer leaks the host `TMPDIR` into the container, so a
  `TMPDIR`-derived default can land on a path that is not mounted inside the
  container (a first `gettempdir()`-based iteration failed exactly this way; per-job
  `TMPDIR` isolation was deliberately given up for cab-mount consistency).
  `getuid()` survives containers where `$USER` doesn't; `setdefault` lets an explicit
  env (native export, stimela `backend.*.env`) win. Same-user concurrent runs share
  one numba cache safely (atomic temp-file + `os.replace` writes; stable keys since
  D17), so isolation is per-user only. A user-facing `--numba-cache-dir` option was
  rejected: stimela invokes the core functions directly, so a parameter arrives after
  module-level imports pulled numba in, and a cab default cannot be computed by
  stimela formulas.
- **Consequences:** Overriding a cache location is env-var-only, and `TMPDIR` does
  not move the defaults (pinned by `tests/test_numba_cache_dir.py`). A containerised
  override outside `/tmp` additionally needs its mount expressed on the stimela side
  (backend `env` today; the cab-level env mechanism when it lands).
- **Source:** issue #270; `src/pfb_imaging/__init__.py`; `Dockerfile`;
  `tests/test_numba_cache_dir.py`; commits `61d96f5`, `83be23f`.

### D19 — Image-space arrays on the hci path are (Y, X)-ordered end to end

- **Context:** The `hci` BeamWizard beam path historically carried a
  transpose+flip "hack to get the images to align" (`547458f`), later removed
  (`330bc5d`), and then bypassed reprojection entirely (`a516530`) during the
  jagged-beam-gain investigation (breifast#208). The hack compensated three real
  bugs in `reproject_and_interp_scat_beam` (transposed array feed, target
  `crpix` off by one, wrong target `CDELT1` sign) and was only approximately
  correct because the MeerKAT beam is nearly circular — measured errors: 4.3 %
  of peak (circular), 21 % (elliptical), rephasing offsets applied along the
  wrong axis; it also required square images.
- **Decision:** Cube/FITS **(Y, X)** order is canonical for every image-space
  array on the hci path — beam maps (`get_rotation_averaged_beam`, native since
  meerkat-beams `616906b`; `reproject_and_interp_scat_beam`, fixed to the
  measured reproject semantics with the 1D `l_beam`/`m_beam` coords, signed
  cdelt/crpix, target WCS = the hci output header) *and* `stokes_image`'s
  working arrays (`residual`/`psf`/`pbeam`) and cube outputs. **No data-moving
  transposes and no flips exist.** ducc's x-major world is confined to the
  `vis2dirty` call sites, which fill the `(ny, nx)` buffers through zero-copy
  transposed views (`dirty=buf.T`; ducc accepts strided output). The other
  x-major seam is `fitcleanbeam` — shared with the legacy `.dds` path, its PA
  convention defined by its input axes — called with `yx_order=True`, an
  explicit flag that adapts via an internal zero-copy view and returns
  identical parameters for either order.
- **Rationale:** Every layer keeps the index order its producer defines
  (astropy/reproject, the wizard, the cube and FITS are (Y, X); only ducc and
  legacy `fitcleanbeam` are x-major), so orientation is auditable at two
  explicit seams instead of smeared across compensating transposes and hacks.
  Conventions were pinned by measurement, not derivation:
  image-and-beam-orientation.md.
- **Consequences:** Non-square images work. The refactor was verified
  output-equivalent against the pre-refactor code on the test MS (cube/psf to
  single-precision threading noise ~1e-7; `psf_pa` bitwise). The `.dt` imager
  path has since followed (D20); only the legacy `.dds` reference code keeps
  wgridder (X, Y) arrays. The zarr-beam branch (`reproject_and_interp_beam` +
  its surviving hack + the feed→sky parity question) was the one exception on
  the hci path; D25 deleted it, so no transpose or flip remains there either.
  Changing any transpose/flip on this path must keep
  `tests/test_beam_orientation.py` green.
- **Source:** `src/pfb_imaging/utils/beam.py`;
  `src/pfb_imaging/utils/stokes2im.py` (`stokes_image`, `beam_for_band`);
  `src/pfb_imaging/utils/misc.py` (`fitcleanbeam`);
  `tests/test_beam_orientation.py`; image-and-beam-orientation.md; commits
  `547458f`, `330bc5d`, `a516530`; meerkat-beams `616906b` / PR
  landmanbester/meerkat-beams#8; ratt-ru/breifast#208.


### D20 — The imager+deconv (.dt) path is (Y, X)-ordered end to end

- **Context:** #277 makes (Y, X) canonical everywhere when the legacy
  subcommands are retired; the imager previously stored `.dt` image-space
  arrays x-major with dims `("corr", "x", "y")` and the FITS layer axis-swapped
  at write time.
- **Decision:** All image-space arrays on the imager+deconv path are
  `(..., ny, nx)` with `.dt` dims `("corr", "y", "x")` /
  `("corr", "y_psf", "x_psf")` / `("corr", "y_psf", "xo2")`, and the scratch
  `BEAM` is `("corr", "y", "x")` on the output image grid (placed there in
  pass 1; see D21). `nx`/`ny` keep meaning the X/RA and
  Y/Dec pixel counts everywhere — only array-axis order changed. ducc's
  x-major world exists only behind zero-copy `.T` views at the
  `vis2dirty`/`dirty2vis` call sites (input and output; both accept strided
  arrays), `fitcleanbeam` is called with `yx_order=True`, and
  `save_fits(yx_order=True)` writes without axis swaps. The `.mds` stays
  x-major — that convention is now **owned by pfb-model-spec** (whose
  `fit_image_cube`/`eval_coeffs_to_slice`/`model_to_ds` pfb-imaging imports since
  #286); pfb-imaging transposes to/from x-major at the `model_to_ds` (deconv) and
  `.mds`-read (degrid) call sites. A future `.mds` (Y, X) flip is a pfb-model-spec
  spec revision (landmanbester/pfb-model-spec#17), not a pfb-imaging change.
  uv-space grids (COUNTS, weighting) are untouched.
- **Rationale:** Same as D19 — one canonical order shared with
  FITS/astropy/reproject, auditable at explicit seams. Extending it to the
  `.dt` was gated on an on-disk schema change, which the 0.1.0 breaking
  release sanctions.
- **Consequences:** **`.dt` stores written by ≤0.0.x must be regenerated**
  (`pfb imager`) — release-notes line required. Old stores are not rejected on
  open: `x`/`y`/`x_psf`/`y_psf` still exist as dim *names*, and every read is
  positional, so without a guard a square-image pre-switch store would
  deconvolve with silently transposed rasters (model/residual/update FITS
  flipped about the diagonal). `core/deconv.py` therefore asserts
  `first.DIRTY.dims == ("corr", "y", "x")` on open and raises loudly instead.
  Verification method:
  the ground-truth tests (WCS positions/fluxes, brute-force DFT oracle,
  per-Stokes fluxes, deconv recovery — all written order-agnostically via WCS
  and dims names *before* the switch) pass unchanged across it, and non-square
  shapes are pinned in `tests/test_imager_pass2.py` and
  `tests/test_hessian_tree.py`. Known latent debt: the wavelet/psi stack's
  `nxmax`/`nymax` buffer conventions are crossed between the solvers
  (`(..., nymax, nxmax)`) and the band workers (`(..., nxmax, nymax)`) — masked
  by square images, pre-existing, unchanged by this switch.
- **Source:** commits `1a99dfb`, `0aac1d0`, `4b571e9`; `tests/test_imager.py`
  (ground truth + DFT oracle), `tests/test_imager_pol.py`,
  `tests/test_deconv.py`; spec/plan of 2026-07-17 (ephemeral).


### D21 — Mosaics rephase to a common tangent plane; --target is an in-plane offset

- **Context:** On-the-fly mosaicing (#1, #281) needs multiple fields on one
  grid. Ported from the abandoned `imager_rephase_and_interp_beam` branch
  onto the (Y, X) imager.
- **Decision:** Pass 1 rephases data+UVW to a common phase centre
  (`--phase-dir`, defaulting to the field barycentre for multi-field
  selections) BEFORE weighting/averaging/COUNTS, chgcentre-style
  (w-difference phase rotation). Both old and new UVW are synthesized through
  the same casacore-measures call and only the DIFFERENCE is applied — to the
  phases and to the stored coordinates (`uvw + (uvw_new - uvw_old)`) — so the
  measures-vs-MS earth-orientation systematic (~1e-5 relative, scaling with
  baseline length) cancels instead of decorrelating off-axis sources (#280
  remains open for a katpoint-based synthesis). `--target` shifts the image
  centre within the tangent plane via the existing `center_x/center_y`
  machinery; the off-centre PSF ramp is predicted adjoint-by-construction
  (dirty2vis of a unit delta), and `set_wcs` carries the offset as a CRPIX
  shift (CRVAL stays the tangent point — facet convention). **Beams are
  computed and placed on the output image grid in pass 1** (#281): the
  rotation-averaged beam (katbeam, or `BeamWizard.get_rotation_averaged_beam`
  for MeerKAT band names U/L/S0/S4 — signature kept stable so meerkat-beams
  can grow weighted time/freq averaging underneath) is evaluated about the
  FIELD's own pointing on a small grid, then SIN→SIN-reprojected onto the
  mosaic grid (tangent + target CRPIX shift, zero outside coverage) per
  piece, in parallel. Pass 2 consumes the stored `(corr, ny, nx)` BEAM as
  is (no beam interpolation in `grid_partition` any more); pieces of a
  partition share the field so the first piece's beam stands in — replace
  with a weighted mean when time-dependent beams arrive. Verified on 3
  MeerKLASS pointings: each partition's wizard beam peaks within half a
  pixel of its field's predicted position in the 5600² mosaic frame.
- **Rationale/pitfalls (hard-won):** (1) The epoch trap (D13):
  `synthesize_uvw` wants MJD seconds, MSv4 time is unix — `to_mjd_time` at
  the single call site. (2) **Frames about different tangent points are
  mutually rotated** by ~dra*sin(dec) to first order: a full-field pixel-wise
  round-trip comparison CANNOT converge (0.14 px displacement at a 0.5 deg
  radius for a 4 arcmin RA offset at dec 30). The old branch died
  misdiagnosing this as an "RA-axis geometry bug"; measured central-box floor
  is ~2e-4 of the dirty peak, i.e. the rephasing itself is numerically sound.
  Round-trip tests must compare the central box and WCS-mapped source
  positions, never full-field pixels.
- **Consequences:** `.dt` attrs: band/partition `ra/dec` = tangent point,
  `ra0/dec0` = the field's own pointing (kept for the #281 beam
  reprojection), `l0/m0` = target offset. Deconv consumers are unchanged
  (HessianTree/residual_from_partitions read the stored beams and l0/m0
  attrs). Acceptance on real data: `scripts/meerklass_mosaic.py
  --expect-aligned` (3 MeerKLASS OTF pointings; ghosts at the pre-rephasing
  positions collapse to ~1% of source; true positions carry the PB-weighted
  average flux — full mosaic gain needs the #281 beam weighting).
- **Source:** commit 502fe90 (port; original work 7dcc892/649c0ce/9bc16cc/
  ef9ae6e on the abandoned branch); `tests/test_imager.py`
  (rephase round-trip + stokes_vis rephase unit), `tests/test_coords.py`.


### D22 — The wgridder n-term is folded into the stored BEAM; divide_by_n stays False

- **Context:** The measurement equation carries a geometric 1/n(l,m) Jacobian
  (n = sqrt(1−l²−m²)) relative to the phase centre. It was historically
  ignored on this path (`divide_by_n=False` everywhere), biasing the
  deconvolved model by n (~0.2% at a 5° fov edge, ~1.5% at 10°) — relevant
  for wide UHF mosaics. Post-D21 rephasing all partitions share one phase
  centre, so n is a single well-defined function on the common grid.
- **Decision:** Pass 1 stores the **effective image-plane response**
  `BEAM = B/n` (including `1/n` when no aperture beam model is set), computed
  on exactly ducc's pixel coordinates (absolute w.r.t. the phase centre,
  `--target` offset included). Every ducc call on the imager+deconv path
  keeps `divide_by_n=False`. Pieces/partitions carry `beam_includes_n: True`.
- **Rationale** (measured; pinned by `tests/test_hessian_nterm.py`):
  1. **It is exact, not an approximation:** under `do_wgridding=True`, ducc's
     `divide_by_n=True` is precisely `diag(1/n)` on either side (verified
     2e-14), so `diag(B/n)·GᵀWG·diag(B/n)` with `divide_by_n=False` is the
     *identical* physical operator to flipping the flag with beam B.
  2. **It is the optimal Hessian approximation:** `HessianTree` applies
     `B̃ᵀ(PSF ⊛ B̃x)` — with `B̃ = B/n` the diagonal n-factors ride in the beam
     slots and are captured exactly; the folded operator matches the pure
     PSF-convolution baseline to 4 significant digits. Flipping
     `divide_by_n=True` instead buries an image-plane 1/n envelope inside the
     gridded PSF, which a convolution cannot represent: measured 4–25% worse,
     growing with fov.
  3. **It is trap-immune:** ducc's `divide_by_n` silently no-ops when
     `do_wgridding=False`; the fold divides explicitly, so behaviour is
     independent of the wgridding flag.
- **Consequences:** the deconvolved MODEL is in **intrinsic** flux (the
  legacy "reconstructs I/n, multiply by n afterwards" correction is gone —
  `tests/test_deconv.py::test_deconv_groundtruth` asserts intrinsic
  recovery). DIRTY/RESIDUAL are unchanged (no beam or n is ever applied on
  the imaging side). **Consumers must not treat the stored BEAM as the bare
  primary beam** — a future PB-corrected quicklook must use B = BEAM·n, or
  check `beam_includes_n`. `degrid`/`comps2vis` predates beams entirely and
  still predicts unattenuated model vis (pre-existing limitation, unchanged).
  Known residual approximation errors in the PSF-convolution Hessian, now
  documented: the w-term, the `abs(PSFHAT)` rectification (Hermitian-
  positivity for CG) — both far larger than the n-term at any fov, sub-
  percent for realistic decaying PSFs — and **PSF truncation** (see below).
- **PSF truncation vs preconditioner rate/stability (issue #287).**
  `nx_psf = good_size(psf_oversize·nx)` (`utils/misc.py`), default
  `psf_oversize=1.4`, i.e. the shipped PSF is **not** the `2·nx` that makes the
  periodic convolution aliasing-exact — the default is already truncated.
  Truncation degrades **only the preconditioner** (the gradient is exact
  degrid/grid, D23), so it changes convergence rate/stability, never the fixed
  point. Measured (coplanar, isolating truncation; issue #287 has the table):
  the exact-Hessian solution is recovered to ~1e-13 for every `psf_oversize ∈
  [1, 2]`; κ(M⁻¹H) grows ~5.5 (2×) → ~8 (1.4× default) → ~50 (1×); and once
  `λmax(M⁻¹H) > 2/γ ≈ 2.1` the driver's **fixed** `gamma=0.95` outer step
  diverges (measured at `psf_oversize ≲ 1.25`). The default 1.4× is stable but
  ~1.5× slower than an exact 2× PSF. Lowering `psf_oversize` for memory can
  silently cross into instability — a step-size guard is proposed in #287.
- **Source:** `tests/test_hessian_nterm.py` (operator identity + accuracy
  study); `tests/test_preconditioner_consistency.py` (fixed-point invariance;
  issue #287); `utils/stokes2vis_msv4.py` beam block; user-reported
  `divide_by_n`/`do_wgridding` trap.

### D23 — The forward solver consumes the beam-attenuated gradient (BRESIDUAL)

- **Context:** The data-term gradient of `½‖V − G(B·x)‖²_W` is
  `Σ_p B_p·GᵀW(V_p − G(B_p·x))` — it carries an **outer per-partition beam** the
  apparent (once-attenuated) residual lacks. The Hessian applies the beam on both
  sides (`H = B GᵀWG B`), so feeding it the apparent residual makes the update
  over-correct by ~`1/B` where the beam rolls off. Legacy sara did
  `residual *= beam` right before the preconditioner solve; the gendeconv rewrite
  reduced `first()` to cache-only and silently lost it (maintainer-spotted; the
  ground-truth tests run beam≈1/n and could not see it).
- **Decision:** Two residual products, per band. The **apparent** residual
  `Σ_p r_p` remains the user-facing one (FITS, λ/rms schedule, `RESIDUAL`).
  The **gradient** residual `BRESIDUAL = Σ_p B_p·r_p` is what
  `first()`/`forward()` consume. Because it is not derivable from the apparent
  sum when partitions carry distinct beams (mosaics), pass 2 stores
  `BDIRTY = Σ_p B_p·dirty_p` (the model-free term) and
  `residual_from_partitions(..., bdirty=…)` accumulates both residuals in one
  sweep; deconv writes `BRESIDUAL` back for resume.
- **Rationale:** Exact per-partition attenuation (not a band-average
  approximation) at negligible cost — pass 2 has each `dirty_p` in memory and the
  residual loop already visits every partition. λ/rms stay on the apparent
  residual, matching legacy (rms was computed before `residual *= beam`).
- **Consequences:** `.dt` trees without `BDIRTY` are refused ("re-run pfb
  imager"); resuming a deconv started before this change needs a restart
  (`MODEL` without `BRESIDUAL` is refused). Debug aid: `pfb deconv
  --fits-per-partition` writes per-partition dirty/residual/apparent-model FITS
  (re-gridded from the stored `VIS` worker-side, chi2 stats in the headers) to
  localise mosaic misfits to specific partitions; `--debug` additionally logs
  per-partition vis-space chi2 every major iteration and writes the chi2
  trajectories plus baseline-binned residual profiles to
  `<fits_oname>_<suffix>_debug.json`. Guards:
  `tests/test_imager_pass2.py::test_residual_gradient_beam_applied_twice`,
  `tests/test_deconv.py::test_band_workers_load_matches_driver_side` (distinct
  per-partition beams), `test_deconv_requires_bdirty`;
  `tests/test_preconditioner_consistency.py` (with `rmsfactor=0`/`positivity=0`
  the preconditioned cycle's fixed point is the exact-Hessian solution — an
  apparent-vs-beam-attenuated gradient bias would move it, plus an e2e
  noise-floor smoke).
- **Source:** legacy `core/sara.py:280` (`residual *= beam`, 7eb3f1d~1);
  `operators/gridder.residual_from_partitions`; `core/imager._grid_image`;
  `core/deconv.py`; `deconv/pfb.py::first`.


### D24 — hci transient injection: fringe sign, differential rephasing, 1/n, w-term sign

- **Context:** `pfb hci --inject-transients` adds analytic point-source
  transients into the visibilities before imaging
  (`utils/stokes2im.stokes_image`). The source is built in the ORIGINAL
  (field-centre) frame at the MS uvw — so a per-field beam can be applied
  there — then carried to the rephased frame when `--phase-dir` is set. Four
  separate convention traps live in that ~15-line block; each mis-places or
  mis-scales injected sources and none is caught by imaging real data. Two of
  them (the rephasing sign, and the source w-term sign) independently drove
  "localisation error grows with distance from the phase centre" reports
  (breifast#263).
- **Decision:** (1) **Fringe sign.** The data is rephased by
  `exp(+freqfactor·w_diff)` (`freqfactor = -2πi·f/c`); the injected fringe is
  applied as `exp(-freqfactor·phase)`, so `w_diff` must enter the injection
  phase with a **minus** (`phase = -w_diff`) to carry the source with the *same*
  rotation the data got. `+w_diff` leaves the source **coherent** but displaced
  by a constant `-2·(field→tangent)` translation (a whole-image shift, not
  decorrelation). In a mosaic each field then shifts by 2× its offset from the
  common tangent, so the error grows with distance from centre — the #263
  signature. (2) **Differential rephasing** (mirror D21 / #280): synthesize BOTH
  the old and new uvw through the same `synthesize_uvw` call and apply only the
  difference — `w_diff = w_new − w_ref`, `uvw = uvw + (uvw_new − uvw_ref)` — while
  the injection's `uvw_old` stays the **MS's own** uvw. hci previously diffed the
  synthesized new-centre w against the MS's *recorded* w and replaced uvw
  wholesale (`uvw = uvw_new`), leaking the measures-vs-MS earth-orientation
  systematic (~1e-5 of the baseline length, scaling with it) into both the phase
  and the sampling. (3) **1/n.** The RIME point-source visibility is
  `I/n·fringe`; injection now scales `dspec /= n0t` (`n0t = √(1−l²−m²)`, a
  per-source scalar; imaging is `divide_by_n=True`). Amplitude-only — small
  on-axis, growing towards the field edge / at low declination. (4) **Source
  w-term sign.** The source's own w-term is `phase += uvw_old·(n0t−1)` (a
  **plus**). The whole fringe is written in the conjugate convention
  `exp(-freqfactor·phase)`, opposite to `psf_vis`/`explicit_wdegridder`'s
  `exp(+freqfactor·(…−w(n−1)))`. The l/m terms stay consistent because `x0t/y0t`
  are **non-negated** here (vs `psf_vis`'s negated `x0/y0`), but `(n0t−1)` has no
  coordinate to flip, so it must be **added** to match the wgridder forward
  model. Subtracting it (the original code) leaves the source coherent on-axis
  but drifts it off-axis in proportion to `w·(n−1)`.
- **Rationale:** With full uv coverage the injection lands on the correct pixel
  *and* the cube's `RA---SIN`/`DEC--SIN` WCS maps that pixel back to the injected
  `(ra, dec)` to <0.2 px out to 0.5° (guard test), for *both* the sign convention
  and the differential — so traps (1)/(2) were rephasing-only, and a *constant*
  −2× shift is the fingerprint of the rephasing phase applied with the wrong
  sign. The differential is the same measures-vs-MS reasoning as D21. Trap (4) is
  different: it is coverage-dependent. Full synthesis averages `w·(n−1)` down to
  sub-pixel (so the full-synthesis guards below never saw it), but a
  **single-integration snapshot** (`integrations_per_image=1`) at low declination
  is a nearly coplanar array with large correlated w, where the wrong w-sign
  drifts an off-axis source several pixels growing with distance from centre.
- **Consequences:** Traps (1)/(2) changed only the `--phase-dir`/mosaic path.
  Trap (4) changes any run with significant w — negligible for full-synthesis
  imaging, multi-pixel for `hci` snapshot cubes. **Corollary for downstream
  debugging (superseded):** an earlier version of this entry said that offset
  transients on a *single-field* run must be downstream (breifast/WCS) rather
  than in this injection. Trap (4) disproves that — a growing single-field
  offset in a snapshot cube *is* this injection. The reliable discriminator is
  coverage, not field count: reproduce with full uv coverage (error vanishes ⇒
  injection/gridder; error persists ⇒ downstream). Guards:
  `tests/test_hci.py::test_hci_inject_transients_location_vs_distance` and
  `::test_hci_inject_transients_rephased` (full synthesis; catch traps 1–3), plus
  `tests/test_hessian_approx.py::test_inject_transient_fringe_wterm` (snapshot;
  catches trap 4, drifts ≥2 px before the w-sign fix).
- **Source:** commits bb76c03 (1/n), 68d7f19 (sign + tests), 1909bfa
  (differential); the w-term sign fix + snapshot guard this session;
  `utils/stokes2im.stokes_image`; breifast#263, pfb-imaging#280.


### D25 — `hci` beams are meerkat-beams only, with the band named separately

- **Context:** `hci --beam-model` accepted either a MeerKAT band name
  (`U`/`L`/`S0`/`S4`, which built a `BeamWizard`) or a path to an MdV zarr beam cube.
  The zarr branch of `beam_for_band` carried the pre-D19 reproject bugs and the
  transpose+flip hack, its feed→sky parity was unresolved, and the wizard branch of
  the transient-injection beam was an outright `NotImplementedError` — so one option
  selected between one maintained path and one half-correct one.
- **Decision:** `--beam-model` names a *model*, and `meerkat-beams` is the only
  accepted value (anything else raises `NotImplementedError`); the band moves to its
  own `--primary-beam-band` (`U`/`L`/`S0`/`S4`), required when a beam model is given.
  `--beam-model` unset still means no beam correction at all. The zarr branch is
  deleted from both `beam_for_band` and the transient injection, and the transient
  beam is implemented on the wizard path by `beam_gain_for_source`.
- **Rationale:** a deliberate stop-gap. Additional beam models are planned to land
  *inside* meerkat-beams rather than as more branches here, so pfb keeps one beam path
  — one place where orientation, Stokes ordering and parity have to be right (D19) —
  and inherits new models through the wizard. Splitting the band out is what the
  stop-gap needs anyway: overloading a single option for both model and band is what
  made "no beam" and "MeerKAT L-band" indistinguishable to the validation code.
- **Consequences:** callers pass two options instead of one; a beam model with a
  missing or invalid band raises `ValueError` up front rather than deep in a Ray task.
  `utils/beam.reproject_and_interp_beam` is now uncalled (see Known debt). MdV
  feed→sky parity becomes entirely meerkat-beams' question. If a second model ever
  needs a different band vocabulary, `--primary-beam-band` is the option to revisit.
  **Gotcha:** the deprecation branch was first written as
  `if beam_model is not None and beam_model.lower() != "meerkat-beams": raise … else: assert band …`,
  whose `else` also catches `beam_model is None` — it needs three branches, not two.
- **Source:** `src/pfb_imaging/core/hci.py`; `src/pfb_imaging/cli/hci.py`;
  `src/pfb_imaging/utils/stokes2im.py` (`beam_for_band`, `beam_gain_for_source`);
  `tests/test_hci.py`; image-and-beam-orientation.md §5, §7.

### D26 — `--eta-mode` shapes eta over the image, in the preconditioner only

- **Context:** on a real MeerKAT mosaic (750², 4.6° field, 3 bands/3 fields,
  `psf_oversize=2`, beam on) `scripts/max_gamma.py` measured
  `lambda_max(M^-1 H_exact) >= 14.5` at the default `--eta 1e-3`, i.e. the major cycle
  diverges for any `gamma > 0.14` — the default `gamma=0.95` blows up after three
  descending cycles (issue #287). The dominant eigenvector is the near-Nyquist ripple
  seen in real images: 100% of its power in the outer half of the field, 49% above
  half-Nyquist. Its cause is a curvature *mismatch*, not a small `eta`: along that mode
  `v'BCBv/wsum = 3.7e-3` against `v'H_exact v = 6.9e-2`, so **M under-estimates the
  curvature 18x** and `eta` supplies only 21% of `v'Mv`. Uniform `eta` cannot fix it —
  reaching `gamma=1` needs 31x more damping *at that mode*, and applying that
  everywhere flattens M into a scaled identity.
- **Decision:** `--eta-mode` (default None = uniform, unchanged) selects a spatially
  varying `e(x)` with dynamic range `--eta-cap` (default 100): `invbeam`/`invbeam2`
  (`1/B_eff`, `1/B_eff²` on the wsum-weighted mosaic of `B_p²`), `radial`
  (`1 + (cap-1)r²`, `r` in field half-widths) and `radial-invbeam`. Every mode is
  normalised so `e == eta` where the operator is trusted, so `--eta` keeps its meaning
  and `lambda_min(M)` cannot drop. `e` enters **`M` only** — it is absent from
  `gridder.residual_from_partitions`, so the fixed point and the flux scale are
  untouched and a profile only damps each forward update. Built inside each band worker
  from that band's own beams (`operators/hessian.eta_profile`), so no `(ny, nx)` array
  crosses Ray.
- **Rationale:** `lambda_max(M^-1 H)` is monotone decreasing in `M` in the PSD order, so
  raising `e` where `M` is untrustworthy lowers it. Measured (same `.dt`, `eta=1e-3`):

  | eta-mode | lambda_max | gamma_max | CG per solve |
  |---|---|---|---|
  | uniform | 14.48 | 0.138 | 20.9 s |
  | `invbeam` cap=100 | 11.47 | 0.174 | 12.2 s |
  | `radial` cap=100 | 3.07 | 0.652 | 9.3 s |
  | `radial` cap=300 | 2.20 | 0.910 | 9.2 s |
  | `radial` cap=1000 | 1.41 | 1.421 | 9.5 s |

  **Beam-shaped profiles barely help and radial ones do**, which is not obvious: uniform
  `eta` already acts like an effective `eta/B²` (that is why enabling the beam helps at
  all), so the maximiser has already relocated to where the beam is *large* — 90% of its
  power above `B_eff = 0.39`. `invbeam` gives the same 1.3x at cap 10, 100 *and* 1000:
  its dynamic range is spent in the skirt where the mode has no power. The w-mismatch
  instead grows with distance from the **tangent point** and the beam does not oppose it
  there, so a radial profile is the one aimed at the actual mode.
- **Consequences:** end-to-end at the default `gamma=0.95` on that mosaic, uniform `eta`
  reaches rms 2.6e-2 at cycle 3 then diverges (rms x5.2, x5.7, peak 46), while
  `--eta-mode radial --eta-cap 1000` descends monotonically to rms 1.7e-2 with `eps`
  still falling. The diverged model swings ±106 with 7.2% of its power beyond `b_max`;
  the profiled model is +6.2/−0.17 with 0.0%. **CG gets cheaper, not dearer** (2.2x),
  because the profile compresses M's spectrum where it is smallest — the earlier
  "no CG cost" expectation was pessimistic. `lambda_max(M)` rises only 1.68 → 1.98, so
  the backward step's `hess_norm` barely moves. Costs: `||update||` roughly halves, so
  the outer field cleans more slowly per cycle (bought back many times over by a 7x
  larger `gamma`), and `r` is measured from the image centre — with `--target` the
  tangent point is offset by `l0/cell` pixels, which the radial modes ignore. This damps
  the instability; it does not fix `M`. A w-aware or faceted preconditioner is the
  actual fix (#287); `--eta-mode` prices how much of the divergence damping alone can
  reach, so a better `M` has a number to beat.
- **Band consistency:** `radial` is pure image geometry, so `e` is **bit-identical across
  bands**; the beam-driven modes are frequency-dependent and are not. This matters even
  though `e` cannot bias the fixed point: bands couple *only* through the L21 prox (D3),
  so a band-dependent `e` damps some bands' updates more than others and the joint
  sparsity decision is taken on a model whose spectral shape is still converging — a bias
  at any finite iteration count. A band-uniform profile is the safe default;
  band-uniformity for a beam-driven mode would need a driver-side reduction of `B2_eff`
  across bands (each worker sees only its own band). Both halves pinned by
  `tests/test_eta_profile.py::test_radial_is_identical_across_bands_and_beam_modes_are_not`.
  **Since D30** this profile is no longer only a diagonal: `--gp-length-scale` reads `e(x)`
  as a per-pixel inverse signal variance and couples bands through
  `D^½C⁻¹ₙD^½`, making `M` the second band-coupling channel after the L21 prox. The
  workers still apply `e(x)` exactly as described here; the coupling is a driver-side
  remainder.
- **Inspecting it:** with `--eta-mode` set, `deconv` writes `<oname>_<suffix>_eta.fits`
  once per run — a `(band, corr, ny, nx)` cube (band on the FREQ axis, so band-to-band
  variation is visible; a 3D array would land it on STOKES because `to4d` prepends) with
  `ETAMODE`/`ETA`/`ETACAP` in the header, and logs the range plus the band-to-band
  spread. Note the radial modes saturate at the cap on the *inscribed circle* (`r = 1`),
  so the corners out to `r = sqrt(2)` are all clipped to `eta*cap` — with cap=300 that is
  ~15% of `lambda_max(M)`, i.e. the corners are heavily damped by design.
- **Source:** `src/pfb_imaging/operators/hessian.py` (`eta_profile`, `ETA_MODES`);
  `src/pfb_imaging/operators/band_worker.py`; `src/pfb_imaging/deconv/presets.py`;
  `src/pfb_imaging/cli/deconv.py`; `scripts/max_gamma.py` (`denominator_report`);
  `tests/test_eta_profile.py`; issue #287.

### D27 — The imager tree is uniformly at `--precision`; ducc allows no dtype mixing

- **Context:** `--precision single` crashed pass 2 with a bare ducc assertion —
  `get_OptNpArr(...) [with T = float]: incorrect data type` — even though pass 1 had
  correctly written `VIS` c4 / `WEIGHT` f4. `grid_partition` allocated its output buffers
  with the builtin `float` (f8), and `_grid_image`'s band accumulators did the same, so a
  single-precision run mixed f4 data with f8 buffers and (once the buffers were fixed)
  still wrote an f8 band node onto f4 partitions.
- **Decision:** Every float array the imager stores follows the requested precision, in
  both the `.dt` and the `.scratch`; **`UVW`/`FREQ` stay f8** (ducc takes those as double
  regardless) and `MASK` stays u1. Buffers are never allocated with a hardcoded dtype:
  `grid_partition` derives `real_type` from the stored `VIS` and `_grid_image` from the
  first scratch piece, so the data is the single source of truth. `--double-accum` is a
  wgridder-**internal** control (its accumulation registers) and must not leak into a
  stored dtype.
- **Rationale:** ducc0's wgridder templates *all* of its real/complex arrays on one type
  `T` — vis, wgt and the dirty/psf buffer must agree, or it raises rather than upcasting.
  Precision is therefore a whole-tree invariant, not a per-array choice: any f8 array
  reaching a wgridder seam alongside f4 data is a hard error, whichever side is "wrong".
  Accumulating the partition sums at the data precision (rather than f8) is what the flag
  asks for, and costs little: single vs double agree to ~1e-6 of peak on `DIRTY`/`BDIRTY`
  and ~2e-6 on `PSF` at `--epsilon 1e-5`, with `WSUM` matching to 4e-8 — well inside the
  gridding accuracy that `epsilon` already concedes.
- **Consequences:** `--precision single` halves the tree on disk and in every downstream
  read. Single precision needs `--epsilon >~ 1e-5` (ducc's f4 kernels); the FITS path is
  unaffected because `save_fits` casts to f4 anyway. **The deconv consumer is not yet
  single-precision safe** (debt below): `residual_from_partitions` sizes its buffers from
  the band `DIRTY` while the model cube arrives f8, so the same assertion fires at
  `gridder.py`'s exact-residual seam. Read the assertion as "some array in this call
  disagrees with the others" and print the dtypes — the template parameter (`T = float`
  vs `T = double`) tells you which side ducc believed.
- **Source:** `src/pfb_imaging/operators/gridder.py` (`grid_partition`);
  `src/pfb_imaging/core/imager.py` (`_grid_image` accumulators, MFS PSF reduction);
  `src/pfb_imaging/utils/stokes2im.py` (the hci path's `real_type`, the pattern this
  restores); `tests/test_imager_precision.py`.

### D28 — `freq_out` is the effective (weighted) frequency, reduced at three levels

- **Context:** `freq_out` was the band-edge midpoint from a `linspace` over the frequency
  span, but channels are sliced *by count* and assigned to the nearest midpoint, so the
  label was wrong by up to half a channel whenever `nband` did not divide `nchan` (#296).
  Worse, `stokes_vis` evaluated the **primary beam** at that same value: the beam was being
  *computed* at the wrong frequency, not merely reported at one. Flagging makes it worse
  still — a fully flagged channel moves a band's centre of mass by a whole channel width,
  which a frequency-uniform grid cannot represent at all.
- **Decision:** `freq_out` means the **weight-weighted mean frequency of the channels
  actually gridded**, reduced at three levels:
  1. **piece** (`stokes_vis`, pass 1) — `Σ w·mask·ν / Σ w·mask` over the post-averaging
     channel axis, using **natural** weights. This value evaluates the beam *and* is stored.
  2. **partition** (`_concat_pieces`, pass 2) — `wsum_nat`-weighted mean of the pieces'
     frequencies, and of their `BEAM`s.
  3. **band** (`_grid_image`, pass 2) — weighted by the **imaging** `prod["WSUM"]`.
  The band-edge midpoints keep their sole remaining role, band *assignment*, renamed
  `band_centres` (driver) / `freq_nominal` (parameters and attrs) so one name no longer
  means two things. `freq_nominal` is also the fallback when no weight survives.
- **Rationale:** A band product is the wsum-weighted sum of its partitions, so its effective
  frequency is the wsum-weighted mean of theirs — the identical reduction `beam_sum` already
  performs, which is *why* level 3 must use the imaging weights: `BEAM` and `freq_out` must
  not be weighted by different quantities at the same level. Level 2 uses natural weights
  because it runs before gridding, where robust weights do not exist; obtaining them would
  mean holding every piece's `BEAM` resident through gridding instead of freeing it
  (~67 MB per piece at 4096² f4), against `memory-and-ray.md`, to correct a beam that varies
  slowly with frequency. **The mixed basis is deliberate — do not "fix" it into a
  regression.** `freq_out` stays scalar, reduced with corr-summed weights, matching what
  `dt2fits` already does for `freq_mfs`; per-correlation flagging differences are
  second-order for continuum and a per-Stokes frequency axis has no home in FITS.
- **Consequences:** Band frequencies **change value** for uneven splits — measured on
  `tests/data/test_ascii_1h60.0s.MS` at `--channels-per-image 3`: +16.5, +99.9 and
  +33.2 MHz on 100 MHz channels (the middle band dominated by one fully flagged channel).
  Anything comparing against an older `.dt` or FITS will see it. Frequency accumulators are
  **f8 regardless of `--precision`** (f4 resolves 1e9 Hz only to ~64 Hz — cf. D27, where
  everything else follows the data dtype). `dt2fits` orders cube planes by `bandid`, not
  `freq_out`, because data-dependent frequencies could otherwise invert and silently
  reorder planes. `_concat_pieces` also fixes a latent bug independent of #296: the
  partition used to inherit piece 0's `BEAM` wholesale, which was already wrong for
  `BeamWizard` (evaluated with the piece's own timestamps). That path is not exercised by
  current data — `timeid` is keyed on `(scan_name, block)`, so a `(band, time)` node holds
  one scan — but is deliberate future-proofing for MeerKAT+ baseline groups.
  **Still open (#302):** FITS cubes keep a linear `CRVAL3`/`CDELT3` and so cannot represent
  a non-uniform frequency axis; per-plane `FREQ%04d` cards (or `--fits-split-bands`) are a
  separate follow-up.
- **Source:** issue #296; `src/pfb_imaging/utils/stokes2vis_msv4.py` (`stokes_vis`);
  `src/pfb_imaging/core/imager.py` (`_concat_pieces`, `_grid_image`, `band_centres`);
  `src/pfb_imaging/utils/fits.py` (`dt2fits` sort key); commits 44bfbab, 090711e, 39cfb6c,
  5111b13; `tests/test_imager.py::test_stokes_vis_beam_follows_effective_freq`,
  `::test_imager_effective_freq_uneven_bands`,
  `tests/test_imager_pass2.py::test_concat_pieces_weighted_beam_and_freq`.

### D29 — Restore names its flux scale; the MFS clean beam is fitted to the MFS PSF

- **Context:** `restore` was the last `.dds` consumer (#303). Porting it to the `.dt`
  exposed two latent errors. First, the band `MODEL` is intrinsic flux (the forward solve
  fits `V ≈ G(B·m)`, D22/D23) while the band `RESIDUAL` is apparent, once-attenuated flux
  (`operators/gridder.residual_from_partitions`); legacy `restore_image` added them
  directly, which is only correct where `B ≈ 1`. Second, with one PSF per data partition
  the restoring beam was undefined, and the MFS beam was taken as the *mean of the
  per-band fitted Gaussians* — a value with no referent when bands are not homogenised.
- **Decision:** Restore emits three separately-named products, selected by CLI letter and
  stored as distinct band variables: `BIMAGE` (`a`/`A`) `= (B̄·m) ⊗ G + r/wsum`, apparent
  throughout; `IMAGE` (`i`/`I`) `= m ⊗ G + r/(wsum·B̄)`, intrinsic throughout and zeroed
  below `--pb-min`; and `KIMAGE` (`k`/`K`) `= m ⊗ G + r/wsum`, the legacy mixed product,
  kept for visualisation and left as the default. The restoring beam is `PSFPARSN` per
  band — already the fit to the wsum-weighted average of the band's partition PSFs — and
  `fitcleanbeam(Σ_b PSF_b / Σ_b WSUM_b)` for the MFS.
- **Rationale:** `Σ_p dirty_p / Σ_p wsum_p` has effective response exactly `B̄`, the stored
  band `BEAM`, so `BIMAGE / B̄ = IMAGE` is an identity rather than an approximation and
  neither product misstates its scale. For the MFS beam, `r_mfs` *is* the wsum-weighted
  sum of the band residuals, whose effective PSF is the wsum-weighted sum of the band
  PSFs; fitting that sum makes the FITS `BMAJ`/`BMIN` exact with no cross-band
  homogenisation. Averaging band beams is wrong because it is a mean of fits to
  *different* PSFs, a quantity nothing in the image is shaped like — and it is
  **weight-blind**: a band contributing 1% of the weight moves it as much as a band
  contributing 99%. Measured on two circular PSFs at 6 and 18 px FWHM with weights 100
  and 1, `clean_beam` returns ≈6.07 (the heavily-weighted band dominates the summed PSF,
  correctly) while the mean of the per-band fits returns 12.
  **Do not "demonstrate" this with equally weighted Gaussians.** For those the two agree
  closely (6/18 equally weighted gives 12.33 vs 12.0), because `fitcleanbeam` is an L2 fit
  over an `nsigma=10` window, area-dominated at large radius — *not* a half-power width.
  The half-power width of that sum is ≈9.4, which is what makes the naive argument look
  compelling and why it was wrong in the original spec. The divergence that matters comes
  from weighting and from real PSFs with sidelobes.
- **Consequences:** The MFS restored image is **not** the weighted sum of the per-band
  restored images (different resolutions), so `dt2fits` cannot produce it and the driver
  computes it directly — the one place restore does not reuse `dt2fits`. `--drop-bands`
  must filter the `G_mfs` PSF sum, not merely the image sum, or a dropped band still sets
  the restoring beam; the same applies to fully flagged bands, which are skipped
  altogether because `RESIDUAL / WSUM` on a zero wsum puts inf/NaN into the stored products
  *and* into the MFS accumulators. `--outputs i`/`I` changes meaning: same letter,
  intrinsic product; the default moved to `kK` so a default run is unchanged. Restore no
  longer uses Ray (the work is FFT-bound and the driver holds the cubes for the MFS
  anyway), so `--nworkers` and `--ray-address` are gone. `convolve2gaussres` gained
  `yx_order` because `gaussian2d`'s position angle is not transpose-invariant and the `.dt`
  is `(corr, y, x)` (D19/D20). Dropped bands are omitted from cubes rather than zeroed,
  which can leave a non-uniform FITS frequency axis — restore warns and defers to #302.
- **Source:** issue #303; `src/pfb_imaging/core/restore.py`;
  `src/pfb_imaging/utils/restoration.py`; `src/pfb_imaging/utils/misc.py`
  (`convolve2gaussres`); `src/pfb_imaging/utils/fits.py` (`dt2fits` `drop_bands`,
  `psfpars_var`); commits 84dd88b, 8501dee, 4b41d26, a4a6b08, 7b66502, d0bdd6b;
  `tests/test_restore.py`, `tests/test_convolve2gaussres.py`, `tests/test_fits_tree.py`.

### D30 — The preconditioner's frequency prior generalises `--eta` to an nband×nband precision

- **Context:** on a real wide-field mosaic the primary beam tapers the *forward update* at
  the edges of the field, worst at the top of the band, and structure visible in the
  residual never reaches the model (issue #307). The forward step already solves
  `B†HB + K⁻¹` with `K⁻¹ = η·I` (`HessianTree.dot`), so exploiting smoothness in frequency
  is a matter of replacing that scalar with a matrix — not a new mechanism.
- **Decision:** `--gp-length-scale` (default None = off, unchanged behaviour) and
  `--gp-cap` (default 10). The preconditioner becomes
  `M x = M_data x + D^½ C⁻¹ₙ D^½ x`, with `D = diag_b(e_b(x))` the existing
  `eta`/`eta_mode` profile and `C⁻¹ₙ` an `nband×nband` normalised precision. `e(x)` is read
  as the **per-pixel inverse signal variance**, so `K = diag(σ)(C ⊗ I)diag(σ)` with
  `σ² = 1/e` and `K⁻¹_{bb'}(x) = sqrt(e_b e_b') C⁻¹_{bb'}`. Applied through the identity
  `D^½C⁻¹ₙD^½ = D + D^½(C⁻¹ₙ − I)D^½` so the **band workers are untouched** and the driver
  adds only the remainder. Squared exponential on a **linear** frequency metric, length
  scale a fraction of the band span. Zero-`wsum` bands take part (at their `freq_nominal`
  fallback, D28) and are interpolated by the prior.
- **Rationale — the normalisation is the load-bearing part.** `prec = min(λmax/λ, cap)`
  then `prec /= prec.max()`, anchoring `η` to the **roughest frequency mode present** and
  relaxing smoother modes by up to `cap` ("relax"). Three alternatives were rejected:
  - *A spatially uniform `(K⁻¹ − ηI)` coupling on top of `diag(e)`.* Under
    `--eta-mode radial --eta-cap 1000` the diagonal is `1000η` against a coupling of
    `η(cap−1) ≈ 100η`, so the prior becomes **10× more white than correlated exactly at
    the corners where it is needed** — backwards. It is also not any GP's precision
    matrix, so the hyperparameter has no interpretation.
  - *"Tighten" (`[η, η·cap]`, `η` on the smoothest mode).* Makes the field-edge update
    *smaller*, not larger, so it does not address the reported symptom; and
    `λmax(P) = η·eta_cap·gp_cap = 100` against the `λmax(M) ≈ 1.98` D26 measured, i.e. a
    50× `hess_norm` inflation and `√50 ≈ 7×` the CG iterations.
  - *Dividing by `cap` instead of `prec.max()`.* A white kernel has a flat eigenspectrum,
    every ratio is 1, and the result is a uniform `I/cap` — silently weakening `eta`
    everywhere instead of degrading to today's behaviour at `ℓ → 0`.
  A **log-frequency metric was also rejected**: writing `ν = ν̄(1+a)`,
  `log ν_i − log ν_j ≈ (ν_i−ν_j)/ν̄`, so linear *is* the first-order expansion of log and
  over a full 2:1 band the two differ by 4% (`log 2 = 0.6931` vs `2/3 = 0.6667`) — far
  inside the ambiguity in `ℓ`. The GP is over linear flux, so a log metric would not
  encode power laws anyway; that needs a GP over `log S` vs `log ν`, which is nonlinear
  and unavailable (the operator must stay linear and PSD for CG).
- **Consequences:**
  - **The forward CG moves from band-parallel in-worker to cube-level on the driver**
    (`HessTreeRay.cg` branches on the prior). The FFT work is unchanged and still happens
    in the workers; only `cg_maxit` round trips are added, against the `pd_maxit` the
    backward step already pays per major cycle. **Measured cost is not negligible:** on
    `subset_withbeam_I.dt` (3 bands, 750², 3 partitions/band) a `max_gamma` power
    iteration — one exact sweep plus one CG solve — went **28 s → 115 s (4.1×)**. The
    exact sweep is common to both, so the whole difference is in the forward solve.
  - **That difference is three effects, not one**
    (`scripts/profile_freq_correlated_hessian.py`, same tree, 7 threads/worker,
    `--cg-tol 1e-3 --cg-maxit 150`; one forward solve **5.8 s → 20.0 s, 3.44×**):
    1. *the fast path is lost* — **+1.4 s (1.25×)**, one round trip becomes 95, at ~38 ms
       of Ray overhead each (a `pool.hess_dot` costs 70 ms against 31 ms of FFT work);
    2. *conditioning* — **+3.8 s**, CG goes 95 → >150 iterations (it hits `cg_maxit`, so
       the 3.44× is a **floor**). This is `λmin(M)` dropping by up to `gp_cap`, and it is
       the one cost a per-`dot` benchmark cannot see. **Slower CG is the designed
       behaviour, not a bug** — the prior only ever *removes* curvature from `M` (the
       roughest mode is anchored at `η` and smoother ones are relaxed toward `η/cap`), so
       `cond(M)` rises by up to `gp_cap` and CG needs ~`√gp_cap` more iterations. It
       shows up only where the data term has no curvature of its own: on a fully sampled
       toy the same prior costs 20 → 22 iterations, on one with unsampled uv cells
       72 → 240. Note this is the **opposite** of `eta_profile`, which is normalised so
       `λmin(M)` cannot degrade — the two knobs pull opposite ways on CG. Pinned by
       `test_prior_lowers_lambda_min_only_where_the_data_lacks_curvature` and
       `test_prior_needs_more_cg_iterations_and_that_is_expected`;
    3. *costlier dots* — originally **+8.4 s**, of which only ~0.5 s was the coupling
       arithmetic. The rest was **BLAS spin**: `k = nband` is tiny and `n = npix` huge,
       so `np.matmul` is memory-bound, but OpenBLAS spread it over every core and those
       threads then busy-polled for ~100 ms (`THREAD_TIMEOUT`) — straight through the
       *next* `ray.get`, while the workers needed the cores. Proven by inserting a sleep
       between the matmul and the round trip: driver CPU during the round trip decayed
       1598 → 1499 → 1242 → 705 → 14 ms as the sleep went 0 → 5 → 20 → 50 → 100 ms.
       **Fixed** by `gauss.eta_freq_mul` (below): the solve is now **6.3 s → 12.7 s
       (2.03×)** and this term is +0.7 s.
  - **The coupling term is `operators/gauss.eta_freq_mul`, a fused numba kernel**, not
    numpy. Band-major inside a 2048-pixel tile: band-major makes the inner loop
    unit-stride and vectorisable, the tile keeps a band slice of `out` in L2 across the
    `(b,c)` loops, and without the tile the kernel re-streams `out` `nband` times and
    loses to numpy above ~1024². At 8 bands × 4096² numpy takes 287 ms on 11.5 cores
    against 83 ms on 15.6. It holds **no scratch cubes** — the numpy form's two
    `(nband, ny, nx)` buffers were 7.6 GB at 8 × 8000². Numba's TBB pool does not spin
    into the next round trip (measured to 22 threads). `rarg_numba_patterns.load_data`
    was tried and rejected: gathering a pixel's band column into a tuple blocks
    vectorisation, and the result neither vectorises nor parallelises.
  - **At production size the binding cost is the cube-level CG's transfer and driver
    memory, not the coupling term.** Measured at 8 × 8000² (a 3.81 GB f8 cube), on an
    echo actor with the FFT removed: one cube-level round trip is **1.63 s** — `ray.put`
    of the band slices is 0.28 s and dispatch 0.15 s, so the *return* path (worker
    result → plasma → the driver's `out[b] = res[0]` memcpy) dominates. The worker's
    read of a task argument *is* zero-copy (`writeable=False`, a plasma view), but the
    round trip is not zero-copy end to end: there are three copies per band slice and
    only that one is free. So a 150-iteration forward solve moves **1.12 TB** through
    the object store and spends **~4 min in transfer alone** before any gridding. The
    band-parallel path pays that once, not 150 times. Driver memory is the other half:
    `pcg_numba` holds 7 cubes (`b, x, r, p, xp, rp, aopp`) = **26.7 GB** at that size,
    against 3.3 GB per worker for the in-worker path. `BandWorkerPool.hess_dot`
    allocates its output with `np.empty_like`, not `zeros_like` — every band is
    overwritten, and zeroing cost 1.07 s per call at this size.
  - `λmax(D^½C⁻¹ₙD^½) = λmax(D)` **exactly**, so the prior's own contribution to `M`'s
    spectrum has the same ceiling as the `η·I` it replaces. Pinned by
    `test_prior_stats_report_the_spectrum_and_its_contribution_to_m`. Note this **bounds**
    `λmax(M)`, it does not fix it: `C⁻¹ ⪯ I` gives `M_gp ⪯ M`, so `λmax` can only fall.
    It does fall, because the eigenvectors do not align and `M`'s top mode is
    frequency-flat — precisely the mode relaxed to `1/gp_cap`. The drop is at most
    `η(1 − 1/gp_cap)`, ~9e-4 against the `λmax(M) ≈ 1.98` D26 measured, so `hess_norm`
    and the primal-dual step sizes move by <0.1% and in the conservative direction
    (a norm that is too large means steps that are too small). The cache keys on
    `gp_length_scale`/`gp_cap` regardless, so nothing rests on the approximation.
  - `λmin(M)` drops by up to `gp_cap`, eroding the stable `γ`. The erosion is bounded by
    the `η` fraction of `v'Mv` along the maximising direction — D26 measured 21% for the
    binding mode, predicting `14.7 → 18.2` (24%) at `gp_cap=10`, not 10×.
    **Measured** on `subset_withbeam_I.dt` at `--eta 1e-3 --gp-length-scale 0.5
    --gp-cap 10` (precision spectrum `[0.107, 1.000]`, i.e. the cap fully saturated):
    `λmax(M⁻¹H_exact)` **14.48 → 17.15 (+18.5%)**, so `γ` must shrink 15.6%
    (`0.124 → 0.105`). The bound held and was slightly pessimistic. `λmax(M)` was
    **unchanged (1.654 → 1.667, +0.8%, within the power-method tolerance)** — the
    empirical confirmation of the `prec.max()` normalisation. **Measure with
    `scripts/max_gamma.py` before raising the cap.**
  - **The erosion lands in a mode the solver does not travel in, and the step it does
    take gets longer.** The maximising eigenvector stayed pinned at the field edge but
    moved to a much rougher spatial mode (high-frequency power fraction 0.49 → 0.83,
    `η`'s share of `v'Mv` there 21.1% → 12.7%), while the Rayleigh quotients along the
    stored `UPDATE` and `DIRTY` — the practically binding directions — were unchanged
    (0.798 → 0.800, 0.819 → 0.820). Solving `u = M⁻¹·BRESIDUAL` both ways on the same rhs:
    the update rotates **28.8°**, its norm grows **1.62×**, and per band the gain is
    1.54 / 1.57 / **1.74** ascending in frequency — largest at the top of the band, which
    is exactly the symptom #307 reported. Net of the 15.6% `γ` cut that is **≈1.37×
    effective step overall and ≈1.47× in the top band.** The update's power moves into
    the smoothest frequency eigenmode (41% → 59%) and out of the roughest (25% → 9%), so
    genuinely rough spectra are approached *more slowly*; by D22 that is a rate effect,
    never a bias.
  - The driver-side remainder is **negative** semi-definite (`C⁻¹ₙ`'s eigenvalues are in
    `(0, 1]`); the total operator is still symmetric positive definite, so CG applies, but
    nothing may assume that term alone is PSD.
  - Interpolated flux in zero-`wsum` bands enters L21's 2-norm over bands, so a
    joint-sparsity decision can be taken partly on a band with no data. Accepted: the loop
    already extrapolates into those bands via `model_to_ds`'s `nbasisf` refit.
  - **The fixed-point test cannot guard the prior's sign.** Flipping `dot`'s `+=` to `-=`
    yields `M_data + D + D^½(I − C⁻¹ₙ)D^½`, still SPD, so D22 says it reaches the same
    fixed point — and it does (verified). The prior term's value is pinned separately
    against an explicit dense formula by `test_prior_term_matches_the_dense_congruence`,
    and structurally by `test_prior_matches_the_kronecker_spectrum`: with one partition
    shared by every band, `M_data = I⊗A` and `P = ηC⁻¹ₙ⊗I` commute, so the whole spectrum
    must be `α_k + η·p_j`. That closed form catches the sign flip, a one-sided congruence
    (dropping either `D^½`), a missing `−I`, and any band/pixel axis mix-up in the
    `reshape(nband, -1)` — all four verified by mutation.
  - `hess_norm` is now cache-keyed on the M-defining options
    (`eta`, `eta_mode`, `eta_cap`, `gp_length_scale`, `gp_cap`), fixing a **pre-existing
    bug**: changing `--eta` between runs on the same `.dt` silently reused a stale norm.
    Trees written before the key existed carry no `hess_norm_opts` and are re-estimated.
    **Expect this once per existing tree:** the first `deconv` run on a `.dt` written
    before this PR misses the cache by construction and pays a power-method estimate
    (a handful of cube-level round trips at production size). It is logged —
    "Preconditioner options changed since the cached hess_norm was written;
    re-estimating" — and it is a one-off, not a per-run regression.
  - Second band-coupling channel in `M` (see D26's band-consistency note — bands
    previously coupled only through the L21 prox, D3).
  - **Attribution:** `--nbasisf < nband` already smooths the *model* in frequency every
    major cycle (`core/deconv.py`, `model_to_ds`). Run GP experiments at the default
    full-rank `nbasisf` or the two effects cannot be separated.
- **Source:** issue #307; `src/pfb_imaging/operators/hessian.py` (`freq_correlation`,
  `freq_precision`, `HessTreeRay`); `src/pfb_imaging/deconv/presets.py`;
  `src/pfb_imaging/core/deconv.py` (`_M_OPTS`, `_m_signature`, `_cached_hess_norm`);
  `src/pfb_imaging/cli/deconv.py`; `tests/test_freq_precision.py`,
  `tests/test_hess_tree_ray.py` (including the Kronecker-spectrum, band-vs-pixel coupling,
  spatially-varying-eta congruence and CG-iteration guards), `tests/test_deconv_hess_norm_cache.py`,
  `tests/test_pfb_solver.py`, `tests/test_preconditioner_consistency.py`,
  `tests/test_deconv.py::test_deconv_driver_runs_with_the_frequency_prior`;
  `scripts/max_gamma.py --gp-length-scale/--gp-cap` (the γ measurements above, on
  `subset_withbeam_I.dt`, `--eta 1e-3`, 11 vs 9 power iterations to `--tol 5e-3`);
  `scripts/profile_freq_correlated_hessian.py` (the cost decomposition above);
  `src/pfb_imaging/operators/gauss.py` (`eta_freq_mul`), `tests/test_eta_freq_mul.py`.

### D31 — Resolution changes use the closed-form Gaussian transform ratio, never a sampled division

- **Context:** `convolve2gaussres` took an image from resolution `gausspari` to `gaussparf`
  by FFT-ing both sampled `gaussian2d` kernels and dividing (`convkernhat[msk] =
  gausskernhat[msk] / thiskernhat[msk]`, masked only by `> 0.0`). Issue #312 found this
  displacing sources: a delta at (170, 260) restored to (146, 260), with −0.81 of peak in
  ringing. It reaches users through `restore_products` whenever a restoring resolution is
  asked for — `pfb restore --gausspar` and the lowest-resolution mode.
- **Decision:** the ratio of two Gaussian transforms is itself a Gaussian, so write it
  down: `Kf(k)/Ki(k) = sqrt(|Σf|/|Σi|)·exp(−2π²·kᵀ(Σf−Σi)k)`, evaluated directly on the
  padded rfft grid (`gauss_cov`, `gauss_ratio_hat`). `gaussian2d`'s support returns to
  `nfwhm` major-axis FWHMs, and the parameter is renamed from `nsigma` to say so. A
  non-positive-semi-definite `Σf − Σi` now raises `ValueError` instead of being applied.
  The `gausspari=None` path (a plain convolution by the sampled kernel) is untouched.
- **Rationale:** there were two independent error sources and the support width only fixes
  one. (1) Truncating at 5 σ leaves a step of `exp(−12.5) = 3.7e-6` of peak, whose spectral
  ripple dwarfs the transform's own ~1e-14 floor near Nyquist; at 5 FWHM (11.8 σ) the step
  is ~4e-31 and the ripple is gone. This is the half issue #312 diagnosed, and the half
  spimple fixed (`landmanbester/spimple@27a4bc8`). (2) A *sampled* Gaussian's DFT sinks
  into FFT round-off before Nyquist regardless of support, so past that point the quotient
  is noise over noise — and unbounded, since pfb's mask is `> 0.0`, not spimple's `> 1e-10`.
  Source (2) gets *worse* as the beam is better sampled: measured on a band-limited sky at
  `--super-resolution-factor` 2/3/4 with a matched 2·srf px beam, the two-step invariant
  (`sky→gi→gf` vs `sky→gf`) broke by 4.0e-10, 1.8e-3 and 1.4e-4 of peak. The closed-form
  ratio has neither problem (8.1e-10 at srf 2, ~6e-16 above — a kernel-aliasing floor),
  conserves flux exactly (the sampled division was off by +6% at a 6 px beam and −25% at
  12 px), and costs two FFTs less per plane.
- **Consequences:** the support width no longer carries correctness for the deconvolution
  path — the remaining `gaussian2d` callers only ever *multiply*, where a 3.7e-6 truncation
  step is harmless — but it is kept wide for spimple parity and because nothing gains from
  narrowing it. The new check immediately exposed that `restoration.lowest_resolution`
  had been producing invalid targets all along — rewritten in D32. `nsigma` survives
  in `fitcleanbeam`, where it does mean standard deviations.
- **Source:** issue #312; landmanbester/spimple#50 and `landmanbester/spimple@27a4bc8`
  (where the same regression was found first); the `gaussian2d` regression entered in
  `1bde45d` (#218), which fixed a genuine width bug and narrowed the support in passing;
  `src/pfb_imaging/utils/misc.py` (`gauss_cov`, `gauss_ratio_hat`, `convolve2gaussres`,
  `gaussian2d`); `src/pfb_imaging/utils/restoration.py`;
  `tests/test_convolve2gaussres.py::test_convolve2gaussres_preserves_position_when_deconvolving`,
  `…_conserves_flux_through_a_resolution_change`, `…_two_step_matches_direct_across_srf`,
  `…_refuses_to_sharpen`, `…_rejects_xy_ordered_grids`.

### D32 — The common restoring resolution is a Loewner envelope, not a max of axes

- **Context:** `--gausspar 0 0 0` homogenises every band to a common resolution — the
  input a spectral-index fit needs (spimple). `restoration.lowest_resolution` built that
  target as `nanmax(emaj)`, `nanmax(emin)`, `nanmean(pa)`. D31's semi-definiteness check
  turned what had been silent corruption into a visible failure, and it fires on real
  data: of six representative band sets, only the one with perfectly aligned position
  angles produced a valid target.
- **Decision:** the target is the smallest ellipse that dominates every input in the
  **Loewner order** (`Σf − Σj ⪰ 0` for every input `j`), which is the exact condition for
  `convolve2gaussres` to be a convolution from all of them. Candidate shapes are formed
  from the max axes at each of several orientations (the circular mean of the input PAs,
  plus each input's own PA) crossed with five axis ratios from the widest input's to
  circular; each is inflated by the smallest scalar that makes it dominate — the largest
  generalised eigenvalue of the pencil `(Σj, shape)`, closed form for 2×2 — and the
  smallest resulting ellipse wins. `resolution_deficit` exposes the same quantity so the
  explicit `--gausspar` branch can fail early with the viable floor instead of failing
  inside an FFT. The MFS native beam is now part of the input set.
- **Rationale:** "at least as wide as every input" is a statement about covariances, not
  about the axes separately — a rotated ellipse pokes out diagonally, so matching axis by
  axis is necessary but **not sufficient**. (Concretely: eigenvalues 4 and 1 at 45° project
  to 2.5 on both coordinate axes, and `2.5·I` does not dominate it.) Three separate
  defects were in that one line. (1) Rotation, above. (2) `nanmean` on position angles,
  which are defined mod π: PAs of 0.05 and π − 0.05 are near-identical orientations that
  average to π/2, orthogonal to both — the circular mean on the doubled angle fixes it.
  (3) The MFS beam is `fitcleanbeam(Σ_b PSF_b)`, a fit to the summed PSF and not an
  average of the band fits (D29), so nothing bounds it by the per-band envelope, yet it is
  reconvolved to the same target. The candidate sweep matters because neither extreme is
  right alone: with aligned PAs the max-axis ellipse is exactly optimal (bit-identical to
  the old answer, area ratio 1.000), while over a ~1.5 rad PA spread a *circular* beam is
  tighter than any scaling of the elongated one (2.00× the old area rather than 2.79×).
- **Consequences:** with aligned PAs nothing changes. Otherwise the restoring beam grows —
  measured at 1.02× to 2.0× in area over the six cases — which is a real resolution cost
  and the honest price of a target every band can actually reach. A `1 + 1e-6` margin on
  the scale keeps the binding input inside D31's tolerance. The search is
  `(n + 1) × 5 × n` 2×2 eigenproblems per correlation, microseconds. `--gausspar` with an
  impossible value now raises before any FFT, naming the shortfall factor and the floor.
- **Source:** issue #312; `src/pfb_imaging/utils/restoration.py` (`lowest_resolution`,
  `resolution_deficit`, `_mean_pa`); `src/pfb_imaging/core/restore.py`;
  `tests/test_restore.py::test_lowest_resolution_dominates_every_input` (the six cases),
  `…_averages_position_angles_modulo_pi`, `…_takes_max_axes_when_the_angles_agree`,
  `test_restore_zero_gausspar_handles_bands_at_different_angles`,
  `test_restore_gausspar_sharper_than_the_data_raises`.

### D33 — `CRESIDUAL` records what the resolution change did to the residual

- **Context:** the restored image is `MODEL ⊗ G + (RESIDUAL/WSUM) ⊗ [G/PSFPARSN_b]`, and the
  second term is computed inside `restore_products` and discarded. Nothing in the tree
  records it, so "what did homogenisation do to the residual" could only be answered by
  redoing the convolution with the exact `G` and `PSFPARSN_b` of that run — which is what
  `scripts/check_spi_ripples.py` has to do, and why it needs a rebuild-vs-`IMAGE` check at
  all. The question matters because that term is the leading suspect for the ripples left in
  a per-pixel spectral index fit (#312).
- **Decision:** `--outputs s`/`S` stores it as band variable `CRESIDUAL`, **apparent**
  (pre-beam-division) and at the restoring resolution. Off by default — it is another cube
  per band. The `C` prefix means convolved, alongside the tree's existing `B` for
  beam-attenuated.
- **Rationale:** apparent because image-plane noise is flat on that scale and `BEAM` is
  already on the node, so the intrinsic form is one divide away; the reverse is not true
  where the beam is small. Storing it makes the tree self-describing: `MODEL ⊗ G + CRESIDUAL`
  reproduces `KIMAGE` exactly, which is the property the tests pin. It also gives
  `spifit` the array its SNR cut is actually applied to — the rms currently comes from
  `std(RESIDUAL/WSUM)`, the *raw* residual, while the image being thresholded contains the
  convolved one, whose rms differs by a band-dependent factor because the broadening needed
  to reach `G` does. That is tidiness rather than a ripple source: the mask is a single
  min-over-bands cut, so it shifts which pixels are fitted, not their spectra.
- **Consequences:** `restore` still never writes `RESIDUAL` — only `PSFPARSF` and the
  `PRODUCT_VARS` entries, so the raw residual survives untouched and `--outputs F` keeps
  FFT-ing the raw array. The MFS `CRESIDUAL` FITS is written from the direct MFS path, not
  by summing bands, for the same reason the restored MFS image is (D29): with no
  homogenisation the per-band `CRESIDUAL` sit at different resolutions. When `gaussparf`
  equals a band's own resolution the convolution is skipped and `CRESIDUAL` is that band's
  residual — correct, since the restoring resolution *is* native there.
- **Source:** issue #312; `src/pfb_imaging/utils/restoration.py` (`PRODUCT_VARS`,
  `restore_products`); `src/pfb_imaging/core/restore.py`; `src/pfb_imaging/cli/restore.py`;
  `scripts/check_spi_ripples.py`; `tests/test_restore.py::test_restore_cresidual_completes_the_restored_image`,
  `…_is_apparent_and_not_the_raw_residual`, `test_restore_without_s_writes_no_cresidual`.

### D34 — `--eta-in-grad` moves the prior from the preconditioner into the objective

- **Context:** the `eta` term (and, since D30, the whole frequency prior) entered `M` only,
  never `residual_from_partitions`. That is what makes it a *preconditioner*: preconditioned
  Richardson converges to `A⁻¹b` for any nonsingular `M` (D22), so the prior reshaped every
  forward update while leaving the fixed point and the flux scale exactly where they were
  (`test_frequency_prior_does_not_move_the_fixed_point`). Issue #310 asked what happens when
  it enters the objective instead.
- **Decision:** `--eta-in-grad` (default **off**, so nothing changes unasked). The objective
  gains `½ mᵀK⁻¹m`, whose gradient is `K⁻¹m`, and in pfb's sign convention (`residual =
  −grad`) both the apparent and beam-attenuated gradients lose that term. `K⁻¹` is applied by
  `HessTreeRay.prior_dot`, which is the *same* term `dot` adds — never a re-derived `eta*x`,
  which would drift the moment `--gp-length-scale` or `--eta-mode` is on.
- **Rationale — two things the correction deliberately does not do.** (1) **No beam.**
  `bresidual` carries the outer per-partition beam because the data Hessian is `B GᵀWG B`
  (D23), but the prior acts on the *intrinsic* model, so `K⁻¹` has no beam on either side and
  the identical term comes off both gradients. (2) **No rescaling.** The residuals reaching
  the solver are already divided by the total wsum, which is the operator `prior_dot` is
  defined on (`--eta` is a fraction of the total wsum, D4).
- **Consequences:** with the flag, "converged" means the *regularised* gradient is small, and
  `--rmsfactor`'s λ is derived from it — so the L1 threshold shifts too. `RESIDUAL`/`BRESIDUAL`
  keep storing the **pure data term** whatever the flag: a resumed run reads `BRESIDUAL` and
  applies the prior itself, so an eta-inclusive stored gradient would double-count on every
  restart. The flag therefore changes what is *reported* (FITS, rms, λ), not what is stored.
  Combined with D30 this is the knob that makes the frequency prior actual regularisation
  rather than preconditioning — the two are different experiments.
- **Source:** issue #310; `src/pfb_imaging/core/deconv.py` (`_grad_with_prior`);
  `src/pfb_imaging/operators/hessian.py` (`prior_dot`, `_prior_s`);
  `tests/test_hess_tree_ray.py::test_prior_dot_is_exactly_the_non_data_part_of_m`,
  `tests/test_deconv.py::test_eta_in_grad_changes_the_update_once_the_model_is_nonzero`,
  `…_is_a_no_op_on_the_first_step_from_a_zero_model`, `test_grad_with_prior_subtracts_the_prior_term_on_the_normalised_scale`.

### D35 — `--mop` stores the near-perfect-residual model, solved fresh at the final model

- **Context:** a prior necessarily makes the residual look worse — that is what a prior does.
  But `A(m + M⁻¹r) = Am + AM⁻¹r ≈ Am + r = data` wherever `M ≈ A`, so `m + M⁻¹r` is the model
  whose residual is near perfect. Issue #311 asked for that pair as a stored product.
- **Decision:** `--mop` (default **on**) writes `MODEL_MOPPED`/`RESIDUAL_MOPPED` to each band
  node plus `_model_mopped`/`_residual_mopped` MFS FITS. The update is **solved fresh** after
  the loop, not taken from the loop's last forward step: that update was computed at the
  *pre-backward* model, so `model + update` is the near-perfect point only if the prox barely
  moved — true at convergence, false on a `niter` or divergence exit. `bresidual` is already
  the gradient against the final model, so the extra cost is one CG solve and **no** extra
  gridding for the gradient (one sweep is still needed for `RESIDUAL_MOPPED` itself).
- **Rationale:** `restore` is already parameterised on its input variables, so
  `pfb restore --model-name MODEL_MOPPED --residual-name RESIDUAL_MOPPED` builds restored
  images — and thence a spectral index map — with **no restore change at all**
  (`test_restore_consumes_the_mopped_products` keeps that true). This is also where D30's
  frequency prior reaches the *output*: the deconvolved model's fixed point is
  prior-independent without D34, but the mop update **is** `M⁻¹r` and therefore carries the
  band coupling directly.
- **Consequences:** `MODEL` is untouched — the mopped model is `MODEL` plus a least-squares
  update, so it is neither sparse nor a component model, and it does **not** go through
  `model_to_ds`; there is no `.mds`/`degrid` path for it.
  **The mop right-hand side is the pure data gradient, never the `--eta-in-grad`-corrected
  one.** "Near perfect residual" means the *data* residual is near zero, so the direction
  wanted is `M⁻¹r_data`. Solving against `r_data − K⁻¹m` instead collapses the mop to nothing
  exactly where it is wanted: at a regularised fixed point that gradient is ~0, so
  `MODEL_MOPPED → MODEL`. It also keeps `MODEL_MOPPED` meaning the same thing with and
  without D34, which is what makes the two comparable
  (`test_mop_uses_the_data_gradient_not_the_eta_corrected_one`). The mop write goes to the band nodes
  *after* the major-cycle write, and `to_zarr(mode="a")` replaces attrs wholesale, so it
  extends the attrs the loop last wrote (`final_attrs`) rather than the band's original ones —
  building it from `band_attrs` silently dropped `niters`/`rms`/`hess_norm`/`hess_norm_opts`,
  which would restart a resumed run from iteration 0 (caught by
  `test_mop_preserves_the_run_attrs`). **Interpretation warning:** the mopped residual is
  near zero by construction, which removes the D33 ripple source (a residual reconvolved from
  a Gaussian fit to a non-Gaussian dirty beam) — so a cleaner spectral index map off mopped
  products is *not* by itself evidence about the prior.
- **Source:** issue #311; `src/pfb_imaging/core/deconv.py`; `src/pfb_imaging/cli/deconv.py`;
  `tests/test_deconv.py::test_deconv_groundtruth` (the residual-reduction claim, on
  consistent data — see the note there on why the synthetic tree cannot carry it),
  `…test_mop_writes_mopped_products_by_default`, `…test_no_mop_writes_neither_product`,
  `…test_mop_leaves_the_deconvolved_model_alone`, `…test_mop_preserves_the_run_attrs`,
  `tests/test_restore.py::test_restore_consumes_the_mopped_products`.

## Known debt

- **`HessTreeRay.cg`'s two branches have opposite `x0` aliasing, and the caller only
  happens to be safe.** The uncoupled branch (`BandWorkerPool.hess_cg`) allocates a fresh
  `out` and leaves `x0` untouched; the coupled branch hands `x0` to `pcg_numba`, which
  binds it as the iterate and mutates it in place, so the returned array *is* `x0`. The
  one caller, `PFBSolver.forward`, passes `x0 = self._update` and immediately rebinds
  `self._update` to the return value, which is correct either way — but only by accident,
  and a second caller that keeps its `x0` would get branch-dependent behaviour with no
  error. Both docstrings warn; that is mitigation, not a fix. **Follow-up:** make the two
  branches agree, preferably by having the coupled branch copy (matching
  `_BandWorkerImpl.cg`, which already copies because Ray hands it a read-only view) and
  dropping the warnings. Found reviewing #308; not fixed there because it changes the
  contract of a frozen oracle's caller and deserves its own PR with a test that pins the
  aliasing on both branches.
- `opt/primal_dual.py::primal_dual_numba` contains two `pdb.set_trace()` breakpoints
  (zero-model and NaN-eps paths) — hangs unattended runs if triggered. Kept because the
  function is a frozen oracle; remove if it ever stops being one.
- `stokes2vis_msv4._release_ms_caches` clears xarray-ms's private
  `Multiton._INSTANCE_CACHE` (see `memory-and-ray.md` layer 3). Needs an upstream
  xarray-ms TTL/eviction knob; delete the helper when one exists.
- Very large nband on small clusters: band-worker claims (1e-2 each) plus the
  `num_cpus = max(nworkers, nband+1)` sizing are untested beyond ~tens of bands;
  revisit scheduling at nband ≳ 50.
- v1 driver limits: single time node and single correlation asserted
  (`core/deconv.py`); joint-pol and dynamic models are future work.
- kclean is designed for (OneShot backward + Clark/Hogbom-as-prox fits the Protocols)
  but not implemented; `PRESETS` has only `sara`/`ista`.
- The PD inner loop round-trips the driver per iteration (grad + 2 Psi calls per band);
  visible overhead on small images (~3 s/major-cycle at 308²). Acceptable at production
  scale; an in-worker backward loop would change the prox's band coupling and is NOT
  planned.
- **Phase-centre comparison, to re-land with mosaicing (issue #1).** `core/init.py`'s
  single-field guard (`_phase_dirs_agree` + `tests/test_phase_dir_agreement.py`, commit
  `6be3ea7`) went away with the legacy retirement (#277) — the code no longer exists on
  this branch; `6be3ea7` is the only copy. Do **not** port it back as a rejection: it refused multiple phase centres, which is precisely what
  mosaicing does (the imager rephases them to a common tangent plane, D21). What must
  survive is the *comparison technique* (see the gotcha below). Its future home is the
  field-identity test in `core/imager.py`, currently
  `np.unique(np.round(field_centres, 12))` — exact equality in disguise. That is
  survivable today (`radec_barycentre` averages unit vectors, so a seam-split or
  noise-split pair still resolves to the right tangent point; the cost is a spurious
  "multiple fields" rephase of what is one field), but it becomes load-bearing the
  moment mosaicing has to decide which fields group together. Give it a real tolerance
  then, and make that tolerance a wrapped magnitude. Lift the helper and its tests from
  `6be3ea7`.
- **`deconv` cannot read a single-precision tree (D27).** The imager honours
  `--precision` end to end, but `gridder.residual_from_partitions` sizes `convim`/`tmp`
  from the band `DIRTY` while the driver hands it an f8 model, so ducc rejects the mixed
  call at the exact-residual seam; `band_worker`'s `--fits-per-partition` path allocates
  f8 `dirty_p`/`resid_p` the same way. `HessianTree` survives only by accident (its f8
  `xpad`/`xhat` scratch upcasts a c4 `psfhat` silently through in-place `*=`). Fixing it
  means casting at the ducc seams while letting the image-space cubes stay f8 — deliberately
  deferred, since the second-order schemes want double anyway. Until then a
  single-precision tree is imager/FITS-only.
- `utils/beam.reproject_and_interp_beam` is dead code — uncalled since D25 deleted the
  zarr-beam branch, and still carrying the pre-D19 reproject bugs it was written
  against. Delete it once it is clear raw MdV zarr beams are not coming back; if they
  are, rewrite it along D19 lines and settle the feed-plane→sky parity question first
  (image-and-beam-orientation.md §5).

## Recurring gotchas

- **psi/psih naming is inverted between the two legacy PD implementations** —
  `primal_dual(psi=synthesis, psih=analysis)` vs `primal_dual_numba(psih=synthesis,
  psi=analysis)`. Read call sites, not names.
- **`pcg_numba` mutates `x0` in place** (returns the same buffer).
- **`psf_oversize` truncates the preconditioner PSF** (`nx_psf =
  good_size(psf_oversize·nx)`, default 1.4, not 2). It only affects the
  preconditioner rate/stability, never the fixed point (D22, issue #287), but a
  *low* value inflates `λmax(M⁻¹H)` past `2/γ` and the fixed-`gamma` outer step
  **diverges** (the `diverge_count` terminator fires only after the fact). A
  `gamma=1` / low-`psf_oversize` divergence is a preconditioner-conditioning
  symptom, not a bug in the operator.
- **Generated CLI annotations lie about optionality.** An optional cab input with
  `choices` round-trips through hip-cargo to `Literal["…"] = None` — a type that
  excludes the value it defaults to (landmanbester/hip-cargo#90; writing the honest
  `Literal[…] | None` currently crashes `generate-cabs`). Branch on `is None` first;
  do not trust the annotation to tell you a value cannot be `None`.
- **Compare phase centres as wrapped magnitudes, never signed differences.**
  `np.any((a - b) > tol)` only fires when `a` is larger on *every* axis, so it silently
  accepts half of all mismatches — two fields 1.1 rad apart passed the `init` guard
  this way for as long as it existed (`6be3ea7`). A naive `np.abs` is not the fix
  either: `construct_mappings` normalises ra into [0, 2pi), so a pair straddling
  RA = 0 reads as ~2pi apart when it is 2e-9. Wrap first, then take magnitudes:
  `np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)`. Dec never wraps, so applying it
  elementwise to the (ra, dec) pair is safe.
- **`convolve2gaussres` reads the pixel size off `xx`/`yy` on the `gausspari` path,**
  so the grids must be `np.meshgrid(x, y, indexing="ij")`. numpy's *default* is
  `indexing="xy"`, which transposes both, makes the inferred spacings zero and every
  frequency infinite — an all-NaN image. It now raises instead; before the closed form
  (D31) the kernel was evaluated on the grids themselves and the mistake was invisible.
- **Never divide two sampled kernels' FFTs.** A sampled Gaussian's DFT sinks into
  round-off before Nyquist, so the quotient is noise over noise there — and the
  error grows as the kernel is *better* sampled, which is the opposite of the
  intuition. When the quotient has a closed form (Gaussians do), evaluate it; see
  D31. The same reasoning applies to any "divide by the transform of a model
  kernel" step, and widening the kernel's support only fixes the truncation half.
- **Warm-cache timing:** back-to-back runs on the same MS read from page cache
  (stimela stats `R GB` ≈ 0); only compare wall times at matching cache state.
- **stimela deconv memory stats are dominated by fixed Ray overhead** on small tests
  (plasma reservation + resident worker processes), not data — don't chase them below
  ~1 GB/process.
- **Transient-injection checks must read the raw `cube`, not `cube_mean`.** With zeroed
  base data (`data_column="DATA-DATA"`), hci's per-bin RMS flag (`rms > 1.5·median_rms`,
  median ≈ 0) flags exactly the bright transient bins and suppresses them in
  `cube_mean`; the raw `cube` is unaffected. (`utils/transients.py`; the designed
  end-to-end test is still unimplemented — see `docs/look-ahead.md`.)
- **A resumed deconv run continues from the tree's MODEL/UPDATE/niters.** After a
  crashed or diverged run, reset by deleting `MODEL`/`MODEL_BEST`/`RESIDUAL`/`UPDATE`
  arrays and the `niters`/`rms`/`rmax`/`hess_norm` attrs from each band group (zarr),
  then `zarr.consolidate_metadata`.
