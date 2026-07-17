---
type: Design Ledger
title: Design decisions, known debt and recurring gotchas
description: Context/Decision/Rationale/Consequences ledger for pfb-imaging's load-bearing choices, plus the debt list and the gotchas that have already cost real debugging sessions.
tags: [design, decisions, debt, gotchas, ray, deconvolution, imager]
timestamp: 2026-07-18T00:15:00Z
last_verified_commit: 502fe90
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
  `eta_b = eta·wsum_b/wsum_tot`; `HessianTree` consumes `abs(PSFHAT)`.
- **Rationale:** Matches legacy sara (`wsums /= wsum; abspsf /= wsum; eta*wsums`), so
  hyperparameters (`eta`, `rmsfactor`) mean the same thing on both paths.
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
- **Decision:** `DeconvSolver.first(residual)` caches (and may preprocess) the
  residual; `forward(residual)`'s argument is Protocol-shape only and is NOT read —
  calling `forward` before `first` raises RuntimeError.
- **Rationale:** Keeps a seam for cube-level beam handling without smuggling it into
  the forward solver.
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
  wgridder (X, Y) arrays. The zarr-beam branch
  (`reproject_and_interp_beam` + its surviving hack + the feed→sky parity
  question) is untouched, documented debt. Changing any transpose/flip on this
  path must keep `tests/test_beam_orientation.py` green.
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
  beam is `("corr", "m_beam", "l_beam")`. `nx`/`ny` keep meaning the X/RA and
  Y/Dec pixel counts everywhere — only array-axis order changed. ducc's
  x-major world exists only behind zero-copy `.T` views at the
  `vis2dirty`/`dirty2vis` call sites (input and output; both accept strided
  arrays), `fitcleanbeam` is called with `yx_order=True`, and
  `save_fits(yx_order=True)` writes without axis swaps. The `.mds` stays
  x-major (degrid/model2comps convention) behind transposed views at the
  `fit_image_cube`/`eval_coeffs_to_slice` boundary until the pfb-model-spec
  migration. uv-space grids (COUNTS, weighting) are untouched.
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
  shift (CRVAL stays the tangent point — facet convention). Beam handling is
  UNCHANGED pending #281: the stored katbeam/unity BEAM is field-centred and
  is not reprojected, so beam_model + rephasing is only approximate.
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

## Known debt

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
- The zarr-beam branch of `beam_for_band` (`reproject_and_interp_beam`) still carries
  the transpose+flip hack and the pre-D19 reproject bugs, and the MdV feed-plane→sky
  parity question ("transmissive or receptive?") is unresolved. Only correct-ish for
  square images and near-circular beams; fix along D19 lines once parity is settled
  (image-and-beam-orientation.md §5).

## Recurring gotchas

- **psi/psih naming is inverted between the two legacy PD implementations** —
  `primal_dual(psi=synthesis, psih=analysis)` vs `primal_dual_numba(psih=synthesis,
  psi=analysis)`. Read call sites, not names.
- **`pcg_numba` mutates `x0` in place** (returns the same buffer).
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
