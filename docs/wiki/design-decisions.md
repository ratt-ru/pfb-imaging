---
type: Design Ledger
title: Design decisions, known debt and recurring gotchas
description: Context/Decision/Rationale/Consequences ledger for pfb-imaging's load-bearing choices, plus the debt list and the gotchas that have already cost real debugging sessions.
tags: [design, decisions, debt, gotchas, ray, deconvolution, imager]
timestamp: 2026-07-16T10:30:00Z
last_verified_commit: 9a46876
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

### D2 — Legacy code is untouched and serves as the test oracle

- **Context:** Rewrites of numerical code need ground truth.
- **Decision:** `core/sara.py`, `core/kclean.py`, the `.dds` consumers and the legacy
  `opt` functions (`primal_dual`, `primal_dual_numba`, `pcg*`, `fista`) are not
  modified (behaviour-wise); new implementations are validated against them in a
  three-tier pyramid (unit rdiff < 1e-10, operator equality, e2e on a real MS).
- **Rationale:** Mirrors the `init`+`grid` → `imager` strategy, which caught real bugs
  at every tier.
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
- **Source:** `utils/weighting.box_sum_counts`; `core/grid.py`; `tests/test_weighting.py`.

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

### D18 — NUMBA_CACHE_DIR defaults to a per-user temp directory

- **Context:** `NUMBA_CACHE_DIR` was hard-coded to `/tmp/numba` (Dockerfile ENV plus
  implicit cab outputs as mount hints). On shared hosts the first user to create it
  owned it; everyone else got cryptic `PermissionError`s (issue #270). Bare `/tmp` is
  no fix: numba nests `<srcdirname>_<sha1(source dir)>` subdirs under the cache root,
  and identical install paths (guaranteed inside containers) collide one level down.
- **Decision:** `pfb_imaging/__init__.py` sets
  `NUMBA_CACHE_DIR=<tempfile.gettempdir()>/numba-cache-<uid>` via
  `os.environ.setdefault`, and `set_envs` forwards it to child processes. The
  Dockerfile ENV is gone. The implicit `numba-cache-dir` cab outputs remain solely as
  mount hints (`write_parent` mounts `/tmp` read-write).
- **Rationale:** The package `__init__` runs before any submodule import, so the value
  is set before numba can be imported from any entry point — CLI, stimela-called core
  functions, Ray workers, tests — with no numba import deferrals. `gettempdir()`
  honours per-job `TMPDIR`; `getuid()` survives containers where `$USER` doesn't;
  `setdefault` lets an explicit env (native export, stimela `backend.*.env`) win.
  Same-user concurrent runs share one cache safely (atomic temp-file + `os.replace`
  writes; stable keys since D17), so isolation is per-user only. A user-facing
  `--numba-cache-dir` option was rejected: stimela invokes the core functions
  directly, so a parameter arrives after module-level imports pulled numba in, and a
  cab default cannot be computed by stimela formulas.
- **Consequences:** Overriding the cache location is env-var-only. A containerised
  override outside `/tmp` additionally needs its mount expressed on the stimela side
  (backend `env` today; the cab-level env mechanism when it lands).
- **Source:** issue #270; `src/pfb_imaging/__init__.py`; `Dockerfile`;
  `tests/test_numba_cache_dir.py`.

### D19 — Beam maps are (Y, X)-ordered; one documented transpose to wgridder order

- **Context:** The `hci` BeamWizard beam path historically carried a
  transpose+flip "hack to get the images to align" (`547458f`), later removed
  (`330bc5d`), and then bypassed reprojection entirely (`a516530`) during the
  jagged-beam-gain investigation (breifast#208). The hack compensated three real
  bugs in `reproject_and_interp_scat_beam` (transposed array feed, target
  `crpix` off by one, wrong target `CDELT1` sign) and was only approximately
  correct because the MeerKAT beam is nearly circular — measured errors: 4.3 %
  of peak (circular), 21 % (elliptical), rephasing offsets applied along the
  wrong axis; it also required square images.
- **Decision:** Beam maps flow through the pipeline in cube/FITS **(Y, X)**
  order end to end — `get_rotation_averaged_beam` (native since meerkat-beams
  `616906b`) → `reproject_and_interp_scat_beam` (fixed to the measured
  reproject semantics; takes the 1D `l_beam`/`m_beam` coords so cdelt/crpix are
  signed and direction-agnostic; target WCS equals the hci output header) —
  with exactly **one** transpose to wgridder (X, Y) order at the end of
  `beam_for_band`'s wizard branch, commented and citing the wiki page. No flips
  anywhere.
- **Rationale:** Every layer keeps the index order its producer defines
  (astropy/reproject and the wizard are (Y, X); ducc's dirty images are
  (X, Y)), so orientation is auditable at one seam instead of smeared across
  compensating hacks. Conventions were pinned by measurement, not derivation:
  see image-and-beam-orientation.md.
- **Consequences:** Non-square images work. The zarr-beam branch
  (`reproject_and_interp_beam` + its surviving hack + the feed→sky parity
  question) is untouched, documented debt. Changing any transpose/flip on this
  path must keep `tests/test_beam_orientation.py` green.
- **Source:** `src/pfb_imaging/utils/beam.py`;
  `src/pfb_imaging/utils/stokes2im.py` (`beam_for_band`);
  `tests/test_beam_orientation.py`; image-and-beam-orientation.md; commits
  `547458f`, `330bc5d`, `a516530`; meerkat-beams `616906b` / PR
  landmanbester/meerkat-beams#8; ratt-ru/breifast#208.

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
