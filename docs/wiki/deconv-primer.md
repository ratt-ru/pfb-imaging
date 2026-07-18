---
type: Domain Primer
title: Deconvolution primer — the PFB framework, math to code
description: Maps the preconditioned forward-backward algorithm, the SARA prior and their numerical conventions onto the pfb deconv code, including the constants that break convergence when wrong.
tags: [deconvolution, sara, primal-dual, forward-backward, protocols, conventions]
timestamp: 2026-07-18T17:00:00Z
last_verified_commit: 9e0ee75
---

# Deconvolution primer — the PFB framework, math to code

Read this before touching `deconv/`, `opt/`, `prox/` or `core/deconv.py`. It maps the
math to the code and records the conventions that are load-bearing: several constants
here have "diverges when wrong" status, discovered the hard way.

## The problem and the major cycle

Interferometric imaging solves `V = R x + noise` for the sky image `x`, where `R`
degrids an image to visibilities and `W` are the imaging weights. The data-fidelity
Hessian `H = Rᵀ W R` acts, to good approximation, as convolution by the PSF. `pfb deconv`
minimises `½‖V − Rx‖²_W + λ R(x)` with a preconditioned forward-backward (PFB) major
cycle (`core/deconv.py`):

1. **first(residual)** — per-iteration preprocessing hook (currently caches the
   residual; historically applied a cube-level beam). `forward()` consumes this cache,
   NOT its own argument — calling `forward()` without `first()` raises.
2. **forward** — `update ≈ H⁻¹ residual` by per-band conjugate gradients
   (`opt/pcg.PCG` → in-worker CG on `HessianTree`). This is the preconditioned gradient.
3. **backward** — `model = argmin_x ½γ⁻¹‖x − x̃‖²_H + λ R(x)` with
   `x̃ = model + γ·update`, solved by primal-dual or forward-backward against the grad
   closure `∇(x) = −H(x̃ − x)/γ` (`deconv/pfb.PFBSolver.forward` builds it).
4. **exact residual** — degrid/grid of the model against the stored per-partition
   inputs (`operators/gridder.residual_from_partitions`), never a PSF convolution and
   never recomputing the PSF.
5. **last()** — arms/refreshes ℓ1 reweighting once past `l1_reweight_from`.

**λ schedule:** `lam = rmsfactor · rms(residual_mfs)` each major iteration, with
`init_factor` (default 0.5) applied **only at the very first iteration of a fresh run**
(`iter0 == 0 and k == 0`, `core/deconv.py`). Legacy `core/sara.py` had `if iter0 == 0`,
which silently applied the factor to every iteration of a fresh run; fixed in `52d5fb1`.
Old sara results predate that fix — expect its residuals to be *lower* at matched
iteration count (it ran with λ effectively halved).

## The SARA prior and the constants that matter

`R(x) = ‖W Ψᵀ x‖_{2,1}` — the 2-norm runs over the **band** axis (bands couple only
here), and Ψ is the concatenation of `nbasis` orthonormal bases (`self` + Daubechies,
`bases` option). Because each basis is orthonormal, `Ψ Ψᵀ = nbasis · I`, so:

- **`nu = nbasis` — not 1.0.** The primal-dual step sizes are
  `sigma = hessnorm/(2γ)/nu` and `tau = 0.98/(hessnorm/(2γ) + sigma·nu²)`
  (`opt/primal_dual.py`; the 0.98 holds for `PrimalDual` and `primal_dual_numba` —
  the legacy allocating `primal_dual` uses 0.9). With the tight-frame default
  `nu=1.0` the dual step is
  ~nbasis× too large and the backward solve **diverges on multi-band data** (single-band
  can survive on stability margin, which is why tests missed it). `deconv/presets.py`
  `make_sara` passes `nu=len(bases)` to `L21`; legacy `core/sara.py` passes
  `nu=nbasis` to `primal_dual`. Guarded by
  `tests/test_pfb_solver.py::test_make_sara_sets_dictionary_nu`.
- **`hess_norm`** is the spectral norm of the *cube* Hessian (max over bands), power-
  methoded when not given, inflated ×1.05, cached in the band-node attrs as
  `hess_norm`. It is data-dependent: a regenerated tree needs a fresh estimate.

## Normalisation conventions (total-wsum)

All image-space products in the `.dt` are stored **raw** (un-normalised); normalisation
by the TOTAL weight sum happens at the point of use. The pieces must agree:

- `wsum_tot = Σ_bands Σ_partitions wsum_p`; `residual = RESIDUAL_raw / wsum_tot`.
- Each band's `HessianTree` gets `wsum=wsum_tot` as an explicit override (not its own
  band's sum) — `deconv/presets._build_hess`, mirroring legacy `abspsf /= wsum`.
- Tikhonov: `--eta` is a **fraction of the total wsum** — on the total-wsum-normalised
  band operators the term is a uniform `+eta·x` (raw units `eta·wsum_tot`), identical
  for every band and invariant to how the data is split into bands. (The earlier
  legacy-matching `eta_b = eta·wsum_b/wsum_tot` weakened the damping as band count
  grew; retired with legacy sara — see D4.)
- `HessianTree` consumes `abs(PSFHAT)`: the stored `PSFHAT` is the raw complex FFT of
  the PSF; without `abs()` the "Hessian" carries a phase, is not Hermitian-positive,
  and CG breaks. The `abs()` happens worker-side in
  `operators/band_worker._BandWorkerImpl.load_band`.

## Backward solvers

Both are concrete, final classes in `opt/` sharing the lifecycle
`__init__(options) → setup(prox, hessnorm) → set_grad(grad) → solve(x, lam)` with
internal warm-started state and `reset()`.

- **`PrimalDual`** (`opt/primal_dual.py`): owns its dual `self._v` (warm-started across
  major cycles). Dual update prefers the regulariser's fused `dual_update` fast path
  (sniffed via `hasattr`; `L21` wraps `dual_update_numba_fast`), else generic Moreau:
  `v = ṽ − σ·prox_{g/σ}(ṽ/σ)`.
- **`ForwardBackward`** (`opt/forward_backward.py`): generic tight-frame primal prox
  written ONCE against the Regulariser Protocol:
  `x + (1/ν)·Ψ(prox_g(Ψᵀx) − Ψᵀx)`.
- **Positivity** is a plain in-place callable (`prox/positivity.positivity_prox(mode)`,
  modes 0/1/2 → None/`positivity`/`positivity_band`) passed as `primal_prox` to the
  backward solver's constructor.

## Reweighting

`L21` owns all reweighting state. `PFBSolver.last()` — once `_iter ≥ l1_reweight_from`
— calls `init_reweighting(update)` (per-basis `rms_comps = std` of the nonzero
coefficients of the CG update; empty bases fall back to 1.0) then `update_weights(model)`
(`l1reweight_func`). This re-estimates `rms_comps` **every major iteration**, matching
legacy sara. Inside the backward solve, `ReweightOnConverge` (`deconv/pfb.py`) fires on
inner convergence: reweight and continue, up to `maxreweight` consecutive times.

**Semantics trap:** `PFBSolver.reweight_active` means "the driver should STOP at
convergence" — it returns True when there is nothing to trigger (plain regulariser or
`l1_reweight_from < 0`), else defers to the regulariser. The driver contract is
`if not reweight_active: trigger_reweight() else: break` (`core/deconv.py`).

## Composition: Protocols and the code map

Seams are `typing.Protocol` classes satisfied structurally — **never introduce ABCs**
(architecture.md §5; issue #185; design-decisions.md D1).
Conformance is enforced at the seams by `operators.require_protocol` (TypeError naming
missing members).

| Seam | Protocol | Implementations |
|------|----------|-----------------|
| Hessian | `operators.LinearOperator` (`dot`/`hdot`, allocating) | `HessTreeRay`, `HessianTree`, `HessPSF` |
| Dictionary | `operators.PsiOperator` (in-place `dot`/`hdot` + shape attrs) | `PsiNocopytRay`, `IdentityPsi`, `Psi` |
| Regulariser | `deconv.Regulariser` (`psi`, `nu`, in-place `prox(v, vout, lam, sigma)`) | `prox.L21`, `prox.L1` |
| Forward | `opt.ForwardSolver` (`solve(hess, residual, x0)`) | `PCG` |
| Backward | `opt.BackwardSolver` (`setup/set_grad/solve/reset`) | `PrimalDual`, `ForwardBackward` |
| Whole solver | `deconv.DeconvSolver` (`first/forward/backward/last`) | `PFBSolver` (escape hatch: implement directly) |

`deconv/presets.py` holds the registry (`{"sara", "ista"}`); a new algorithm = a
regulariser + a factory (see `deconv/README.md`).

## Ray topology

Bands couple only through the prox, so everything else is per-band: one
`operators.band_worker.BandWorkerPool` worker process per band co-locates the band's
`HessianTree` (with in-worker CG — one Ray dispatch per forward solve, not per CG
iteration), the wavelet jitclass, and the pinned gridding inputs for the exact residual.
Workers read their own band's vis-scale data straight from the `.dt` store
(`load_bands`); the driver handles image-scale cubes only. The backward loop runs
cube-level in the driver: each PD/FB iteration fans out one `hess.dot` (grad) and two
Psi calls per band — on small images this dispatch overhead is visible (~ms per call ×
`pd_maxit` inner iterations); at production image sizes compute dominates. Scheduling and
memory discipline: see `memory-and-ray.md`.

## Legacy code: oracles, not dead code

`opt.primal_dual.primal_dual{,_numba}`, `opt.pcg.pcg_numba` and `opt.fista.fista` are
the unit-level validation oracles for the new framework — **do not modify their
behaviour**. (The e2e oracles `core/sara.py`/`core/kclean.py` were retired with the
legacy pipeline in 0.1.0 (#277, D2); e2e correctness is now pinned by the ground-truth
tests against an injected sky in `tests/test_imager*.py`/`tests/test_deconv.py`.)
Traps when reading them:

- Argument naming is inverted between the two PD implementations: in `primal_dual`,
  `psi` is SYNTHESIS (coeffs→image, allocating) and `psih` is ANALYSIS; in
  `primal_dual_numba`, `psih` is SYNTHESIS and `psi` is ANALYSIS (both in-place,
  two-arg). Check the call sites, not the names.
- `pcg_numba` binds `x = x0` and updates it **in place** — callers see their `x0`
  mutated (the Ray band worker copies before calling for exactly this reason).
- `primal_dual_numba` contains `pdb.set_trace()` breakpoints on its zero-model and
  NaN-eps paths — it will hang unattended runs if those trigger.
