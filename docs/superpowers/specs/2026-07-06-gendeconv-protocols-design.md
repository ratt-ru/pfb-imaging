# General Deconvolution Interface via Protocols — Design

**Issue:** [#185](https://github.com/ratt-ru/pfb-imaging/issues/185) — Generalize the
deconvolution interface + standardise operator methods via Protocols.
**Branch:** `gendeconv`.

## Goal

A general deconvolution interface `deconv(hess, forward_alg, backward_alg, prox)` where the
four pieces compose freely: `hess` is the data-fidelity Hessian operator, `forward_alg`
solves the (preconditioned) forward step (CG now, BFGS later), `backward_alg` solves the
backward step (primal-dual, forward-backward/FISTA, ADMM later), and `prox` is the
regulariser (l21/SARA, l1/ISTA, TV later, Clark/Hogbom-as-prox for kclean). Interfaces are
formalised as `typing.Protocol` classes so contributors implement plain classes with the
required methods — **no inheritance from abstract base classes anywhere**.

The new `pfb deconv` sub-command consumes the imager's partitioned `xarray.DataTree`
(`<out>_<P>.dt`, PR #252) and distributes work with Ray from the outset. Existing algorithms
(`core/sara.py`, `core/kclean.py`, the `.dds` path, and the legacy `opt` functions) remain
untouched as test oracles, mirroring the `init`+`grid` → `imager` strategy.

## Why the current branch state is unsatisfying (diagnosis)

Both ABC layers on `gendeconv` exist because the coupling between an algorithm and a
regulariser is done by **subclassing** (template-method pattern):

1. `opt` layer: `ForwardBackward.solve()`/`PrimalDual.solve()` are fixed loops with abstract
   `prox()`/`dual_step()`/`prox_primal()`. Every (algorithm × regulariser) pairing needs a
   subclass (`L21ForwardBackward`, `L21PrimalDual`) — an M×N explosion, with the two
   `on_converge` reweighting bodies already verbatim duplicates.
2. `deconv` layer: `SARABase` couples regulariser setup to the opt backend via abstract
   `backward()` — one subclass per backend, and each `backward()` re-syncs
   `l1weight`/`reweighter` into the inner solver (a symptom of weights living in two places).

## The unifying observation

Every regulariser in scope decomposes as **R(x) = g(Ψᵀx)** with Ψ possibly the identity and
g having a cheap coefficient-domain prox. Both backward algorithms can *derive* what they
need from the pair (Ψ, prox_g):

- **Primal-dual** does its dual update via Moreau decomposition:
  `v = ṽ − σ·prox_{g/σ}(ṽ/σ)` — needs nothing else.
- **Forward-backward** needs a primal prox, but the tight-frame trick
  `x + (1/ν)·Ψ(prox_g(Ψᵀx) − Ψᵀx)` is generic in (Ψ, prox_g, ν) — written once in the
  solver, not per regulariser.
- **Positivity** is an indicator prox with Ψ = identity — a plain callable, not a magic int.
- **kclean**: Clark/Hogbom are approximate proxes with Ψ = identity; its backward step is
  *apply the prox once* (a trivial `OneShot` backward solver).

So algorithms become concrete, final classes that take the problem as data, and Protocols
describe the seams.

## Decisions (settled during design review)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Shape of `prox` | Decomposed regulariser object: `psi` + `nu` + coefficient-domain `prox`, owning its weight/reweighting state. Optional fused fast paths sniffed via `hasattr`. |
| 2 | kclean | Design for it (`OneShot` + `ClarkProx` mapping must fit the Protocols); implement and validate in a follow-up PR. This PR: sara-pd, sara-fb, ista. |
| 3 | Operator ownership | `deconv(hess, forward_alg, backward_alg, prox)` — the Hessian is an explicit fourth piece; `forward_alg` receives it per `solve` call. |
| 4 | Backward-solver state | Internal: PD owns its dual `v` (lazily allocated, warm-started across major cycles); uniform `solve(x, lam) -> x`; `reset()` for cold starts. |
| 5 | `on_converge` | **Not** in the Protocol (PFBSolver never calls it; `OneShot` cannot honour it). Optional constructor callback `on_converge(x, k, eps) -> bool` on the concrete iterative solvers; return `False` to continue (inner reweighting), `True` to stop. Built once by `PFBSolver` (`ReweightOnConverge`), which also uses it for verbosity-dependent convergence metrics. |
| 6 | Layering | Two-tier: `DeconvSolver` Protocol (`first/forward/backward/last`) is all the driver knows; `PFBSolver` is the one concrete composed implementation; algorithms that don't decompose implement `DeconvSolver` directly (escape hatch). |
| 7 | Data + distribution | `.dt` DataTree input via native `xr.open_datatree`; Ray band-actors from the outset; `PsiNocopytRay` (`operators/psi.py`) for wavelets. |

## Architecture

```
core/deconv.py driver        — .dt I/O, major cycle, lam schedule, FITS/mds output
  └─> DeconvSolver Protocol  — first/forward/backward/last (unchanged, escape hatch)
        └─ PFBSolver          — ONE concrete class composing the four pieces
             ├─> hess         : LinearOperator Protocol   (HessTreeRay, Ray-backed)
             ├─> forward_alg  : ForwardSolver Protocol    (PCG — per-band, in-actor)
             ├─> backward_alg : BackwardSolver Protocol   (PrimalDual | ForwardBackward)
             └─> prox         : Regulariser Protocol      (L21 | L1 | later ClarkProx)
```

**Ray topology.** The structural fact exploited: in the whole minor cycle, **bands couple
only through the prox** (l21 sums over bands; band-joint positivity). Hessian dots, CG
solves and residuals are per-band independent. Therefore:

- `HessTreeRay` mirrors the `PsiNocopytRay` actor pattern: band actors round-robin, each
  holding a `HessianTree` built from its band node's partition `PSFHAT`/`BEAM`/`WSUM`
  (sum over partitions sequential inside the actor, per architecture.md §8), with warmup
  and the preallocated FFT scratch `HessianTree` already carries for this purpose.
- The **forward CG runs inside the actors** (one Ray dispatch per major cycle; each band
  iterates to its own convergence), not a cube-level CG with a round-trip per iteration.
- The **backward loop stays cube-level** in the driver process: per iteration one
  `hess.dot` fan-out (grad) and two `PsiNocopytRay` fan-outs (analysis/synthesis), with the
  numba prox/dual-update running cube-level where the band coupling lives.
- **Residual between major cycles**: one Ray task per band node calling
  `gridder.residual_from_partitions` (exact degrid/grid path; PSF never recomputed).

**Per-major-cycle data flow:**

```
residual (from .dt / residual_from_partitions)
  → solver.first(residual)                      # beam handling
  → update = forward_alg.solve(hess, residual)  # per-band CG in actors
  → grad closure built from hess.dot            # xtilde = model + gamma*update
  → model = backward_alg.solve(model, lam)      # PD/FB loop, prox = regulariser
  → model → .dt band nodes + mds + FITS
  → residual_from_partitions per band (Ray)
  → solver.last()                               # regulariser weight updates
```

## Protocols

All `@runtime_checkable`, satisfied structurally (nothing inherits from them). Array
conventions are part of each contract; the two operator families genuinely differ and stay
honest rather than forcibly uniform. `runtime_checkable` isinstance checks verify method
*names* only (documentation-grade guards, like the driver's existing
`isinstance(solver, DeconvSolver)`); signatures are enforced by tests.

```python
# operators/__init__.py
@runtime_checkable
class LinearOperator(Protocol):
    """Hermitian image-space operator (Hessian family). Allocating style."""
    def dot(self, x: np.ndarray) -> np.ndarray: ...
    def hdot(self, x: np.ndarray) -> np.ndarray: ...   # = dot when Hermitian

@runtime_checkable
class PsiOperator(Protocol):
    """Analysis/synthesis pair. In-place style (fills preallocated outputs).
    Shape attributes are part of the contract:
    nband, nbasis, nxmax, nymax, nx, ny.
    """
    def dot(self, x, alphao) -> None: ...      # image -> coefficients
    def hdot(self, alpha, xo) -> None: ...     # coefficients -> image
```

```python
# deconv/__init__.py — DeconvSolver (first/forward/backward/last) unchanged, plus:
@runtime_checkable
class Regulariser(Protocol):
    """Separable regulariser R(x) = g(Psi^T x). Owns its own state (weights)."""
    psi: PsiOperator      # IdentityPsi for image-domain regularisers
    nu: float             # spectral norm of psi

    def prox(self, v, vout, lam, sigma=1.0) -> None: ...
        # vout = prox_{(lam/sigma) g}(v), coefficient domain, in-place
```

Optional sniffed extensions on a regulariser (not Protocol-required):
`dual_update(vp, v, lam, sigma)` — fused fast path PD prefers when present
(`dual_update_numba_fast` goes here); reweighting trio `init_reweighting(update)` /
`update_weights(x)` / `reweight_active`.

```python
# opt/__init__.py
@runtime_checkable
class ForwardSolver(Protocol):
    def solve(self, hess: LinearOperator, residual, x0=None) -> np.ndarray: ...
        # returns update ≈ hess^{-1} residual

@runtime_checkable
class BackwardSolver(Protocol):
    def setup(self, prox: Regulariser, hessnorm: float) -> None: ...
        # once, by PFBSolver: bind regulariser, compute step sizes, size buffers
    def set_grad(self, grad) -> None: ...        # each major cycle
    def solve(self, x, lam) -> np.ndarray: ...   # aux state internal, warm-started
    def reset(self) -> None: ...                 # drop warm-start state
```

Lifecycle is explicit: constructor = algorithm options only; `setup()` = problem binding;
`set_grad()` = per major cycle; `solve()` = iterate. This is what lets the four pieces
arrive independently — `PFBSolver` does the wiring, and no (algorithm × regulariser)
classes exist.

**Positivity**: backward solvers take an optional `primal_prox` callable (in-place
`x -> x`); `_nb_positivity` / `_nb_positivity_band` become the two stock choices in
`prox/positivity.py`. The existing `Preconditioner` Protocol (with `idot`) survives only
for the legacy `.dds` path; `ProxOperatorProtocol` (`prox`/`hprox`) is deleted.

## Concrete components

### Operators (`operators/`)

- **`HessTreeRay`** (new, `operators/hessian.py`): cube-level `LinearOperator`.
  `_HessBandActorImpl` actors round-robin over band nodes (arrays through the object store
  once at construction), each exposing:
  - `dot(x_b)` — one `HessianTree.dot` (used by the cube `dot(x)` fan-out: grad, power
    method);
  - `cg(r_b, x0_b, tol, maxit)` — full per-band CG iterating `HessianTree.dot` locally.

  `hessnorm` via the existing `power_method` against `HessTreeRay.dot`.
- **`IdentityPsi`** (new, `operators/psi.py`, ~15 lines): trivial `PsiOperator`
  (`nbasis=1`, `nxmax/nymax = nx/ny`, copy semantics). Makes ISTA a configuration.
- **`PsiNocopytRay`** used as-is.

### Algorithms (`opt/`) — concrete, final

- **`ForwardBackward`**: constructor(`tol, maxit, report_freq, verbosity, acceleration,
  on_converge, primal_prox`); `setup(prox, hessnorm)` sizes coefficient buffers from
  `prox.psi` shapes and computes the step; solve loop contains the generic tight-frame prox
  written once against the `Regulariser` Protocol. Replaces `L21ForwardBackward`.
- **`PrimalDual`**: same lifecycle; owns dual `self._v` (lazily sized in `setup`,
  warm-started, dropped by `reset()`); dual update prefers fused `reg.dual_update`, falls
  back to generic Moreau via `reg.prox`. Replaces `L21PrimalDual`.
- **`PCG`** (`ForwardSolver`): `solve(hess, residual, x0)` sniffs the distributed fast path
  (`hasattr(hess, "cg")` → one task per band actor with PCG's own tol/maxit), falls back
  to generic cube-level CG over `hess.dot` for any plain `LinearOperator`.
- Positivity kernels move `opt/primal_dual.py` → `prox/positivity.py` (public names).
- Legacy functions (`primal_dual`, `primal_dual_numba`, `fista`, `pcg_*`) untouched.

### Regularisers (`prox/`)

- **`L21`** (`prox/l21.py`): holds `psi` (the injected `PsiNocopytRay`), `nu`, and
  `l1weight` as its own state. `prox` wraps `prox_21m_numba`; `dual_update` wraps
  `dual_update_numba_fast`. Reweighting (`rms_comps` estimation from the update,
  `l1reweight_func` wiring, `update_weights(model)`, `reweight_active`) moves here from
  `SARABase.last()` — weights live in exactly one object.
- **`L1`** (`prox/l1.py`): weighted soft-threshold with `psi = IdentityPsi` (ISTA).

### Composition layer (`deconv/`)

- **`PFBSolver`** (`deconv/pfb.py`) — the one concrete `DeconvSolver`.
  `__init__(hess, forward_alg, backward_alg, prox, *, model, update, gamma,
  hessnorm=None, l1_reweight_from=..., verbosity=...)`:
  power method if `hessnorm is None`; `backward_alg.setup(prox, hessnorm)`; builds
  `ReweightOnConverge` (single implementation of the consecutive-reweight counter,
  `maxreweight` cap, `reg.update_weights(x)` call, and verbosity-dependent convergence
  metrics) when the regulariser exposes reweighting, and installs it by plain attribute
  assignment (`backward_alg.on_converge = ...`, only if the solver's own is `None`) — the
  hook is a concrete-class attribute, not a Protocol member. `primal_prox` (positivity) is
  a backward-solver constructor option, not a `PFBSolver` parameter. Methods:
  - `first(residual)` — store the residual as-is in v1: per-partition beams are applied
    *inside* `HessianTree.dot` and `residual_from_partitions`, so no cube-level beam
    weighting happens here (the hook stays for algorithms that need one);
  - `forward(residual)` — `update = forward_alg.solve(hess, residual, x0=update)`;
    `xtilde = model + γ·update`; grad closure from `hess.dot`;
    `backward_alg.set_grad(grad)`; return update;
  - `backward(lam)` — `model = backward_alg.solve(model, lam)`;
  - `last()` — delegate weight updates to the regulariser past `l1_reweight_from`;
  - forwards `reweight_active` / `trigger_reweight` to the regulariser (driver sniffing
    unchanged).
- **`deconv/presets.py`** — registry `{"sara": make_sara, "ista": make_ista}`; each
  factory (~30 lines) assembles `(HessTreeRay, PCG, PrimalDual|ForwardBackward, L21|L1)
  → PFBSolver` from CLI options + tree metadata. Contributors add an algorithm by writing
  a regulariser and registering a factory. `"kclean"` slots in later
  (`OneShot` + `ClarkProx`).

### Deleted from this branch

`deconv/sara.py` (`SARABase`), `deconv/sara_fb.py`, `deconv/sara_pd.py`. Every line
redistributes: precond/psi construction → factories; tight-frame prox → `ForwardBackward`;
reweighting state → `L21`; `on_converge` duplicates → `ReweightOnConverge`; grad/xtilde
wiring → `PFBSolver.forward`. `HessPSF` (dot+idot+modes) stops being the model for new
code — the Hessian is *just* `dot`; inversion belongs to the `ForwardSolver`.
`tests/test_primal_dual.py` is rewritten against the new pieces.

## Driver (`core/deconv.py` rewrite)

Major-cycle skeleton (lam schedule, model→mds fit, best-model tracking, divergence counter,
FITS outputs) carries over; the data layer and solver construction change:

- **Input**: `xr.open_datatree(f"{basename}_{suffix}.dt")` — native DataTree API only.
  Band nodes supply `DIRTY`/`RESIDUAL`/`WSUM` (+ `MODEL`/`UPDATE` on resume); partition
  children go to the `HessTreeRay` factory. Up-front asserts: single time node, single
  corr (v1, matching current deconv's limits).
- **Ray**: `init_ray(nworkers)` + `worker_process_setup_hook` as in `core/imager.py`;
  actor pools constructed once before the loop.
- **Solver**: `solver = PRESETS[minor_cycle](tree_meta, opts)`;
  `isinstance(solver, DeconvSolver)` guard and `reweight_active`/`trigger_reweight`
  sniffing unchanged.
- **Residual**: one Ray task per band node → `residual_from_partitions`.
- **Write-back**: `MODEL`/`UPDATE`/`MODEL_BEST` + `hess_norm`/`niters` attrs into band
  nodes via `ds.to_zarr(store, group="band…", mode="a")`; FITS via `dt2fits`/`rdt2fits`
  with `time_is_unix=True`.
- **Memory discipline** (docs/msv4-memory-patterns.md applies to the new Ray surfaces):
  actors return post-gc `{pid, rss_gb, peak_gb}` telemetry on request, printed at
  `verbosity > 1`; per-band task payloads are plain numpy, never live Dataset objects.

**CLI (`cli/deconv.py`)**: keeps `minor_cycle` (registry key) and `opt_backend` (backward
solver within a preset), gains `--nworkers`. Help text per python-standards §2.1
round-trip rules; cabs regenerated.

## Testing strategy — three tiers, tightest first

**Tier 1 — unit, no data, no Ray:**
- Protocol conformance (`isinstance`) for every implementation.
- Generic tight-frame prox in `ForwardBackward` vs the old hand-coded
  `L21ForwardBackward.prox` (ported into the test as reference) — bitwise-close.
- Moreau consistency: PD with fused `dual_update` vs generic `reg.prox` path — identical
  trajectories.
- PD/FB on a synthetic quadratic + l1 problem with known soft-threshold solution;
  `ReweightOnConverge` counter/cap logic; `IdentityPsi` round-trip;
  `L1` + `IdentityPsi` ≡ direct soft-thresholding (ISTA identity).

**Tier 2 — operator equivalence (the sharp instrument):**
- `HessTreeRay.dot` ≡ `HessPSF.dot` given the same `psfhat`/`beam` (single partition,
  `eta=0`) — direct numerical equality.
- In-actor `cg` ≡ legacy `pcg` on the same operator/rhs.
- Multi-partition `HessianTree` sum ≡ single-partition on concatenated data.

**Tier 3 — end-to-end vs legacy oracles** (`tests/data/test_ascii_1h60.0s.MS`, session Ray
fixture, small maxit):
- `pfb imager` → `.dt`; legacy `init`+`grid` → `.dds` (equivalence already tested). New
  `deconv` (sara-pd) on `.dt` vs legacy `core/sara.py` on `.dds` with matching parameters
  and a **fixed shared `hess_norm`**: one major cycle compared tightly, N cycles within a
  looser tolerance.
- Same for sara-fb; ista against a direct `fista` reference run.

kclean validation is deferred with its implementation.

## Non-goals (v1)

- kclean implementation (designed for; follow-up PR).
- Multi-time (dynamic) models — single time node asserted, as in current deconv.
- Joint-polarisation deconvolution — single corr asserted.
- Merging the `HessTreeRay` and `PsiNocopytRay` actor pools (memory optimisation, later).
- Removing the legacy `.dds` deconvolution paths (they are the oracles).

## File map

| Path | Change |
|------|--------|
| `operators/__init__.py` | `LinearOperator`, `PsiOperator` Protocols; `ProxOperatorProtocol` deleted; `Preconditioner` kept (legacy) |
| `operators/hessian.py` | + `_HessBandActorImpl`, `HessTreeRay` |
| `operators/psi.py` | + `IdentityPsi` |
| `opt/__init__.py` | `ForwardSolver`, `BackwardSolver` Protocols |
| `opt/forward_backward.py` | concrete `ForwardBackward` (generic tight-frame prox, `on_converge` callback) |
| `opt/primal_dual.py` | concrete `PrimalDual` (internal dual, Moreau/fused paths); positivity kernels move out |
| `opt/pcg.py` | + `PCG` ForwardSolver (duck-typed `hess.cg` fast path) |
| `prox/positivity.py` | new; public positivity kernels |
| `prox/l21.py`, `prox/l1.py` | new; `L21`, `L1` regularisers |
| `deconv/__init__.py` | + `Regulariser` Protocol (DeconvSolver unchanged) |
| `deconv/pfb.py` | new; `PFBSolver`, `ReweightOnConverge` |
| `deconv/presets.py` | new; minor-cycle registry + factories |
| `deconv/sara.py`, `deconv/sara_fb.py`, `deconv/sara_pd.py` | **deleted** |
| `core/deconv.py` | rewritten: `.dt` input, Ray, presets |
| `cli/deconv.py` + cab | `--nworkers`; help round-trip rules |
| `tests/test_primal_dual.py` | rewritten against new pieces |
| `tests/` | + tier-1/2/3 tests per strategy above |
