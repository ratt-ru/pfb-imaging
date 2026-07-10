# Protocol-Based General Deconv Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A composable deconvolution interface `deconv(hess, forward_alg, backward_alg, prox)` with `typing.Protocol` seams (no ABCs), consuming the imager's `.dt` DataTree with Ray band-actors, validated against the legacy sara path.

**Architecture:** Concrete final algorithm classes take the problem as data (grad callable + a decomposed `Regulariser` R(x) = g(Ψᵀx)); Protocols document the seams; one concrete `PFBSolver` composes the four pieces behind the existing `DeconvSolver` Protocol. Spec: `docs/superpowers/specs/2026-07-06-gendeconv-protocols-design.md`.

**Tech Stack:** numpy, numba (existing kernels), Ray actors, xarray DataTree (zarr), ducc0 FFT/gridder, Typer/hip-cargo CLI.

## Global Constraints

- **No ABCs anywhere in new code.** Interfaces are `@runtime_checkable` `typing.Protocol` classes, satisfied structurally — nothing inherits from them.
- **Legacy paths untouched** (they are the test oracles): `core/sara.py`, `core/kclean.py`, `core/grid.py`, `core/init.py`, the module-level functions `primal_dual`, `primal_dual_numba`, `fista`, `pcg`, `pcg_numba`, `pcg_psf`, `pcg_dds`, all `.dds` consumers, and `deconv/{hogbom,clark,nnls,fskclean}.py`.
- **Prox semantics (exact contract):** `reg.prox(v, vout, lam, sigma=1.0)` computes `vout = prox_{(lam/sigma)·g}(v/sigma)` in-place — this matches `prox_21m_numba(v, result, lam, sigma, weight)` exactly. With `sigma=1`: `vout = prox_{lam·g}(v)`.
- **Array conventions:** image cubes `(nband, nx, ny)` float64; coefficient buffers `(nband, nbasis, nymax, nxmax)` float64, where `PsiOperator` exposes `nband, nbasis, nymax, nxmax, nx, ny` as attributes.
- **casacore-free imaging path:** no module-scope `daskms`/`africanus`/`casacore` imports in any file this plan touches (`.claude/rules/architecture.md` §3/§8). The new `core/deconv.py` must NOT import `DaskMSStore` or `xds_from_url`.
- **Native DataTree API only** for the `.dt`: `xr.open_datatree(store, engine="zarr", chunks=None)`, `ds.to_zarr(store, group=..., mode="a")`. No wrapper helpers.
- **Ray memory discipline** (`docs/msv4-memory-patterns.md`): task payloads are plain numpy or small selected-variable Datasets `ray.put` once before the major loop; never re-serialise per iteration.
- **Lint after every task:** `uv run ruff format . && uv run ruff check . --fix`
- **Commits:** Conventional Commits, first line < 72 chars, imperative. Pre-commit `generate-cabs` hook may fail without `hip-cargo` on PATH — commit with `--no-verify` and regenerate cabs manually when CLI files change (testing-and-ci §3.1): `uv run hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs`
- **CLI help text** must round-trip (python-standards §2.1): one sentence per line, no sentence > 120 chars rendered, no `e.g.` in help strings. After CLI edits run `uv run pytest tests/test_roundtrip.py`.
- **Test suite:** single session `uv run pytest -v tests/` (arcae + python-casacore coexist). The session-scoped autouse `manage_ray` fixture in `tests/conftest.py` provides Ray.
- **Interim breakage rule:** `core/deconv.py` lazily imports `deconv.sara_pd`/`deconv.sara_fb` *inside* its function body; Tasks 5–6 delete those modules, leaving `core/deconv.py` broken at runtime (not at import) until Task 11 rewrites it. No test exercises `core/deconv.py` before Task 12, so the suite stays green throughout.

---

### Task 1: Protocols (LinearOperator, PsiOperator, Regulariser, ForwardSolver, BackwardSolver)

**Files:**
- Modify: `src/pfb_imaging/operators/__init__.py` (add `LinearOperator`, `PsiOperator`; keep `Preconditioner` and `PsiOperatorProtocol` for now — deleted in Task 13)
- Modify: `src/pfb_imaging/opt/__init__.py` (replace `OptimiserProtocol` with `ForwardSolver`, `BackwardSolver`)
- Modify: `src/pfb_imaging/deconv/__init__.py` (add `Regulariser`; `DeconvSolver` unchanged)
- Test: `tests/test_protocols.py`

**Interfaces:**
- Consumes: existing `HessianTree`, `HessPSF` (`operators/hessian.py`), `Psi` (`operators/psi.py`), `DeconvSolver` (`deconv/__init__.py`).
- Produces: `pfb_imaging.operators.LinearOperator` (methods `dot(x)`, `hdot(x)` — allocating style), `pfb_imaging.operators.PsiOperator` (attrs `nband, nbasis, nymax, nxmax, nx, ny`; methods `dot(x, alphao)`, `hdot(alpha, xo)` — in-place style), `pfb_imaging.deconv.Regulariser` (attrs `psi`, `nu`; method `prox(v, vout, lam, sigma=1.0)`), `pfb_imaging.opt.ForwardSolver` (`solve(hess, residual, x0=None)`), `pfb_imaging.opt.BackwardSolver` (`setup(prox, hessnorm)`, `set_grad(grad)`, `solve(x, lam)`, `reset()`). Every later task type-checks against these names.

- [ ] **Step 1: Confirm `OptimiserProtocol` and `ProxOperatorProtocol` are unused**

Run: `grep -rn "OptimiserProtocol\|ProxOperatorProtocol" src/ tests/ --include="*.py" | grep -v "__init__.py"`
Expected: no output (only the definitions exist). If anything imports them, stop and report.

- [ ] **Step 2: Write the failing test**

Create `tests/test_protocols.py`:

```python
"""Protocol conformance for the composable deconv framework (issue #185)."""

import numpy as np

from pfb_imaging.deconv import DeconvSolver, Regulariser
from pfb_imaging.operators import LinearOperator, PsiOperator
from pfb_imaging.operators.hessian import HessianTree, HessPSF
from pfb_imaging.operators.psi import Psi
from pfb_imaging.opt import BackwardSolver, ForwardSolver


def _delta_part(nx_psf, ny_psf):
    """Single partition with a delta-function psfhat (identity convolution)."""
    psfhat = np.ones((1, nx_psf, ny_psf // 2 + 1))
    beam = np.ones((1, nx_psf // 2, ny_psf // 2))
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([1.0])}


def test_hessians_satisfy_linear_operator():
    nx = ny = 8
    hess = HessianTree([_delta_part(2 * nx, 2 * ny)], nx, ny, 2 * nx, 2 * ny)
    assert isinstance(hess, LinearOperator)
    abspsf = np.ones((1, 2 * nx, ny + 1))
    assert isinstance(HessPSF(nx, ny, abspsf, eta=1.0), LinearOperator)


def test_psi_satisfies_psi_operator():
    psi = Psi(1, 32, 32, ("self", "db1"), 2, 1)
    assert isinstance(psi, PsiOperator)


def test_protocols_reject_nonconforming():
    class Empty:
        pass

    assert not isinstance(Empty(), LinearOperator)
    assert not isinstance(Empty(), PsiOperator)
    assert not isinstance(Empty(), Regulariser)
    assert not isinstance(Empty(), ForwardSolver)
    assert not isinstance(Empty(), BackwardSolver)
    assert not isinstance(Empty(), DeconvSolver)


def test_structural_regulariser_conformance():
    class Toy:
        def __init__(self):
            self.psi = None
            self.nu = 1.0

        def prox(self, v, vout, lam, sigma=1.0):
            np.copyto(vout, v)

    assert isinstance(Toy(), Regulariser)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_protocols.py -v`
Expected: FAIL with `ImportError: cannot import name 'LinearOperator'`

- [ ] **Step 4: Implement the Protocols**

In `src/pfb_imaging/operators/__init__.py`, **delete** the `ProxOperatorProtocol` class (verified unused in Step 1) and **add** after `Preconditioner`:

```python
@runtime_checkable
class LinearOperator(Protocol):
    """Hermitian image-space operator (Hessian family). Allocating style.

    Methods:

    - dot: apply the operator; returns a new array
    - hdot: apply the adjoint (same as dot for Hermitian operators)
    """

    def dot(self, x): ...

    def hdot(self, x): ...


@runtime_checkable
class PsiOperator(Protocol):
    """Analysis/synthesis operator pair. In-place style.

    Shape attributes are part of the contract; coefficient buffers have
    shape ``(nband, nbasis, nymax, nxmax)`` and images ``(nband, nx, ny)``.

    Methods:

    - dot: analysis, image to coefficients, fills ``alphao`` in-place
    - hdot: synthesis, coefficients to image, fills ``xo`` in-place
    """

    nband: int
    nbasis: int
    nymax: int
    nxmax: int
    nx: int
    ny: int

    def dot(self, x, alphao): ...

    def hdot(self, alpha, xo): ...
```

In `src/pfb_imaging/opt/__init__.py`, **replace** the `OptimiserProtocol` class with:

```python
@runtime_checkable
class ForwardSolver(Protocol):
    """Solves the forward (preconditioned gradient) step.

    Methods:

    - solve: return ``update ≈ hess^{-1} residual``; ``hess`` satisfies
      the ``LinearOperator`` Protocol and is passed per call
    """

    def solve(self, hess, residual, x0=None): ...


@runtime_checkable
class BackwardSolver(Protocol):
    """Solves the backward (proximal) step.

    Lifecycle: constructor takes algorithm options only; ``setup`` binds
    the regulariser and step sizes once; ``set_grad`` is called each major
    cycle; ``solve`` iterates.  Auxiliary state (e.g. a dual variable) is
    internal and warm-started across calls; ``reset`` drops it.

    Methods:

    - setup: bind regulariser and hessnorm, size buffers
    - set_grad: set the gradient of the smooth data-fidelity term
    - solve: run the solve loop, return the updated x
    - reset: drop warm-start state
    """

    def setup(self, prox, hessnorm): ...

    def set_grad(self, grad): ...

    def solve(self, x, lam): ...

    def reset(self): ...
```

In `src/pfb_imaging/deconv/__init__.py`, add after `DeconvSolver` (import `Any` from typing for the `psi` annotation):

```python
@runtime_checkable
class Regulariser(Protocol):
    """A separable regulariser ``R(x) = g(Psi^T x)``.  Owns its own state.

    ``prox(v, vout, lam, sigma)`` computes ``vout = prox_{(lam/sigma) g}(v/sigma)``
    in the coefficient domain, in-place (matches ``prox_21m_numba``).

    Optional extensions sniffed by consumers (not required):

    - ``dual_update(vp, v, lam, sigma)``: fused primal-dual fast path
    - ``init_reweighting(update)`` / ``update_weights(x)`` /
      ``reweight_active``: iterative reweighting state
    """

    psi: Any  # PsiOperator; identity implementation for image-domain regularisers
    nu: float  # spectral norm of psi

    def prox(self, v, vout, lam, sigma=1.0): ...
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_protocols.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/operators/__init__.py src/pfb_imaging/opt/__init__.py src/pfb_imaging/deconv/__init__.py tests/test_protocols.py
git commit --no-verify -m "feat(deconv): add Protocol seams for composable deconvolution"
```

---

### Task 2: IdentityPsi + prox/positivity.py

**Files:**
- Modify: `src/pfb_imaging/operators/psi.py` (add `IdentityPsi` at the end)
- Create: `src/pfb_imaging/prox/positivity.py`
- Modify: `src/pfb_imaging/opt/primal_dual.py` (replace kernel definitions with imports)
- Test: `tests/test_protocols.py` (extend)

**Interfaces:**
- Consumes: `PsiOperator` Protocol (Task 1); the `_nb_positivity`/`_nb_positivity_band` numba kernel bodies currently in `opt/primal_dual.py:38-57`.
- Produces: `IdentityPsi(nband, nx, ny)` — trivial `PsiOperator` with `nbasis=1, nymax=nx, nxmax=ny` (the trailing `(nymax, nxmax)` coefficient axes hold `(nx, ny)` directly; the names follow the existing `(nband, nbasis, nymax, nxmax)` buffer convention). `prox/positivity.py` exports `positivity(x)`, `positivity_band(x)` (in-place njit kernels) and `positivity_prox(mode: int)` returning `None`/`positivity`/`positivity_band` for modes 0/1/2.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_protocols.py`)

```python
def test_identity_psi_roundtrip():
    from pfb_imaging.operators.psi import IdentityPsi

    nband, nx, ny = 2, 8, 6
    psi = IdentityPsi(nband, nx, ny)
    assert isinstance(psi, PsiOperator)
    assert (psi.nbasis, psi.nymax, psi.nxmax) == (1, nx, ny)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((nband, nx, ny))
    alpha = np.zeros((nband, 1, nx, ny))
    xo = np.zeros_like(x)
    psi.dot(x, alpha)
    psi.hdot(alpha, xo)
    np.testing.assert_array_equal(xo, x)


def test_positivity_prox_modes():
    from pfb_imaging.prox.positivity import positivity, positivity_band, positivity_prox

    assert positivity_prox(0) is None
    assert positivity_prox(1) is positivity
    assert positivity_prox(2) is positivity_band

    x = np.array([[[1.0, -1.0], [2.0, -0.5]]])  # (1, 2, 2)
    positivity(x)
    np.testing.assert_array_equal(x, [[[1.0, 0.0], [2.0, 0.0]]])

    # band mode: zero the pixel across all bands if non-positive in any band
    y = np.stack([np.full((2, 2), 1.0), np.array([[1.0, -1.0], [1.0, 1.0]])])
    positivity_band(y)
    assert y[0, 0, 1] == 0.0 and y[1, 0, 1] == 0.0
    assert y[0, 0, 0] == 1.0 and y[1, 1, 1] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_protocols.py -v -k "identity_psi or positivity"`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement**

Append to `src/pfb_imaging/operators/psi.py`:

```python
class IdentityPsi:
    """Trivial PsiOperator for image-domain regularisers (l1/ISTA, positivity).

    Coefficient buffers have shape ``(nband, 1, nx, ny)``: the trailing
    ``(nymax, nxmax)`` axes hold ``(nx, ny)`` directly (names follow the
    existing buffer convention).
    """

    def __init__(self, nband, nx, ny):
        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nbasis = 1
        self.nymax = nx
        self.nxmax = ny

    def dot(self, x, alphao):
        alphao[:, 0, :, :] = x

    def hdot(self, alpha, xo):
        xo[...] = alpha[:, 0, :, :]
```

Create `src/pfb_imaging/prox/positivity.py` (move the kernel bodies verbatim from `opt/primal_dual.py:38-57`, renamed):

```python
"""Positivity constraints as stock primal proximal operators.

These are indicator-function proxes applied in-place in the image domain;
backward solvers take one as their optional ``primal_prox`` callable.
"""

from numba import njit, prange

_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


@njit(**_FAST_JIT)
def positivity(x):
    """Clamp negative values to zero (mode 1)."""
    n = x.size
    x_f = x.ravel()
    for i in prange(n):
        if x_f[i] < 0.0:
            x_f[i] = 0.0


@njit(**_FAST_JIT)
def positivity_band(x):
    """Zero a pixel in all bands where any band is non-positive (mode 2)."""
    nband, nx, ny = x.shape
    for i in prange(nx):
        for j in range(ny):
            for b in range(nband):
                if x[b, i, j] <= 0.0:
                    for bb in range(nband):
                        x[bb, i, j] = 0.0
                    break


def positivity_prox(mode: int):
    """Map the CLI positivity mode to a primal_prox callable (or None)."""
    if mode == 0:
        return None
    if mode == 1:
        return positivity
    if mode == 2:
        return positivity_band
    raise ValueError(f"Unknown positivity mode {mode}")
```

In `src/pfb_imaging/opt/primal_dual.py`, delete the `_nb_positivity` and `_nb_positivity_band` function definitions and add near the top (keeps `deconv/sara_fb.py`/`sara_pd.py` imports working until Tasks 5–6 delete them):

```python
from pfb_imaging.prox.positivity import positivity as _nb_positivity  # noqa: F401
from pfb_imaging.prox.positivity import positivity_band as _nb_positivity_band  # noqa: F401
```

- [ ] **Step 4: Run the full suite selection to verify nothing broke**

Run: `uv run pytest tests/test_protocols.py tests/test_primal_dual.py -v`
Expected: PASS (old primal-dual tests still green via the aliases)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/operators/psi.py src/pfb_imaging/prox/positivity.py src/pfb_imaging/opt/primal_dual.py tests/test_protocols.py
git commit --no-verify -m "feat(prox): add IdentityPsi and public positivity proxes"
```

---

### Task 3: L1 regulariser (prox/l1.py)

**Files:**
- Create: `src/pfb_imaging/prox/l1.py`
- Test: `tests/test_regularisers.py`

**Interfaces:**
- Consumes: `Regulariser` Protocol, `IdentityPsi` (Tasks 1–2).
- Produces: `L1(psi, nu=1.0)` with attribute `weight` (ndarray `(nbasis, nymax, nxmax)`, initialised to ones) and `prox(v, vout, lam, sigma=1.0)` = element-wise weighted soft threshold. This is the ISTA regulariser when `psi` is `IdentityPsi`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_regularisers.py`:

```python
"""Regulariser implementations against the Regulariser Protocol contract."""

import numpy as np
from numpy.testing import assert_allclose

from pfb_imaging.deconv import Regulariser
from pfb_imaging.operators.psi import IdentityPsi


def test_l1_is_soft_threshold():
    from pfb_imaging.prox.l1 import L1

    nband, nx, ny = 2, 6, 5
    reg = L1(IdentityPsi(nband, nx, ny))
    assert isinstance(reg, Regulariser)

    rng = np.random.default_rng(1)
    v = rng.standard_normal((nband, 1, nx, ny))
    vout = np.zeros_like(v)
    lam, sigma = 0.3, 2.0
    reg.prox(v, vout, lam, sigma=sigma)

    # prox_{(lam/sigma) l1}(v/sigma) elementwise
    vs = v / sigma
    expected = np.sign(vs) * np.maximum(np.abs(vs) - lam / sigma, 0.0)
    assert_allclose(vout, expected, rtol=1e-14)


def test_l1_weighting():
    from pfb_imaging.prox.l1 import L1

    reg = L1(IdentityPsi(1, 2, 2))
    reg.weight[...] = 10.0  # threshold everything
    v = np.ones((1, 1, 2, 2))
    vout = np.zeros_like(v)
    reg.prox(v, vout, 1.0)
    assert_allclose(vout, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_regularisers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pfb_imaging.prox.l1'`

- [ ] **Step 3: Implement**

Create `src/pfb_imaging/prox/l1.py`:

```python
"""Weighted l1 regulariser g(alpha) = ||W alpha||_1 (ISTA when psi is IdentityPsi)."""

import numpy as np


class L1:
    """Satisfies the ``Regulariser`` Protocol.

    Args:
        psi: Operator satisfying ``PsiOperator`` (typically ``IdentityPsi``).
        nu: Spectral norm of ``psi`` (default 1.0).
    """

    def __init__(self, psi, nu: float = 1.0):
        self.psi = psi
        self.nu = nu
        self.weight = np.ones((psi.nbasis, psi.nymax, psi.nxmax))

    def prox(self, v, vout, lam, sigma=1.0):
        """vout = prox_{(lam/sigma) ||W .||_1}(v/sigma), in-place."""
        np.divide(v, sigma, out=vout)
        thresh = (lam / sigma) * self.weight  # broadcasts over the band axis
        np.copysign(np.maximum(np.abs(vout) - thresh, 0.0), vout, out=vout)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_regularisers.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/prox/l1.py tests/test_regularisers.py
git commit --no-verify -m "feat(prox): add L1 regulariser (ISTA building block)"
```

---

### Task 4: L21 regulariser (prox/l21.py)

**Files:**
- Create: `src/pfb_imaging/prox/l21.py`
- Test: `tests/test_regularisers.py` (extend)

**Interfaces:**
- Consumes: `prox_21m_numba`, `dual_update_numba_fast` (`prox/prox_21m.py`), `l1reweight_func` (`utils/misc.py:745`, signature `l1reweight_func(model, psih=None, outvar=None, rmsfactor=None, rms_comps=None, alpha=4)`), `PsiOperator` shape attrs.
- Produces: `L21(psi, bases, nu=1.0, rmsfactor=1.0, alpha=2.0)` with: attr `l1weight` `(nbasis, nymax, nxmax)` ones-initialised; `prox(v, vout, lam, sigma=1.0)`; fused fast path `dual_update(vp, v, lam, sigma=1.0)`; reweighting trio `init_reweighting(update)`, `update_weights(x)`, property `reweight_active`. This is the entire reweighting state formerly in `SARABase.last()` (`deconv/sara.py:197-217`) — weights now live in exactly one object.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_regularisers.py`)

```python
class SlicePsi:
    """Embeds the image into a larger coefficient grid (test helper)."""

    def __init__(self, nband, nx, ny, nbasis, nymax, nxmax):
        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nbasis = nbasis
        self.nymax = nymax
        self.nxmax = nxmax

    def dot(self, x, v):
        v[:] = 0.0
        for b in range(self.nbasis):
            v[:, b, : self.nx, : self.ny] = x

    def hdot(self, v, xout):
        xout[:] = v[:, :, : self.nx, : self.ny].sum(axis=1)


def _l21_reg(nband=2, nx=8, ny=8, nbasis=2, npad=4):
    from pfb_imaging.prox.l21 import L21

    psi = SlicePsi(nband, nx, ny, nbasis, nx + npad, ny + npad)
    return L21(psi, bases=("self", "db1"), rmsfactor=1.0, alpha=2.0), psi


def test_l21_prox_matches_kernel():
    from pfb_imaging.prox.prox_21m import prox_21m_numba

    reg, psi = _l21_reg()
    rng = np.random.default_rng(2)
    v = rng.standard_normal((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))
    got = np.zeros_like(v)
    want = np.zeros_like(v)
    reg.prox(v, got, 0.7, sigma=1.5)
    prox_21m_numba(v, want, 0.7, sigma=1.5, weight=reg.l1weight)
    assert_allclose(got, want, rtol=1e-14)


def test_l21_dual_update_matches_kernel():
    from pfb_imaging.prox.prox_21m import dual_update_numba_fast

    reg, psi = _l21_reg()
    rng = np.random.default_rng(3)
    shape = (psi.nband, psi.nbasis, psi.nymax, psi.nxmax)
    vp, v = rng.standard_normal(shape), rng.standard_normal(shape)
    vp2, v2 = vp.copy(), v.copy()
    reg.dual_update(vp, v, 0.4, sigma=2.0)
    dual_update_numba_fast(vp2, v2, 0.4, sigma=2.0, weight=reg.l1weight)
    assert_allclose(v, v2, rtol=1e-14)


def test_l21_reweighting_lifecycle():
    reg, psi = _l21_reg()
    assert not reg.reweight_active
    w0 = reg.l1weight.copy()

    rng = np.random.default_rng(4)
    update = rng.standard_normal((psi.nband, psi.nx, psi.ny))
    model = np.abs(rng.standard_normal((psi.nband, psi.nx, psi.ny)))

    reg.init_reweighting(update)
    assert reg.reweight_active
    reg.update_weights(model)
    assert reg.l1weight.shape == w0.shape
    assert not np.array_equal(reg.l1weight, w0)
    # high-SNR coefficients get weights near (1+rmsfactor)/large, i.e. weights in (0, 1+rmsfactor]
    assert (reg.l1weight > 0).all() and (reg.l1weight <= 1.0 + reg.rmsfactor).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_regularisers.py -v -k l21`
Expected: FAIL with `ModuleNotFoundError: No module named 'pfb_imaging.prox.l21'`

- [ ] **Step 3: Implement**

Create `src/pfb_imaging/prox/l21.py`:

```python
"""Weighted l21 regulariser over a wavelet dictionary (the SARA prior)."""

from functools import partial

import numpy as np

from pfb_imaging.prox.prox_21m import dual_update_numba_fast, prox_21m_numba
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.misc import l1reweight_func

log = pfb_logging.get_logger("L21")


class L21:
    """Satisfies the ``Regulariser`` Protocol; owns the l1-reweighting state.

    R(x) = ||W Psi^T x||_{2,1} with the 2-norm over the band axis.

    Args:
        psi: Operator satisfying ``PsiOperator`` (e.g. ``Psi``/``PsiNocopytRay``).
        bases: Wavelet basis names, one per ``psi.nbasis`` (for logging).
        nu: Spectral norm of ``psi`` (default 1.0; tight frame).
        rmsfactor: Threshold factor in the reweighting formula.
        alpha: Exponent in the reweighting formula.
    """

    def __init__(self, psi, bases, nu: float = 1.0, rmsfactor: float = 1.0, alpha: float = 2.0):
        self.psi = psi
        self.nu = nu
        self.bases = tuple(bases)
        self.rmsfactor = rmsfactor
        self.alpha = alpha
        self.l1weight = np.ones((psi.nbasis, psi.nymax, psi.nxmax))
        self._outvar = np.zeros((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))
        self._reweighter = None

    def prox(self, v, vout, lam, sigma=1.0):
        """vout = prox_{(lam/sigma) ||W .||_{21}}(v/sigma), in-place."""
        prox_21m_numba(v, vout, lam, sigma=sigma, weight=self.l1weight)

    def dual_update(self, vp, v, lam, sigma=1.0):
        """Fused primal-dual dual update (fast path sniffed by PrimalDual)."""
        dual_update_numba_fast(vp, v, lam, sigma=sigma, weight=self.l1weight)

    @property
    def reweight_active(self) -> bool:
        """True once reweighting has been initialised."""
        return self._reweighter is not None

    def init_reweighting(self, update):
        """Estimate per-basis component rms from the update and arm reweighting."""
        self.psi.dot(update, self._outvar)
        tmp = np.sum(self._outvar, axis=0)
        rms_comps = np.ones(self.psi.nbasis, dtype=float)
        for i, base in enumerate(self.bases):
            tmpb = tmp[i]
            rms_comps[i] = np.std(tmpb[tmpb != 0])
            log.info(f"rms_comps for base {base} is {rms_comps[i]:.3e}")
        self._reweighter = partial(
            l1reweight_func,
            psih=self.psi.dot,
            outvar=self._outvar,
            rmsfactor=self.rmsfactor,
            rms_comps=rms_comps,
            alpha=self.alpha,
        )

    def update_weights(self, x):
        """Recompute l1 weights from the current model/iterate."""
        self.l1weight = self._reweighter(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_regularisers.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/prox/l21.py tests/test_regularisers.py
git commit --no-verify -m "feat(prox): add L21 regulariser owning reweighting state"
```

---

### Task 5: Concrete ForwardBackward (generic tight-frame prox) + delete sara_fb.py

**Files:**
- Rewrite: `src/pfb_imaging/opt/forward_backward.py`
- Delete: `src/pfb_imaging/deconv/sara_fb.py` (its only consumer was the old ABC)
- Test: `tests/test_forward_backward.py`

**Interfaces:**
- Consumes: `Regulariser` (Task 1), `L1`/`IdentityPsi` (Tasks 2–3), `L21` (Task 4), `_nb_norm_diff`/`_nb_any_nonzero` from `opt/primal_dual.py`.
- Produces: concrete `ForwardBackward(tol=1e-5, maxit=1000, report_freq=10, verbosity=1, gamma=1.0, acceleration=True, on_converge=None, primal_prox=None)` satisfying `BackwardSolver`: `setup(prox, hessnorm)` (computes `step = 2*gamma/hessnorm`, sizes coefficient buffers from `prox.psi`), `set_grad(grad)`, `solve(x, lam) -> x`, `reset()`. `on_converge(x, k, eps) -> bool` fires when `eps < tol`; return `False` to keep iterating (inner reweighting), `True`/absent to stop. The tight-frame prox `x + (1/nu)*Psi(prox_g(Psi^T x) - Psi^T x)` is written ONCE here, generic in the regulariser.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forward_backward.py`:

```python
"""Concrete ForwardBackward solver (composition, no subclassing)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.operators.psi import IdentityPsi
from pfb_imaging.opt import BackwardSolver
from pfb_imaging.opt.forward_backward import ForwardBackward
from pfb_imaging.prox.l1 import L1

pmp = pytest.mark.parametrize


@pmp("acceleration", [True, False])
@pmp("lam", [0.1, 1.0])
def test_lasso_analytic(acceleration, lam):
    """min 0.5||x - b||^2 + lam||x||_1 -> soft threshold of b."""
    nband, nx, ny = 1, 50, 4
    rng = np.random.default_rng(0)
    b = rng.standard_normal((nband, nx, ny))

    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-10, maxit=5000, verbosity=0, gamma=0.45, acceleration=acceleration)
    assert isinstance(fb, BackwardSolver)
    fb.setup(reg, hessnorm=1.0)  # step = 0.9 < 1/L
    fb.set_grad(lambda x: x - b)
    x = fb.solve(np.zeros_like(b), lam)

    x_star = np.sign(b) * np.maximum(np.abs(b) - lam, 0.0)
    assert_allclose(x, x_star, atol=1e-4)


def test_generic_tight_frame_matches_handcoded_l21():
    """The generic tight-frame prox reproduces the old L21ForwardBackward.prox."""
    from pfb_imaging.prox.l21 import L21
    from pfb_imaging.prox.prox_21m import prox_21m_numba
    from tests.test_regularisers import SlicePsi

    nband, nx, ny, nbasis = 2, 8, 8, 2
    psi = SlicePsi(nband, nx, ny, nbasis, nx + 4, ny + 4)
    reg = L21(psi, bases=("self", "db1"))
    nu = 2.0
    reg.nu = nu

    fb = ForwardBackward(verbosity=0)
    fb.setup(reg, hessnorm=1.0)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((nband, nx, ny))
    lam = 0.3

    # reference: the deleted L21ForwardBackward.prox, hand-coded
    alpha = np.zeros((nband, nbasis, psi.nymax, psi.nxmax))
    buf = np.zeros_like(alpha)
    xout = np.zeros_like(x)
    psi.dot(x, alpha)
    prox_21m_numba(alpha, buf, fb.step * lam, sigma=1.0, weight=reg.l1weight)
    buf -= alpha
    psi.hdot(buf, xout)
    want = x + xout / nu

    got = fb._apply_prox(x.copy(), lam)
    assert_allclose(got, want, rtol=1e-13)


def test_on_converge_continues_iteration():
    calls = []

    def cb(x, k, eps):
        calls.append(k)
        return len(calls) >= 3  # keep going twice, stop on the third event

    nband, nx, ny = 1, 10, 1
    b = np.ones((nband, nx, ny))
    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-6, maxit=1000, verbosity=0, gamma=0.45, on_converge=cb)
    fb.setup(reg, hessnorm=1.0)
    fb.set_grad(lambda x: x - b)
    fb.solve(np.zeros_like(b), 0.1)
    assert len(calls) == 3


def test_primal_prox_applied():
    from pfb_imaging.prox.positivity import positivity

    nband, nx, ny = 1, 20, 1
    b = np.linspace(-1, 1, nx).reshape(nband, nx, ny)
    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-10, maxit=2000, verbosity=0, gamma=0.45, primal_prox=positivity)
    fb.setup(reg, hessnorm=1.0)
    fb.set_grad(lambda x: x - b)
    x = fb.solve(np.zeros_like(b), 0.05)
    assert (x >= 0).all()


def test_solve_raises_without_setup_or_grad():
    fb = ForwardBackward(verbosity=0)
    with pytest.raises(RuntimeError, match="setup"):
        fb.solve(np.zeros((1, 2, 2)), 1.0)
    fb.setup(L1(IdentityPsi(1, 2, 2)), hessnorm=1.0)
    with pytest.raises(RuntimeError, match="set_grad"):
        fb.solve(np.zeros((1, 2, 2)), 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forward_backward.py -v`
Expected: FAIL (old ForwardBackward is an ABC requiring a `prox` override; constructor signature differs)

- [ ] **Step 3: Rewrite `src/pfb_imaging/opt/forward_backward.py`** (full replacement)

```python
"""Forward-backward splitting (with optional FISTA acceleration).

Concrete solver satisfying the ``BackwardSolver`` Protocol: the regulariser
arrives as data via ``setup`` — no subclassing.  The tight-frame proximal
composition ``x + (1/nu) * Psi(prox_g(Psi^T x) - Psi^T x)`` is implemented
once here, generically for any ``Regulariser``.
"""

from time import time

import numpy as np

from pfb_imaging.opt.primal_dual import _nb_any_nonzero, _nb_norm_diff
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("FB")


class ForwardBackward:
    """Forward-backward solver for ``min_x f(x) + lam * R(x)``.

    At each iteration computes ``x = tight_frame_prox(y - step * grad(y), lam)``
    with optional FISTA momentum on ``y``.

    Args:
        tol: Convergence tolerance on relative primal change.
        maxit: Maximum iterations.
        report_freq: Log every this many iterations at verbosity > 1.
        verbosity: 0 silent, 1 convergence message, 2 per-iter logging.
        gamma: Step-size safety factor; ``step = 2 * gamma / hessnorm``.
        acceleration: Enable FISTA momentum (ISTA when False).
        on_converge: Optional ``cb(x, k, eps) -> bool`` fired when
            ``eps < tol``; return False to continue iterating.
        primal_prox: Optional in-place image-domain prox (e.g. positivity)
            applied after the tight-frame step.
    """

    def __init__(
        self,
        tol: float = 1e-5,
        maxit: int = 1000,
        report_freq: int = 10,
        verbosity: int = 1,
        gamma: float = 1.0,
        acceleration: bool = True,
        on_converge=None,
        primal_prox=None,
    ):
        self.tol = tol
        self.maxit = maxit
        self.report_freq = report_freq
        self.verbosity = verbosity
        self.gamma = gamma
        self.acceleration = acceleration
        self.on_converge = on_converge
        self.primal_prox = primal_prox
        self._grad = None
        self._reg = None

    def setup(self, prox, hessnorm: float) -> None:
        """Bind the regulariser, compute the step size, size buffers."""
        self._reg = prox
        self.hessnorm = hessnorm
        self.step = 2.0 * self.gamma / hessnorm
        psi = prox.psi
        self._alpha = np.zeros((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))
        self._alpha_buf = np.zeros_like(self._alpha)
        self._xout = np.zeros((psi.nband, psi.nx, psi.ny))

    def set_grad(self, grad) -> None:
        """Set the gradient of the smooth data-fidelity term."""
        self._grad = grad

    def reset(self) -> None:
        """No warm-start state beyond x itself."""

    def _apply_prox(self, x, lam):
        """Generic tight-frame prox of ``lam * g(Psi^T x)``, then primal_prox."""
        reg = self._reg
        reg.psi.dot(x, self._alpha)
        reg.prox(self._alpha, self._alpha_buf, self.step * lam, sigma=1.0)
        self._alpha_buf -= self._alpha
        reg.psi.hdot(self._alpha_buf, self._xout)
        x += self._xout / reg.nu
        if self.primal_prox is not None:
            self.primal_prox(x)
        return x

    def solve(self, x, lam: float):
        """Run the forward-backward loop; returns the final iterate."""
        if self._reg is None:
            raise RuntimeError("regulariser not bound; call setup() before solve()")
        if self._grad is None:
            raise RuntimeError("grad not set; call set_grad() before solve()")

        xp = x.copy()
        y = x.copy()
        t = 1.0
        eps = 1.0
        tii = time()
        for k in range(self.maxit):
            x = y - self.step * self._grad(y)
            x = self._apply_prox(x, lam)

            eps = _nb_norm_diff(x, xp) if _nb_any_nonzero(x) else 1.0
            if eps < self.tol:
                if self.on_converge is None or self.on_converge(x, k, eps):
                    break

            if self.acceleration:
                tp = t
                t = (1.0 + np.sqrt(1.0 + 4.0 * tp**2)) / 2.0
                y = x + (tp - 1.0) / t * (x - xp)
            else:
                np.copyto(y, x)
            np.copyto(xp, x)

            if not k % self.report_freq and self.verbosity > 1:
                log.info(f"At iteration {k} eps = {eps:.3e}")

        ttot = time() - tii
        if self.verbosity > 1:
            log.info(f"Total time: {ttot:.3f}s  ({ttot / max(k + 1, 1) * 1e3:.1f} ms/iter)")
        if k == self.maxit - 1:
            if self.verbosity:
                log.info(f"Max iters reached. eps = {eps:.3e}")
        elif self.verbosity:
            log.info(f"Success, converged after {k} iterations")
        return x
```

- [ ] **Step 4: Delete the old subclass module**

```bash
git rm src/pfb_imaging/deconv/sara_fb.py
```

(`core/deconv.py`'s lazy `from pfb_imaging.deconv.sara_fb import ...` is inside its function body — import-time green, rewritten in Task 11.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_forward_backward.py tests/test_protocols.py tests/test_regularisers.py -v`
Expected: PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add -A src/pfb_imaging/opt/forward_backward.py src/pfb_imaging/deconv/ tests/test_forward_backward.py
git commit --no-verify -m "refactor(opt): concrete ForwardBackward with generic tight-frame prox"
```

---

### Task 6: Concrete PrimalDual + delete sara_pd.py/sara.py + rewrite test_primal_dual.py

**Files:**
- Modify: `src/pfb_imaging/opt/primal_dual.py` (replace the `PrimalDual` ABC class only; module-level functions `primal_dual`/`primal_dual_numba` and njit kernels stay byte-identical)
- Delete: `src/pfb_imaging/deconv/sara_pd.py`, `src/pfb_imaging/deconv/sara.py`
- Rewrite: `tests/test_primal_dual.py`

**Interfaces:**
- Consumes: `Regulariser`, `L1`, `L21`, `IdentityPsi`; kernels `_nb_extrapolate_dual`, `_nb_primal_step`, `_nb_norm_diff`, `_nb_any_nonzero` (same module).
- Produces: concrete `PrimalDual(tol=1e-5, maxit=1000, report_freq=10, verbosity=1, gamma=1.0, sigma=None, on_converge=None, primal_prox=None)` satisfying `BackwardSolver`. `setup(prox, hessnorm)` computes `sigma = hessnorm/(2*gamma)/nu` (when None) and `tau = 0.98/(hessnorm/(2*gamma) + sigma*nu**2)`, allocates the internal dual `self._v` (warm-started across solves; `reset()` zeros it). The dual step prefers the fused `reg.dual_update(vp, v, lam, sigma)` when present, else generic Moreau via `reg.prox`.

- [ ] **Step 1: Rewrite `tests/test_primal_dual.py`** (full replacement)

```python
"""Concrete PrimalDual solver: composition, Moreau/fused equivalence, legacy oracle."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.operators.psi import IdentityPsi
from pfb_imaging.opt import BackwardSolver
from pfb_imaging.opt.primal_dual import PrimalDual, primal_dual_numba
from pfb_imaging.prox.l1 import L1
from pfb_imaging.prox.l21 import L21
from pfb_imaging.prox.positivity import positivity
from tests.test_regularisers import SlicePsi

pmp = pytest.mark.parametrize


def _l21_problem(nband, nx, ny, nbasis=2, npad=0, seed=42):
    """Small diagonal-Hessian imaging problem (mirrors the old test helper)."""
    nymax, nxmax = nx + npad, ny + npad
    rng = np.random.default_rng(seed)
    diag = rng.uniform(0.5, 2.0, size=(nband, nx, ny))
    diag_sq = diag * diag
    hessnorm = float(np.max(diag_sq))
    dirty = rng.uniform(1.0, 5.0, size=(nband, nx, ny))
    diag_dirty = diag * dirty

    def grad(x):
        return diag_sq * x - diag_dirty

    psi = SlicePsi(nband, nx, ny, nbasis, nymax, nxmax)
    l1weight = rng.uniform(0.01, 0.1, size=(nbasis, nymax, nxmax))
    return psi, grad, l1weight, hessnorm


@pmp("n", [50, 200])
@pmp("lam", [0.1, 1.0])
def test_lasso_identity_analytic(n, lam):
    """PD + L1 + IdentityPsi recovers the soft-threshold solution (Moreau path)."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((1, n, 1))
    x_star = np.sign(b) * np.maximum(np.abs(b) - lam, 0.0)

    reg = L1(IdentityPsi(1, n, 1))
    pd = PrimalDual(tol=1e-10, maxit=5000, verbosity=0)
    assert isinstance(pd, BackwardSolver)
    pd.setup(reg, hessnorm=1.0)
    pd.set_grad(lambda x: x - b)
    x = pd.solve(np.zeros_like(b), lam)
    assert_allclose(x, x_star, atol=1e-4)


@pmp("nband", [1, 3])
@pmp("nx", [16, 32])
@pmp("positivity_mode", [0, 1])
def test_l21_matches_primal_dual_numba(nband, nx, positivity_mode):
    """New PrimalDual + L21 reproduces the legacy primal_dual_numba trajectories."""
    ny = nx
    psi, grad, l1weight, hessnorm = _l21_problem(nband, nx, ny)
    lam, tol, maxit = 0.05, 1e-8, 30

    def psi_func(x, v):
        psi.dot(x, v)

    def psih_func(v, xout):
        psi.hdot(v, xout)

    shape_v = (nband, psi.nbasis, psi.nymax, psi.nxmax)
    x_ref, v_ref = primal_dual_numba(
        np.zeros((nband, nx, ny)),
        np.zeros(shape_v),
        lam,
        psih_func,  # 4th positional (named psih) is the SYNTHESIS op in the body
        psi_func,  # 5th positional (named psi) is the ANALYSIS op in the body
        hessnorm,
        prox=None,
        l1weight=l1weight,
        reweighter=None,
        grad=grad,
        nu=1.0,
        tol=tol,
        maxit=maxit,
        positivity=positivity_mode,
        verbosity=0,
    )

    reg = L21(psi, bases=("self", "db1"))
    reg.l1weight = l1weight
    pd = PrimalDual(tol=tol, maxit=maxit, verbosity=0, primal_prox=positivity if positivity_mode else None)
    pd.setup(reg, hessnorm)
    pd.set_grad(grad)
    x_new = pd.solve(np.zeros((nband, nx, ny)), lam)

    def rdiff(a, b):
        return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)

    assert rdiff(x_new, x_ref) < 1e-10
    assert rdiff(pd._v, v_ref) < 1e-10


def test_fused_and_moreau_paths_agree():
    """PD via reg.dual_update (fused) == PD via generic reg.prox (Moreau)."""
    nband, nx = 2, 16
    psi, grad, l1weight, hessnorm = _l21_problem(nband, nx, nx)
    reg = L21(psi, bases=("self", "db1"))
    reg.l1weight = l1weight

    class MoreauOnly:
        """Same regulariser with the fused fast path hidden."""

        def __init__(self, inner):
            self.psi = inner.psi
            self.nu = inner.nu
            self.prox = inner.prox

    lam, tol, maxit = 0.05, 1e-8, 25
    x0 = np.zeros((nband, nx, nx))

    pd1 = PrimalDual(tol=tol, maxit=maxit, verbosity=0)
    pd1.setup(reg, hessnorm)
    pd1.set_grad(grad)
    x_fused = pd1.solve(x0.copy(), lam)

    pd2 = PrimalDual(tol=tol, maxit=maxit, verbosity=0)
    pd2.setup(MoreauOnly(reg), hessnorm)
    pd2.set_grad(grad)
    x_moreau = pd2.solve(x0.copy(), lam)

    assert_allclose(x_fused, x_moreau, rtol=1e-10, atol=1e-12)


def test_dual_warm_start_and_reset():
    nband, nx = 1, 8
    psi, grad, l1weight, hessnorm = _l21_problem(nband, nx, nx)
    reg = L21(psi, bases=("self", "db1"))
    reg.l1weight = l1weight
    pd = PrimalDual(tol=1e-8, maxit=20, verbosity=0)
    pd.setup(reg, hessnorm)
    pd.set_grad(grad)
    pd.solve(np.zeros((nband, nx, nx)), 0.05)
    assert np.any(pd._v)  # dual retained for warm start
    pd.reset()
    assert not np.any(pd._v)


def test_solve_raises_without_setup_or_grad():
    pd = PrimalDual(verbosity=0)
    with pytest.raises(RuntimeError, match="setup"):
        pd.solve(np.zeros((1, 2, 2)), 1.0)
    pd.setup(L1(IdentityPsi(1, 2, 2)), hessnorm=1.0)
    with pytest.raises(RuntimeError, match="set_grad"):
        pd.solve(np.zeros((1, 2, 2)), 1.0)
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_primal_dual.py -v`
Expected: FAIL (old PrimalDual is an ABC with different constructor/solve signatures)

- [ ] **Step 3: Replace the `PrimalDual` class in `src/pfb_imaging/opt/primal_dual.py`**

Delete the entire old `class PrimalDual(ABC)` block (and the now-unused `from abc import ABC, abstractmethod` and `from pfb_imaging.operators import PsiOperatorProtocol` imports) and add:

```python
class PrimalDual:
    """Primal-dual solver for ``min_x f(x) + lam * g(Psi^T x)``.

    Concrete class satisfying the ``BackwardSolver`` Protocol; the
    regulariser arrives via ``setup``.  The dual update prefers the fused
    ``reg.dual_update(vp, v, lam, sigma)`` fast path when the regulariser
    provides one, falling back to the generic Moreau decomposition
    ``v = vtilde - sigma * prox_{(lam/sigma) g}(vtilde/sigma)`` via
    ``reg.prox``.  The dual variable is internal state, warm-started
    across ``solve`` calls; ``reset()`` zeros it.

    Args:
        tol: Convergence tolerance on relative primal change.
        maxit: Maximum iterations.
        report_freq: Log every this many iterations at verbosity > 1.
        verbosity: 0 silent, 1 convergence message, 2 per-iter logging.
        gamma: Step-size safety factor.
        sigma: Dual step size; computed from hessnorm/gamma/nu when None.
        on_converge: Optional ``cb(x, k, eps) -> bool`` fired when
            ``eps < tol``; return False to continue iterating.
        primal_prox: Optional in-place image-domain prox (e.g. positivity).
    """

    def __init__(
        self,
        tol: float = 1e-5,
        maxit: int = 1000,
        report_freq: int = 10,
        verbosity: int = 1,
        gamma: float = 1.0,
        sigma: float | None = None,
        on_converge=None,
        primal_prox=None,
    ):
        self.tol = tol
        self.maxit = maxit
        self.report_freq = report_freq
        self.verbosity = verbosity
        self.gamma = gamma
        self._sigma_opt = sigma
        self.on_converge = on_converge
        self.primal_prox = primal_prox
        self._grad = None
        self._reg = None
        self._v = None

    def setup(self, prox, hessnorm: float) -> None:
        """Bind the regulariser, compute step sizes, allocate the dual."""
        self._reg = prox
        self.hessnorm = hessnorm
        nu = prox.nu
        sigma = self._sigma_opt
        if sigma is None:
            sigma = hessnorm / (2.0 * self.gamma) / nu
        self.sigma = sigma
        self.tau = 0.98 / (hessnorm / (2.0 * self.gamma) + sigma * nu**2)
        psi = prox.psi
        self._v = np.zeros((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))

    def set_grad(self, grad) -> None:
        """Set the gradient of the smooth data-fidelity term."""
        self._grad = grad

    def reset(self) -> None:
        """Drop the warm-started dual variable."""
        if self._v is not None:
            self._v[...] = 0.0

    def _dual_step(self, xp, v, vp, lam):
        """Analysis + dual proximal update; fused fast path when available."""
        reg = self._reg
        reg.psi.dot(xp, v)
        if hasattr(reg, "dual_update"):
            reg.dual_update(vp, v, lam, sigma=self.sigma)
        else:
            # generic Moreau: v holds Psi^T xp on entry
            vtilde = vp + self.sigma * v
            reg.prox(vtilde, v, lam, sigma=self.sigma)
            np.subtract(vtilde, self.sigma * v, out=v)

    def solve(self, x, lam: float):
        """Run the primal-dual loop; returns the final primal iterate."""
        if self._reg is None:
            raise RuntimeError("regulariser not bound; call setup() before solve()")
        if self._grad is None:
            raise RuntimeError("grad not set; call set_grad() before solve()")

        xp = x.copy()
        v = self._v
        vp = v.copy()
        xout = np.zeros_like(x)

        eps = 1.0
        tii = time()
        for k in range(self.maxit):
            self._dual_step(xp, v, vp, lam)
            _nb_extrapolate_dual(v, vp)
            self._reg.psi.hdot(vp, xout)
            xout += self._grad(xp)
            _nb_primal_step(x, xp, xout, self.tau)
            if self.primal_prox is not None:
                self.primal_prox(x)

            eps = _nb_norm_diff(x, xp) if _nb_any_nonzero(x) else 1.0
            if eps < self.tol:
                if self.on_converge is None or self.on_converge(x, k, eps):
                    break

            np.copyto(xp, x)
            np.copyto(vp, v)

            if not k % self.report_freq and self.verbosity > 1:
                log.info(f"At iteration {k} eps = {eps:.3e}")

        ttot = time() - tii
        if self.verbosity > 1:
            log.info(f"Total time: {ttot:.3f}s  ({ttot / max(k + 1, 1) * 1e3:.1f} ms/iter)")
        if k == self.maxit - 1:
            if self.verbosity:
                log.info(f"Max iters reached. eps = {eps:.3e}")
        elif self.verbosity:
            log.info(f"Success, converged after {k} iterations")
        return x
```

- [ ] **Step 4: Delete the old subclass modules**

```bash
git rm src/pfb_imaging/deconv/sara_pd.py src/pfb_imaging/deconv/sara.py
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_primal_dual.py tests/test_forward_backward.py tests/test_protocols.py -v`
Expected: PASS. Also confirm the legacy oracle still imports: `uv run python -c "from pfb_imaging.opt.primal_dual import primal_dual, primal_dual_numba"`

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add -A src/pfb_imaging/opt/primal_dual.py src/pfb_imaging/deconv/ tests/test_primal_dual.py
git commit --no-verify -m "refactor(opt): concrete PrimalDual with internal warm-started dual"
```

---

### Task 7: PCG ForwardSolver

**Files:**
- Modify: `src/pfb_imaging/opt/pcg.py` (append the `PCG` class; existing functions untouched)
- Test: `tests/test_pcg_solver.py`

**Interfaces:**
- Consumes: `pcg_numba(aop, b, x0=None, precond=None, tol=1e-5, maxit=500, minit=100, verbosity=1, report_freq=10, ...)` (same module), `ForwardSolver`/`LinearOperator` Protocols.
- Produces: `PCG(tol=1e-3, maxit=150, minit=1, verbosity=0, report_freq=10)` satisfying `ForwardSolver`. `solve(hess, residual, x0=None)`: if `hasattr(hess, "cg")` delegate to the operator's distributed per-band CG (`hess.cg(residual, x0=x0, tol=..., maxit=..., minit=...)`), else run `pcg_numba` over `hess.dot` on the whole cube.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pcg_solver.py`:

```python
"""PCG ForwardSolver: generic path and duck-typed distributed fast path."""

import numpy as np
from numpy.testing import assert_allclose

from pfb_imaging.opt import ForwardSolver
from pfb_imaging.opt.pcg import PCG


class DiagOp:
    def __init__(self, d):
        self.d = d

    def dot(self, x):
        return self.d * x

    def hdot(self, x):
        return self.dot(x)


def test_pcg_solves_diagonal_system():
    rng = np.random.default_rng(0)
    d = rng.uniform(1.0, 3.0, size=(2, 8, 8))
    b = rng.standard_normal((2, 8, 8))
    pcg = PCG(tol=1e-12, maxit=500, minit=1, verbosity=0)
    assert isinstance(pcg, ForwardSolver)
    x = pcg.solve(DiagOp(d), b)
    assert_allclose(x, b / d, atol=1e-8)


def test_pcg_delegates_to_operator_cg():
    class FakeHess:
        def __init__(self):
            self.called_with = None

        def dot(self, x):  # pragma: no cover - must not be used
            raise AssertionError("generic path used despite cg fast path")

        def cg(self, rhs, x0=None, tol=None, maxit=None, minit=None):
            self.called_with = (tol, maxit, minit)
            return rhs * 2.0

    hess = FakeHess()
    pcg = PCG(tol=1e-4, maxit=77, minit=3, verbosity=0)
    out = pcg.solve(hess, np.ones((1, 2, 2)))
    assert hess.called_with == (1e-4, 77, 3)
    assert_allclose(out, 2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pcg_solver.py -v`
Expected: FAIL with `ImportError: cannot import name 'PCG'`

- [ ] **Step 3: Implement** (append to `src/pfb_imaging/opt/pcg.py`)

```python
class PCG:
    """Conjugate-gradient ForwardSolver: ``update ≈ hess^{-1} residual``.

    Satisfies the ``ForwardSolver`` Protocol.  When the operator exposes a
    ``cg`` method (the distributed per-band fast path of ``HessTreeRay``),
    the solve is delegated to it with this solver's controls; otherwise a
    generic cube-level CG runs over ``hess.dot``.

    Args:
        tol: CG convergence tolerance.
        maxit: Maximum CG iterations.
        minit: Minimum CG iterations.
        verbosity: 0 silent, > 1 per-iteration reporting.
        report_freq: Reporting cadence at verbosity > 1.
    """

    def __init__(
        self,
        tol: float = 1e-3,
        maxit: int = 150,
        minit: int = 1,
        verbosity: int = 0,
        report_freq: int = 10,
    ):
        self.tol = tol
        self.maxit = maxit
        self.minit = minit
        self.verbosity = verbosity
        self.report_freq = report_freq

    def solve(self, hess, residual, x0=None):
        """Solve ``hess @ update = residual`` for update."""
        if hasattr(hess, "cg"):
            return hess.cg(residual, x0=x0, tol=self.tol, maxit=self.maxit, minit=self.minit)
        return pcg_numba(
            hess.dot,
            residual,
            x0=x0,
            tol=self.tol,
            maxit=self.maxit,
            minit=self.minit,
            verbosity=self.verbosity,
            report_freq=self.report_freq,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pcg_solver.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/opt/pcg.py tests/test_pcg_solver.py
git commit --no-verify -m "feat(opt): add PCG ForwardSolver with duck-typed distributed path"
```

---

### Task 8: HessTreeRay band actors (+ HessianTree wsum override)

**Files:**
- Modify: `src/pfb_imaging/operators/hessian.py` (add optional `wsum` arg to `HessianTree.__init__`; append `_HessBandActorImpl` and `HessTreeRay`)
- Test: `tests/test_hess_tree_ray.py`

**Interfaces:**
- Consumes: `HessianTree` (same file), `pcg_numba` (lazy import inside methods — `import ray` also stays lazy per architecture §3), the `_PsiBandActorImpl` TBB-loading pattern (`operators/psi.py:682-695`), session `manage_ray` fixture in tests.
- Produces: `HessianTree(partitions, nx, ny, nx_psf, ny_psf, eta=0.0, nthreads=1, wsum=None)` — `wsum` overrides the sum-over-partitions normalisation (the factory passes the TOTAL wsum so per-band operators match the legacy total-normalised convention). `HessTreeRay(partitions_per_band, nx, ny, nx_psf, ny_psf, etas=0.0, nthreads=1, wsums=None, cg_tol=1e-3, cg_maxit=150, cg_minit=1, cg_verbose=0)` — cube-level `LinearOperator` with `dot(x)`/`hdot(x)` on `(nband, nx, ny)` cubes and the distributed `cg(rhs, x0=None, tol=None, maxit=None, minit=None)` fast path. One actor per band (v1; `nband == 1` falls back to a local `HessianTree`, no Ray). `etas`/`wsums` are scalars or per-band sequences.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hess_tree_ray.py`:

```python
"""HessTreeRay: Ray band-actor Hessian vs local HessianTree/HessPSF (tier 2)."""

import numpy as np
import pytest
from ducc0.fft import r2c
from numpy.testing import assert_allclose

from pfb_imaging.operators import LinearOperator
from pfb_imaging.operators.hessian import HessianTree, HessPSF, HessTreeRay

pmp = pytest.mark.parametrize


def _rand_part(rng, nx, ny, nx_psf, ny_psf, wsum=1.0):
    """Partition dict with a positive real psfhat (|FT of a random psf|)."""
    psf = rng.uniform(0.0, 1.0, size=(nx_psf, ny_psf))
    psfhat = np.abs(r2c(psf, axes=(0, 1), forward=True, inorm=0))[None]
    beam = np.ones((1, nx, ny))
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([wsum])}


def test_wsum_override():
    rng = np.random.default_rng(0)
    nx = ny = 8
    part = _rand_part(rng, nx, ny, 2 * nx, 2 * ny, wsum=4.0)
    x = rng.standard_normal((1, nx, ny))
    default = HessianTree([part], nx, ny, 2 * nx, 2 * ny).dot(x)
    overridden = HessianTree([part], nx, ny, 2 * nx, 2 * ny, wsum=8.0).dot(x)
    assert_allclose(overridden, default / 2.0, rtol=1e-13)


@pmp("nband", [1, 3])
def test_dot_matches_local_hessian_tree(nband):
    rng = np.random.default_rng(1)
    nx = ny = 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny) for _ in range(2)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.01)
    assert isinstance(hess, LinearOperator)

    x = rng.standard_normal((nband, nx, ny))
    got = hess.dot(x)
    want = np.zeros_like(x)
    for b in range(nband):
        local = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=0.01)
        want[b] = local.dot(x[b])[0]
    assert_allclose(got, want, rtol=1e-12)


def test_dot_matches_hess_psf_single_partition():
    """Single partition, unit wsum, no beam, eta=0: HessTreeRay == HessPSF."""
    rng = np.random.default_rng(2)
    nband, nx, ny = 2, 16, 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    abspsf = np.concatenate([p[0]["psfhat"] for p in parts], axis=0)

    tree = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.0)
    ref = HessPSF(nx, ny, abspsf, eta=0.0)

    x = rng.standard_normal((nband, nx, ny))
    assert_allclose(tree.dot(x), ref.dot(x).copy(), rtol=1e-12)


@pmp("nband", [1, 3])
def test_cg_matches_local_pcg(nband):
    from pfb_imaging.opt.pcg import pcg_numba

    rng = np.random.default_rng(3)
    nx = ny = 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.5, cg_tol=1e-8, cg_maxit=200)

    rhs = rng.standard_normal((nband, nx, ny))
    got = hess.cg(rhs)
    for b in range(nband):
        local = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=0.5)
        want_b = pcg_numba(lambda z: local.dot(z)[0], rhs[b], tol=1e-8, maxit=200, minit=1, verbosity=0)
        assert_allclose(got[b], want_b, rtol=1e-6, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hess_tree_ray.py -v`
Expected: FAIL with `ImportError: cannot import name 'HessTreeRay'` (and `TypeError` for the new `wsum` kwarg)

- [ ] **Step 3: Add the `wsum` override to `HessianTree.__init__`**

Replace the wsum block in `HessianTree.__init__` (`operators/hessian.py:469-471`) and extend the signature:

```python
    def __init__(self, partitions, nx, ny, nx_psf, ny_psf, eta=0.0, nthreads=1, wsum=None):
```

```python
        self.ncorr = partitions[0]["wsum"].size
        if wsum is None:
            self.wsum = np.zeros(self.ncorr)
            for p in partitions:
                self.wsum += p["wsum"]
        else:
            # explicit normalisation (e.g. TOTAL wsum across all bands so the
            # per-band operator matches the legacy total-normalised convention)
            self.wsum = np.broadcast_to(np.asarray(wsum, dtype=float), (self.ncorr,)).copy()
```

Also update the class docstring Args to document `wsum`.

- [ ] **Step 4: Append `_HessBandActorImpl` and `HessTreeRay` to `operators/hessian.py`**

```python
class _HessBandActorImpl:
    """Ray actor holding one band's HessianTree (see _PsiBandActorImpl).

    The HessianTree (with its preallocated FFT scratch) is built once in
    the actor process; per-call payloads are plain numpy arrays.
    """

    def __init__(self, partitions, nx, ny, nx_psf, ny_psf, eta, nthreads, wsum=None):
        # Load TBB in this actor process (ctypes.CDLL in the main process
        # does not carry over to forked/spawned Ray workers).
        import ctypes
        import importlib.metadata

        import numba

        dist = importlib.metadata.distribution("tbb")
        for f in dist.files:
            if str(f).endswith("/libtbb.so"):
                ctypes.CDLL(str(dist.locate_file(f).resolve()))
                break
        numba.set_num_threads(min(nthreads, numba.config.NUMBA_NUM_THREADS))

        self._hess = HessianTree(partitions, nx, ny, nx_psf, ny_psf, eta=eta, nthreads=nthreads, wsum=wsum)
        self._nx = nx
        self._ny = ny

    def dot(self, x):
        return self._hess.dot(x)

    def cg(self, rhs, x0, tol, maxit, minit, verbosity):
        from pfb_imaging.opt.pcg import pcg_numba

        return pcg_numba(
            lambda z: self._hess.dot(z)[0],
            rhs,
            x0=x0,
            tol=tol,
            maxit=maxit,
            minit=minit,
            verbosity=verbosity,
        )

    def warmup(self):
        """Trigger JIT/FFT-plan warmup so the first real call is fast."""
        self._hess.dot(np.zeros((self._nx, self._ny)))

    def get_mem(self):
        """Post-gc memory telemetry (docs/msv4-memory-patterns.md)."""
        import gc
        import os
        import resource

        import psutil

        gc.collect()
        return {
            "pid": os.getpid(),
            "rss_gb": psutil.Process().memory_info().rss / 2**30,
            "peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30,
        }


class HessTreeRay:
    """Cube-level Hessian distributing per-band HessianTree over Ray actors.

    Satisfies the ``LinearOperator`` Protocol on ``(nband, nx, ny)`` cubes.
    One actor per band (bands couple only through the prox, so Hessian
    applications and CG solves are embarrassingly parallel over bands).
    ``cg`` is the distributed fast path sniffed by ``opt.pcg.PCG``: each
    band iterates its own CG to convergence inside its actor — one Ray
    dispatch per forward solve, not per CG iteration.

    For ``nband == 1`` a local HessianTree is used (no Ray overhead).

    Args:
        partitions_per_band: list (over bands) of partition-dict lists, each
            dict with ``psfhat``/``beam``/``wsum`` as for ``HessianTree``.
        nx, ny, nx_psf, ny_psf: image/PSF geometry.
        etas: Tikhonov parameter, scalar or per-band sequence.
        nthreads: total FFT threads, divided across actors.
        wsums: optional normalisation override, scalar or per-band sequence
            (pass the total wsum for the legacy convention).
        cg_tol, cg_maxit, cg_minit, cg_verbose: defaults for ``cg``.
    """

    def __init__(
        self,
        partitions_per_band,
        nx,
        ny,
        nx_psf,
        ny_psf,
        etas=0.0,
        nthreads=1,
        wsums=None,
        cg_tol=1e-3,
        cg_maxit=150,
        cg_minit=1,
        cg_verbose=0,
    ):
        self.nband = len(partitions_per_band)
        self.nx = nx
        self.ny = ny
        self.cg_tol = cg_tol
        self.cg_maxit = cg_maxit
        self.cg_minit = cg_minit
        self.cg_verbose = cg_verbose
        etas = np.broadcast_to(np.asarray(etas, dtype=float), (self.nband,))
        if wsums is None:
            wsums = [None] * self.nband
        else:
            wsums = np.broadcast_to(np.asarray(wsums, dtype=float), (self.nband,))

        if self.nband == 1:
            self._local = HessianTree(
                partitions_per_band[0], nx, ny, nx_psf, ny_psf, eta=etas[0], nthreads=nthreads, wsum=wsums[0]
            )
            self._actors = None
        else:
            import ray

            self._local = None
            cpus_per_actor = max(1, nthreads // self.nband)
            actor_cls = ray.remote(num_cpus=cpus_per_actor)(_HessBandActorImpl)
            self._actors = [
                actor_cls.remote(
                    partitions_per_band[b], nx, ny, nx_psf, ny_psf, etas[b], cpus_per_actor, wsums[b]
                )
                for b in range(self.nband)
            ]
            ray.get([a.warmup.remote() for a in self._actors])

    def dot(self, x):
        out = np.zeros_like(x)
        if self._actors is None:
            out[0] = self._local.dot(x[0])[0]
            return out
        import ray

        refs = [self._actors[b].dot.remote(x[b]) for b in range(self.nband)]
        for b, res in enumerate(ray.get(refs)):
            out[b] = res[0]
        return out

    def hdot(self, x):
        return self.dot(x)

    def cg(self, rhs, x0=None, tol=None, maxit=None, minit=None):
        """Distributed per-band CG solve of ``hess @ update = rhs``."""
        tol = self.cg_tol if tol is None else tol
        maxit = self.cg_maxit if maxit is None else maxit
        minit = self.cg_minit if minit is None else minit
        out = np.zeros_like(rhs)
        if self._actors is None:
            from pfb_imaging.opt.pcg import pcg_numba

            x00 = None if x0 is None else x0[0]
            out[0] = pcg_numba(
                lambda z: self._local.dot(z)[0],
                rhs[0],
                x0=x00,
                tol=tol,
                maxit=maxit,
                minit=minit,
                verbosity=self.cg_verbose,
            )
            return out
        import ray

        refs = [
            self._actors[b].cg.remote(
                rhs[b], None if x0 is None else x0[b], tol, maxit, minit, self.cg_verbose
            )
            for b in range(self.nband)
        ]
        for b, res in enumerate(ray.get(refs)):
            out[b] = res
        return out

    def get_mem(self):
        """Per-actor post-gc memory telemetry (empty for the local path)."""
        if self._actors is None:
            return []
        import ray

        return ray.get([a.get_mem.remote() for a in self._actors])
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_hess_tree_ray.py tests/test_hessian_tree.py -v`
Expected: PASS (Ray tests use the session `manage_ray` fixture; nband=1 cases run Ray-free)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/operators/hessian.py tests/test_hess_tree_ray.py
git commit --no-verify -m "feat(operators): HessTreeRay band-actor Hessian with in-actor CG"
```

---

### Task 9: ReweightOnConverge + PFBSolver (deconv/pfb.py)

**Files:**
- Create: `src/pfb_imaging/deconv/pfb.py`
- Test: `tests/test_pfb_solver.py`

**Interfaces:**
- Consumes: `DeconvSolver`/`Regulariser` Protocols, `ForwardSolver`/`BackwardSolver` implementations (Tasks 5–7), `power_method_numba` (`opt/power_method.py:40`, returns `(beta, b)`).
- Produces: `ReweightOnConverge(regulariser, maxreweight=20, verbosity=1)` — callable `(x, k, eps) -> bool` with `reset()`; the single implementation of the consecutive-reweight counter formerly duplicated in `sara_fb.py`/`sara_pd.py`. `PFBSolver(hess, forward_alg, backward_alg, prox, *, model, update, gamma=1.0, hessnorm=None, l1_reweight_from=5, maxreweight=20, pm_tol=1e-3, pm_maxit=100, pm_verbose=1, pm_report_freq=25, verbosity=1)` — the ONE concrete `DeconvSolver` (`first/forward/backward/last`), attribute `hess_norm`, plus `reweight_active` property and `trigger_reweight()` for the driver's sniffing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pfb_solver.py`:

```python
"""PFBSolver: composition of (hess, forward, backward, prox) behind DeconvSolver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_imaging.deconv import DeconvSolver
from pfb_imaging.operators.psi import IdentityPsi
from pfb_imaging.opt.forward_backward import ForwardBackward
from pfb_imaging.opt.pcg import PCG
from pfb_imaging.prox.l1 import L1


class IdOp:
    """Identity Hessian."""

    def dot(self, x):
        return x.copy()

    def hdot(self, x):
        return x.copy()


def _solver(b, lam_unused=None, **kwargs):
    from pfb_imaging.deconv.pfb import PFBSolver

    nband, nx, ny = b.shape
    reg = L1(IdentityPsi(nband, nx, ny))
    fb = ForwardBackward(tol=1e-10, maxit=5000, verbosity=0, gamma=0.45)
    return PFBSolver(
        IdOp(),
        PCG(tol=1e-12, maxit=200, minit=1, verbosity=0),
        fb,
        reg,
        model=np.zeros_like(b),
        update=np.zeros_like(b),
        gamma=1.0,
        hessnorm=1.0,
        l1_reweight_from=-1,  # reweighting disabled
        **kwargs,
    )


def test_satisfies_deconv_solver_protocol():
    b = np.zeros((1, 4, 4))
    assert isinstance(_solver(b), DeconvSolver)


def test_one_major_cycle_identity_hess():
    """With H = I and model0 = 0: update = residual = b, xtilde = b,
    and backward solves min 0.5||x-b||^2 + lam||x||_1 -> soft(b, lam)."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((1, 30, 4))
    lam = 0.3

    solver = _solver(b)
    solver.first(b)
    update = solver.forward(b)
    assert_allclose(update, b, atol=1e-10)  # I^{-1} b

    model = solver.backward(lam)
    assert_allclose(model, np.sign(b) * np.maximum(np.abs(b) - lam, 0.0), atol=1e-4)

    solver.last()  # reweighting disabled: must be a no-op
    assert solver.reweight_active is False


def test_power_method_when_hessnorm_none():
    b = np.zeros((1, 8, 8))
    from pfb_imaging.deconv.pfb import PFBSolver

    reg = L1(IdentityPsi(1, 8, 8))
    solver = PFBSolver(
        IdOp(),
        PCG(verbosity=0),
        ForwardBackward(verbosity=0),
        reg,
        model=b.copy(),
        update=b.copy(),
        hessnorm=None,
        pm_verbose=0,
        l1_reweight_from=-1,
    )
    # spectral norm of I is 1; the 1.05 safety factor is applied
    assert solver.hess_norm == pytest.approx(1.05, rel=1e-2)


def test_reweight_on_converge_counter():
    from pfb_imaging.deconv.pfb import ReweightOnConverge

    class StubReg:
        reweight_active = True

        def __init__(self):
            self.calls = 0

        def update_weights(self, x):
            self.calls += 1

    reg = StubReg()
    cb = ReweightOnConverge(reg, maxreweight=2, verbosity=0)
    x = np.zeros(3)
    assert cb(x, 10, 1e-6) is False and reg.calls == 1  # first reweight, continue
    assert cb(x, 11, 1e-6) is False and reg.calls == 2  # consecutive -> count 1
    assert cb(x, 12, 1e-6) is False and reg.calls == 3  # consecutive -> count 2 == max
    assert cb(x, 13, 1e-6) is True  # cap reached -> stop
    cb.reset()
    assert cb(x, 20, 1e-6) is False  # counter cleared

    reg.reweight_active = False
    cb2 = ReweightOnConverge(reg, maxreweight=2, verbosity=0)
    assert cb2(x, 0, 1e-6) is True  # not armed -> stop at convergence


def test_trigger_reweight_arms_last():
    rng = np.random.default_rng(1)
    b = rng.standard_normal((1, 16, 16))
    from pfb_imaging.deconv.pfb import PFBSolver
    from pfb_imaging.prox.l21 import L21
    from tests.test_regularisers import SlicePsi

    psi = SlicePsi(1, 16, 16, 2, 20, 20)
    reg = L21(psi, bases=("self", "db1"))
    solver = PFBSolver(
        IdOp(),
        PCG(tol=1e-10, maxit=100, minit=1, verbosity=0),
        ForwardBackward(tol=1e-8, maxit=500, verbosity=0, gamma=0.45),
        reg,
        model=np.zeros_like(b),
        update=np.zeros_like(b),
        hessnorm=1.0,
        l1_reweight_from=100,  # far in the future
    )
    solver.first(b)
    solver.forward(b)
    solver.backward(0.1)
    solver.last()
    assert not solver.reweight_active  # threshold not reached
    solver.trigger_reweight()
    solver.last()
    assert solver.reweight_active  # armed by trigger
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pfb_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pfb_imaging.deconv.pfb'`

- [ ] **Step 3: Implement**

Create `src/pfb_imaging/deconv/pfb.py`:

```python
"""PFBSolver: the concrete DeconvSolver composing (hess, forward, backward, prox)."""

import numpy as np

from pfb_imaging.opt.power_method import power_method_numba as power_method
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("PFB")


class ReweightOnConverge:
    """on_converge callback driving inner l1 reweighting.

    Fired by an iterative BackwardSolver when ``eps < tol``.  While the
    regulariser's reweighting is armed and the consecutive-reweight cap is
    not reached, reweights and returns False (keep iterating); otherwise
    returns True (stop).  Single implementation of the counter formerly
    duplicated in sara_fb.py/sara_pd.py.

    Args:
        regulariser: Object with ``reweight_active`` and ``update_weights``.
        maxreweight: Maximum consecutive reweighting steps.
        verbosity: > 1 logs convergence-event metrics.
    """

    def __init__(self, regulariser, maxreweight: int = 20, verbosity: int = 1):
        self.reg = regulariser
        self.maxreweight = maxreweight
        self.verbosity = verbosity
        self._num = 0
        self._last_iter = 0

    def reset(self) -> None:
        """Clear the consecutive-reweight counter (call before each solve)."""
        self._num = 0
        self._last_iter = 0

    def __call__(self, x, k: int, eps: float) -> bool:
        if self.reg.reweight_active and self._num < self.maxreweight:
            self.reg.update_weights(x)
            if k - self._last_iter == 1:
                self._num += 1
            else:
                self._num = 0
            self._last_iter = k
            if self.verbosity > 1:
                log.info(f"Reweighted at iteration {k}, eps = {eps:.3e}, consecutive = {self._num}")
            return False
        if self._num >= self.maxreweight and self.verbosity:
            log.info("Maximum reweighting steps reached")
        return True


class PFBSolver:
    """Preconditioned forward-backward solver from four composable pieces.

    Satisfies the ``DeconvSolver`` Protocol consumed by the outer major-cycle
    loop.  All wiring between the pieces lives here: the grad closure built
    from ``hess.dot``, the ``backward_alg.setup`` binding, and the
    ``ReweightOnConverge`` installation (plain attribute assignment on the
    concrete solver — the hook is not part of any Protocol).

    Args:
        hess: Data-fidelity Hessian satisfying ``LinearOperator``.
        forward_alg: ``ForwardSolver`` for ``update ≈ hess^{-1} residual``.
        backward_alg: ``BackwardSolver`` for the proximal step.
        prox: ``Regulariser``; may expose the optional reweighting trio.
        model: Initial model image ``(nband, nx, ny)``.
        update: Initial update image ``(nband, nx, ny)``.
        gamma: PFB step size for ``xtilde = model + gamma * update``.
        hessnorm: Spectral norm of ``hess`` (power method when None).
        l1_reweight_from: Arm reweighting after this many major cycles
            (negative disables).
        maxreweight: Cap for ``ReweightOnConverge``.
        pm_tol, pm_maxit, pm_verbose, pm_report_freq: power-method controls.
        verbosity: Logging level.
    """

    def __init__(
        self,
        hess,
        forward_alg,
        backward_alg,
        prox,
        *,
        model: np.ndarray,
        update: np.ndarray,
        gamma: float = 1.0,
        hessnorm: float | None = None,
        l1_reweight_from: int = 5,
        maxreweight: int = 20,
        pm_tol: float = 1e-3,
        pm_maxit: int = 100,
        pm_verbose: int = 1,
        pm_report_freq: int = 25,
        verbosity: int = 1,
    ):
        self.hess = hess
        self.forward_alg = forward_alg
        self.backward_alg = backward_alg
        self.reg = prox
        self._model = model.copy()
        self._update = update.copy()
        self._residual = np.zeros_like(model)
        self._gamma = gamma
        self._l1_reweight_from = l1_reweight_from
        self._iter = 0

        if hessnorm is None:
            log.info("Finding spectral norm of Hessian approximation")
            hessnorm, _ = power_method(
                hess.dot,
                model.shape,
                tol=pm_tol,
                maxit=pm_maxit,
                verbosity=pm_verbose,
                report_freq=pm_report_freq,
            )
            hessnorm *= 1.05
        self.hess_norm = hessnorm
        log.info(f"Using hess_norm = {hessnorm:.3e}")

        backward_alg.setup(prox, hessnorm)

        self._reweight_cb = None
        if hasattr(prox, "update_weights") and hasattr(prox, "reweight_active"):
            self._reweight_cb = ReweightOnConverge(prox, maxreweight=maxreweight, verbosity=verbosity)
            if getattr(backward_alg, "on_converge", None) is None:
                backward_alg.on_converge = self._reweight_cb

    # --- DeconvSolver interface ---

    def first(self, residual: np.ndarray) -> None:
        """Store the residual (per-partition beams are applied inside hess)."""
        self._residual = residual

    def forward(self, residual: np.ndarray) -> np.ndarray:
        """Forward solve; builds the grad closure for the backward step."""
        x0 = self._update if self._update.any() else None
        self._update = self.forward_alg.solve(self.hess, self._residual, x0=x0)
        xtilde = self._model + self._gamma * self._update

        def grad(x):
            return -self.hess.dot(xtilde - x) / self._gamma

        self.backward_alg.set_grad(grad)
        return self._update

    def backward(self, lam: float) -> np.ndarray:
        """Backward (proximal) solve; returns the updated model."""
        if self._reweight_cb is not None:
            self._reweight_cb.reset()
        self._model = self.backward_alg.solve(self._model, lam)
        self._iter += 1
        return self._model

    def last(self) -> None:
        """Arm/refresh l1 reweighting once the threshold is reached."""
        if not hasattr(self.reg, "init_reweighting"):
            return
        if self._l1_reweight_from < 0 or self._iter < self._l1_reweight_from:
            return
        log.info("Computing L1 weights")
        self.reg.init_reweighting(self._update)
        self.reg.update_weights(self._model)

    # --- driver sniffing (matches the legacy SARABase extras) ---

    @property
    def reweight_active(self) -> bool:
        """True once l1 reweighting has been armed."""
        return getattr(self.reg, "reweight_active", True)

    def trigger_reweight(self) -> None:
        """Force reweighting to arm at the next :meth:`last` call."""
        self._l1_reweight_from = self._iter
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pfb_solver.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/deconv/pfb.py tests/test_pfb_solver.py
git commit --no-verify -m "feat(deconv): PFBSolver composition and ReweightOnConverge"
```

---

### Task 10: Preset factories and registry (deconv/presets.py)

**Files:**
- Create: `src/pfb_imaging/deconv/presets.py`
- Test: `tests/test_pfb_solver.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 2–9 plus `PsiNocopytRay` (`operators/psi.py:745`, constructor `PsiNocopytRay(nband, nx, ny, bases, nlevel, nthreads, nactors=None)`).
- Produces: `PRESETS: dict[str, callable]` with keys `"sara"`, `"ista"`. Factory contract (both):
  `make_sara(partitions_per_band, geometry, model, update, opts) -> PFBSolver` where
  `partitions_per_band` = list (over bands, bandid-sorted) of partition-dict lists (`psfhat`/`beam`/`wsum` per dict);
  `geometry` = dict with keys `nx, ny, nx_psf, ny_psf`;
  `model`/`update` = `(nband, nx, ny)` arrays;
  `opts` = dict containing at least `bases, nlevels, rmsfactor, alpha, gamma, positivity, eta, l1_reweight_from, hess_norm, opt_backend, acceleration, nthreads, verbosity, pd_tol, pd_maxit, pd_verbose, pd_report_freq, fb_tol, fb_maxit, fb_verbose, fb_report_freq, cg_tol, cg_maxit, cg_verbose, cg_report_freq, pm_tol, pm_maxit, pm_verbose, pm_report_freq` (the driver passes its `locals()` copy, so extra keys are fine).
  Normalisation convention (tier-3 critical): with `wsum_b` = per-band sum of partition wsums and `wsum_tot = Σ_b wsum_b`, the factory passes `wsums=wsum_tot` (override) and `etas = opts["eta"] * (wsum_b / wsum_tot)` to `HessTreeRay` — reproducing the legacy `HessPSF(abspsf/wsum_tot, eta=eta*wsums_norm)` semantics.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pfb_solver.py`)

```python
def _delta_partitions(nband, nx, ny, wsum=1.0):
    """Delta-function psfhat partitions: the Hessian acts as (wsum_b/wsum_tot)*I + eta."""
    psfhat = np.ones((1, 2 * nx, ny + 1))
    beam = np.ones((1, nx, ny))
    return [[{"psfhat": psfhat.copy(), "beam": beam.copy(), "wsum": np.array([wsum])}] for _ in range(nband)]


def test_presets_registry_builds_deconv_solvers():
    from pfb_imaging.deconv.presets import PRESETS

    nband, nx, ny = 1, 16, 16
    geometry = {"nx": nx, "ny": ny, "nx_psf": 2 * nx, "ny_psf": 2 * ny}
    model = np.zeros((nband, nx, ny))
    opts = dict(
        bases=["self", "db1"], nlevels=2, rmsfactor=1.0, alpha=2.0, gamma=1.0,
        positivity=1, eta=0.5, l1_reweight_from=-1, hess_norm=1.0,
        opt_backend="primal-dual", acceleration=True, nthreads=1, verbosity=0,
        pd_tol=1e-6, pd_maxit=50, pd_verbose=0, pd_report_freq=100,
        fb_tol=1e-6, fb_maxit=50, fb_verbose=0, fb_report_freq=100,
        cg_tol=1e-6, cg_maxit=50, cg_verbose=0, cg_report_freq=100,
        pm_tol=1e-3, pm_maxit=50, pm_verbose=0, pm_report_freq=100,
    )

    for name in ("sara", "ista"):
        solver = PRESETS[name](_delta_partitions(nband, nx, ny), geometry, model.copy(), model.copy(), dict(opts))
        assert isinstance(solver, DeconvSolver), name
        rng = np.random.default_rng(0)
        residual = rng.standard_normal((nband, nx, ny))
        solver.first(residual)
        update = solver.forward(residual)
        assert np.isfinite(update).all()
        out = solver.backward(1e-3)
        assert np.isfinite(out).all()
        solver.last()


def test_make_sara_opt_backend_selects_solver():
    from pfb_imaging.deconv.presets import make_sara
    from pfb_imaging.opt.forward_backward import ForwardBackward as FB
    from pfb_imaging.opt.primal_dual import PrimalDual as PD

    geometry = {"nx": 16, "ny": 16, "nx_psf": 32, "ny_psf": 32}
    model = np.zeros((1, 16, 16))
    base_opts = dict(
        bases=["self"], nlevels=1, rmsfactor=1.0, alpha=2.0, gamma=1.0,
        positivity=0, eta=0.5, l1_reweight_from=-1, hess_norm=1.0,
        acceleration=True, nthreads=1, verbosity=0,
        pd_tol=1e-6, pd_maxit=10, pd_verbose=0, pd_report_freq=100,
        fb_tol=1e-6, fb_maxit=10, fb_verbose=0, fb_report_freq=100,
        cg_tol=1e-6, cg_maxit=10, cg_verbose=0, cg_report_freq=100,
        pm_tol=1e-3, pm_maxit=10, pm_verbose=0, pm_report_freq=100,
    )
    s_pd = make_sara(_delta_partitions(1, 16, 16), geometry, model.copy(), model.copy(),
                     dict(base_opts, opt_backend="primal-dual"))
    s_fb = make_sara(_delta_partitions(1, 16, 16), geometry, model.copy(), model.copy(),
                     dict(base_opts, opt_backend="forward-backward"))
    assert isinstance(s_pd.backward_alg, PD)
    assert isinstance(s_fb.backward_alg, FB)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pfb_solver.py -v -k preset`
Expected: FAIL with `ModuleNotFoundError: No module named 'pfb_imaging.deconv.presets'`

- [ ] **Step 3: Implement**

Create `src/pfb_imaging/deconv/presets.py`:

```python
"""Minor-cycle preset factories: assemble PFBSolver from CLI options.

Contributors add a deconvolution algorithm by writing a Regulariser and
registering a factory here ("kclean" will slot in as OneShot + ClarkProx).
"""

import numpy as np

from pfb_imaging.deconv.pfb import PFBSolver
from pfb_imaging.operators.hessian import HessTreeRay
from pfb_imaging.operators.psi import IdentityPsi, PsiNocopytRay
from pfb_imaging.opt.forward_backward import ForwardBackward
from pfb_imaging.opt.pcg import PCG
from pfb_imaging.opt.primal_dual import PrimalDual
from pfb_imaging.prox.l1 import L1
from pfb_imaging.prox.l21 import L21
from pfb_imaging.prox.positivity import positivity_prox


def _build_hess(partitions_per_band, geometry, opts):
    """HessTreeRay with the legacy total-wsum normalisation convention."""
    wsum_b = np.array([sum(float(np.sum(p["wsum"])) for p in parts) for parts in partitions_per_band])
    wsum_tot = wsum_b.sum()
    etas = opts["eta"] * wsum_b / wsum_tot
    return HessTreeRay(
        partitions_per_band,
        geometry["nx"],
        geometry["ny"],
        geometry["nx_psf"],
        geometry["ny_psf"],
        etas=etas,
        nthreads=opts["nthreads"],
        wsums=wsum_tot,
        cg_tol=opts["cg_tol"],
        cg_maxit=opts["cg_maxit"],
        cg_verbose=opts["cg_verbose"],
    )


def _build_backward(opts):
    """Backward solver from opt_backend; primal_prox from the positivity mode."""
    pprox = positivity_prox(opts["positivity"])
    if opts["opt_backend"] == "primal-dual":
        return PrimalDual(
            tol=opts["pd_tol"],
            maxit=opts["pd_maxit"],
            verbosity=opts["pd_verbose"],
            report_freq=opts["pd_report_freq"],
            gamma=opts["gamma"],
            primal_prox=pprox,
        )
    if opts["opt_backend"] == "forward-backward":
        return ForwardBackward(
            tol=opts["fb_tol"],
            maxit=opts["fb_maxit"],
            verbosity=opts["fb_verbose"],
            report_freq=opts["fb_report_freq"],
            gamma=opts["gamma"],
            acceleration=opts["acceleration"],
            primal_prox=pprox,
        )
    raise ValueError(f"Unknown opt_backend '{opts['opt_backend']}'")


def _common_kwargs(model, update, opts):
    return dict(
        model=model,
        update=update,
        gamma=opts["gamma"],
        hessnorm=opts["hess_norm"],
        l1_reweight_from=opts["l1_reweight_from"],
        pm_tol=opts["pm_tol"],
        pm_maxit=opts["pm_maxit"],
        pm_verbose=opts["pm_verbose"],
        pm_report_freq=opts["pm_report_freq"],
        verbosity=opts["verbosity"],
    )


def make_sara(partitions_per_band, geometry, model, update, opts):
    """SARA: l21 over a wavelet dictionary, PD or FB backward."""
    nband = len(partitions_per_band)
    bases = tuple(opts["bases"]) if not isinstance(opts["bases"], str) else tuple(opts["bases"].split(","))
    psi = PsiNocopytRay(nband, geometry["nx"], geometry["ny"], bases, opts["nlevels"], opts["nthreads"])
    reg = L21(psi, bases, rmsfactor=opts["rmsfactor"], alpha=opts["alpha"])
    hess = _build_hess(partitions_per_band, geometry, opts)
    fwd = PCG(tol=opts["cg_tol"], maxit=opts["cg_maxit"], verbosity=opts["cg_verbose"], report_freq=opts["cg_report_freq"])
    return PFBSolver(hess, fwd, _build_backward(opts), reg, **_common_kwargs(model, update, opts))


def make_ista(partitions_per_band, geometry, model, update, opts):
    """ISTA: image-domain l1, forward-backward without acceleration."""
    nband = len(partitions_per_band)
    reg = L1(IdentityPsi(nband, geometry["nx"], geometry["ny"]))
    hess = _build_hess(partitions_per_band, geometry, opts)
    fwd = PCG(tol=opts["cg_tol"], maxit=opts["cg_maxit"], verbosity=opts["cg_verbose"], report_freq=opts["cg_report_freq"])
    bwd = ForwardBackward(
        tol=opts["fb_tol"],
        maxit=opts["fb_maxit"],
        verbosity=opts["fb_verbose"],
        report_freq=opts["fb_report_freq"],
        gamma=opts["gamma"],
        acceleration=False,
        primal_prox=positivity_prox(opts["positivity"]),
    )
    return PFBSolver(hess, fwd, bwd, reg, **_common_kwargs(model, update, opts))


PRESETS = {"sara": make_sara, "ista": make_ista}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pfb_solver.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/deconv/presets.py tests/test_pfb_solver.py
git commit --no-verify -m "feat(deconv): sara/ista preset factories and registry"
```

---

### Task 11: Driver rewrite (core/deconv.py) + CLI + cabs

**Files:**
- Rewrite: `src/pfb_imaging/core/deconv.py`
- Modify: `src/pfb_imaging/cli/deconv.py`
- Regenerate: `src/pfb_imaging/cabs/deconv.yml`
- Test: `tests/test_roundtrip.py` (existing; must stay green)

**Interfaces:**
- Consumes: `PRESETS` (Task 10), `DeconvSolver`, `residual_from_partitions(dirty, parts, model, cell_rad, nthreads=1, epsilon=1e-7, do_wgridding=True, double_accum=True)` (`operators/gridder.py:928` — parts are xr Datasets with `UVW/WEIGHT/MASK/FREQ/BEAM` + `l0`/`m0` attrs; returns `(corr, nx, ny)` raw residual), `dt2fits(store_url, column, outname, norm_wsum=True, ...)` (`utils/fits.py:405`), `init_ray`/`setup_ray_worker`/`set_envs` (`pfb_imaging/__init__.py`), `set_output_names` (`utils/naming.py`), `fit_image_cube`/`eval_coeffs_to_slice` (`utils/modelspec.py`), `wgridder_conventions` (`operators/gridder.py`).
- Produces: `deconv(output_filename, suffix="main", ..., minor_cycle="sara", opt_backend="primal-dual", nworkers=1, ray_address="local", ...)` reading `<basename>_<suffix>.dt`, writing `MODEL`/`UPDATE`/`MODEL_BEST`/`RESIDUAL` + `hess_norm`/`niters`/`rms`/`rmax` attrs back into band nodes, `<basename>_<suffix>_model.mds`, and FITS via `dt2fits`.
- `.dt` facts (written by `core/imager.py:138-194`): band nodes named `band{b:04d}_time{t:04d}` with vars `DIRTY/RESIDUAL/PSF/PSFPARSN/WSUM` (dims `(corr, x, y)`; `WSUM` `(corr,)`) and attrs `bandid, timeid, freq_out, time_out, ra, dec, cell_rad, niters`; partition children `part{p:04d}` with vars `VIS/WEIGHT/MASK/UVW/FREQ/PSF/PSFHAT/BEAM/PSFPARSN` and attrs incl. `wsum` (list) and `l0`/`m0`.

- [ ] **Step 1: Rewrite `src/pfb_imaging/core/deconv.py`** (full replacement; the mds model-fit block is carried over from the current file lines 267-336 with `.dt`-appropriate attrs)

```python
import time
from copy import deepcopy

import numpy as np
import psutil
import ray
import xarray as xr
from ducc0.misc import resize_thread_pool

from pfb_imaging import init_ray, pfb_version, set_envs, setup_ray_worker
from pfb_imaging.deconv import DeconvSolver
from pfb_imaging.deconv.presets import PRESETS
from pfb_imaging.operators.gridder import residual_from_partitions, wgridder_conventions
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import dt2fits, save_fits, set_wcs
from pfb_imaging.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
from pfb_imaging.utils.naming import set_output_names

log = pfb_logging.get_logger("DECONV")


def _band_residual(dirty, parts, model, cell_rad, nthreads, epsilon, do_wgridding, double_accum):
    """Ray task: exact residual for one band node (raw, un-normalised)."""
    return residual_from_partitions(
        dirty,
        parts,
        model,
        cell_rad,
        nthreads=nthreads,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        double_accum=double_accum,
    )


def deconv(
    output_filename: str,
    suffix: str = "main",
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
    minor_cycle: str = "sara",
    opt_backend: str = "primal-dual",
    bases: list[str] = ["self", "db1", "db2", "db3"],
    nlevels: int = 3,
    l1_reweight_from: int = 5,
    alpha: float = 2.0,
    hess_norm: float | None = None,
    rmsfactor: float = 1.0,
    eta: float = 0.001,
    gamma: float = 0.95,
    nbasisf: int | None = None,
    positivity: int = 1,
    niter: int = 10,
    tol: float = 0.0005,
    diverge_count: int = 5,
    rms_outside_model: bool = False,
    init_factor: float = 0.5,
    verbosity: int = 1,
    nthreads: int | None = None,
    nworkers: int = 1,
    ray_address: str = "local",
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    pd_tol: float = 0.0003,
    pd_maxit: int = 1000,
    pd_verbose: int = 1,
    pd_report_freq: int = 100,
    fb_tol: float = 0.0003,
    fb_maxit: int = 1000,
    fb_verbose: int = 1,
    fb_report_freq: int = 100,
    acceleration: bool = True,
    pm_tol: float = 0.001,
    pm_maxit: int = 500,
    pm_verbose: int = 1,
    pm_report_freq: int = 100,
    cg_tol: float = 0.001,
    cg_maxit: int = 150,
    cg_verbose: int = 1,
    cg_report_freq: int = 10,
):
    """
    General preconditioned forward-backward deconvolution of the imager DataTree.

    The minor_cycle preset assembles (hess, forward_alg, backward_alg, prox) into
    a PFBSolver; any object satisfying the DeconvSolver Protocol can drive the loop.
    """
    opts_dict = locals().copy()

    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )
    opts_dict["output_filename"] = output_filename
    opts_dict["fits_output_folder"] = fits_output_folder
    opts_dict["log_directory"] = log_directory

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2
    else:
        ncpu = np.minimum(nthreads, psutil.cpu_count(logical=False))
    opts_dict["nthreads"] = nthreads
    log.info(f"Using {nthreads} threads total")
    resize_thread_pool(nthreads)
    set_envs(nthreads, ncpu, log)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/deconv_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.log_options_dict(opts_dict, title="DECONV options")

    basename = output_filename
    fits_oname = f"{fits_output_folder}/{oname}"
    dt_name = f"{basename}_{suffix}.dt"

    time_start = time.time()

    init_ray(nworkers, ray_address=ray_address, log=log)

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    image_names = sorted(n for n in dt.children if n.startswith("band"))
    if not image_names:
        log.error_and_raise(f"No band nodes found in {dt_name}", ValueError)

    timeids = {int(dt[n].ds.attrs["timeid"]) for n in image_names}
    if len(timeids) > 1:
        log.error_and_raise("Only static models currently supported", NotImplementedError)

    nodes = sorted(image_names, key=lambda n: int(dt[n].ds.attrs["bandid"]))
    first = dt[nodes[0]].ds
    if first.corr.size > 1:
        log.error_and_raise("Joint polarisation deconvolution not yet supported", NotImplementedError)

    nband = len(nodes)
    nx, ny = first.x.size, first.y.size
    nx_psf, ny_psf = first.x_psf.size, first.y_psf.size
    cell_rad = first.attrs["cell_rad"]
    cell_deg = np.rad2deg(cell_rad)
    radec = [first.attrs["ra"], first.attrs["dec"]]
    freq_out = np.array([dt[n].ds.attrs["freq_out"] for n in nodes])
    time_out = np.array([first.attrs["time_out"]])
    iter0 = int(first.attrs.get("niters", 0))
    geometry = {"nx": nx, "ny": ny, "nx_psf": nx_psf, "ny_psf": ny_psf}

    # --- load band cubes and partition inputs (selective, memory-disciplined) ---
    residual_raw = np.zeros((nband, nx, ny))
    model = np.zeros((nband, nx, ny))
    update = np.zeros((nband, nx, ny))
    dirty_raw = np.zeros((nband, nx, ny))
    wsums = np.zeros(nband)
    partitions_per_band = []  # plain-numpy dicts for HessTreeRay
    parts_refs = []  # per-band ray.put of the gridding-input Datasets

    for b, n in enumerate(nodes):
        band = dt[n]
        bds = band.ds
        dirty_raw[b] = bds.DIRTY.values[0]
        residual_raw[b] = bds.RESIDUAL.values[0] if "RESIDUAL" in bds else bds.DIRTY.values[0]
        if "MODEL" in bds:
            model[b] = bds.MODEL.values[0]
        if "UPDATE" in bds:
            update[b] = bds.UPDATE.values[0]
        wsums[b] = bds.WSUM.values[0]

        ph = []
        pg = []
        for cname in sorted(band.children):
            child = band[cname].ds
            pds = child[["UVW", "WEIGHT", "MASK", "FREQ", "BEAM"]].load()
            pds.attrs.update(child.attrs)
            ph.append(
                {
                    "psfhat": child.PSFHAT.values,
                    "beam": pds.BEAM.values,
                    "wsum": np.asarray(child.attrs["wsum"]),
                }
            )
            pg.append(pds)
        partitions_per_band.append(ph)
        parts_refs.append(ray.put(pg))  # serialised once, reused every major cycle

    wsum = wsums.sum()
    residual = residual_raw / wsum
    residual_mfs = np.sum(residual, axis=0)
    fsel = wsums > 0
    model_mfs = np.mean(model[fsel], axis=0)
    if nbasisf is None:
        nbasisf = int(np.sum(fsel))

    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, np.mean(freq_out), casambm=False)

    # hess_norm from the tree cache when available; solver estimates it otherwise
    if hess_norm is None and "hess_norm" in first.attrs:
        hess_norm = first.attrs["hess_norm"]
        opts_dict["hess_norm"] = hess_norm
        log.info(f"Using previously estimated hess_norm of {hess_norm:.3e}")

    if minor_cycle not in PRESETS:
        log.error_and_raise(f"Unknown minor_cycle '{minor_cycle}'", ValueError)
    solver = PRESETS[minor_cycle](partitions_per_band, geometry, model, update, opts_dict)
    if not isinstance(solver, DeconvSolver):
        raise TypeError(f"Solver must be a DeconvSolver, got {type(solver)}")

    _residual_remote = ray.remote(_band_residual)

    if rms_outside_model and model.any():
        rms = np.std(residual_mfs[model_mfs == 0])
    else:
        rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms, best_rmax = rms, rmax
    best_model = model.copy()
    diverge_count_curr = 0
    log.info(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}")

    for k in range(niter):
        log.info("Solving for update")
        solver.first(residual)
        update = solver.forward(residual)
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs, fits_oname + f"_{suffix}_update_{iter0 + k + 1}.fits", hdr_mfs)

        modelp = deepcopy(model)
        lam = (init_factor if iter0 == 0 and k == 0 else 1.0) * rmsfactor * rms
        log.info(f"Solving for model with lambda = {lam:.3e}")
        model = solver.backward(lam)

        # write component model (carried over from the legacy driver; .dt-native attrs)
        log.info(f"Writing model to {basename}_{suffix}_model.mds")
        try:
            coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
                time_out,
                freq_out[fsel],
                model[None, fsel, :, :],
                wgt=(wsums / wsum)[None, fsel],
                nbasisf=nbasisf,
                method="Legendre",
                sigmasq=1e-6,
            )
            flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
            coeff_dataset = xr.Dataset(
                data_vars={"coefficients": (("par", "comps"), coeffs)},
                coords={
                    "location_x": (("x",), x_index),
                    "location_y": (("y",), y_index),
                    "params": (("par",), params),
                    "times": (("t",), time_out),
                    "freqs": (("f",), freq_out),
                },
                attrs={
                    "pfb-imaging-version": pfb_version,
                    "spec": "genesis",
                    "cell_rad_x": cell_rad,
                    "cell_rad_y": cell_rad,
                    "npix_x": nx,
                    "npix_y": ny,
                    "texpr": texpr,
                    "fexpr": fexpr,
                    "center_x": x0,
                    "center_y": y0,
                    "flip_u": flip_u,
                    "flip_v": flip_v,
                    "flip_w": flip_w,
                    "ra": radec[0],
                    "dec": radec[1],
                    "stokes": product,
                    "parametrisation": expr,
                },
            )
            coeff_dataset.to_zarr(f"{basename}_{suffix}_model.mds", mode="w")

            for b in range(nband):
                model[b] = eval_coeffs_to_slice(
                    time_out[0], freq_out[b], coeffs, x_index, y_index, expr, params,
                    texpr, fexpr, nx, ny, cell_rad, cell_rad, x0, y0,
                    nx, ny, cell_rad, cell_rad, x0, y0,
                )
        except Exception as e:
            log.info(f"Exception {e} raised during model fit.")

        model_mfs = np.mean(model[fsel], axis=0)
        save_fits(model_mfs, fits_oname + f"_{suffix}_model_{iter0 + k + 1}.fits", hdr_mfs)

        log.info("Computing residual")
        refs = [
            _residual_remote.remote(
                dirty_raw[b][None], parts_refs[b], model[b][None], cell_rad,
                nthreads, epsilon, do_wgridding, double_accum,
            )
            for b in range(nband)
        ]
        for b, res in enumerate(ray.get(refs)):
            residual_raw[b] = res[0]
        residual = residual_raw / wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs, fits_oname + f"_{suffix}_residual_{iter0 + k + 1}.fits", hdr_mfs)

        # post-iteration hook (e.g. arming l1 reweighting)
        solver.last()

        # per-actor post-gc memory telemetry (docs/msv4-memory-patterns.md)
        if verbosity > 1 and hasattr(getattr(solver, "hess", None), "get_mem"):
            for m in solver.hess.get_mem():
                log.info(f"hess actor pid {m['pid']} rss {m['rss_gb']:.2f} GB peak {m['peak_gb']:.2f} GB")

        rmsp, rmaxp = rms, rmax
        if rms_outside_model:
            rms = np.std(residual_mfs[model_mfs == 0])
        else:
            rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        if rms < best_rms:
            best_rms, best_rmax = rms, rmax
            best_model = model.copy()

        hess_norm = getattr(solver, "hess_norm", hess_norm)

        # write back into the band nodes (native DataTree API)
        for b, n in enumerate(nodes):
            data_vars = {
                "MODEL": (("corr", "x", "y"), model[b][None]),
                "UPDATE": (("corr", "x", "y"), update[b][None]),
                "RESIDUAL": (("corr", "x", "y"), residual_raw[b][None]),
            }
            if (model == best_model).all():
                data_vars["MODEL_BEST"] = (("corr", "x", "y"), best_model[b][None])
            ds_out = xr.Dataset(
                data_vars,
                attrs={"rms": best_rms, "rmax": best_rmax, "niters": iter0 + k + 1, "hess_norm": hess_norm},
            )
            ds_out.to_zarr(dt_name, group=n, mode="a")

        log.info(f"Iter {iter0 + k + 1}: peak residual = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}")

        if eps < tol:
            if not getattr(solver, "reweight_active", True):
                solver.trigger_reweight()  # reweight instead of stopping
            else:
                log.info(f"Converged after {iter0 + k + 1} iterations.")
                break

        if (rms > rmsp) and (rmax > rmaxp):
            diverge_count_curr += 1
            if diverge_count_curr > diverge_count:
                log.info("Algorithm is diverging. Terminating.")
                break

    if fits_mfs or fits_cubes:
        log.info(f"Writing fits files to {fits_oname}_{suffix}")
        for column, norm in (("RESIDUAL", True), ("MODEL", False), ("UPDATE", False)):
            dt2fits(
                dt_name,
                column,
                f"{fits_oname}_{suffix}",
                norm_wsum=norm,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
            )
            log.info(f"Done writing {column}")

    log.info(f"All done after {time.time() - time_start:.1f}s")
```

Note the deliberate deletions vs the old driver: no `daskms`/`DaskMSStore`/`xds_from_url` imports, no `hess_approx` option (the PSF approximation is the only v1 Hessian), no per-write thread executor (blocking `to_zarr` per band is fine at this cadence).

- [ ] **Step 2: Sanity-check the module imports**

Run: `uv run python -c "from pfb_imaging.core.deconv import deconv; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Update `src/pfb_imaging/cli/deconv.py`**

Exact edits (three call-sites each for removed/added params — the signature, the `preflight_remote_must_exist` dict, the `deconv_core(...)` call, and the `run_in_container` dict):

1. Change the `dds-out` stimela output to the DataTree (deconv updates it in place):

```python
@stimela_output(
    dtype="Directory",
    name="dt-out",
    info="DataTree dataset updated in place.",
    implicit="{current.output-filename}_{current.product}_{current.suffix}.dt",
    must_exist=False,
)
```

2. `minor_cycle` becomes `Literal["sara", "ista"]` with help:

```python
    minor_cycle: Annotated[
        Literal["sara", "ista"],
        typer.Option(
            help="Which minor cycle algorithm to use.",
            rich_help_panel="PFB",
        ),
    ] = "sara",
```

3. **Delete** the `hess_approx` parameter (and its entries in all three dicts).

4. **Add** after `nthreads` (and to all three dicts):

```python
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of Ray workers.",
            rich_help_panel="Performance",
        ),
    ] = 1,
    ray_address: Annotated[
        str,
        typer.Option(
            help="Address of the Ray cluster to connect to.",
            rich_help_panel="Performance",
        ),
    ] = "local",
```

5. Update the cab `info` string and docstring to: `"General deconvolution of the imager DataTree via composable forward/backward algorithms."`

- [ ] **Step 4: Regenerate cabs and verify the round-trip**

```bash
uv run hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs
uv run pytest tests/test_roundtrip.py -v
```
Expected: PASS. If the deconv module fails the line-count comparison, check for help sentences rendering > 120 chars (python-standards §2.1) and split them.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/core/deconv.py src/pfb_imaging/cli/deconv.py src/pfb_imaging/cabs/deconv.yml
git commit --no-verify -m "feat(deconv): DataTree-native Ray-distributed deconv driver"
```

---

### Task 12: Tier-3 end-to-end equivalence test (tests/test_deconv.py)

**Files:**
- Create: `tests/test_deconv.py`

**Interfaces:**
- Consumes: `core.imager.imager` (invocation pattern: `tests/test_imager.py:63-74`), `core.init.init` + `core.grid.grid` + `core.sara.sara` (invocation pattern: `tests/test_sara.py:105-149`; `sara` accepts `hess_norm`), the new `core.deconv.deconv` (Task 11), session fixtures `ms_name`, `ms_meta`, `image_geometry` from `tests/conftest.py`.
- Strategy: run both pipelines on the SAME simulated data with **natural weighting** (`robustness=None`) and a **pinned shared `hess_norm`** (removes power-method nondeterminism), reweighting disabled, one major cycle → tight comparison. The imager↔init+grid dirty-image equivalence is already covered by `tests/test_imager.py::test_imager_matches_init_grid_single_field`.

- [ ] **Step 1: Write the test**

Create `tests/test_deconv.py`. Copy the model-visibility simulation preamble verbatim from `tests/test_sara.py::test_sara` (the block from the fixture unpacking down to `dask.compute(writes)` — it simulates point sources into the `DATA` column; roughly lines 22–99) into a module-level helper `def _simulate_data(ms_name, ms_meta, image_geometry):` returning `fov`. Add the module imports the test needs (`numpy as np`, `xarray as xr`, `pathlib.Path`, plus whatever the copied simulation preamble uses). Then:

```python
def test_deconv_matches_legacy_sara(ms_name, ms_meta, image_geometry, tmp_path):
    """New .dt deconv (sara/PD) vs legacy .dds sara: same data, pinned hess_norm."""
    from pfb_imaging.core.deconv import deconv as deconv_core
    from pfb_imaging.core.grid import grid as grid_core
    from pfb_imaging.core.imager import imager as imager_core
    from pfb_imaging.core.init import init as init_core
    from pfb_imaging.core.sara import sara as sara_core
    from pfb_imaging.utils.naming import xds_from_url

    fov = _simulate_data(ms_name, ms_meta, image_geometry)

    common = dict(
        niter=1,
        gamma=1.0,
        eta=0.5,
        rmsfactor=1.0,
        init_factor=1.0,
        l1_reweight_from=100,  # disabled within one major cycle
        bases="self,db1",
        nlevels=2,
        positivity=1,
        hess_norm=None,  # first legacy run estimates; then we pin (below)
        pd_tol=1e-4,
        pd_maxit=200,
        cg_tol=1e-6,
        cg_maxit=100,
        pm_tol=1e-4,
        pm_maxit=200,
        nthreads=2,
        do_wgridding=True,
        epsilon=1e-7,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
    )

    # --- legacy path: init + grid + sara on the .dds ---
    out_legacy = str(tmp_path / "legacy")
    init_core([ms_name], out_legacy, data_column="DATA", flag_column="FLAG",
              max_field_of_view=fov * 1.1, overwrite=True, channels_per_image=1,
              keep_ray_alive=True, nthreads=2)
    grid_core(out_legacy, field_of_view=fov, fits_mfs=False, psf=True, residual=False,
              noise=False, nthreads=2, overwrite=True, robustness=None,
              do_wgridding=True, keep_ray_alive=True)
    sara_core(out_legacy, **{k: v for k, v in common.items() if k in
              sara_core.__code__.co_varnames})
    dds, _ = xds_from_url(f"{out_legacy}_I_main.dds")
    hess_norm = dds[0].hess_norm  # pin for the new path
    model_legacy = np.stack([ds.MODEL.values[0] for ds in sorted(dds, key=lambda d: d.freq_out)])

    # --- new path: imager + deconv on the .dt ---
    out_new = str(tmp_path / "new")
    imager_core([Path(ms_name)], out_new, channels_per_image=1, product="I",
                field_of_view=fov, robustness=None, fits_mfs=False,
                overwrite=True, keep_ray_alive=True)
    deconv_core(out_new, minor_cycle="sara", opt_backend="primal-dual",
                bases=["self", "db1"],
                **{k: v for k, v in common.items() if k not in ("bases", "hess_norm")},
                hess_norm=float(hess_norm))

    dt = xr.open_datatree(f"{out_new}_I.dt", engine="zarr", chunks=None)
    nodes = sorted((n for n in dt.children if n.startswith("band")),
                   key=lambda n: int(dt[n].ds.attrs["bandid"]))
    model_new = np.stack([dt[n].ds.MODEL.values[0] for n in nodes])
    for n in nodes:
        assert "MODEL" in dt[n].ds and "UPDATE" in dt[n].ds
        assert dt[n].ds.attrs["niters"] == 1

    rdiff = np.linalg.norm(model_new - model_legacy) / np.linalg.norm(model_legacy)
    # one major cycle, pinned hess_norm, natural weights: the paths share the
    # same kernels; residual slack covers CG minit/backtrack differences.
    assert rdiff < 1e-2, f"model mismatch: rdiff = {rdiff:.3e}"
```

Adjust the `common`-dict filtering to the actual `sara_core`/`deconv_core` signatures (`sara` has no `nworkers`/`opt_backend`; `deconv` has no `hess_approx`). The band ordering key on the legacy side is `ds.freq_out`; on the new side `bandid`.

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_deconv.py -v` (first run downloads test data; needs the session Ray fixture)
Expected: PASS. If `rdiff` marginally exceeds 1e-2, investigate parameter mismatches FIRST (gamma, eta, positivity, pd defaults — `sigma`/`tau` constants must match); only relax the tolerance with a comment explaining the residual source, and never beyond 5e-2.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v tests/`
Expected: all green.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add tests/test_deconv.py
git commit --no-verify -m "test(deconv): end-to-end equivalence vs legacy sara path"
```

---

### Task 13: Cleanup — old Protocol names, docs, full verification

**Files:**
- Modify: `src/pfb_imaging/operators/__init__.py` (delete `PsiOperatorProtocol`)
- Rewrite: `src/pfb_imaging/deconv/README.md`
- Modify: `.claude/rules/architecture.md` (§5 pointer to the new framework)

**Interfaces:**
- Consumes: all previous tasks complete.
- Produces: no stale names; documentation matching reality.

- [ ] **Step 1: Delete `PsiOperatorProtocol` after confirming nothing imports it**

Run: `grep -rn "PsiOperatorProtocol" src/ tests/ --include="*.py"`
Expected: only the definition in `operators/__init__.py` (its importers were rewritten/deleted in Tasks 5–6). Delete the class. If anything still imports it, migrate that import to `PsiOperator` first.

- [ ] **Step 2: Rewrite `src/pfb_imaging/deconv/README.md`**

```markdown
# Deconvolution module

The general deconvolution interface composes four pieces behind the
`DeconvSolver` Protocol (see `docs/superpowers/specs/2026-07-06-gendeconv-protocols-design.md`
and issue #185):

    PFBSolver(hess, forward_alg, backward_alg, prox)

- `hess` satisfies `operators.LinearOperator` (e.g. `HessTreeRay`, Ray band-actors)
- `forward_alg` satisfies `opt.ForwardSolver` (e.g. `PCG`)
- `backward_alg` satisfies `opt.BackwardSolver` (`PrimalDual`, `ForwardBackward`)
- `prox` satisfies `deconv.Regulariser` (`prox.L21`, `prox.L1`; owns its weights)

No inheritance anywhere: the interfaces are `typing.Protocol` classes, satisfied
structurally. To contribute an algorithm:

1. Write a regulariser: a plain class with `psi`, `nu` and
   `prox(v, vout, lam, sigma=1.0)` = `prox_{(lam/sigma) g}(v/sigma)` in-place.
   Optional fast paths (`dual_update`) and reweighting hooks
   (`init_reweighting`/`update_weights`/`reweight_active`) are sniffed, not required.
2. Register a factory in `deconv/presets.py` mapping CLI options to a `PFBSolver`.
3. Algorithms that do not decompose as forward/prox implement the
   `DeconvSolver` Protocol (`first/forward/backward/last`) directly instead.

The legacy per-algorithm drivers (`core/sara.py`, `core/kclean.py`, reading the
`.dds`) remain as validation oracles; `pfb deconv` reads the imager `.dt`.
```

- [ ] **Step 3: Update `.claude/rules/architecture.md` §5**

Replace the §5 body ("Implement as callable classes with `dot`...") with:

```markdown
Operators are callable classes with `dot` (forward/analysis) and `hdot`
(adjoint/synthesis) methods. The composable deconvolution framework
(`pfb deconv`, issue #185) formalises its seams as `typing.Protocol` classes —
`LinearOperator`/`PsiOperator` (`operators/__init__.py`), `ForwardSolver`/
`BackwardSolver` (`opt/__init__.py`), `Regulariser`/`DeconvSolver`
(`deconv/__init__.py`). **Never introduce ABCs for these seams**; implementations
are plain classes satisfying the Protocols structurally, composed by
`deconv/pfb.PFBSolver` and the `deconv/presets.py` registry. Design:
`docs/superpowers/specs/2026-07-06-gendeconv-protocols-design.md`.
```

- [ ] **Step 4: Full verification**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs
git diff --stat src/pfb_imaging/cabs/   # must be clean
uv run pytest -v tests/
```
Expected: cabs unchanged, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit --no-verify -m "docs(deconv): document Protocol framework; drop stale protocol names"
```

---

## Execution notes

- Task order is strict up to Task 9 (each consumes the previous); Tasks 10–13 are sequential after that. Task 8 (`HessTreeRay`) and Task 9 (`PFBSolver`) can run in either order relative to each other, but nothing else may be reordered.
- The kclean mapping (`OneShot` backward + `ClarkProx`) is deliberately OUT of scope (spec non-goal); do not add it.
- If a numerical tolerance in Tasks 6/8/12 fails marginally, the first suspect is a convention mismatch (sigma/tau constants, wsum normalisation, band ordering) — do not loosen tolerances until the convention is verified against the legacy code path cited in the task.
