# radiomesh-backed weight_data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sympy-lambdified stokes functions inside `weight_data` with radiomesh's
pre-generated expression functions so the numba disk cache works across processes (fixes
pfb-imaging#273, settles #183).

**Architecture:** Two repos. radiomesh's `gen_expr.py` generator gains diag-jones and minvar
variants (surgical: generator + regenerated module + one oracle test). pfb-imaging's
`weight_data` `@overload` then selects plain radiomesh functions instead of lambdifying sympy —
closure cells hold only plain module-level functions and ints, which is what makes numba's
cache key stable. Follow-up commits flip the outer njit to `cache=True` and canonicalise the
`stokes_image` inputs to kill signature churn.

**Tech Stack:** numba (`@overload`, `register_jitable`), sympy (generator/test-only),
rarg-numba-patterns (`load_data`), Ray, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-weight-data-radiomesh-design.md`

## Global Constraints

- **Cache-safety rule (load-bearing):** closure cells of any `@overload` impl may contain only
  plain module-level functions and ints — never njit `Dispatcher` objects (their pickle
  embeds a per-process UUID and poisons the cache key).
- radiomesh repo: `/home/bester/software/radiomesh`, 2-space indent, ruff line length 88,
  changes stay surgical (generator + regenerated file + one test module). Run its tooling
  via `uv run` **inside that directory**.
- pfb-imaging repo: `/home/bester/software/pfb-imaging`, branch `issue273`. After any code
  change: `uv run ruff format . && uv run ruff check . --fix`. Conventional commit messages,
  first line < 72 chars, one mechanism per commit.
- No CLI modules are touched → no cab regeneration needed. No `docs/wiki/` page references
  `weight_data`/`stokes_funcs` (checked) → no wiki maintenance needed.
- `weight_data`'s Python-visible behaviour (12-arg signature, output shapes/dtypes, error
  types) must not change; consumers `stokes2im.py`, `stokes2vis.py`, `stokes2vis_msv4.py`
  are not modified except commit 3's call-site normalisation in `stokes2im.py`.
- Local runs may need `NUMBA_THREADING_LAYER=workqueue` set *after* importing `pfb_imaging`
  (its `__init__` forces `tbb`, which is broken in some dev venvs); subprocess test snippets
  below do this explicitly.

---

### Task 1: radiomesh — failing oracle test for the new generated functions

**Files:**
- Create: `/home/bester/software/radiomesh/radiomesh/tests/test_stokes_expr.py`

**Interfaces:**
- Produces (once Task 2 makes it pass): module `radiomesh.generated._stokes_expr` gains, for
  `POL ∈ {LINEAR, CIRCULAR}` and `S ∈ {I, Q, U, V}`:
  - `{POL}_VIS_DIAGJONES_{S}(v00, v01, v10, v11, jp00, jp11, jq00, jq11)`
  - `{POL}_WEIGHT_DIAGJONES_{S}(w00, w01, w10, w11, jp00, jp11, jq00, jq11)`
  - `{POL}_WEIGHT_MINVAR_DIAGJONES_{S}(w00, w01, w10, w11, jp00, jp11, jq00, jq11)`
  - `CONVERT_FNS` keys `("VIS", POL, "DIAGJONES", S)`, `("WEIGHT", POL, "DIAGJONES", S)`,
    `("WEIGHT_MINVAR", POL, "DIAGJONES", S)`.

- [ ] **Step 1: Create a branch in the radiomesh clone**

```bash
cd /home/bester/software/radiomesh
git checkout main && git pull
git checkout -b stokes-diag-minvar
```

- [ ] **Step 2: Write the failing test**

Create `radiomesh/tests/test_stokes_expr.py` (2-space indent, radiomesh style). The oracle is
the generator's own symbolic derivation (`sympy_expressions`) — the numeric comparison
exercises the diag substitution, minvar term extraction and code printing independently.

```python
import numpy as np
import pytest
import sympy

from radiomesh.generated._stokes_expr import CONVERT_FNS
from radiomesh.scripts.gen_expr import sympy_expressions

STOKES_SCHEMA = ["I", "Q", "U", "V"]

VIS_DIAG_ARGS = "v00 v01 v10 v11 jp00 jp11 jq00 jq11"
WEIGHT_DIAG_ARGS = "w00 w01 w10 w11 jp00 jp11 jq00 jq11"


def diag_expressions(pol_type):
  """Diag-jones oracle: the generator's symbolic derivation with
  off-diagonal jones terms substituted to zero."""
  schema, C, W, _, _ = sympy_expressions(pol_type)
  assert schema == STOKES_SCHEMA
  jp01, jp10, jq01, jq10 = sympy.symbols("jp01 jp10 jq01 jq10", real=False)
  diag = {jp01: 0, jp10: 0, jq01: 0, jq10: 0}
  return sympy.simplify(C.subs(diag)), sympy.simplify(W.subs(diag))


def random_args(names, rng):
  values = {}
  for name in names.split():
    if name.startswith("w"):
      values[name] = rng.uniform(0.1, 2.0)
    else:
      values[name] = rng.uniform(-1, 1) + rng.uniform(-1, 1) * 1j
  return values


@pytest.mark.parametrize("pol_type", ["linear", "circular"])
@pytest.mark.parametrize("stokes", STOKES_SCHEMA)
def test_vis_diagjones(pol_type, stokes):
  rng = np.random.default_rng(42)
  C_diag, _ = diag_expressions(pol_type)
  expr = C_diag[STOKES_SCHEMA.index(stokes)]
  oracle = sympy.lambdify(
    sympy.symbols(VIS_DIAG_ARGS), expr, modules=[{"conjugate": np.conjugate}]
  )
  fn = CONVERT_FNS[("VIS", pol_type.upper(), "DIAGJONES", stokes)]
  for _ in range(5):
    values = random_args(VIS_DIAG_ARGS, rng)
    np.testing.assert_allclose(fn(**values), oracle(**values), rtol=1e-12)


@pytest.mark.parametrize("pol_type", ["linear", "circular"])
@pytest.mark.parametrize("stokes", STOKES_SCHEMA)
def test_weight_diagjones(pol_type, stokes):
  rng = np.random.default_rng(43)
  _, W_diag = diag_expressions(pol_type)
  expr = W_diag[STOKES_SCHEMA.index(stokes)]
  oracle = sympy.lambdify(
    sympy.symbols(WEIGHT_DIAG_ARGS), expr, modules=[{"conjugate": np.conjugate}]
  )
  fn = CONVERT_FNS[("WEIGHT", pol_type.upper(), "DIAGJONES", stokes)]
  for _ in range(5):
    values = random_args(WEIGHT_DIAG_ARGS, rng)
    np.testing.assert_allclose(fn(**values), np.real(oracle(**values)), rtol=1e-12)


@pytest.mark.parametrize("pol_type", ["linear", "circular"])
@pytest.mark.parametrize("stokes", STOKES_SCHEMA)
def test_weight_minvar_diagjones(pol_type, stokes):
  """Minvar oracle mirrors pfb-imaging's stokes_funcs construction:
  4 * Min(*expand(element).args) with Min -> np.minimum."""
  rng = np.random.default_rng(44)
  _, W_diag = diag_expressions(pol_type)
  element = sympy.expand(W_diag[STOKES_SCHEMA.index(stokes)])
  expr = 4 * sympy.Min(*(element.args if isinstance(element, sympy.Add) else (element,)))
  oracle = sympy.lambdify(
    sympy.symbols(WEIGHT_DIAG_ARGS),
    expr,
    modules=[{"Min": np.minimum, "conjugate": np.conjugate}],
  )
  fn = CONVERT_FNS[("WEIGHT_MINVAR", pol_type.upper(), "DIAGJONES", stokes)]
  for _ in range(5):
    values = random_args(WEIGHT_DIAG_ARGS, rng)
    np.testing.assert_allclose(fn(**values), np.real(oracle(**values)), rtol=1e-12)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /home/bester/software/radiomesh
uv run pytest radiomesh/tests/test_stokes_expr.py -x -q
```

Expected: FAIL/ERROR with `KeyError: ('VIS', 'LINEAR', 'DIAGJONES', 'I')` (the new
`CONVERT_FNS` keys don't exist yet).

---

### Task 2: radiomesh — extend the generator, regenerate, make the test pass

**Files:**
- Modify: `/home/bester/software/radiomesh/radiomesh/scripts/gen_expr.py`
- Regenerate: `/home/bester/software/radiomesh/radiomesh/generated/_stokes_expr.py`
- Test: `radiomesh/tests/test_stokes_expr.py` (from Task 1)

**Interfaces:**
- Consumes: `sympy_expressions(pol)` (existing, returns `(schema, C, W, C_nojones, W_nojones)`).
- Produces: the generated functions and `CONVERT_FNS` keys listed in Task 1's Interfaces.

- [ ] **Step 1: Add diag argument lists next to the existing ones in `gen_expr.py`**

After the existing `VIS_FN_ARGUMENTS` line (around line 30):

```python
DIAG_JONES_ARGUMENTS = ["jp00", "jp11", "jq00", "jq11"]
DIAG_WEIGHT_FN_ARGUMENTS = WEIGHT_ARGUMENTS + DIAG_JONES_ARGUMENTS
DIAG_VIS_FN_ARGUMENTS = VIS_ARGUMENTS + DIAG_JONES_ARGUMENTS
```

- [ ] **Step 2: Emit the diag and minvar functions in `generate_expression`**

Inside the `for pol_type in POLARISATION_TYPES:` loop, after the existing four emission loops
(after the `WEIGHT_NOJONES` block, before the loop ends), insert:

```python
    # Diagonal-jones variants: substitute off-diagonal jones terms to zero
    # before simplification. Required by pfb-imaging's weight_data (diag
    # jones path) and by the minvar weights, which cannot be derived from
    # the full-jones expressions (Min over expansion terms would select
    # the structurally-zero cross terms).
    jp01, jp10 = sympy.symbols("jp01 jp10", real=False)
    jq01, jq10 = sympy.symbols("jq01 jq10", real=False)
    diag_subs = {jp01: 0, jp10: 0, jq01: 0, jq10: 0}
    _, coh_full, wgt_full, _, _ = sympy_expressions(pol_type)
    coh_diag = sympy.simplify(coh_full.subs(diag_subs))
    wgt_diag = sympy.simplify(wgt_full.subs(diag_subs))

    for stokes, coh in zip(stokes_schema, coh_diag):
      fn_name = f"{pol_type.upper()}_VIS_DIAGJONES_{stokes.upper()}"
      key = ("VIS", pol_type.upper(), "DIAGJONES", stokes.upper())
      conv_fns[key] = fn_name
      lines.append(f"def {fn_name}({', '.join(DIAG_VIS_FN_ARGUMENTS)}):\n")
      lines.append(f"  return {subs_sympy(coh)}\n")
      lines.append("\n")

    for stokes, wgt in zip(stokes_schema, wgt_diag):
      fn_name = f"{pol_type.upper()}_WEIGHT_DIAGJONES_{stokes.upper()}"
      key = ("WEIGHT", pol_type.upper(), "DIAGJONES", stokes.upper())
      conv_fns[key] = fn_name
      lines.append(f"def {fn_name}({', '.join(DIAG_WEIGHT_FN_ARGUMENTS)}):\n")
      lines.append(f"  return ({subs_sympy(wgt)}).real\n")
      lines.append("\n")

    for stokes, wgt in zip(stokes_schema, wgt_diag):
      # Minimum-variance weights: 4 * min over the expansion terms of the
      # diag weight expression, each term cast to real (the terms are
      # real-valued products; builtin min cannot order complex values).
      fn_name = f"{pol_type.upper()}_WEIGHT_MINVAR_DIAGJONES_{stokes.upper()}"
      key = ("WEIGHT_MINVAR", pol_type.upper(), "DIAGJONES", stokes.upper())
      conv_fns[key] = fn_name
      expanded = sympy.expand(wgt)
      terms = expanded.args if isinstance(expanded, sympy.Add) else (expanded,)
      term_srcs = ", ".join(f"({subs_sympy(t)}).real" for t in terms)
      lines.append(f"def {fn_name}({', '.join(DIAG_WEIGHT_FN_ARGUMENTS)}):\n")
      lines.append(f"  return 4 * min({term_srcs})\n")
      lines.append("\n")
```

Note: `sympy_expressions` is re-called here rather than reusing the loop's `coh_jones`
variables because the existing emission loops shadow them during iteration (`for stokes,
coh_jones in zip(...)` rebinds the name).

- [ ] **Step 3: Regenerate the module**

```bash
cd /home/bester/software/radiomesh
uv run radiomesh gen-expr
git diff --stat radiomesh/generated/_stokes_expr.py
```

Expected: the file grows by 24 new functions (2 pols × 4 stokes × 3 families) and the
`CONVERT_FNS` dict gains 24 entries. Eyeball a couple of the new function bodies: they must
reference only their parameters and `conj`/`min` (no `Abs`, `re`, `im`, `Min`).

- [ ] **Step 4: Run the oracle test — must pass now**

```bash
uv run pytest radiomesh/tests/test_stokes_expr.py -q
```

Expected: 24 passed.

- [ ] **Step 5: Run radiomesh's existing test suite and lint**

```bash
uv run pytest radiomesh/tests/ -q -x --ignore=radiomesh/tests/test_benchmark_es_kernel.py
uv run ruff format radiomesh/ && uv run ruff check radiomesh/ --fix
git diff --stat
```

Expected: existing tests pass (the change is additive); ruff clean.

- [ ] **Step 6: Commit and push, capture the sha**

```bash
cd /home/bester/software/radiomesh
git add radiomesh/scripts/gen_expr.py radiomesh/generated/_stokes_expr.py radiomesh/tests/test_stokes_expr.py
git commit -m "Generate diag-jones and minvar stokes expression variants"
git push -u origin stokes-diag-minvar
git rev-parse HEAD
```

Record the printed sha — Task 3 pins it in pfb-imaging's `pyproject.toml`. (The branch goes
to the user for review/merge into radiomesh main; the pin is swapped to a version once
radiomesh releases.)

---

### Task 3: pfb-imaging — pin radiomesh and sync the environment

**Files:**
- Modify: `/home/bester/software/pfb-imaging/pyproject.toml` (the `full` extra, lines 31-51)
- Modify: `/home/bester/software/pfb-imaging/uv.lock` (via `uv lock`)

**Interfaces:**
- Consumes: the radiomesh commit sha recorded in Task 2 Step 6 (`<SHA>` below).
- Produces: `radiomesh.generated._stokes_expr.CONVERT_FNS` importable in pfb's venv.

- [ ] **Step 1: Add the pin**

In `pyproject.toml`, inside the `full = [...]` list (keep alphabetical-ish placement near
`ray`), add — substituting the actual sha from Task 2:

```toml
    "radiomesh @ git+https://github.com/ratt-ru/radiomesh.git@<SHA>",
```

- [ ] **Step 2: Lock and sync**

```bash
cd /home/bester/software/pfb-imaging
uv lock && uv sync --all-extras
uv run python -c "from radiomesh.generated._stokes_expr import CONVERT_FNS; print(len(CONVERT_FNS))"
```

Expected: prints `56` (32 original + 24 new entries). If the GitHub branch isn't pushed yet,
this fails — Task 2 must complete first. Do **not** commit yet; this lands with Task 5's
commit (the pin is only meaningful together with the code that uses it).

---

### Task 4: pfb-imaging — expression selector with failing tests first

**Files:**
- Modify: `/home/bester/software/pfb-imaging/src/pfb_imaging/utils/stokes.py`
- Create: `/home/bester/software/pfb-imaging/tests/test_stokes_selector.py`

**Interfaces:**
- Consumes: `radiomesh.generated._stokes_expr.CONVERT_FNS` (Task 3).
- Produces: `stokes_expr_funcs(product, pol, nc, wgt_mode, jones_ndim) -> (vis_fns, wgt_fns)`
  in `pfb_imaging.utils.stokes` — two equal-length tuples of **plain** functions, ordered
  I, Q, U, V. Diag fns take 8 args (`v00, v01, v10, v11, jp00, jp11, jq00, jq11` /
  `w00, w01, w10, w11, jp00, jp11, jq00, jq11`); full-jones fns take 12
  (`..., jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11`). Also produces the side effect
  that every selected function is `register_jitable`'d (callable from nopython code).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stokes_selector.py`:

```python
import pytest
from radiomesh.generated import _stokes_expr as se

from pfb_imaging.utils.stokes import stokes_expr_funcs


def test_selects_diag_l2_in_iquv_order():
    vis_fns, wgt_fns = stokes_expr_funcs("VQI", "linear", "4", "l2", 5)
    # always ordered I, Q, U, V regardless of product string order
    assert vis_fns == (
        se.LINEAR_VIS_DIAGJONES_I,
        se.LINEAR_VIS_DIAGJONES_Q,
        se.LINEAR_VIS_DIAGJONES_V,
    )
    assert wgt_fns == (
        se.LINEAR_WEIGHT_DIAGJONES_I,
        se.LINEAR_WEIGHT_DIAGJONES_Q,
        se.LINEAR_WEIGHT_DIAGJONES_V,
    )


def test_selects_minvar_weights():
    vis_fns, wgt_fns = stokes_expr_funcs("I", "circular", "4", "minvar", 5)
    assert vis_fns == (se.CIRCULAR_VIS_DIAGJONES_I,)
    assert wgt_fns == (se.CIRCULAR_WEIGHT_MINVAR_DIAGJONES_I,)


def test_selects_full_jones():
    vis_fns, wgt_fns = stokes_expr_funcs("IQUV", "linear", "4", "l2", 6)
    assert vis_fns[0] is se.LINEAR_VIS_JONES_I
    assert wgt_fns[3] is se.LINEAR_WEIGHT_JONES_V
    assert len(vis_fns) == len(wgt_fns) == 4


@pytest.mark.parametrize(
    ("product", "pol", "nc", "wgt_mode", "jones_ndim", "exc"),
    [
        ("I", "elliptical", "4", "l2", 5, ValueError),      # unknown pol
        ("I", "linear", "3", "l2", 5, ValueError),          # unsupported nc
        ("I", "linear", "4", "huber", 5, ValueError),       # unknown mode
        ("Q", "circular", "2", "l2", 5, ValueError),        # Q needs 4 corr (circular)
        ("U", "linear", "2", "l2", 5, ValueError),          # U needs 4 corr
        ("V", "linear", "2", "l2", 5, ValueError),          # V needs 4 corr (linear)
        ("IX", "linear", "4", "l2", 5, ValueError),         # unknown product letter
        ("I", "linear", "1", "l2", 5, ValueError),          # 1 corr unsupported
        ("I", "linear", "4", "minvar", 6, NotImplementedError),  # minvar full-jones
        ("I", "linear", "2", "l2", 6, ValueError),          # full jones needs 4 corr
        ("I", "linear", "4", "l2", 4, ValueError),          # bad jones ndim
    ],
)
def test_selector_validation(product, pol, nc, wgt_mode, jones_ndim, exc):
    with pytest.raises(exc):
        stokes_expr_funcs(product, pol, nc, wgt_mode, jones_ndim)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/bester/software/pfb-imaging
uv run pytest tests/test_stokes_selector.py -q
```

Expected: FAIL — `ImportError: cannot import name 'stokes_expr_funcs'`.

- [ ] **Step 3: Implement the selector and delete the sympy derivation**

In `src/pfb_imaging/utils/stokes.py`:

1. Replace the module imports — delete `import sympy as sm`, `from numba import njit`,
   `from numba.core import types`, `from sympy.physics.quantum import TensorProduct`; add:

```python
from numba.extending import register_jitable

from radiomesh.generated._stokes_expr import CONVERT_FNS
```

   (keep `import string`, `import numpy as np` — the numpy helpers stay).

2. Delete the entire `stokes_funcs` function (lines 74-314 in the current file).

3. Add in its place:

```python
# Make every generated expression function callable from nopython code.
# register_jitable returns the original plain function, so CONVERT_FNS
# values stay plain module-level functions: capturing them in an @overload
# impl closure keeps numba's cache key stable across processes (issue #273
# in pfb-imaging) — never replace these with njit dispatchers.
for _fn in set(CONVERT_FNS.values()):
    register_jitable(inline="always")(_fn)


def stokes_expr_funcs(product, pol, nc, wgt_mode, jones_ndim):
    """Select radiomesh per-Stokes expression functions for weight_data.

    Args:
        product: Stokes product string, subset of "IQUV" (any order).
        pol: Polarisation type, "linear" or "circular".
        nc: Number of correlations as a string, "2" or "4".
        wgt_mode: Weighting mode, "l2" or "minvar".
        jones_ndim: 5 for diagonal jones, 6 for full 2x2 jones.

    Returns:
        Tuple of (vis_fns, wgt_fns): equal-length tuples of plain functions
        ordered I, Q, U, V. Diag functions take
        (x00, x01, x10, x11, jp00, jp11, jq00, jq11); full-jones functions
        take (x00, x01, x10, x11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11).

    Raises:
        ValueError: For invalid product/pol/nc/mode/ndim combinations.
        NotImplementedError: For minvar with full 2x2 jones.
    """
    if pol not in ("linear", "circular"):
        raise ValueError(f"Unknown polarisation type {pol}")
    if nc not in ("1", "2", "4"):
        raise ValueError(f"Unsupported number of correlations {nc}")
    if wgt_mode not in ("l2", "minvar"):
        raise ValueError(f"Unknown weighting mode {wgt_mode}")

    # this ensures that outputs are always ordered as [I, Q, U, V]
    stokes = []
    if "I" in product:
        stokes.append("I")
    if "Q" in product:
        if pol == "circular" and nc == "2":
            raise ValueError("Q is not available in circular polarisation with 2 correlations")
        stokes.append("Q")
    if "U" in product:
        if nc == "2":
            raise ValueError(f"U is not available in {pol} polarisation with 2 correlations")
        stokes.append("U")
    if "V" in product:
        if pol == "linear" and nc == "2":
            raise ValueError("V is not available in linear polarisation with 2 correlations")
        stokes.append("V")
    remprod = product.strip("IQUV")
    if len(remprod):
        raise ValueError(f"Unknown polarisation product {remprod}")

    polu = pol.upper()
    if jones_ndim == 6:
        if wgt_mode == "minvar":
            raise NotImplementedError("Minvar weighting not yet implemented for full-Stokes")
        if nc != "4":
            raise ValueError("Full 2x2 jones mode requires 4 correlation data")
        wkey, jmode = "WEIGHT", "JONES"
    elif jones_ndim == 5:
        if nc not in ("2", "4"):
            raise ValueError(
                f"Selected product is only available from 2 or 4 correlation data while you have ncorr={nc}."
            )
        wkey = "WEIGHT_MINVAR" if wgt_mode == "minvar" else "WEIGHT"
        jmode = "DIAGJONES"
    else:
        raise ValueError("Jones term has incorrect number of dimensions")

    vis_fns = tuple(CONVERT_FNS[("VIS", polu, jmode, s)] for s in stokes)
    wgt_fns = tuple(CONVERT_FNS[(wkey, polu, jmode, s)] for s in stokes)
    return vis_fns, wgt_fns
```

- [ ] **Step 4: Run the selector tests — must pass**

```bash
uv run pytest tests/test_stokes_selector.py -q
```

Expected: all pass. (`utils/weighting.py` still imports `stokes_funcs` and is now broken —
fixed in Task 5; that's why there is no commit here.)

- [ ] **Step 5: Verify register_jitable returned the original functions**

```bash
uv run python -c "
from radiomesh.generated._stokes_expr import CONVERT_FNS
import pfb_imaging.utils.stokes  # triggers register_jitable loop
from radiomesh.generated._stokes_expr import LINEAR_VIS_JONES_I
assert CONVERT_FNS[('VIS', 'LINEAR', 'JONES', 'I')] is LINEAR_VIS_JONES_I
print('plain functions preserved')
"
```

Expected: `plain functions preserved`. If this assertion fails, the numba version's
`register_jitable` wraps functions — STOP and re-evaluate the cache-key stability before
proceeding (see Global Constraints).

---

### Task 5: pfb-imaging — rewrite the weight_data overload on radiomesh functions (commit 1)

**Files:**
- Modify: `/home/bester/software/pfb-imaging/src/pfb_imaging/utils/weighting.py` (lines 1-15
  imports; lines 306-411: `nb_weight_data_impl` and `weight_data_np`)
- Modify: `/home/bester/software/pfb-imaging/pyproject.toml` (add `rarg-numba-patterns` to `full`)
- Test: existing `tests/test_polproducts.py`, `tests/test_imager.py` (oracles — not modified)

**Interfaces:**
- Consumes: `stokes_expr_funcs` (Task 4), `rarg_numba_patterns.load_data(array, index, ndata,
  axis)` → UniTuple, radiomesh generated function signatures (Task 1 Interfaces).
- Produces: `weight_data(data, weight, flag, jones, tbin_idx, tbin_counts, ant1, ant2, pol,
  product, nc, wgt_mode) -> (vis, wgt)` — unchanged 12-arg signature; `vis` shape
  `(nrow, nchan, ns)` dtype `data.dtype`, `wgt` same shape dtype `data.real.dtype`.

- [ ] **Step 1: Add the rarg-numba-patterns dependency**

In `pyproject.toml`'s `full` extra (next to the radiomesh pin from Task 3):

```toml
    "rarg-numba-patterns>=0.0.1",
```

Then `uv lock && uv sync --all-extras`.

- [ ] **Step 2: Update weighting.py imports**

Replace `from pfb_imaging.utils.stokes import stokes_funcs` with:

```python
from numba.extending import overload, register_jitable
from rarg_numba_patterns import load_data

from pfb_imaging.utils.stokes import stokes_expr_funcs
```

(delete the old `from numba.extending import overload` line if it becomes duplicated).

- [ ] **Step 3: Add the correlation adapters at module level in weighting.py**

Directly above `weight_data` (line ~250). These consume the UniTuple from `load_data` and
produce the full 4-correlation arguments the generated functions take:

```python
@register_jitable(inline="always")
def _corr2_to_full_vis(v):
    # unsampled cross-hand visibilities are zero (legacy stokes_funcs convention)
    return v[0], 0j, 0j, v[1]


@register_jitable(inline="always")
def _corr2_to_full_wgt(w):
    # unsampled cross-hand weights are unity (legacy stokes_funcs convention)
    return w[0], 1.0, 1.0, w[1]


@register_jitable(inline="always")
def _corr4_passthrough(x):
    return x[0], x[1], x[2], x[3]
```

- [ ] **Step 4: Replace `nb_weight_data_impl` (keep `weight_data` and `_weight_data_impl` as-is)**

Replace the whole `@overload(...) def nb_weight_data_impl(...)` block (currently lines
306-365) with the following. Notes baked into the design: closure cells hold only plain
functions (`vf*/wf*/vext/wext`) and the int `ns` — the cache-safety rule; unused per-stokes
slots alias slot 0 so a single impl body types for every `ns` (the `if ns > k` branches are
compile-time constant and fold away); the old `_impl.returns = types.Tuple(...)` line is
**dropped** — verified inert (numba never reads a `.returns` attribute; output dtypes come
from the allocations, unchanged).

```python
@overload(_weight_data_impl, prefer_literal=True, jit_options={**JIT_OPTIONS, "parallel": True})
def nb_weight_data_impl(
    data,
    weight,
    flag,
    jones,
    tbin_idx,
    tbin_counts,
    ant1,
    ant2,
    pol,
    product,
    nc,
    wgt_mode,
):
    try:
        vis_fns, wgt_fns = stokes_expr_funcs(
            product.literal_value,
            pol.literal_value,
            nc.literal_value,
            wgt_mode.literal_value,
            jones.ndim,
        )
    except Exception as e:
        raise numba.core.errors.TypingError(f"Failed in overload resolution: {e}") from e

    ns = len(vis_fns)
    NC = int(nc.literal_value)
    if NC == 2:
        vext, wext = _corr2_to_full_vis, _corr2_to_full_wgt
    else:
        vext = _corr4_passthrough
        wext = _corr4_passthrough
    vf0, wf0 = vis_fns[0], wgt_fns[0]
    vf1, wf1 = (vis_fns[1], wgt_fns[1]) if ns > 1 else (vis_fns[0], wgt_fns[0])
    vf2, wf2 = (vis_fns[2], wgt_fns[2]) if ns > 2 else (vis_fns[0], wgt_fns[0])
    vf3, wf3 = (vis_fns[3], wgt_fns[3]) if ns > 3 else (vis_fns[0], wgt_fns[0])

    if jones.ndim == 5:  # DIAG mode

        def _impl(
            data,
            weight,
            flag,
            jones,
            tbin_idx,
            tbin_counts,
            ant1,
            ant2,
            pol,
            product,
            nc,
            wgt_mode,
        ):
            # for dask arrays we need to adjust the chunks to
            # start counting from zero
            tbin_idx = tbin_idx - tbin_idx.min()
            nt = np.shape(tbin_idx)[0]
            nrow, nchan, _ = data.shape
            vis = np.zeros((nrow, nchan, ns), dtype=data.dtype)
            wgt = np.zeros((nrow, nchan, ns), dtype=data.real.dtype)

            for t in prange(nt):
                for row in range(tbin_idx[t], tbin_idx[t] + tbin_counts[t]):
                    p = int(ant1[row])
                    q = int(ant2[row])
                    for chan in range(nchan):
                        if flag[row, chan].any():
                            continue
                        jp00 = jones[t, p, chan, 0, 0]
                        jp11 = jones[t, p, chan, 0, 1]
                        jq00 = jones[t, q, chan, 0, 0]
                        jq11 = jones[t, q, chan, 0, 1]
                        v00, v01, v10, v11 = vext(load_data(data, (row, chan), NC, -1))
                        w00, w01, w10, w11 = wext(load_data(weight, (row, chan), NC, -1))
                        vis[row, chan, 0] = vf0(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                        wgt[row, chan, 0] = wf0(w00, w01, w10, w11, jp00, jp11, jq00, jq11)
                        if ns > 1:
                            vis[row, chan, 1] = vf1(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                            wgt[row, chan, 1] = wf1(w00, w01, w10, w11, jp00, jp11, jq00, jq11)
                        if ns > 2:
                            vis[row, chan, 2] = vf2(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                            wgt[row, chan, 2] = wf2(w00, w01, w10, w11, jp00, jp11, jq00, jq11)
                        if ns > 3:
                            vis[row, chan, 3] = vf3(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                            wgt[row, chan, 3] = wf3(w00, w01, w10, w11, jp00, jp11, jq00, jq11)

            return (vis, wgt)

    else:  # full 2x2 jones mode

        def _impl(
            data,
            weight,
            flag,
            jones,
            tbin_idx,
            tbin_counts,
            ant1,
            ant2,
            pol,
            product,
            nc,
            wgt_mode,
        ):
            # for dask arrays we need to adjust the chunks to
            # start counting from zero
            tbin_idx = tbin_idx - tbin_idx.min()
            nt = np.shape(tbin_idx)[0]
            nrow, nchan, _ = data.shape
            vis = np.zeros((nrow, nchan, ns), dtype=data.dtype)
            wgt = np.zeros((nrow, nchan, ns), dtype=data.real.dtype)

            for t in prange(nt):
                for row in range(tbin_idx[t], tbin_idx[t] + tbin_counts[t]):
                    p = int(ant1[row])
                    q = int(ant2[row])
                    for chan in range(nchan):
                        if flag[row, chan].any():
                            continue
                        jp00 = jones[t, p, chan, 0, 0, 0]
                        jp01 = jones[t, p, chan, 0, 0, 1]
                        jp10 = jones[t, p, chan, 0, 1, 0]
                        jp11 = jones[t, p, chan, 0, 1, 1]
                        jq00 = jones[t, q, chan, 0, 0, 0]
                        jq01 = jones[t, q, chan, 0, 0, 1]
                        jq10 = jones[t, q, chan, 0, 1, 0]
                        jq11 = jones[t, q, chan, 0, 1, 1]
                        v00, v01, v10, v11 = vext(load_data(data, (row, chan), NC, -1))
                        w00, w01, w10, w11 = wext(load_data(weight, (row, chan), NC, -1))
                        vis[row, chan, 0] = vf0(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        wgt[row, chan, 0] = wf0(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        if ns > 1:
                            vis[row, chan, 1] = vf1(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                            wgt[row, chan, 1] = wf1(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        if ns > 2:
                            vis[row, chan, 2] = vf2(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                            wgt[row, chan, 2] = wf2(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        if ns > 3:
                            vis[row, chan, 3] = vf3(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                            wgt[row, chan, 3] = wf3(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)

            return (vis, wgt)

    return _impl
```

- [ ] **Step 5: Delete dead code**

- Delete `weight_data_np` (weighting.py lines 368-411 in the current file) — it consumed the
  old lambdified functions and has no callers (`grep -rn weight_data_np src/ tests/` must
  come back empty afterwards).
- In `weighting.py`, remove `types` from the `from numba import ...` line **only if** the
  module no longer uses it (`grep -n 'types\.' src/pfb_imaging/utils/weighting.py`).

- [ ] **Step 6: Quick numeric smoke test (identity jones, unit weights)**

```bash
cd /home/bester/software/pfb-imaging
uv run python -c "
import os
import numpy as np
import pfb_imaging
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
from pfb_imaging.utils.weighting import weight_data

nrow, nchan = 6, 4
rng = np.random.default_rng(0)
for nc in (2, 4):
    data = (rng.normal(size=(nrow, nchan, nc)) + 1j * rng.normal(size=(nrow, nchan, nc))).astype(np.complex64)
    weight = np.ones((nrow, nchan, nc), np.float32)
    flag = np.zeros((nrow, nchan, nc), bool)
    jones = np.ones((1, 3, nchan, 1, 2), np.complex64)
    tbin_idx = np.array([0], np.int32); tbin_counts = np.array([nrow], np.int32)
    ant1 = np.array([0, 0, 0, 1, 1, 2], np.int32); ant2 = np.array([1, 2, 2, 2, 0, 0], np.int32)
    for mode in ('l2', 'minvar'):
        vis, wgt = weight_data(data, weight, flag, jones, tbin_idx, tbin_counts, ant1, ant2, 'linear', 'I', str(nc), mode)
        # identity jones, unit weights: I = (v00 + v11) / 2, weight I = 4 min(w) or sum(w)
        expected = 0.5 * (data[..., 0] + data[..., -1])
        np.testing.assert_allclose(vis[..., 0], expected, rtol=1e-6)
        assert vis.dtype == np.complex64 and wgt.dtype == np.float32
        print(f'nc={nc} mode={mode} OK, wgt[0,0,0]={wgt[0,0,0]}')
"
```

Expected: four `OK` lines; l2 weight is 2.0 (w00 + w11) and minvar weight 4.0
(4·min(1,1)) for unit weights — print and eyeball against those values.

Contingency: if `load_data` fails typing with `RequireLiteralValue` on `ndata` (the closure
int `NC` not literalised on some numba version), hoist the extraction into the two adapter
call sites by making `NC` a plain constant per branch of the overload — or fall back to
plain indexing (`data[row, chan, 0]`, ...) inside `_impl`; the cache-safety rule is
unaffected either way.

- [ ] **Step 7: Run the consumer oracle tests**

```bash
uv run pytest tests/test_stokes_selector.py tests/test_polproducts.py tests/test_imager.py -q
```

Expected: all pass. `test_polproducts[do_gains=True]` is the strong oracle — it corrupts
visibilities with random gains and checks the imaged IQUV point-source fluxes recover.

- [ ] **Step 8: Lint and commit (commit 1)**

```bash
uv run ruff format . && uv run ruff check . --fix
git add pyproject.toml uv.lock src/pfb_imaging/utils/stokes.py src/pfb_imaging/utils/weighting.py tests/test_stokes_selector.py
git commit -m "feat(weighting): radiomesh expressions replace sympy in weight_data

The @overload impl previously closed over freshly-lambdified njit
dispatchers whose pickled bytes embed a per-process UUID, so the numba
disk cache never hit and every compile appended a new .nbc to the cache
dir (issue #273). Closure cells now hold only plain module-level
functions from radiomesh.generated._stokes_expr (stable cache keys) and
overload resolution no longer runs sympy (~seconds per fresh Ray
worker)."
```

---

### Task 6: pfb-imaging — end-to-end cacheability + #273 regression test (commit 2)

**Files:**
- Modify: `/home/bester/software/pfb-imaging/src/pfb_imaging/utils/weighting.py:256` (the
  `weight_data` decorator)
- Create: `/home/bester/software/pfb-imaging/tests/test_weight_data_cache.py`

**Interfaces:**
- Consumes: `weight_data` (Task 5).
- Produces: nothing new — behavioural guarantee that a second process loads the compile from
  disk instead of recompiling.

- [ ] **Step 1: Write the failing regression test**

Create `tests/test_weight_data_cache.py`:

```python
"""Cross-process numba cache regression test for issue #273.

weight_data's compile chain must be servable from the on-disk numba cache:
a second process compiling the same signature must LOAD, not recompile and
re-save. The old sympy-lambdified implementation failed this (closure cells
held dispatchers whose pickle embeds a per-process UUID), making the cache
write-only and growing /tmp/numba without bound.
"""

import os
import subprocess
import sys

SNIPPET = """
import os
os.environ["NUMBA_CACHE_DIR"] = {cache_dir!r}
os.environ["NUMBA_DEBUG_CACHE"] = "1"
import numpy as np
import pfb_imaging  # noqa: F401  (forces NUMBA_THREADING_LAYER=tbb)
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
from pfb_imaging.utils.weighting import weight_data

nrow, nchan, ncorr = 6, 4, 2
data = np.zeros((nrow, nchan, ncorr), np.complex64)
weight = np.ones((nrow, nchan, ncorr), np.float32)
flag = np.zeros((nrow, nchan, ncorr), bool)
jones = np.ones((1, 3, nchan, 1, 2), np.complex64)
tbin_idx = np.array([0], np.int32)
tbin_counts = np.array([nrow], np.int32)
ant1 = np.array([0, 0, 0, 1, 1, 2], np.int32)
ant2 = np.array([1, 2, 2, 2, 0, 0], np.int32)
weight_data(data, weight, flag, jones, tbin_idx, tbin_counts,
            ant1, ant2, "linear", "I", "2", "minvar")
"""


def _run(cache_dir):
    env = os.environ.copy()
    env.pop("NUMBA_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET.format(cache_dir=cache_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def _nbc_count(cache_dir):
    return sum(1 for _, _, files in os.walk(cache_dir) for f in files if f.endswith(".nbc"))


def test_weight_data_cross_process_cache(tmp_path):
    cache_dir = str(tmp_path / "numba_cache")

    out1 = _run(cache_dir)
    n1 = _nbc_count(cache_dir)
    assert n1 > 0, f"first process saved nothing to the cache:\n{out1}"

    out2 = _run(cache_dir)
    n2 = _nbc_count(cache_dir)
    assert "data loaded from" in out2, f"second process never loaded from cache:\n{out2}"
    assert n2 == n1, f"second process wrote new cache entries ({n1} -> {n2}):\n{out2}"
```

- [ ] **Step 2: Run it to see the current failure mode**

```bash
cd /home/bester/software/pfb-imaging
uv run pytest tests/test_weight_data_cache.py -q
```

Expected: FAIL on `n2 == n1` **or** on `"data loaded from"` — the outer `weight_data` is
still `cache=False`, so the second process recompiles it (the overload impl may load). If it
unexpectedly passes, check that `NUMBA_CACHE_DIR` was really honoured (a `.nbi` file must
exist under `tmp_path`).

- [ ] **Step 3: Flip the outer decorator**

`src/pfb_imaging/utils/weighting.py:256`:

```python
@njit(nogil=True, cache=True, parallel=False)
def weight_data(
```

- [ ] **Step 4: Run the regression test — must pass**

```bash
uv run pytest tests/test_weight_data_cache.py -q
```

Expected: PASS. Second process output contains `data loaded from` lines and the `.nbc` count
is unchanged.

- [ ] **Step 5: Commit (commit 2)**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/weighting.py tests/test_weight_data_cache.py
git commit -m "perf(weighting): cache weight_data compiles across processes

With the overload impl cacheable (previous commit) the outer njit can be
cache=True: fresh Ray workers warm-start weight_data from the on-disk
cache instead of paying a full compile. Adds the cross-process cache
regression test for issue #273."
```

---

### Task 7: pfb-imaging — canonicalise stokes_image's weight_data inputs (commit 3)

**Files:**
- Modify: `/home/bester/software/pfb-imaging/src/pfb_imaging/utils/stokes2im.py` (module-level
  helper + the `weight_data` call at lines 446-459)
- Modify: `/home/bester/software/pfb-imaging/tests/test_weighting.py` (append one test)

**Interfaces:**
- Consumes: `weight_data` (Task 5).
- Produces: `as_contiguous_readonly_view(a) -> np.ndarray` in `pfb_imaging.utils.stokes2im` —
  C-contiguous readonly view (copy only when non-contiguous).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weighting.py`:

```python
def testas_contiguous_readonly_view_flags():
    from pfb_imaging.utils.stokes2im import as_contiguous_readonly_view

    # contiguous input: no copy, readonly view
    a = np.ones((4, 3, 2), dtype=np.float32)
    v = as_contiguous_readonly_view(a)
    assert v.flags["C_CONTIGUOUS"] and not v.flags["WRITEABLE"]
    assert np.shares_memory(a, v)
    assert a.flags["WRITEABLE"]  # original untouched

    # non-contiguous input (like the jones swapaxes view): copied contiguous
    b = np.ones((4, 3, 2), dtype=np.complex64).swapaxes(0, 1)
    assert not b.flags["C_CONTIGUOUS"]
    w = as_contiguous_readonly_view(b)
    assert w.flags["C_CONTIGUOUS"] and not w.flags["WRITEABLE"]
    assert not np.shares_memory(b, w)
```

Note `tests/test_weighting.py` imports `numpy as np` already; check its import block and
match style.

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_weighting.py::testas_contiguous_readonly_view_flags -q
```

Expected: FAIL — `ImportError: cannot import name 'as_contiguous_readonly_view'`.

- [ ] **Step 3: Implement the helper and normalise the call site**

In `src/pfb_imaging/utils/stokes2im.py`, at module level (after the imports):

```python
def as_contiguous_readonly_view(a):
    """C-contiguous readonly view of ``a`` (copies only if non-contiguous).

    weight_data never mutates its inputs, and numba specialises on each
    array's layout and writeable flag: normalising here means one compiled
    signature per dtype configuration instead of one per readonly/layout
    permutation of the eight array arguments (issue #273).
    """
    a = np.ascontiguousarray(a)
    v = a.view()
    v.flags.writeable = False
    return v
```

and change the `weight_data` call (currently lines 446-459) to:

```python
    data, weight = weight_data(
        as_contiguous_readonly_view(data),
        as_contiguous_readonly_view(weight),
        as_contiguous_readonly_view(flag),
        as_contiguous_readonly_view(jones),
        as_contiguous_readonly_view(tbin_idx),
        as_contiguous_readonly_view(tbin_counts),
        as_contiguous_readonly_view(ant1),
        as_contiguous_readonly_view(ant2),
        poltype,
        product,
        str(ncorr),
        wgt_mode,
    )
```

(`data`/`weight` are rebound by the return; `flag`, `ant1`, `ant2` etc. keep their original
writable bindings for the code after the call.)

- [ ] **Step 4: Run the test — must pass**

```bash
uv run pytest tests/test_weighting.py -q
```

Expected: PASS (all of test_weighting.py, including the pre-existing tests).

- [ ] **Step 5: Commit (commit 3)**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/pfb_imaging/utils/stokes2im.py tests/test_weighting.py
git commit -m "perf(hci): canonicalise weight_data inputs in stokes_image

numba specialises on layout and the writeable flag, and Ray hands tasks a
per-task mix of readonly zero-copy arrays and writable local copies, so
stokes_image triggered a fresh weight_data compile (and cache write) per
readonly/layout permutation. Feed it C-contiguous readonly views instead:
one signature per dtype configuration (issue #273)."
```

---

### Task 8: pfb-imaging — hci consumer test (l2 + minvar)

**Files:**
- Create: `/home/bester/software/pfb-imaging/tests/test_hci.py`

**Interfaces:**
- Consumes: `pfb_imaging.core.hci.hci` (existing), session fixtures `ms_name` and the
  autouse `manage_ray` fixture from `tests/conftest.py`.
- Produces: pytest coverage for the hci consumer of `weight_data` (previously only the
  manual `test_hci.yml` stimela recipe).

- [ ] **Step 1: Write the test**

Create `tests/test_hci.py`:

```python
"""In-process smoke test for the hci sub-command (issue #273 consumer coverage).

Runs the Ray-distributed batch_stokes_image/stokes_image path on the small
test MS for both weighting modes and sanity-checks the output zarr cube.
minvar is the mode that exercised the weight_data recompile bug in
production.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.core.hci import hci as hci_core

pmp = pytest.mark.parametrize


@pmp("wgt_mode", ("l2", "minvar"))
def test_hci_writes_cube(wgt_mode, ms_name, tmp_path):
    outname = str(tmp_path / f"test_hci_{wgt_mode}.zarr")

    hci_core(
        [Path(ms_name)],
        outname,
        product="I",
        data_column="DATA",
        integrations_per_image=8,
        images_per_chunk=4,
        max_simul_chunks=2,
        field_of_view=1.0,
        precision="single",
        wgt_mode=wgt_mode,
        overwrite=True,
        keep_ray_alive=True,
    )

    ds = xr.open_zarr(outname, chunks=None)
    cube = ds.cube.values      # (STOKES, FREQ, TIME, Y, X)
    wsum = ds.weight.values    # (STOKES, FREQ, TIME)

    assert (wsum > 0).any(), "no data was imaged"
    imaged = cube[wsum > 0]
    assert np.isfinite(imaged).all()
    assert np.abs(imaged).max() > 0
```

- [ ] **Step 2: Run it**

```bash
cd /home/bester/software/pfb-imaging
uv run pytest tests/test_hci.py -q
```

Expected: 2 passed (this is new coverage, so it should pass immediately; if dimension names
differ from `cube`/`weight`, read the dataset construction in
`pfb_imaging/core/hci.py::make_dummy_dataset` and fix the variable names in the test — not
the core).

- [ ] **Step 3: Commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add tests/test_hci.py
git commit -m "test(hci): in-process cube smoke test for l2 and minvar weighting"
```

---

### Task 9: Final verification

**Files:** none modified (verification only).

- [ ] **Step 1: Full pfb-imaging suite**

```bash
cd /home/bester/software/pfb-imaging
uv run pytest tests/ -q
```

Expected: everything passes. Watch `test_roundtrip.py` in particular (it fails if anything
accidentally touched CLI help text or cabs — nothing in this plan should).

- [ ] **Step 2: Full radiomesh suite (once more, on the pushed branch)**

```bash
cd /home/bester/software/radiomesh
uv run pytest radiomesh/tests/ -q --ignore=radiomesh/tests/test_benchmark_es_kernel.py
```

Expected: all pass.

- [ ] **Step 3: Lint both repos, confirm clean trees**

```bash
uv run ruff format . && uv run ruff check . && git status --short
cd /home/bester/software/radiomesh && uv run ruff format radiomesh/ && uv run ruff check radiomesh/ && git status --short
```

Expected: no diffs, clean trees.

- [ ] **Step 4: Review the commit stack**

```bash
cd /home/bester/software/pfb-imaging && git log --oneline f698702..HEAD
```

Expected shape (docs commits from the design phase, then):

```
test(hci): in-process cube smoke test for l2 and minvar weighting
perf(hci): canonicalise weight_data inputs in stokes_image
perf(weighting): cache weight_data compiles across processes
feat(weighting): radiomesh expressions replace sympy in weight_data
```

Do **not** push or open PRs — the user reviews the radiomesh branch and decides merge order
(radiomesh first, then swap the pfb git pin to a released version or merged-main sha).

## Post-plan notes

- **Perf sanity (optional but recommended):** the scratchpad repro from the #273
  investigation measured ~6 s cold compile per fresh process for the old path. After Task 6,
  re-run the equivalent snippet twice and confirm the second process' first call drops to
  well under a second (cache load).
- **Follow-ups out of scope:** the hci Ray dispatch pattern (#237), the shared `/tmp/numba`
  default (#270), swapping the radiomesh git pin to a version pin after release.
