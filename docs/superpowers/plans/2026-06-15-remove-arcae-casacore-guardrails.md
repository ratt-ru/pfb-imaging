# Remove arcae ⊥ python-casacore guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Now that arcae 0.5.2 lets arcae and python-casacore coexist in one process, remove the test-suite, CI and documentation guardrails that kept them in separate processes — without touching the (deliberately retained) casacore-free imaging-path import discipline.

**Architecture:** Bump the dependency to pull arcae ≥ 0.5.2, prove coexistence holds in our env (hard gate), then collapse the two-invocation test split into a single `pytest tests/`, convert the equivalence test from a subprocess to in-process calls, and reframe the docs from "they segfault together (arcae#72)" to "they coexist; the imaging path stays casacore-free by choice." The `conftest.py` `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` setting and all source-level deferred imports are left untouched.

**Tech Stack:** Python 3.11–3.13, uv, pytest, Ray, arcae / xarray-ms, daskms, GitHub Actions.

**Scope (decided during brainstorming):** tests / CI / docs only. The imaging-path source modules (`stokes2vis_msv4`, `operators/gridder`, `operators/hessian`, `utils/fits`, `misc`/`beam`) keep their deferred `africanus`/`daskms`/`casacore` imports, `to_unix_time`, and `scipy.constants` — these stay as a lightweight-install / fast-startup choice. `conftest.py` is **not** changed (its `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` is orthogonal — a uv/Ray worker-venv fix, confirmed by sjperkins).

**Branch:** continue on the existing `imager` branch (this is part of PR #252). Commit after each task.

---

## Files touched

- Modify: `pyproject.toml` (bump `xarray-ms`, add explicit `arcae` pin) + regenerate `uv.lock`
- Modify: `tests/test_imager.py` (subprocess → in-process equivalence test; docstring; imports)
- Modify: `.github/workflows/ci.yml:167-193` (two test steps → one)
- Modify: `.github/workflows/publish.yml:122-146` (two test steps → one)
- Modify: `CLAUDE.md` (reframe the ⚠️ arcae ⊥ python-casacore note)
- Modify: `.claude/rules/testing-and-ci.md` (§1 isolation block; §5 ci.yml bullet)
- Modify: `.claude/rules/architecture.md` (§3 item 3; §8 note)
- Modify: `.github/copilot-instructions.md` (lazy-import bullet; ⚠️ note; Run tests block)
- Annotate (light, historical): `docs/superpowers/specs/2026-06-04-imager-datatree-design.md`, `docs/look-ahead.md`, `docs/superpowers/plans/2026-06-04-imager-datatree.md`
- **Unchanged:** `tests/conftest.py`, all `src/pfb_imaging/**` source modules.

---

## Task 1: Bump dependency and PROVE coexistence (hard gate)

**Files:**
- Modify: `pyproject.toml:53-55` (the `[project.optional-dependencies].full` MSv4 entries)
- Regenerate: `uv.lock`

> ⚠️ **This task is a gate.** Every later task assumes arcae ≥ 0.5.2 makes `pytest tests/` run without segfaulting. If Step 5 or Step 6 segfaults / crashes, **STOP** and report — the premise is invalid and the rest of the plan must not proceed.

- [ ] **Step 1: Add the arcae pin and bump xarray-ms**

In `pyproject.toml`, replace these three lines (currently at 53-55):

```toml
    "xarray-ms>=0.5.4 ; python_version >= '3.11'",
    "msv4-utils>=0.0.3 ; python_version >= '3.11'",
    "xarray-kat>=0.0.6 ; python_version >= '3.11'",
```

with (adds an explicit `arcae` pin and bumps `xarray-ms` to the release that requires `arcae>=0.5.2`):

```toml
    "arcae>=0.5.2,<0.6.0 ; python_version >= '3.11'",
    "xarray-ms>=0.5.5 ; python_version >= '3.11'",
    "msv4-utils>=0.0.3 ; python_version >= '3.11'",
    "xarray-kat>=0.0.6 ; python_version >= '3.11'",
```

- [ ] **Step 2: Regenerate the lock file**

Run: `uv lock`
Expected: completes without conflicts; `git diff uv.lock` shows `arcae` moving to `0.5.2` (and `xarray-ms` to `>=0.5.5`).

- [ ] **Step 3: Sync the full environment**

Run: `uv sync --extra full --group dev --group test`
Expected: completes; arcae 0.5.2 installed.

- [ ] **Step 4: Verify the installed arcae version**

Run: `uv run python -c "import arcae; print(arcae.__version__)"`
Expected: prints `0.5.2` (or higher within `<0.6.0`).

- [ ] **Step 5: Coexistence import smoke test (the crux)**

Import both stacks — arcae *and* the casacore-pulling legacy path — in one process:

Run:
```bash
uv run python -c "import arcae, xarray_ms; from pfb_imaging.core.init import init; from pfb_imaging.core.imager import imager; print('coexist-ok')"
```
Expected: prints `coexist-ok` and exits 0. **A segfault / non-zero exit here means coexistence does NOT hold — STOP.**

- [ ] **Step 6: Run the FULL suite as a single command**

Run:
```bash
uv run pytest -v tests/
```
Expected: the whole suite passes in one process (arcae tests in `test_imager.py` alongside the casacore tests). No segfault. This is the behaviour the guardrails existed to prevent; confirming it here is what licenses the rest of the plan.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: require arcae>=0.5.2 for python-casacore coexistence"
```

---

## Task 2: Convert the equivalence test from subprocess to in-process

**Files:**
- Modify: `tests/test_imager.py` (module docstring lines 1-7; imports lines 9-13; `test_imager_matches_init_grid_single_field` lines 124-231)

**Why:** the subprocess existed only to keep python-casacore (`init`/`grid`) out of the arcae process. With coexistence we call the core functions directly. **Critical:** `init`/`grid` call `ray.shutdown()` unless `keep_ray_alive=True`; the session `manage_ray` fixture owns the Ray cluster, so both in-process calls **must** pass `keep_ray_alive=True` (mirroring the existing `imager_core(...)` call in this test).

- [ ] **Step 1: Replace the module docstring**

Replace lines 1-7 (the current docstring) with:

```python
"""End-to-end smoke test for the MSv4 DataTree imager (.dt product).

This module loads the arcae ``xarray-ms:msv2`` engine. As of arcae 0.5.2
(ratt-ru/arcae#211, #212) arcae and python-casacore coexist in one process, so
this file runs in the same ``pytest tests/`` session as the casacore-based
tests, and the equivalence test calls the legacy ``init``+``grid`` reference
in-process (it previously ran them in a subprocess for arcae#72 isolation).
"""
```

- [ ] **Step 2: Drop the now-unused imports**

Replace lines 9-13:

```python
import glob
import os
import subprocess
import sys
from pathlib import Path
```

with (the converted test no longer shells out, so `os`/`subprocess`/`sys` are unused; `glob` is still used by `test_imager_writes_dt_tree`):

```python
import glob
from pathlib import Path
```

- [ ] **Step 3: Replace the equivalence test body**

Replace the entire `test_imager_matches_init_grid_single_field` function (lines 124-231) with:

```python
def test_imager_matches_init_grid_single_field(ms_name, tmp_path):
    """imager .dt dirty matches the legacy init+grid .dds dirty (natural weights).

    As of arcae 0.5.2 (ratt-ru/arcae#211, #212) arcae and python-casacore coexist
    in one process, so the casacore-based init+grid reference runs **in-process**
    alongside the arcae-based imager (it used to run in a subprocess purely to keep
    casacore out of the arcae process). All three calls share the session Ray
    cluster, so each passes keep_ray_alive=True to avoid tearing it down mid-suite.

    KNOWN LIMITATION (FIXME): the shared test MS downloaded from Google Drive
    currently has an all-zero DATA column, so in CI both pipelines produce an
    identically-zero dirty image and this test only verifies that the two paths
    *agree* (and run end to end) -- it is a no-op on the actual gridding maths.
    It becomes a real equivalence check only when the MS carries visibilities
    (older locally-cached copies still do, which is why it is meaningful there).
    To make it meaningful everywhere, populate DATA with a deterministic model
    (predict point sources with ducc0 ``dirty2vis`` as test_kclean/test_sara do)
    -- ideally once per session in conftest. Fully flagging one band at the same
    time would additionally exercise the band-dropping path. Until then the
    comparison is deliberately divide-by-zero-safe (see below) so a zero-signal MS
    does not masquerade as a NaN failure.
    """
    from pfb_imaging.core.grid import grid
    from pfb_imaging.core.init import init

    nx = ny = 256

    base_leg = str(tmp_path / "legacy")
    init(
        [Path(ms_name)],
        base_leg,
        channels_per_image=2,
        integrations_per_image=-1,
        product="I",
        overwrite=True,
        keep_ray_alive=True,
    )
    grid(
        base_leg,
        nx=nx,
        ny=ny,
        psf=False,
        residual=False,
        noise=False,
        beam=False,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    base_new = str(tmp_path / "imager")
    imager_core(
        [Path(ms_name)],
        base_new,
        channels_per_image=2,
        integrations_per_image=-1,
        product="I",
        nx=nx,
        ny=ny,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    leg = _mfs_dirty_from_dds(base_leg + "_I_main.dds")
    new = _mfs_dirty_from_dt(base_new + "_I.dt")
    assert new.shape == leg.shape == (nx, ny)

    # finiteness is checked per-image first so a CI failure names the offending
    # path (legacy vs imager) instead of collapsing both into a single nan diff
    assert np.isfinite(leg).all(), "legacy init+grid MFS dirty contains NaN/Inf (see diagnostics above)"
    assert np.isfinite(new).all(), "imager .dt MFS dirty contains NaN/Inf (see diagnostics above)"

    # Both images are already wsum-normalised (sum DIRTY / sum WSUM), summed over
    # all output-image nodes, so the comparison is MFS and independent of how the
    # bands are indexed (imager keeps the original band id, init+grid reindexes
    # contiguously when a band is dropped). init+grid and imager share the wsum
    # convention, so they agree to ~1e-12 on real data. Use the 1 + x shift idiom
    # (cf. test_sara) so the assertion stays well-defined when both images are
    # identically zero -- an MS with no signal in DATA, or a fully-flagged band --
    # rather than dividing by a zero peak.
    assert_allclose(1 + new, 1 + leg, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 4: Lint (catches any leftover unused import)**

Run: `uv run ruff check tests/test_imager.py`
Expected: no `F401` (unused import) errors; clean.

- [ ] **Step 5: Run the converted test**

Run: `uv run pytest -v "tests/test_imager.py::test_imager_matches_init_grid_single_field"`
Expected: PASS (runs init+grid+imager all in one process, shares session Ray).

- [ ] **Step 6: Run the whole test_imager.py module**

Run: `uv run pytest -v tests/test_imager.py`
Expected: all three tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_imager.py
git commit -m "test: run init+grid equivalence reference in-process"
```

---

## Task 3: Collapse the two CI test invocations into one

**Files:**
- Modify: `.github/workflows/ci.yml:167-193`
- Modify: `.github/workflows/publish.yml:122-146`

- [ ] **Step 1: ci.yml — replace the two test steps with one**

Replace the block at `.github/workflows/ci.yml:167-193` (the comment + the "Run tests (MSv4 / arcae)" and "Run tests (MSv2 / daskms)" steps) with:

```yaml
      - name: Run tests
        if: env.SKIP_CHECKS != 'true'
        run: |
          export RAY_NUM_CPUS=2
          export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
          export OMP_NUM_THREADS=1
          export MKL_NUM_THREADS=1
          export OPENBLAS_NUM_THREADS=1
          export NUMEXPR_NUM_THREADS=1

          uv run --frozen pytest -v tests/
```

- [ ] **Step 2: publish.yml — replace the two test steps with one**

Replace the block at `.github/workflows/publish.yml:122-146` (same comment + two steps, but **without** the `if: env.SKIP_CHECKS` guard — publish runs on tags) with:

```yaml
      - name: Run tests
        run: |
          export RAY_NUM_CPUS=2
          export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
          export OMP_NUM_THREADS=1
          export MKL_NUM_THREADS=1
          export OPENBLAS_NUM_THREADS=1
          export NUMEXPR_NUM_THREADS=1

          uv run --frozen pytest -v tests/
```

- [ ] **Step 3: Validate the YAML parses**

Run:
```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/publish.yml')); print('yaml-ok')"
```
Expected: prints `yaml-ok`.

- [ ] **Step 4: Confirm the old split is fully gone**

Run: `grep -rn "ignore=tests/test_imager\|Run tests (MSv" .github/workflows/`
Expected: no matches (empty output).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/publish.yml
git commit -m "ci: run the test suite as a single pytest invocation"
```

---

## Task 4: Reframe the active documentation

These docs are read by humans and agents. **Keep** the casacore-free import discipline guidance; **change** its rationale from "they segfault (arcae#72)" to "they coexist; the imaging path stays casacore-free by choice"; **remove** every "do not run `pytest tests/`" / two-invocation instruction.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.claude/rules/testing-and-ci.md`
- Modify: `.claude/rules/architecture.md`
- Modify: `.github/copilot-instructions.md`

- [ ] **Step 1: CLAUDE.md — reframe the ⚠️ note**

Replace the paragraph beginning `**⚠️ arcae ⊥ python-casacore:**` (in the "MSv4 DataTree imager" section) — currently:

```markdown
**⚠️ arcae ⊥ python-casacore:** the MSv4 reader (`arcae`) and python-casacore cannot share a
process (arcae#72). The imaging path is kept casacore-free — **never** add top-level
`africanus`/`daskms`/`casacore` imports to imaging-path modules (`stokes2vis_msv4`,
`operators/gridder`, `operators/hessian`, `utils/fits`, and the `misc`/`beam` helpers); defer them
into the functions that need them. Consequently **do not run `pytest tests/` as one command** — it
segfaults; `tests/test_imager.py` runs in its own pytest invocation (see
`.claude/rules/testing-and-ci.md`).
```

with:

```markdown
**arcae + python-casacore:** as of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and
python-casacore coexist in one process, so the whole suite runs as a single `pytest tests/`. The
imaging path (`stokes2vis_msv4`, `operators/gridder`, `operators/hessian`, `utils/fits`, and the
`misc`/`beam` helpers) is nonetheless kept **casacore-free by choice** — for a lightweight CLI
install and fast startup — so prefer deferring any `africanus`/`daskms`/`casacore` import into the
function that needs it rather than adding it at module scope. This is a hygiene preference now, not
a hard segfault constraint.
```

- [ ] **Step 2: .claude/rules/testing-and-ci.md — replace the §1 isolation subsection**

Replace the whole `### arcae / python-casacore test isolation (CRITICAL)` subsection (from that heading down to and including the three bullet points ending `... see .claude/rules/architecture.md §3/§8).`) with:

```markdown
### arcae / python-casacore coexistence

As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and python-casacore coexist in one
process. Run the whole suite as a single command:

```bash
uv run pytest -v tests/
```

`tests/test_imager.py` (arcae / `pfb imager`) and the casacore-based tests (which pull in
python-casacore via `construct_mappings`/daskms in `ms_meta`) therefore run together in one
pytest session, sharing the session Ray fixture. New tests need no special placement. The
imaging-path modules (`operators/gridder`, `operators/hessian`, `utils/fits`, …) remain
casacore-free by design (see `.claude/rules/architecture.md` §3/§8) — a lightweight-install
preference, not an isolation requirement.
```

- [ ] **Step 3: .claude/rules/testing-and-ci.md — fix the §5 ci.yml bullet**

Replace the `* **`ci.yml`**:` bullet (the one mentioning "two separate pytest invocations") — currently:

```markdown
* **`ci.yml`**: Code quality (ruff) and tests across Python 3.10-3.12. Tests run as two separate
  pytest invocations (`test_imager.py` alone, then the rest with `--ignore=tests/test_imager.py`)
  for the arcae/casacore isolation described in §1.
```

with:

```markdown
* **`ci.yml`**: Code quality (ruff) and tests across Python 3.11-3.13. The whole suite runs as a
  single `pytest tests/` invocation (arcae and python-casacore coexist as of arcae 0.5.2; see §1).
```

- [ ] **Step 4: .claude/rules/architecture.md — reframe §3 item 3**

Replace item 3 under "## 3. Import Style" — currently:

```markdown
3. **python-casacore-pulling imports on the MSv4 imaging path.** `africanus`, `daskms` and `casacore` transitively import python-casacore, which **cannot coexist with the `arcae` `xarray-ms` engine** used to read MSv4 data in the same process (arcae#72). Modules on the imager/gridding/FITS path must therefore keep these imports *out of module scope* — defer them into the functions that need them. Existing examples: the daskms imports in `construct_mappings` (`utils/misc.py`) and the `africanus.rime` imports in `interp_beam` (`utils/beam.py`). Use `from scipy.constants import c as lightspeed` (not `africanus.constants`) and `utils/misc.to_unix_time` (not `casacore.quanta.quantity`). See §8.
```

with:

```markdown
3. **python-casacore-pulling imports on the MSv4 imaging path.** `africanus`, `daskms` and `casacore` transitively import python-casacore. As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) this no longer clashes with the `arcae` `xarray-ms` engine in one process, but the imager/gridding/FITS path still keeps these imports *out of module scope* — deferred into the functions that need them — to keep the path lightweight (fast CLI startup, lightweight install). Existing examples: the daskms imports in `construct_mappings` (`utils/misc.py`) and the `africanus.rime` imports in `interp_beam` (`utils/beam.py`). Prefer `from scipy.constants import c as lightspeed` (not `africanus.constants`) and `utils/misc.to_unix_time` (not `casacore.quanta.quantity`). See §8.
```

- [ ] **Step 5: .claude/rules/architecture.md — reframe the §8 note**

Replace the paragraph beginning `**arcae ⊥ python-casacore (CRITICAL).**` — currently:

```markdown
**arcae ⊥ python-casacore (CRITICAL).** arcae and python-casacore cannot live in one process
(arcae#72). The whole imaging path (`stokes2vis_msv4`, `operators/gridder`, `operators/hessian`,
`utils/fits`, and the `misc`/`beam` helpers they touch) is kept casacore-free so arcae and ducc0
coexist — keep it that way (see §3 for the import discipline). This also dictates test layout
(see `.claude/rules/testing-and-ci.md`).
```

with:

```markdown
**arcae + python-casacore.** As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and
python-casacore coexist in one process, so the suite runs as a single `pytest tests/` and the
imager↔`init`+`grid` equivalence test runs both paths in-process. The whole imaging path
(`stokes2vis_msv4`, `operators/gridder`, `operators/hessian`, `utils/fits`, and the `misc`/`beam`
helpers they touch) is nonetheless kept casacore-free *by choice* (lightweight install, fast
startup) — keep it that way (see §3 for the import discipline).
```

- [ ] **Step 6: .github/copilot-instructions.md — reframe the lazy-import bullet (line ~52)**

Replace the bullet beginning `- **python-casacore-pulling imports on the MSv4 imaging path.**` — currently:

```markdown
- **python-casacore-pulling imports on the MSv4 imaging path.** `africanus`, `daskms` and `casacore` pull in python-casacore, which **cannot coexist with the `arcae` MSv4 reader in one process** (arcae#72). Modules on the imager/gridding/FITS path keep these out of module scope and import them inside the functions that need them (see `construct_mappings` in `utils/misc.py`, `interp_beam` in `utils/beam.py`). Use `scipy.constants` for `lightspeed` and `utils/misc.to_unix_time` (not `casacore.quanta`).
```

with:

```markdown
- **python-casacore-pulling imports on the MSv4 imaging path.** `africanus`, `daskms` and `casacore` pull in python-casacore. As of **arcae 0.5.2** this coexists with the `arcae` MSv4 reader in one process, but the imager/gridding/FITS path still keeps these out of module scope (deferred into the functions that need them — see `construct_mappings` in `utils/misc.py`, `interp_beam` in `utils/beam.py`) to keep the path lightweight. Prefer `scipy.constants` for `lightspeed` and `utils/misc.to_unix_time` (not `casacore.quanta`).
```

- [ ] **Step 7: .github/copilot-instructions.md — reframe the ⚠️ note (lines ~73-75)**

Replace:

```markdown
**⚠️ arcae ⊥ python-casacore (arcae#72):** the imaging path is deliberately casacore-free so arcae
and ducc0 coexist. Never add top-level `africanus`/`daskms`/`casacore` imports to imaging-path
modules. This also splits the test suite (see Run tests below).
```

with:

```markdown
**arcae + python-casacore (arcae 0.5.2+):** arcae and python-casacore now coexist in one process.
The imaging path is still kept casacore-free *by choice* (lightweight install / fast startup), so
prefer deferring `africanus`/`daskms`/`casacore` imports into functions rather than module scope.
```

- [ ] **Step 8: .github/copilot-instructions.md — fix the "Run tests" block (lines ~104-108)**

Replace:

```markdown
**Run tests** — do NOT run `pytest tests/` as one command (arcae and python-casacore can't share a process, arcae#72). Use two invocations (as CI does):
```bash
uv run pytest -v tests/test_imager.py                 # arcae only
uv run pytest -v tests/ --ignore=tests/test_imager.py # everything else (casacore)
```
```

with:

```markdown
**Run tests** — the whole suite runs as one command (arcae and python-casacore coexist as of arcae 0.5.2):
```bash
uv run pytest -v tests/
```
```

- [ ] **Step 9: Confirm no stale instructions remain in active docs**

Run:
```bash
grep -rn "ignore=tests/test_imager\|do not run .pytest tests\|cannot share a process\|in its own pytest invocation" CLAUDE.md .claude/rules/ .github/copilot-instructions.md
```
Expected: no matches (empty output).

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md .claude/rules/testing-and-ci.md .claude/rules/architecture.md .github/copilot-instructions.md
git commit -m "docs: arcae 0.5.2 coexists with casacore; drop test-split guardrails"
```

---

## Task 5: Annotate historical design docs (light touch, do not rewrite)

These are dated, point-in-time records. Add a brief "superseded" note at the stale spots so the history stays honest without being rewritten.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-04-imager-datatree-design.md`
- Modify: `docs/look-ahead.md`
- Modify: `docs/superpowers/plans/2026-06-04-imager-datatree.md`

- [ ] **Step 1: design spec — annotate the casacore-isolation claim**

In `docs/superpowers/specs/2026-06-04-imager-datatree-design.md`, immediately **before** the line starting `**python-casacore must stay off the imaging import path` insert:

```markdown
> **Superseded (2026-06-15, arcae 0.5.2):** arcae and python-casacore now coexist in one process
> (ratt-ru/arcae#211, #212), so the process-isolation rationale below no longer holds. The imaging
> path is still kept casacore-free, but now *by choice* (lightweight install), not necessity. See
> the PR #252 review responses in `docs/msv4-review-request.md`.

```

- [ ] **Step 2: design spec — annotate the "Single-field equivalence" bullet**

Replace the bullet — currently:

```markdown
- **Single-field equivalence** (`tests/test_imager.py`): `imager` MFS `DIRTY` matches `init`+`grid`
  within a peak-normalised tolerance. Because arcae and python-casacore cannot share a process
  (arcae#72), the casacore-based `init`+`grid` reference runs in a **subprocess** while `imager`
  runs in-process.
```

with:

```markdown
- **Single-field equivalence** (`tests/test_imager.py`): `imager` MFS `DIRTY` matches `init`+`grid`
  within a peak-normalised tolerance. As of arcae 0.5.2 both paths run **in-process** (the
  casacore-based `init`+`grid` reference previously ran in a subprocess for arcae#72 isolation).
```

- [ ] **Step 3: design spec — annotate the "Test isolation" note**

Replace — currently:

```markdown
**Test isolation:** `tests/test_imager.py` (arcae) runs in its own pytest invocation;
everything else runs with `--ignore=tests/test_imager.py` (see `.claude/rules/testing-and-ci.md`).
```

with:

```markdown
**Test isolation (superseded 2026-06-15):** originally `tests/test_imager.py` (arcae) ran in its
own pytest invocation, with everything else under `--ignore=tests/test_imager.py`. As of arcae
0.5.2 the suite runs as a single `pytest tests/` and the equivalence reference runs in-process
(see `.claude/rules/testing-and-ci.md`).
```

- [ ] **Step 4: look-ahead.md — update the §1 populate-step constraint**

In `docs/look-ahead.md`, replace the bullet — currently:

```markdown
- Constraint: writing the MS needs casacore (daskms) or arcae, neither of which
  may be imported in the `test_imager.py` (arcae) process (arcae#72). So either
  write via arcae once arcae writes are trusted, or run the populate step in a
  **subprocess** so the arcae process stays casacore-free.
```

with:

```markdown
- As of arcae 0.5.2 (ratt-ru/arcae#211, #212) casacore (daskms) and arcae coexist in one
  process, so the populate step can write the MS via daskms/casacore directly in the
  `test_imager.py` process — no subprocess isolation needed.
```

- [ ] **Step 5: imager-datatree plan — add a top-of-file superseded banner**

In `docs/superpowers/plans/2026-06-04-imager-datatree.md`, immediately after the H1 title line (the first `# ...` line), insert a blank line then:

```markdown
> **Note (2026-06-15):** the casacore-isolation steps in this completed plan (e.g. "run MSv4 reads
> in a process that never imports the casacore gridding path") were superseded by arcae 0.5.2,
> which lets arcae and python-casacore coexist in one process. The imaging path is still kept
> casacore-free by choice. See `docs/superpowers/plans/2026-06-15-remove-arcae-casacore-guardrails.md`.
```

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-06-04-imager-datatree-design.md docs/look-ahead.md docs/superpowers/plans/2026-06-04-imager-datatree.md
git commit -m "docs: annotate historical specs/plans superseded by arcae 0.5.2"
```

---

## Task 6: Final lint + full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Lint and format the whole repo**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: clean (no errors). If `ruff format` reflows anything, it should only be the edited `tests/test_imager.py`.

- [ ] **Step 2: Run the full suite as one command (final gate)**

Run: `uv run pytest -v tests/`
Expected: full suite PASSES in a single process — the end-state behaviour the whole change enables.

- [ ] **Step 3: Confirm conftest.py and source were left untouched**

Run: `git diff --name-only origin/main...HEAD -- tests/conftest.py 'src/pfb_imaging/**'`
Expected: empty (this change set touches neither `conftest.py` nor any source module). If non-empty, review — only the dependency/test/CI/doc files in this plan should have changed.

- [ ] **Step 4: Commit any residual lint changes**

```bash
git add -A
git commit -m "chore: ruff format after guardrail removal" || echo "nothing to commit"
```

---

## Self-Review (author checklist — completed)

**1. Spec coverage** — every approved design item maps to a task:
- Dependency bump (explicit `arcae>=0.5.2` + `xarray-ms>=0.5.5`) → Task 1.
- Validation gate (coexistence import + full suite) → Task 1 Steps 5-6 (and Task 6 Step 2).
- CI merge in **both** `ci.yml` and `publish.yml` → Task 3.
- Equivalence test subprocess → in-process → Task 2.
- Docs reframed (CLAUDE.md, testing-and-ci.md ×2, architecture.md ×2, copilot-instructions.md ×3) → Task 4.
- Historical-doc annotation (design spec, look-ahead, 2026-06-04 plan) → Task 5.
- `conftest.py` RAY env var **kept** / source import discipline **kept** → asserted in Task 6 Step 3; no task modifies them.

**2. Placeholder scan** — no TBD/TODO/"handle edge cases"; every code/doc step shows the exact old→new text or exact command + expected output.

**3. Consistency** — `keep_ray_alive=True` used consistently for all three in-process calls (Task 2); `arcae>=0.5.2,<0.6.0` / `xarray-ms>=0.5.5` identical wording in Task 1 and Task 4 rationale; `pytest -v tests/` is the single command used everywhere.

**Known risk (flagged for the executor):** the whole plan is gated on Task 1 Steps 5-6. If coexistence does not hold in our environment despite arcae 0.5.2, STOP after Task 1 and report — do not proceed to remove the guardrails.
