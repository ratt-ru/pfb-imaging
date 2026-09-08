# Testing & CI/CD Guidelines

Read this when editing `tests/**/*.py` or `.github/workflows/*.yml` files.

## 1. Test Infrastructure

* Tests are parametrized with `pytest.mark.parametrize`.
* Test data in `tests/data/` is downloaded automatically from Google Drive on first run.
* Session-scoped fixtures in `conftest.py` for efficient data reuse.

### Dependency groups: one `dev` group, `full` is the heavy axis

There is a single `dev` dependency group (ruff, pre-commit, pytest, tbump, stimela — the
former `test` group was folded into it; nothing ever installed one without the other).
The distinction that *is* load-bearing is the `full` extra:

```bash
uv sync --group dev                 # lint/cab tooling only — the Code Quality job
uv sync --extra full --group dev    # + the scientific stack — tests, and local work
```

**Never put `pfb-imaging[full]` into the `dev` group.** `dev` is a uv default group, so
doing so drags ray/ducc0/jax/dask-ms/africanus into `uv sync --group dev` — i.e. into the
Code Quality job, whose entire body is `ruff format --check .` and `ruff check .`, and into
`update-cabs`, which only needs hip-cargo. Jobs that need the stack name `--extra full`
explicitly.

### arcae / python-casacore coexistence

As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and python-casacore coexist in one
process, so there is one pytest session — no special placement for new tests, and no
casacore-related import restrictions (the historical casacore-free discipline was retired —
wiki design-decisions D14).

### Fast by default, slow in CI

`pyproject.toml`'s `addopts` carries `-m "not slow"`, so the bare command is the fast loop:

```bash
uv run pytest tests/          # fast loop: 654 tests, ~151 s
uv run pytest -m slow tests/  # only the deselected 35, ~450 s
uv run pytest -m "" tests/    # everything, ~570 s (what CI runs)
```

A command-line `-m` overrides the one in `addopts` (pytest keeps a single value, last wins).
Both `ci.yml` and `publish.yml` therefore pass `-m ""` — a release must be gated on the whole
suite. `ci.yml` also runs `pytest -m slow --collect-only` as a guard, so a broken override
cannot silently drop the slow set everywhere at once (pytest exits 5 when a selection collects
nothing).

**Marking rule: a test is `slow` when its _cheapest_ parametrisation costs ≥2 s.** The
"cheapest" qualifier is load-bearing. Several functions look expensive but are only carrying a
one-off warm-up — JIT, beam-model load, Ray spin-up — attributed to whichever param ran first
(`test_beam`: 4.40 s then 11 × 0.00 s; `test_psi`: 4.15 s then 23 × ~0.01 s). That cost is
sticky to the *run*, not the test: marking such a function evicts its tests without removing
the time, which simply reattaches to whatever runs next. Confirmed in practice — deselecting
`test_hci_channels_per_bin_invariance_no_beam` pushed the hci warm-up onto
`test_hci_produces_expected_output_structure`, which went from cheap to 10.19 s. Do not chase
those; it is whack-a-mole.

`addopts` also carries `--durations=10` so a genuinely newly-slow test surfaces in the fast
loop's own output instead of quietly rotting there.

**Do not make `conftest.manage_ray` opt-in to save its ~4.5 s.** No test calls Ray directly;
that autouse fixture pre-seeds `ray.init` with a specific `runtime_env`
(`worker_process_setup_hook`, and the `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` workaround `conftest.py`
documents as a hang-cause), which the source-level `ray.init(ignore_reinit_error=True)` in
`src/pfb_imaging/__init__.py` then attaches to. Opt-in means a test that needs Ray but forgets
to request the fixture silently gets a differently-configured cluster, or hangs. The same
reasoning rules out `pytest-xdist`: each worker would stand up its own Ray cluster.

## 2. Commit Messages

* Use [Conventional Commits](https://www.conventionalcommits.org/) format: `<type>: <description>`
* Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `ci`, `deps`, `chore`
* Keep the first line under 72 characters. Use imperative mood.
* Optional scope: `feat(init): add support for new data column`
* Changelog is auto-generated from these prefixes via git-cliff.

## 3. Mandatory Linting

Always run linting after adding or modifying any code:
`uv run ruff format . && uv run ruff check . --fix`

### 3.1 Cab Sync Before Committing CLI Changes

The pre-commit `generate-cabs` hook needs `hip-cargo` on PATH and may fail in
environments where it isn't. Regenerate manually and confirm a clean diff instead:
`uv run hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs`
(a stale cab fails CI via `tests/test_roundtrip.py`; see python-standards §2.1 for the
help-text formatting constraints that test imposes).

## 4. CI Workflow and `[skip checks]`

The CI pipeline uses a custom `[skip checks]` tag (not GitHub's `[skip ci]`).
* The `update-cabs` workflow commits with `[skip checks]` after regenerating cab definitions on merge to main.
* Each CI job checks the commit message via `gh api` and sets `SKIP_CHECKS=true` to skip heavy steps while still reporting success for branch protection.

## 5. GitHub Actions Workflows

* **`ci.yml`**: Code quality (ruff) and tests across Python 3.11-3.13. The whole suite runs as a
  single `pytest tests/` invocation (arcae and python-casacore coexist as of arcae 0.5.2; see §1).
* **`publish.yml`**: PyPI publishing on version tags. Runs quality + tests before publishing.
* **`publish-container.yml`**: Build and push container images to GHCR.
* **`update-cabs.yml`**: Regenerate cab definitions on push to `main`. Uses `landman-ci-bot` GitHub App for auth.

## 6. Releases

```bash
tbump <new_version>
```

This generates a changelog (git-cliff), updates version strings and `_container_image.py` tag, regenerates cabs, creates a git tag, and triggers publish workflows.

## 7. Contributing Workflow

1. Create a feature branch: `git checkout -b your-feature-name`
2. Update `CONTAINER_IMAGE` tag in `src/pfb_imaging/_container_image.py` to match your branch name.
3. Make changes and ensure tests pass.
4. Commit using conventional commit messages.
5. Push and create a pull request.

The `update-cabs` workflow resets the tag to `latest` automatically on merge to `main`.
