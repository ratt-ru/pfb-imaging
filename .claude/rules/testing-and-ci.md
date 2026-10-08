# Testing & CI/CD Guidelines

Read this when editing `tests/**/*.py` or `.github/workflows/*.yml` files.

## 1. Test Infrastructure

* Tests are parametrized with `pytest.mark.parametrize`.
* Test data in `tests/data/` is downloaded automatically from Google Drive on first run,
  **unconditionally** — the MS is read through arcae, so the casacore-free legs need it too.
* Session-scoped fixtures in `conftest.py` for efficient data reuse.

### What skips without the `[casacore]` extra

`ms_name` is only a path, and the test MS is *downloaded*, not built — `imager` and `degrid`
read it through arcae. So **`ms_name` does not require the extra**, and MSv2-backed tests run
on a `[full]`-only install. It used to `importorskip("daskms")`, which cascaded a skip to all
77 of them; that made the aarch64 `--extra full` leg — the one we gate on — skip every test
that grids a visibility, so a green arm tick said nothing about whether imaging worked there
(#330).

What skips now is only what genuinely needs the extra, and each declares it for itself:

| | needs | why |
|---|---|---|
| `ms_meta`, and `sky_truth` through it | dask-ms | reads the MS with `xds_from_ms` |
| the `pctable` fixture | python-casacore | casacore's *write* API — `addrows`, `putcell` on subtables, `removecols` — which arcae does not expose |
| `drop_column`, `make_multi_spw_ms` | python-casacore | same |
| `test_stokes_vis_beam_on_image_grid` | pyrap | rephasing calls africanus' `synthesize_uvw` |
| `test_hci.py`, `test_imager_pol.py` | dask-ms | module-scope imports, so `collect_ignore` rather than a skip (pytest reports a collection-time ImportError as an error, not a skip) |

A test needing casacore's write API takes the `pctable` fixture rather than importing it —
one reason string, one place to change.

`tests/test_naming.py` used to carry the same cascade at module scope -- one
`importorskip("daskms.fsspec_store")` skipped all 13 of its tests although only the two
`DaskMSStore`-equivalence ones needed dask-ms. Those two were deleted and the skip
removed, so its 11 remaining tests (pure `fsspec`) run on every leg.

**Shared session fixtures (D44).** `band_pool` hands out one `BandWorkerPool` per
`(nband, nthreads)`; `gt_dt`/`gt_deconv_dt` image and deconvolve the ground-truth sky once;
`sky_truth` is session-scoped; `manage_ray` runs `ray.init(num_cpus=2)`. The rules that come
with that, each established by breaking it:

* Two `HessTreeRay` facades on one pool **must** share their `init_hess` args or be used
  strictly sequentially (`freq_prec` is driver-side and safe to differ).
* **`keep_ray_alive=True` is mandatory** in any test calling `imager_core`/`hci_core` — without
  it the driver calls `ray.shutdown()` and tears down the session cluster and every actor cached
  by `band_pool` (which rebuilds its cache if Ray is down, but without the session's
  `num_cpus`/`runtime_env`).
* **Do not convert `nband == 1` call sites** — that branch runs in-process and never imports Ray.
* **Do not make `conftest.manage_ray` opt-in** to save its ~4.5 s; the same reasoning rules out
  `pytest-xdist`.
* **Any test that writes to the session MS takes the `writable_ms` fixture** (a function-scoped
  copy) instead of `ms_name`, and uses it for *every* access in that test — including writes
  through dask-ms `xds_to_table`, which a grep for `putcol` will not find. A forgotten writer
  passes in the fast loop and the slow loop separately and fails only under `-m ""`.

`tests/test_band_pool.py` pins the reuse equivalence. Why each of these holds: **D44**.

### Dependency groups: one `dev` group, `full` is the heavy axis

There is a single `dev` dependency group (ruff, pre-commit, pytest, tbump, stimela — the
former `test` group was folded into it; nothing ever installed one without the other).
The distinction that *is* load-bearing is the `full` extra:

```bash
uv sync --group dev                 # lint/cab tooling only — the Code Quality job
uv sync --extra all --group dev    # + the scientific stack — tests, and local work
```

**Never put `pfb-imaging[full]` into the `dev` group.** `dev` is a uv default group, so
doing so drags ray/ducc0/jax/dask-ms/africanus into `uv sync --group dev` — i.e. into the
Code Quality job, whose entire body is `ruff format --check .` and `ruff check .`, and into
`update-cabs`, which only needs hip-cargo. Jobs that need the stack name `--extra all`
explicitly.

### arcae / python-casacore coexistence

As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and python-casacore coexist in one
process, so there is one pytest session — no special placement for new tests, and no
casacore-related import restrictions (the historical casacore-free discipline was retired —
wiki design-decisions D14).

### Fast by default — and the fast loop is the only loop you run

`pyproject.toml`'s `addopts` carries `-m "not slow"`, so the bare command is the loop:

```bash
uv run pytest tests/          # 840 tests, ~115 s -- run THIS
```

(One of those skips without the `[casacore]` extra; the rest pass. Counts here are *collected*
counts, which `scripts/check_docs.py` measures and checks — pass/skip totals are not pinned
because they depend on which extras are installed.)

**Never run the slow set unless you changed a shared fixture.** `uv run pytest -m slow tests/`
runs the 49 deselected ones and costs ~465 s; `uv run pytest -m "" tests/` costs ~9.6 min. CI
already runs the fast set on every push across six legs — x86_64 3.11/3.12/3.13 and aarch64,
each with `--extra all` and `--extra full` — and the slow set separately (see below).
Reproducing one leg locally costs more than a push and covers less. Run fast, push, read the
result.

**The one exception is a change to `conftest.py`'s session fixtures.** Slow tests consume them
most heavily and are the least likely to have been re-run: a fixture-basename change reached
final review in #336 with two broken assertions (`test_deconv.py`, `test_imager.py`) that only
a full run caught. Change a session fixture, run the slow set once. Otherwise, don't.

### The CI split: fast on every push, slow on demand (#338)

A command-line `-m` overrides the one in `addopts` (pytest keeps a single value, last wins).
Which workflow passes what is the whole design:

| workflow | selection | when |
|---|---|---|
| `ci.yml` | the fast set (`addopts`, no override) | every push and PR, six legs |
| `acceptance.yml` | `-m slow` | push to `main`; `/test-acceptance` on a PR; `workflow_dispatch` |
| `publish.yml` | `-m ""` | version tags — a release is gated on the whole suite |

The slow set is 49 of the 889 collected tests and ~80% of the suite's wall time, so running it
on all six legs of every push was the bulk of the repo's CI bill. `ci.yml` still runs
`pytest -m slow --collect-only` as a guard, so a broken marker or `-m` override cannot silently
empty `acceptance.yml` (pytest exits 5 when a selection collects nothing) — that guard matters
more now that no `ci.yml` leg collects a slow test. `addopts` carries `--durations=10` so a
newly-slow test surfaces in the fast loop's own output.

**A green PR no longer means the pipeline still images a sky.** Comment `/test-acceptance` on
any PR that touches the imager, deconv, degrid or gridding paths — it is authorized to
OWNER/MEMBER/COLLABORATOR, reports a commit status named `acceptance`, and replies on the PR
with the result. `acceptance.yml`'s install steps are a subset of `ci.yml`'s and must stay in
step with them.

**Marking rule: a test is `slow` when its _cheapest_ parametrisation costs ≥ 2 s.** The
"cheapest" qualifier is load-bearing and the reasoning is **D45** — marking a warm-up carrier
evicts the test without removing the time. Do not chase warm-up spikes; it is whack-a-mole.

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

* **`ci.yml`**: Code quality (ruff) and the fast test set across Python 3.11-3.13, as a single
  `pytest tests/` invocation (arcae and python-casacore coexist as of arcae 0.5.2; see §1).
* **`acceptance.yml`**: the slow set (`-m slow`) on x86_64 and aarch64. Triggers and the
  reasoning: §1's CI-split table.
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
