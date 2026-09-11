# CLAUDE.md - Project Context for Claude Code

## Project Overview

**pfb-imaging** is a radio interferometric imaging suite based on the preconditioned forward-backward algorithm. It follows the [hip-cargo](https://github.com/landmanbester/hip-cargo) package format: lightweight CLI installation with auto-generated [stimela](https://github.com/caracal-pipeline/stimela) cab definitions and containerised execution. The project prioritizes **simplicity and minimalism** over feature completeness. When in doubt, consult [The Twelve Factor App](https://12factor.net/) for guidance.

*Note: Detailed domain logic, Python standards, and CI/CD rules have been modularized into the `.claude/rules/` directory for progressive disclosure.*

## LLM wiki (`docs/wiki/`)

Deep internal knowledge — the deconvolution math→code primer, the design-decisions
ledger (with known debt and gotchas), and the Ray/memory discipline — lives in
`docs/wiki/` (Open Knowledge Format v0.1; start at `docs/wiki/index.md`, which says
when to read each page). Consult it before touching `deconv/`/`opt/`/`prox/`, before
"fixing" something that looks wrong (it may be a documented decision), and when
debugging memory or Ray behaviour. **Maintenance rule:** any change that invalidates a
wiki page updates the page, its `timestamp` and its `last_verified_commit` stamp in the
same session/PR.

**Specs and plans are ephemeral.** Design specs and implementation plans (the
brainstorming/planning skills write them to `docs/superpowers/specs/` and
`docs/superpowers/plans/`) are working scratch for the duration of a feature branch:
`docs/superpowers/` is gitignored and its files are never committed. Before finishing a
branch, fold any durable knowledge (decisions, rationale, gotchas, layouts) into
`docs/wiki/` — updating the affected pages per the maintenance rule — and let the spec
and plan files die with the branch. Wiki pages and rules files cite code, tests, PRs,
commits and issues as sources, never spec/plan paths.

## MSv4 DataTree imager (`pfb imager`)

`pfb imager` is the MSv4 front-end: a two-pass pipeline producing
a single unified `xarray.DataTree` (`<out>_<PRODUCT>.dt`, one node per `(band,time)` output image with a
`part####` child per data partition) plus a `.scratch` cache. It uses the **native** DataTree API
(`xr.open_datatree`, `ds.to_zarr(group=…)`, `dt.children`) — not the legacy
`xds_from_url`/`xds_from_list` helpers (those remain for the `.dds` consumers). Full detail:
`.claude/rules/architecture.md §8` and `docs/wiki/imager-pipeline.md`.

**`pfb degrid-msv4`** is the second MSv4 front-end: it degrids a `.mds` component model into
MSv4 measurement sets via `xarray-ms` write support and Ray Serve, replacing the dask-ms
`pfb degrid` (#278). All numerics go through `pfb_model_spec.utils.degrid`. Detail:
`.claude/rules/architecture.md` §6 and wiki design-decisions D38-D42.

**arcae + python-casacore:** as of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and
python-casacore coexist in one process, so the whole suite runs as a single `pytest tests/` and
`africanus`/`daskms`/`casacore` imports live at module scope like any other (the old
casacore-free-by-choice discipline was retired — wiki design-decisions D14). Imports are
top-level unless a documented `.claude/rules/architecture.md` §3 exception applies.

**Memory/performance:** the Ray+MSv4 path carries hard-won memory discipline (selective variable
loads, `gc.collect()` at Ray-task boundaries, xarray-ms table-cache eviction, per-task RSS
telemetry in the progress lines). Before touching pass 1/2 or debugging footprint, read
`docs/wiki/memory-and-ray.md` and `.claude/rules/architecture.md` §8 — do not regress these.

## Core Dependencies

* Minimize external dependencies.
* The lightweight install provides CLI and cab definitions only (sole dependency: `hip-cargo`).
* Full scientific stack is optional via `pip install pfb-imaging[full]`.
* Development uses a single `dev` dependency group; the scientific stack stays behind the
  `full` extra (`uv sync --extra full --group dev`). Keeping `pfb-imaging[full]` out of `dev`
  is deliberate — see `.claude/rules/testing-and-ci.md` §1.

## Mandatory Development Workflow

**Always run linting after adding or modifying any code:**

```bash
uv run ruff format . && uv run ruff check . --fix
```

**Tests are fast by default.** `pyproject.toml`'s `addopts` carries `-m "not slow"`, so
`uv run pytest tests/` runs 685 tests in ~170 s. The 41 deselected tests are the end-to-end
pipeline ones (`*_groundtruth`, the imager/deconv/restore/hci and degrid-msv4 drivers, and the
degrid parity/null tests) plus a few whose cost is Ray actor startup or a dense operator build
(the `_build_hess` preset-wiring pair, the frequency-prior fixed-point guard and the
frequency-prior spectrum guards). They are left to CI, which overrides with `-m ""`. Use
`-m ""` locally before finishing a branch — not per-task during development. A test earns the
`slow` marker when its *cheapest* parametrisation costs ≥2 s — see `.claude/rules/testing-and-ci.md` §1
for why "cheapest" matters and why chasing warm-up spikes is whack-a-mole.

## Working Effectively (notes for agents)

Lessons from real debugging sessions in this repo — follow them; they are cheaper than
rediscovery:

* **Profiling loop:** the maintainer drops stimela logs into `tmp/` for comparison (e.g.
  `tmp/logs_dirty_and_init` as the legacy baseline vs `tmp/logs_imager*`). Start with
  `stimela.stats.summary.txt` (wall / CPU% / peak-mem / total-I/O per step), then the per-step
  logs. The imager progress lines carry per-task post-gc RSS telemetry — read it before
  theorising about memory (interpretation guide: `docs/wiki/memory-and-ray.md`).
* **Reproduce locally before proposing fixes:** cluster-scale Ray/xarray behaviour
  (serialisation, lazy loading, retention) reproduces on `tests/data/test_ascii_1h60.0s.MS`
  with a pickle-roundtrip harness — pickling a datatree node is exactly what Ray does to task
  args — plus `psutil` RSS sampling and a manual `gc.collect()` to separate cycle retention
  from below-Python retention.
* **Quantify before ranking hypotheses:** estimate per-task bytes from array shapes/dtypes.
  A hypothesis that does not add up to the measured GB is incomplete — the residual usually
  lives below Python (library caches, allocator arenas) where `gc` and object counting are
  blind.
* **Beware warm-cache timing:** back-to-back runs on the same MS read from page cache (the
  stats `R GB` column shows ~0); only compare wall times at matching cache state.
* **One mechanism per commit,** with the measured before/after in the commit message. It keeps
  cluster-run bisection possible when a change must be re-litigated.
* **`gh issue view` and `gh pr edit` are broken on this machine — use `gh api` instead.** The
  system `gh` is Ubuntu's 2.46.0, which asks GraphQL for the Projects-classic `projectCards`
  field; the API has hard-errored on that since the May 2024 sunset, so both commands die with
  `GraphQL: Projects (classic) is being deprecated … (repository.pullRequest.projectCards)`.
  Upstream feature-detects v1 projects from **2.71.0** (`issue view`) and **2.73.0** (`pr edit`),
  but the machine deliberately stays on the distro package, so treat this as permanent. It is
  the client, not the repo — it reproduces against `cli/cli`. Reach for REST:
  `gh api repos/ratt-ru/pfb-imaging/issues/<n> --jq '{title,state,body}'` to read an issue, and
  `gh api -X PATCH repos/ratt-ru/pfb-imaging/pulls/<n> --input body.json` (a `{"body": …}` file,
  so markdown survives shell quoting) to edit a PR. `gh pr view`/`list`/`checks`/`create`/`merge`
  and every `gh api` call are unaffected, as is CI — both workflow uses are already `gh api`.

## Project Structure

```
pfb-imaging/
├── src/pfb_imaging/
│   ├── __init__.py
│   ├── _container_image.py   # Container image URL (single source of truth)
│   ├── cabs/                 # Generated cab definitions (YAML)
│   ├── cli/                  # Lightweight CLI wrappers
│   │   └── __init__.py       # Main Typer app, registers commands
│   ├── core/                 # Core implementations (lazy-loaded)
│   ├── deconv/               # Composable deconvolution (PFBSolver + presets registry)
│   ├── operators/            # Mathematical operators (gridding, PSF, Psi)
│   ├── opt/                  # Optimization algorithms (PCG, FISTA, primal-dual)
│   ├── prox/                 # Proximal operators
│   ├── utils/                # Utility functions (FITS I/O, naming, weighting)
│   └── wavelets/             # Wavelet transform implementations
├── scripts/                  # Profiling and automation scripts
├── tests/
├── Dockerfile
├── pyproject.toml
├── tbump.toml
├── .pre-commit-config.yaml
└── README.md
```
