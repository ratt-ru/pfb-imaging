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

`pfb imager` is the MSv4 front-end that combines `init`+`grid` into a two-pass pipeline producing
a single unified `xarray.DataTree` (`<out>_<P>.dt`, one node per `(band,time)` output image with a
`part####` child per data partition) plus a `.scratch` cache. It uses the **native** DataTree API
(`xr.open_datatree`, `ds.to_zarr(group=…)`, `dt.children`) — not the legacy
`xds_from_url`/`xds_from_list` helpers (those remain for the `.dds` consumers). Full detail:
`.claude/rules/architecture.md §8` and `docs/wiki/imager-pipeline.md`.

**arcae + python-casacore:** as of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and
python-casacore coexist in one process, so the whole suite runs as a single `pytest tests/`. The
imaging path (`stokes2vis_msv4`, `operators/gridder`, `operators/hessian`, `utils/fits`, and the
`misc`/`beam` helpers) is nonetheless kept **casacore-free by choice** — for a lightweight CLI
install and fast startup — so prefer deferring any `africanus`/`daskms`/`casacore` import into the
function that needs it rather than adding it at module scope. This is a hygiene preference now, not
a hard segfault constraint.

**Memory/performance:** the Ray+MSv4 path carries hard-won memory discipline (selective variable
loads, `gc.collect()` at Ray-task boundaries, xarray-ms table-cache eviction, per-task RSS
telemetry in the progress lines). Before touching pass 1/2 or debugging footprint, read
`docs/wiki/memory-and-ray.md` and `.claude/rules/architecture.md` §8 — do not regress these.

## Core Dependencies

* Minimize external dependencies.
* The lightweight install provides CLI and cab definitions only (sole dependency: `hip-cargo`).
* Full scientific stack is optional via `pip install pfb-imaging[full]`.

## Mandatory Development Workflow

**Always run linting after adding or modifying any code:**

```bash
uv run ruff format . && uv run ruff check . --fix
```

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
│   ├── deconv/               # Deconvolution algorithms (SARA, Hogbom, Clark)
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
