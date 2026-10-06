# CLAUDE.md - Project Context for Claude Code

## Project Overview

**pfb-imaging** is a radio interferometric imaging suite based on the preconditioned forward-backward algorithm. It follows the [hip-cargo](https://github.com/landmanbester/hip-cargo) package format: lightweight CLI installation with auto-generated [stimela](https://github.com/caracal-pipeline/stimela) cab definitions and containerised execution. The project prioritizes **simplicity and minimalism** over feature completeness. When in doubt, consult [The Twelve Factor App](https://12factor.net/) for guidance.

*Note: Detailed domain logic, Python standards, and CI/CD rules have been modularized into the `.claude/rules/` directory for progressive disclosure.*

## LLM wiki (`docs/wiki/`)

Deep internal knowledge that is expensive to re-derive from source. Open Knowledge Format
v0.1; start at `docs/wiki/index.md`, which says when to read each page.

**Maintenance rule:** any change that invalidates a wiki page updates the page, its
`timestamp` and its `last_verified_commit`, in the same session/PR. A stamp asserts the page
was verified against that commit, so do not restamp a page you did not read.

**Specs and plans are ephemeral.** `docs/superpowers/` is gitignored and never committed.
Before finishing a branch, fold durable knowledge into `docs/wiki/` per the maintenance rule
and let the spec and plan die with the branch. Wiki pages and rules files cite code, tests,
PRs, commits and issues as sources — **never spec/plan paths.**

**Upstream MSv4 issues live in `docs/msv4_issues.md`** — every arcae / xarray-ms bug we hit,
filed or not, with a runnable reproducer in `scripts/msv4_issues/`. Add to it rather than
rediscovering.

## Where things are documented

Read the rules file for the files you are editing, and the wiki page for the thing you are
reasoning about. The rules say what you must do; the wiki says why.

| working on | read |
|---|---|
| `src/**/*.py` | `.claude/rules/python-standards.md`, `.claude/rules/architecture.md` |
| `tests/**`, `.github/workflows/**` | `.claude/rules/testing-and-ci.md` |
| `deconv/`, `opt/`, `prox/` | `docs/wiki/deconv-primer.md`, then `docs/wiki/design-decisions.md` |
| the imager or degrid pipeline | `docs/wiki/imager-pipeline.md` |
| memory or Ray behaviour | `docs/wiki/memory-and-ray.md` |
| image or beam orientation | `docs/wiki/image-and-beam-orientation.md` |
| something that looks wrong | `docs/wiki/design-decisions.md` — it may be a documented decision |
| an odd MSv4 write-path failure | `docs/msv4_issues.md` |

**The front-ends.** `pfb imager` (MSv4 -> `.dt`), `pfb deconv` (`.dt` -> model),
`pfb restore` (`.dt` -> restored FITS) and `pfb degrid` (`.mds` -> MSv4), plus `pfb hci` for
high-cadence imaging. `.claude/rules/architecture.md` §6 enumerates the pipeline and §8 the
imager's invariants; `docs/wiki/imager-pipeline.md` has the tree layout.

**Two things that bite hardest, so they are stated here as well as there:** never
blanket-`.load()` an MSv4 node (it reads every correlated-data column), and `gc.collect()` at
every Ray-task boundary. Both are in `.claude/rules/architecture.md` §8 with the rest of the
memory discipline, and `docs/wiki/memory-and-ray.md` has the measured story.

## Core Dependencies

* Minimize external dependencies. The lightweight install provides CLI and cab definitions
  only (sole dependency: `hip-cargo`); the scientific stack is the optional `[full]` extra.
* One `dev` group, with the scientific stack behind extras. Local work and tests want
  **`uv sync --extra all --group dev`** (what CI installs); `--group dev` alone is lint/cab
  tooling only. **Never put `pfb-imaging[full]` into `dev`** —
  `.claude/rules/testing-and-ci.md` §1 says why.

## Mandatory Development Workflow

**Always run linting after adding or modifying any code:**

```bash
uv run ruff format . && uv run ruff check . --fix
```

**Tests are fast by default, and the fast loop is the *only* loop you run locally:**

```bash
uv run pytest tests/
```

**Never run the slow set (`-m slow`) or the full suite (`-m ""`) locally** — the fast set runs
on every push across six legs, which is broader coverage than one local run can give, and the
slow set runs in `acceptance.yml` (on demand via a `/test-acceptance` PR comment, and on every
push to `main`). The single exception is a change to `conftest.py`'s session fixtures. Counts,
timings, the marking rule and the CI split: `.claude/rules/testing-and-ci.md` §1.

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
* **`gh issue view` and `gh pr edit` are permanently broken on this machine — use `gh api`.**
  The distro `gh` (2.46.0) asks GraphQL for the sunset Projects-classic `projectCards` field;
  upstream feature-detects from 2.71.0/2.73.0 but the machine stays on the distro package. Use
  `gh api repos/ratt-ru/pfb-imaging/issues/<n> --jq '{title,state,body}'` to read an issue and
  `gh api -X PATCH repos/ratt-ru/pfb-imaging/pulls/<n> --input body.json` to edit a PR (a file,
  so markdown survives shell quoting). `gh pr view`/`list`/`checks`/`create`/`merge` and every
  `gh api` call are unaffected, as is CI.

## Project Structure

```
src/pfb_imaging/
├── _container_image.py   # container image URL (single source of truth)
├── cabs/                 # generated cab definitions (YAML) -- never edit by hand
├── cli/                  # lightweight Typer wrappers; lazy-import core
├── core/                 # command implementations (imager, deconv, restore, degrid, hci)
├── deconv/               # composable deconvolution (PFBSolver + presets registry)
├── operators/            # gridding, PSF/Hessian, Psi, band workers
├── opt/                  # PCG and primal-dual
├── prox/                 # proximal operators
├── utils/                # FITS I/O, naming, weighting, MSv4 seams
└── wavelets/             # wavelet transforms
```

`scripts/` holds profiling and automation helpers, plus `scripts/msv4_issues/` reproducers
and `scripts/check_docs.py`, which gates this file and the wiki against each other.
