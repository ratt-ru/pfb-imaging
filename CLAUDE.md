# CLAUDE.md - Project Context for Claude Code

## Project Overview

**pfb-imaging** is a radio interferometric imaging suite based on the preconditioned forward-backward algorithm. It follows the [hip-cargo](https://github.com/landmanbester/hip-cargo) package format: lightweight CLI installation with auto-generated [stimela](https://github.com/caracal-pipeline/stimela) cab definitions and containerised execution. The project prioritizes **simplicity and minimalism** over feature completeness. When in doubt, consult [The Twelve Factor App](https://12factor.net/) for guidance.

*Note: Detailed domain logic, Python standards, and CI/CD rules have been modularized into the `.claude/rules/` directory for progressive disclosure.*

## MSv4 DataTree imager (`pfb imager`)

`pfb imager` is the MSv4 front-end that combines `init`+`grid` into a two-pass pipeline producing
a single unified `xarray.DataTree` (`<out>_<P>.dt`, one node per `(band,time)` output image with a
`part####` child per data partition) plus a `.scratch` cache. It uses the **native** DataTree API
(`xr.open_datatree`, `ds.to_zarr(group=…)`, `dt.children`) — not the legacy
`xds_from_url`/`xds_from_list` helpers (those remain for the `.dds` consumers). Full detail:
`.claude/rules/architecture.md §8` and `docs/superpowers/specs/2026-06-04-imager-datatree-design.md`.

**⚠️ arcae ⊥ python-casacore:** the MSv4 reader (`arcae`) and python-casacore cannot share a
process (arcae#72). The imaging path is kept casacore-free — **never** add top-level
`africanus`/`daskms`/`casacore` imports to imaging-path modules (`stokes2vis_msv4`,
`operators/gridder`, `operators/hessian`, `utils/fits`, and the `misc`/`beam` helpers); defer them
into the functions that need them. Consequently **do not run `pytest tests/` as one command** — it
segfaults; `tests/test_imager.py` runs in its own pytest invocation (see
`.claude/rules/testing-and-ci.md`).

## Core Dependencies

* Minimize external dependencies.
* The lightweight install provides CLI and cab definitions only (sole dependency: `hip-cargo`).
* Full scientific stack is optional via `pip install pfb-imaging[full]`.

## Mandatory Development Workflow

**Always run linting after adding or modifying any code:**

```bash
uv run ruff format . && uv run ruff check . --fix
```

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
