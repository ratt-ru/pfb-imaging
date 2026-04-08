# CLAUDE.md - Project Context for Claude Code

## Project Overview

**pfb-imaging** is a radio interferometric imaging suite based on the preconditioned forward-backward algorithm. It follows the [hip-cargo](https://github.com/landmanbester/hip-cargo) package format: lightweight CLI installation with auto-generated [stimela](https://github.com/caracal-pipeline/stimela) cab definitions and containerised execution. The project prioritizes **simplicity and minimalism** over feature completeness. When in doubt, consult [The Twelve Factor App](https://12factor.net/) for guidance.

## Core Philosophy

### 1. Simplicity First
- Keep implementations straightforward and readable
- Avoid over-engineering solutions
- Prefer explicit over implicit behavior
- Don't add features "just in case" - wait for actual need

### 2. Lightweight Dependencies
- Minimize external dependencies
- Only add dependencies when absolutely necessary
- The lightweight install provides CLI and cab definitions only (sole dependency: `hip-cargo`)
- Full scientific stack is optional via `pip install pfb-imaging[full]`

### 3. Modern Python Best Practices
- Python 3.10+ features are allowed and encouraged
- Use type hints consistently
- Follow PEP 8 style (enforced by ruff)
- Prefer functional approaches over classes when possible, except for mathematical operators

## Development Environment Setup

### Installation

```bash
# Clone and install in development mode (lightweight)
git clone https://github.com/ratt-ru/pfb-imaging.git
cd pfb-imaging
uv sync

# Install with full scientific dependencies
uv sync --extra full --extra dev

# Install pre-commit hooks
uv run pre-commit install

# For maximum performance, install ducc in no-binary mode
pip install ducc0 --no-binary ducc0
```

### Running Tests

```bash
# Run all tests
uv run pytest -v tests/

# Run specific test file
uv run pytest -v tests/test_beam.py

# Run tests with pattern matching
uv run pytest -v tests/ -k "beam"
```

**Note**: First test run will automatically download test data from Google Drive.

### Code Quality

```bash
# Format code
uv run ruff format .

# Check and auto-fix linting issues
uv run ruff check . --fix
```

## Project Structure

```
pfb-imaging/
├── src/pfb_imaging/
│   ├── __init__.py
│   ├── cabs/                 # Generated Stimela cab definitions (YAML)
│   ├── cli/                  # Lightweight CLI wrappers (Typer)
│   │   └── __init__.py       # Main Typer app, registers commands
│   ├── core/                 # Core implementations (lazy-loaded from CLI)
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
└── README.rst
```

## CLI Architecture

### hip-cargo Format

The CLI uses Typer with hip-cargo decorators for Stimela cab generation:

```python
from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo import stimela_cab, stimela_output

Directory = NewType("Directory", Path)
URI = NewType("URI", Path)


@stimela_cab(
    name="init",
    info="Parse measurement sets into xarray datasets",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="xds-out",
    info="Output xarray dataset directory",
)
def init(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=Path,
            help="Path to measurement set",
        ),
    ],
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
        ),
    ],
    # ... more parameters
):
    # Lazy import the core implementation
    from pfb_imaging.core.init import init as init_core

    init_core(ms, output_filename, ...)
```

### Command Registration

Commands are registered in `src/pfb_imaging/cli/__init__.py`:

```python
import typer

app = typer.Typer(
    name="pfb",
    help="pfb-imaging: Radio interferometric imaging suite",
    no_args_is_help=True,
)

from pfb_imaging.cli.init import init
app.command(name="init")(init)
```

### Typer Option/Argument Syntax (IMPORTANT)

**NEVER** use `None` as a positional argument to `typer.Option()`:

```python
# WRONG - causes AttributeError
param: Annotated[str | None, typer.Option(None, help="...")] = None

# CORRECT
param: Annotated[str | None, typer.Option(help="...")] = None
```

**Pattern Summary:**
- Required: `Annotated[Type, typer.Option(..., help="...")]` (no `= default`)
- Optional with default: `Annotated[Type, typer.Option(help="...")] = default`
- Optional None: `Annotated[Type | None, typer.Option(help="...")] = None`

### Import Style

**Prefer top-level imports.** Put imports at the top of the file whenever possible.

Lazy (in-function) imports are only acceptable in two cases:

1. **CLI modules** (`src/pfb_imaging/cli/`): these must stay lightweight so that `pfb --help` and Stimela cab generation don't pull in the full scientific stack:
   ```python
   def my_command(...):
       """Command description."""
       from pfb_imaging.core.my_command import my_command as my_command_core
       return my_command_core(...)
   ```

2. **Optional heavy runtimes** (e.g. `ray`) in library modules that are also usable without that runtime. For example, `psi.py` defines `Psi` and `PsiNocopyt` (thread-pool, no Ray needed) alongside `PsiNocopytRay` (requires Ray). Importing `ray` at module level would force it on all users of the module, so `import ray` is deferred to `PsiNocopytRay` methods only.

Scripts, tests, and core modules should use top-level imports.

## Cab Generation and Container Workflow

### Auto-generated Cabs

Stimela cab definitions in `src/pfb_imaging/cabs/*.yml` are auto-generated from CLI functions using hip-cargo. **Never edit these files manually.**

Generation happens automatically via:
- **Pre-commit hook**: regenerates cabs on every commit
- **GitHub Action** (`update-cabs.yml`): updates cabs on push to `main` with the correct image tag
- **Manual**: `hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs`

### Image Resolution

The container image URL (including the tag) is the single source of truth and lives in `pyproject.toml` as a hip-cargo entry point:

```toml
[project.entry-points."hip.cargo"]
container-image = "ghcr.io/ratt-ru/pfb-imaging:<tag>"
```

This is read by `hip-cargo` at runtime via `importlib.metadata` (see `hip_cargo.utils.config.get_container_image`). Both cab generation and container fallback execution use this single value, so the tag portion must stay in sync with your current context.

The tag is managed by three mechanisms:

1. **Feature branches (manual):** When you create a feature branch, edit the `container-image` tag in `pyproject.toml` to match the branch name and run `uv sync` to refresh the installed package metadata. Pre-commit's `generate-cabs` hook then writes cab YAML with the correct branch tag.
2. **Merge to `main` (`update-cabs` workflow):** Resets the `container-image` tag to `latest`, runs `uv sync`, regenerates cab YAML, and commits `pyproject.toml`, `uv.lock`, and `src/pfb_imaging/cabs/*.yml`. The workflow uses `[skip checks]` in its commit messages.
3. **Releases (`tbump`):** Before-commit hooks in `tbump.toml` rewrite the `container-image` tag to the new semver, run `uv sync`, regenerate cab YAML, and stage everything as part of the version-bump commit.

CLI source files are never modified during cab generation — only the YAML cab files and `pyproject.toml` are updated.

### Container Images

Container images are built and published via GitHub Actions (`.github/workflows/publish-container.yml`). The `Dockerfile` installs the full scientific stack so stimela can run any command inside the container.

### Execution Backends

Every CLI command has a `--backend` option (provided by hip-cargo) that controls execution:

- `auto` (default): try native, fall back to container if imports fail
- `native`: run natively, fail if dependencies missing
- `docker`: run inside Docker container
- `podman`: run inside Podman container (daemonless, rootless)
- `apptainer`: run inside Apptainer container (HPC-friendly)
- `singularity`: run inside Singularity container

The `--always-pull-images` flag forces re-pulling the container image.

Both `--backend` and `--always-pull-images` are marked `{"stimela": {"skip": True}}` so they don't appear in the cab YAML — they are infrastructure parameters for local CLI execution only.

Volume mounts are resolved automatically from the function's type hints: input paths mounted read-only, output paths read-write.

## CLI Documentation

The CLI is built with Typer and provides rich, auto-generated documentation:

```bash
# List all commands
pfb --help

# Detailed parameter docs for a command
pfb init --help
```

This shows full parameter documentation with types and defaults directly in the terminal, and is generally more useful than `stimela doc` for exploring available options.

## Processing Pipeline

The system follows a modular command pattern where each processing step is a separate CLI command:

1. **`pfb init`** - Parse measurement sets into xarray datasets (.xds)
2. **`pfb grid`** - Create dirty images, PSFs, and weights (.dds)
3. **`pfb kclean`** - Classical deconvolution (Hogbom/Clark algorithms)
4. **`pfb sara`** - Advanced deconvolution with sparsity constraints
5. **`pfb restore`** - Restore clean components to final image
6. **`pfb degrid`** - Subtract model from visibilities

### Data Flow

- **Input**: Measurement Sets (MS) → **Intermediate**: Xarray datasets (.xds/.dds) → **Output**: FITS files
- **Storage**: Zarr-based distributed arrays for chunked processing
- **Processing**: Dask graphs for lazy evaluation and distributed execution

## Coding Standards

### Type Hints
- Always use type hints for function signatures
- Use `from typing import Any` for generic types
- Assume Python 3.10+; do not import from `typing_extensions` unless required

### Functions Over Classes
- Prefer pure functions where possible
- Use functions for transformations and data processing
- Only use classes when:
  - State management is truly beneficial (e.g., mathematical operators)
  - You need inheritance or polymorphism
  - The maintainer explicitly requests it

### Mathematical Operators (Exception to Functions-First Rule)

Implement mathematical operations as callable classes when they need adjoint operations:

```python
class MyOperator:
    def __init__(self, ...):
        # Setup

    def dot(self, x, out):
        # Forward operation (analysis)

    def hdot(self, x, out):
        # Adjoint operation (synthesis)
```

This pattern is appropriate for linear algebra operators that require both forward and adjoint computations. The Psi wavelet operator family (`Psi`, `PsiNocopyt`, `PsiNocopytRay`) follows this convention.

### Error Handling
- Be explicit about error cases
- Provide helpful error messages
- Let exceptions propagate unless there's a good reason to catch them
- Use `typer.Exit(code=1)` for CLI errors

### Documentation
- Use Google-style docstrings
- Document Args, Returns, and Raises
- Keep docstrings concise but informative
- Add comments when code intent isn't obvious; prefer short inline comments

## What NOT to Do

- Never delete a test unless instructed
- Don't add shallow utility functions that just wrap object methods

**Don't add these without explicit request:**
- Complex abstractions or frameworks
- Configuration systems beyond what exists
- Extensive plugin architectures
- Over-engineered validation frameworks
- Async/await unless performance demands it
- Custom metaclasses or descriptors

**Don't optimize prematurely:**
- Keep code simple first
- Only optimize if there's a measured performance problem
- Readability > performance in most cases

## What TO Do

**Simple, direct implementations:**
```python
# Good: Clear what's happening
def parse_freq_range(freq_range: str) -> tuple[float, float]:
    low, high = freq_range.split(":")
    return float(low), float(high)

# Avoid: Over-engineered
class FrequencyRangeParser:
    def __init__(self, delimiter: str = ":"):
        self._delimiter = delimiter

    def parse(self, freq_range: str) -> tuple[float, float]:
        ...
```

**Explicit over implicit:**
```python
# Good: Clear parameter parsing
scans_list = None
if scans is not None:
    scans_list = [int(x.strip()) for x in scans.split(",")]

# Avoid: Magic inference
scans_list = AutoParser.infer_and_parse(scans)
```

## Performance

### Compute Stack
- Critical loops use Numba JIT compilation with TBB threading
- DUCC0 for high-performance gridding and FFT operations
- Memory-efficient chunked processing for large datasets

### Wavelet Operators
- `Psi`: original copyt implementation with thread pool
- `PsiNocopyt`: polyphase nocopyt DWT, ~1.4x faster round-trip
- `PsiNocopytRay`: Ray actor processes for true parallelism across bands, ~1.6x faster

### Profiling
See `scripts/profiling.md` for a guide to the profiling scripts (convolution kernels, polyphase, DWT, full Psi operator).

### Distributed Computing
- Dask for parallel chunk processing (`--nworkers`)
- Thread-level parallelism for FFTs/gridding (`--nthreads`)
- Ray actors for process-level parallelism in wavelet operators

## Testing Guidelines

### Test Structure
- Tests are parametrized heavily using `pytest.mark.parametrize`
- Test data is automatically downloaded and managed via `conftest.py`
- Tests include both unit tests and full pipeline integration tests

### Test Data Management
- Test data stored in `tests/data/`
- Large datasets (MS files) downloaded automatically from Google Drive
- Session-scoped fixtures for efficient test data reuse

## CI/CD

### GitHub Actions Workflows
- **`ci.yml`**: Code quality (ruff) and tests across Python 3.10-3.12
- **`publish.yml`**: PyPI publishing on version tags
- **`publish-container.yml`**: Build and push container images to GHCR
- **`update-cabs.yml`**: Regenerate cab definitions on push to `main` (uses `landman-ci-bot` GitHub App for auth)

### Releases
Version bumping is managed with `tbump`:

```bash
tbump 0.0.9
```

This updates version strings, creates a git tag, and triggers the publish workflows.

## Questions to Ask Before Implementing

1. **Can this be done with stdlib?** → Use stdlib
2. **Can this be a simple function?** → Make it a function (unless it's an operator)
3. **Is this actually needed now?** → If no, defer
4. **Will this add dependencies?** → Probably avoid
5. **Would the maintainer call this simple?** → If no, simplify

## Maintainer Preferences

- **Background**: Senior scientific software developer
- **Experience**: Python, NumPy, Numba, JAX, Ray, Dask
- **Style**: Functional programming preferred (except for mathematical operators)
- **Philosophy**: Simple, explicit, lightweight

When in doubt, prefer the simpler solution.
