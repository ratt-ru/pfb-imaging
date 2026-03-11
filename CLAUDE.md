# CLAUDE.md - Project Context for Claude Code

## Project Overview

**pfb-imaging** is a radio interferometric imaging suite based on the preconditioned forward-backward algorithm. It's designed for high-performance astronomical data processing with distributed computing capabilities. The project prioritizes **simplicity and minimalism** over feature completeness. When in doubt, consult the principles outlined in [The Twelve Factor App](https://12factor.net/) for guidance.

## Core Philosophy

### 1. Simplicity First
- Keep implementations straightforward and readable
- Avoid over-engineering solutions
- Prefer explicit over implicit behavior
- Don't add features "just in case" - wait for actual need

### 2. Lightweight Dependencies
- Minimize external dependencies
- Only add dependencies when absolutely necessary
- The lightweight install provides CLI and cab definitions only
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
git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git
pip install -e ducc
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
│   │   ├── __init__.py
│   │   ├── init.yaml
│   │   ├── grid.yaml
│   │   └── ...
│   ├── cli/                  # Lightweight CLI wrappers (Typer)
│   │   ├── __init__.py       # Main Typer app, registers commands
│   │   ├── init.py
│   │   ├── grid.py
│   │   └── ...
│   ├── core/                 # Core implementations (lazy-loaded)
│   │   ├── __init__.py
│   │   ├── init.py
│   │   ├── grid.py
│   │   └── ...
│   ├── deconv/               # Deconvolution algorithms (SARA, Hogbom, Clark)
│   ├── operators/            # Mathematical operators (gridding, PSF, FFT)
│   ├── opt/                  # Optimization algorithms (PCG, FISTA, primal-dual)
│   ├── prox/                 # Proximal operators
│   ├── utils/                # Utility functions (FITS I/O, naming, weighting)
│   └── wavelets/             # Wavelet transform implementations
├── scripts/                  # Automation scripts
│   └── generate_cabs.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.rst
```

## CLI Architecture

### Typer-Based Commands

The CLI uses Typer with hip-cargo decorators for Stimela cab generation:

```python
from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo.utils.decorators import stimela_cab, stimela_output

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

### Default Naming Conventions

Output files follow consistent naming patterns:
- XDS datasets: `{output_filename}_{product}.xds`
- DDS datasets: `{output_filename}_{product}_{suffix}.dds`
- Models: `{output_filename}_{product}_{suffix}_model.mds`
- FITS files: Same convention with appropriate extensions

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

    def __call__(self, x):
        # Forward operation

    def adjoint(self, x):
        # Adjoint operation
```

This pattern is appropriate for linear algebra operators that require both forward and adjoint computations.

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

## Distributed Computing

- Built around Dask for scalability
- Use `--nworkers` for parallel chunk processing
- Use `--nthreads` for thread-level parallelism (FFTs, gridding)
- Core functions handle Dask client initialization and cleanup

## Performance Optimization

- Critical loops use Numba JIT compilation
- Memory-efficient chunked processing for large datasets
- DUCC0 for high-performance gridding and FFT operations
- Designed for datasets that may not fit in memory

## Testing Guidelines

### Test Structure
- Tests are parametrized heavily using `pytest.mark.parametrize`
- Test data is automatically downloaded and managed via `conftest.py`
- Tests include both unit tests and full pipeline integration tests

### Test Data Management
- Test data stored in `tests/data/`
- Large datasets (MS files) downloaded automatically from Google Drive
- Session-scoped fixtures for efficient test data reuse

## Special Considerations

### Astronomy-Specific Requirements
- Requires CASA measures data for coordinate transformations
- Handles large radio astronomy datasets (measurement sets)
- Supports full Stokes polarization processing
- Integrates with radio astronomy software ecosystem

### Dependencies
- Heavy reliance on scientific Python stack (NumPy, SciPy, Xarray)
- Dask for distributed computing
- JAX for automatic differentiation
- Specialized astronomy libraries (codex-africanus, dask-ms, katbeam)

## CLI Usage

### Basic Commands

```bash
# List available commands
pfb --help

# Get help for specific command
pfb init --help

# Basic pipeline example
pfb init --ms my_data.ms --output-filename my_output
pfb grid --output-filename my_output
pfb kclean --output-filename my_output
pfb restore --output-filename my_output
```

### Parallelism Control

```bash
# Process with multiple workers
pfb init --nworkers 4 --nthreads 2

# Single worker for debugging
pfb init --nworkers 1
```

## Cab Generation

Stimela cab definitions are auto-generated from CLI functions using hip-cargo:

```bash
# Generate cabs for all CLI commands
uv run python scripts/generate_cabs.py
```

Cabs are stored in `src/pfb_imaging/cabs/` and can be included in Stimela recipes:

```yaml
_include:
  - (pfb_imaging.cabs)init.yaml
```

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
