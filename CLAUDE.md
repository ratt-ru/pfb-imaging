# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pfb-imaging is a radio interferometric imaging suite based on the preconditioned forward-backward algorithm. It's designed for high-performance astronomical data processing with distributed computing capabilities.

## Development Environment Setup

### Installation
```bash
# Clone and install in development mode
pip install -e .

# Install with Poetry (recommended)
poetry install

# For maximum performance, install ducc in no-binary mode
git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git
pip install -e ducc
```

### Running Tests
```bash
# Run all tests
poetry run pytest -v tests/

# Run specific test file
poetry run pytest -v tests/test_beam.py

# Run tests with pattern matching
poetry run pytest -v tests/ -k "beam"
```

**Note**: First test run will automatically download test data from Google Drive, which may take some time.

## Core Architecture

### Worker-Based Processing Pipeline
The system follows a modular worker pattern where each processing step is a separate CLI command:

1. **`pfb init`** - Parse measurement sets into xarray datasets (.xds)
2. **`pfb grid`** - Create dirty images, PSFs, and weights (.dds)
3. **`pfb kclean`** - Classical deconvolution (Hogbom/Clark algorithms)
4. **`pfb sara`** - Advanced deconvolution with sparsity constraints
5. **`pfb restore`** - Restore clean components to final image
6. **`pfb degrid`** - Subtract model from visibilities

### Key Module Structure
- **`pfb/workers/`** - CLI workers (main processing tasks)
- **`pfb/operators/`** - Mathematical operators (gridding, PSF, FFT)
- **`pfb/opt/`** - Optimization algorithms (PCG, FISTA, primal-dual)
- **`pfb/deconv/`** - Deconvolution algorithms (SARA, Hogbom, Clark)
- **`pfb/utils/`** - Utility functions (FITS I/O, naming conventions)
- **`pfb/parser/`** - Configuration schemas and CLI generation
- **`pfb/wavelets/`** - Wavelet transform implementations
- **`pfb/prox/`** - Proximal operators

### Data Flow
- **Input**: Measurement Sets (MS) → **Intermediate**: Xarray datasets (.xds/.dds) → **Output**: FITS files
- **Storage**: Zarr-based distributed arrays for chunked processing
- **Processing**: Dask graphs for lazy evaluation and distributed execution

## Configuration System

### Schema-Driven Configuration
Configuration is managed through YAML schemas in `pfb/parser/`. Each worker has its own schema that automatically generates CLI parameters:

```python
# Example worker structure
@cli.command()
@clickify_parameters(schema.worker_name)
def worker_name(**kw):
    # Worker implementation
```

### Default Naming Conventions
Output files follow consistent naming patterns:
- XDS datasets: `{output_filename}_{product}.xds`
- DDS datasets: `{output_filename}_{product}_{suffix}.dds`
- Models: `{output_filename}_{product}_{suffix}_model.mds`
- FITS files: Same convention with appropriate extensions

## Key Development Patterns

### Mathematical Operators
Implement mathematical operations as callable classes:
```python
class MyOperator:
    def __init__(self, ...):
        # Setup
    
    def __call__(self, x):
        # Forward operation
        
    def adjoint(self, x):
        # Adjoint operation
```

### Distributed Computing
- Built around Dask for scalability
- Use `--nworkers` for parallel chunk processing
- Use `--nthreads-per-worker` for thread-level parallelism
- Workers handle Dask client initialization and cleanup

### Performance Optimization
- Critical loops use Numba JIT compilation
- Memory-efficient chunked processing for large datasets
- DUCC0 for high-performance gridding and FFT operations

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

### Performance Requirements
- Designed for datasets that may not fit in memory
- Supports scaling from laptop to supercomputer
- Memory management through intelligent chunking
- Optimized for both single-machine and distributed execution

### Dependencies
- Heavy reliance on scientific Python stack (NumPy, SciPy, Xarray)
- Dask for distributed computing
- JAX for automatic differentiation
- Specialized astronomy libraries (codex-africanus, dask-ms, katbeam)

## CLI Usage

### Basic Commands
```bash
# List available workers
pfb --help

# Get help for specific worker
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
pfb init --nworkers 4 --nthreads-per-worker 2

# Single worker for debugging
pfb init --nworkers 1
```

## Configuration Files

The project uses Poetry for dependency management (`pyproject.toml`) and includes comprehensive metadata for the scientific Python ecosystem. Worker configurations are defined in YAML files within `pfb/parser/` and automatically generate CLI interfaces.