# Contributing to PFB-Imaging

Thank you for your interest in contributing to PFB-Imaging! This guide will help you get started with development and contributing to the project.

## Getting Started

### Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pfb-imaging.git
   cd pfb-imaging
   ```

2. **Install in development mode**:
   ```bash
   poetry install --with docs
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Dependencies

Install additional development tools:

```bash
poetry install --with dev,docs,test
```

This includes:
- Testing frameworks (pytest, pytest-cov)
- Code formatting (black, isort)
- Linting (flake8, mypy)
- Documentation tools (mkdocs, mkdocs-material)

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Follow the coding standards and patterns described below.

### 3. Run Tests

```bash
# Run all tests
poetry run pytest -v

# Run specific test file
poetry run pytest -v tests/test_beam.py

# Run with coverage
poetry run pytest --cov=pfb tests/
```

### 4. Format and Lint Code

```bash
# Format code
poetry run black pfb/ tests/
poetry run isort pfb/ tests/

# Lint code
poetry run flake8 pfb/ tests/
poetry run mypy pfb/
```

### 5. Submit Pull Request

1. Push your changes to your fork
2. Create a pull request against the main repository
3. Fill out the pull request template
4. Wait for review and address feedback

## Coding Standards

### Python Style

- **PEP 8**: Follow Python style guidelines
- **Type hints**: Use type annotations for all functions
- **Docstrings**: Use NumPy-style docstrings
- **Black**: Use Black for code formatting
- **isort**: Use isort for import sorting

### Example Function

```python
import numpy as np
from typing import Tuple, Optional

def gridding_operator(
    visibilities: np.ndarray,
    uvw: np.ndarray,
    weights: np.ndarray,
    nx: int,
    ny: int,
    cell_size: float,
    epsilon: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grid visibilities onto a regular grid.

    Parameters
    ----------
    visibilities : np.ndarray
        Complex visibility data with shape (nvis,)
    uvw : np.ndarray
        UVW coordinates with shape (nvis, 3)
    weights : np.ndarray
        Visibility weights with shape (nvis,)
    nx : int
        Number of pixels in x direction
    ny : int
        Number of pixels in y direction
    cell_size : float
        Cell size in arcseconds
    epsilon : float, optional
        Gridding accuracy, by default 1e-7

    Returns
    -------
    grid : np.ndarray
        Gridded visibilities with shape (nx, ny)
    weights_grid : np.ndarray
        Gridded weights with shape (nx, ny)

    Examples
    --------
    >>> vis = np.random.complex128(1000)
    >>> uvw = np.random.random((1000, 3))
    >>> weights = np.ones(1000)
    >>> grid, wgrid = gridding_operator(vis, uvw, weights, 256, 256, 2.0)
    """
    # Implementation here
    pass
```

### Mathematical Operators

Implement operators as callable classes:

```python
class GriddingOperator:
    """Gridding operator for radio interferometry."""
    
    def __init__(self, uvw: np.ndarray, nx: int, ny: int, cell_size: float):
        """
        Initialize gridding operator.
        
        Parameters
        ----------
        uvw : np.ndarray
            UVW coordinates
        nx, ny : int
            Image dimensions
        cell_size : float
            Cell size in arcseconds
        """
        self.uvw = uvw
        self.nx = nx
        self.ny = ny
        self.cell_size = cell_size
        self._setup_kernel()
    
    def __call__(self, visibilities: np.ndarray) -> np.ndarray:
        """Apply gridding operator (forward operation)."""
        return self._grid(visibilities)
    
    def adjoint(self, image: np.ndarray) -> np.ndarray:
        """Apply adjoint gridding operator (degridding)."""
        return self._degrid(image)
    
    def _setup_kernel(self) -> None:
        """Setup gridding kernel."""
        # Implementation
        pass
    
    def _grid(self, visibilities: np.ndarray) -> np.ndarray:
        """Grid visibilities."""
        # Implementation
        pass
    
    def _degrid(self, image: np.ndarray) -> np.ndarray:
        """Degrid image."""
        # Implementation
        pass
```

## Testing

### Test Structure

Tests are organized by module:

```
tests/
├── test_beam.py          # Beam model tests
├── test_gridding.py      # Gridding operator tests
├── test_deconv.py        # Deconvolution tests
├── test_operators.py     # Mathematical operator tests
├── test_utils.py         # Utility function tests
└── data/                 # Test data
```

### Writing Tests

Use pytest and parametrize tests:

```python
import pytest
import numpy as np
from pfb.operators.gridding import GriddingOperator

class TestGriddingOperator:
    """Test gridding operator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        nvis = 1000
        uvw = np.random.random((nvis, 3))
        visibilities = np.random.complex128(nvis)
        weights = np.ones(nvis)
        return uvw, visibilities, weights
    
    @pytest.mark.parametrize("nx,ny", [(256, 256), (512, 512)])
    def test_gridding_shapes(self, sample_data, nx, ny):
        """Test gridding produces correct shapes."""
        uvw, vis, weights = sample_data
        
        op = GriddingOperator(uvw, nx, ny, cell_size=2.0)
        grid = op(vis)
        
        assert grid.shape == (nx, ny)
        assert grid.dtype == np.complex128
    
    def test_adjoint_property(self, sample_data):
        """Test adjoint property: <Ax, y> = <x, A^H y>."""
        uvw, vis, weights = sample_data
        
        op = GriddingOperator(uvw, 256, 256, cell_size=2.0)
        
        # Create test vectors
        x = np.random.complex128(len(vis))
        y = np.random.complex128((256, 256))
        
        # Test adjoint property
        lhs = np.vdot(op(x), y)
        rhs = np.vdot(x, op.adjoint(y))
        
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)
```

### Test Data

Large test data is automatically downloaded:

```python
# conftest.py handles automatic download
@pytest.fixture(scope="session")
def ms_data():
    """Download and return measurement set data."""
    # Automatically downloads test data
    return download_test_data("test_ms.tar.gz")
```

## Documentation

### Docstring Standards

Use NumPy-style docstrings:

```python
def my_function(param1: int, param2: float) -> bool:
    """
    Brief description of the function.

    Longer description if needed. Can include mathematical
    notation using LaTeX:
    
    .. math::
        x = \arg\min_x \|Ax - b\|^2_2 + \lambda \|x\|_1

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : float
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        If param1 is negative
    TypeError
        If param2 is not a float

    Examples
    --------
    >>> result = my_function(10, 3.14)
    >>> print(result)
    True

    See Also
    --------
    related_function : Related functionality
    """
    pass
```

### Building Documentation

```bash
# Build documentation
poetry run mkdocs build

# Serve documentation locally
poetry run mkdocs serve

# Deploy to GitHub Pages
poetry run mkdocs gh-deploy
```

## Project Structure

Understanding the codebase structure:

```
pfb/
├── workers/              # CLI workers (main entry points)
│   ├── init.py          # Initialize measurement sets
│   ├── grid.py          # Gridding worker
│   ├── kclean.py        # Classical deconvolution
│   └── sara.py          # SARA deconvolution
├── operators/           # Mathematical operators
│   ├── gridding.py      # Gridding operators
│   ├── psf.py           # PSF operators
│   └── fft.py           # FFT operators
├── opt/                 # Optimization algorithms
│   ├── pcg.py           # Preconditioned conjugate gradient
│   └── fista.py         # FISTA algorithm
├── deconv/              # Deconvolution algorithms
│   ├── hogbom.py        # Hogbom CLEAN
│   └── sara.py          # SARA algorithm
├── utils/               # Utility functions
│   ├── fits.py          # FITS I/O
│   └── naming.py        # File naming conventions
└── parser/              # Configuration schemas
    ├── init.yaml        # Init worker schema
    └── grid.yaml        # Grid worker schema
```

## Performance Considerations

### Memory Management

- Use chunked arrays for large datasets
- Implement memory-efficient algorithms
- Profile memory usage with `memory_profiler`

### Parallel Processing

- Use Dask for distributed computing
- Implement thread-safe operations
- Consider NUMA topology for performance

### Numerical Stability

- Use appropriate data types (float64 for coordinates)
- Handle edge cases in algorithms
- Validate numerical accuracy in tests

## Review Process

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and cover new functionality
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Backwards compatibility is maintained

### Code Review

- Focus on correctness and clarity
- Consider performance implications
- Ensure proper error handling
- Verify test coverage

## Getting Help

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Chat**: Join our development chat (link in README)

## Recognition

Contributors are recognized in:
- AUTHORS file
- Release notes
- Documentation acknowledgments

Thank you for contributing to PFB-Imaging!