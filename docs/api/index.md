# API Reference

This section provides comprehensive API documentation for pfb-imaging, automatically generated from the source code docstrings.

## Package Structure

pfb-imaging is organized into several main modules:

### Workers
CLI workers that implement the main processing pipeline:

- **[pfb.workers.init](init.md)** - Initialize measurement sets
- **[pfb.workers.grid](grid.md)** - Gridding and dirty image creation
- **[pfb.workers.kclean](kclean.md)** - Classical deconvolution algorithms
- **[pfb.workers.sara](sara.md)** - SARA deconvolution with sparsity
- **[pfb.workers.restore](restore.md)** - Image restoration

### Operators
Mathematical operators for radio interferometry:

- **[pfb.operators.gridding](gridding.md)** - Gridding and degridding operators
- **[pfb.operators.psf](psf.md)** - Point spread function operators
- **[pfb.operators.fft](fft.md)** - Fast Fourier transform operators

### Optimization
Optimization algorithms for image reconstruction:

- **[pfb.opt.pcg](pcg.md)** - Preconditioned conjugate gradient
- **[pfb.opt.fista](fista.md)** - Fast iterative shrinkage-thresholding

### Deconvolution
Deconvolution algorithms:

- **[pfb.deconv.hogbom](hogbom.md)** - Hogbom CLEAN algorithm
- **[pfb.deconv.clark](clark.md)** - Clark CLEAN algorithm
- **[pfb.deconv.sara](sara.md)** - SARA algorithm implementation

### Utilities
Utility functions and helpers:

- **[pfb.utils.fits](fits.md)** - FITS file I/O
- **[pfb.utils.naming](naming.md)** - File naming conventions
- **[pfb.utils.beam](beam.md)** - Beam model handling

## Usage Examples

### Basic Function Usage

```python
import numpy as np
from pfb.operators.gridding import GriddingOperator

# Create sample data
nvis = 1000
uvw = np.random.random((nvis, 3))
visibilities = np.random.complex128(nvis)

# Initialize gridding operator
op = GriddingOperator(uvw, nx=256, ny=256, cell_size=2.0)

# Grid visibilities
dirty_image = op(visibilities)

# Degrid (adjoint operation)
model_vis = op.adjoint(dirty_image)
```

### Mathematical Operators

All operators follow a consistent interface:

```python
class MyOperator:
    def __call__(self, x):
        """Forward operation."""
        return self.forward(x)
    
    def adjoint(self, x):
        """Adjoint operation."""
        return self.backward(x)
```

### Optimization Algorithms

```python
from pfb.opt.pcg import PCG
from pfb.operators.gridding import GriddingOperator

# Setup problem
A = GriddingOperator(uvw, nx, ny, cell_size)
b = visibilities

# Solve Ax = b
solver = PCG(A, tol=1e-6, maxiter=100)
x = solver.solve(b)
```

## Function Categories

### Core Functions

Functions that implement the main algorithmic components:

- Gridding and degridding operations
- FFT and convolution operations
- Deconvolution algorithms
- Optimization solvers

### Utility Functions

Helper functions for common tasks:

- FITS file I/O
- Coordinate transformations
- Array manipulation
- Logging and progress tracking

### Configuration Functions

Functions for handling configuration and parameters:

- Schema validation
- CLI parameter parsing
- Configuration file loading

## Type Annotations

All functions use type hints for better code documentation and IDE support:

```python
from typing import Optional, Tuple, Union
import numpy as np

def my_function(
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    threshold: float = 0.01
) -> Tuple[np.ndarray, float]:
    """
    Example function with type hints.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    weights : np.ndarray, optional
        Optional weights array
    threshold : float
        Processing threshold
        
    Returns
    -------
    result : np.ndarray
        Processed data
    metric : float
        Quality metric
    """
    pass
```

## Error Handling

All functions include proper error handling:

```python
def safe_function(data: np.ndarray) -> np.ndarray:
    """
    Function with error handling.
    
    Raises
    ------
    ValueError
        If data is empty or has wrong shape
    TypeError
        If data is not a numpy array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    
    return process_data(data)
```

## Performance Notes

### Memory Usage

- Functions use chunked arrays for large datasets
- Memory-efficient algorithms are preferred
- Automatic memory management with context managers

### Parallel Processing

- Thread-safe operations where applicable
- Dask integration for distributed computing
- Numba JIT compilation for performance-critical functions

### Numerical Precision

- Appropriate data types for different operations
- Numerical stability considerations
- Validation of results and convergence

## See Also

- [User Guide](../user-guide/overview.md) - High-level usage patterns
- [Examples](../examples/basic-imaging.md) - Practical examples
- [Theory](../theory/index.md) - Mathematical background