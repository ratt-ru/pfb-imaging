# Installation

## Requirements

PFB-Imaging requires Python 3.10 or later and is tested on Linux systems. The package depends on several scientific Python libraries and specialized radio astronomy tools.

### System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)
- **Memory**: At least 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: Fast SSD recommended for optimal performance

## Installation Methods

### 1. PyPI Installation (Recommended)

The easiest way to install PFB-Imaging is from PyPI:

```bash
pip install pfb-imaging
```

### 2. Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/ratt-ru/pfb-imaging.git
cd pfb-imaging
pip install -e .
```

### 3. Poetry Installation (Advanced)

If you prefer using Poetry for dependency management:

```bash
git clone https://github.com/ratt-ru/pfb-imaging.git
cd pfb-imaging
poetry install
```

## Performance Optimization

### DUCC0 Optimization

For maximum performance, install DUCC0 from source:

```bash
git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git
pip install -e ducc
```

This provides optimized FFT and gridding operations.

### JAX Backend

PFB-Imaging uses JAX for automatic differentiation. For GPU acceleration:

```bash
# For CUDA support
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For TPU support
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## CASA Measures Data

PFB-Imaging requires CASA measures data for coordinate transformations. This is automatically downloaded during the first run, but you can pre-install it:

```bash
mkdir -p ~/measures
curl ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar | tar xvzf - -C ~/measures
echo "measures.directory: ~/measures" > ~/.casarc
```

## Verification

Verify your installation by running:

```bash
pfb --help
```

You should see the PFB-Imaging command-line interface with available workers.

## Docker Installation

For containerized deployment:

```bash
# Build Docker image
docker build -t pfb-imaging .

# Run with data mounted
docker run -v /path/to/data:/data pfb-imaging pfb init --ms /data/my_data.ms
```

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

#### Memory Issues

For large datasets, increase memory limits:

```bash
export NUMBA_CACHE_DIR=/tmp/numba-cache
export OMP_NUM_THREADS=4
```

#### Performance Issues

1. **Use SSD storage** for intermediate files
2. **Adjust chunk sizes** in configuration
3. **Increase number of workers** for parallel processing

```bash
pfb grid --nworkers 8 --nthreads-per-worker 2
```

### Environment Variables

Configure PFB-Imaging behavior with environment variables:

```bash
# Numba cache directory
export NUMBA_CACHE_DIR=/tmp/numba-cache

# OpenMP thread count
export OMP_NUM_THREADS=4

# Dask configuration
export DASK_CONFIG=/path/to/dask.yaml
```

## Dependencies

### Core Dependencies

- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **Dask**: Distributed computing
- **Xarray**: Labeled arrays
- **JAX**: Automatic differentiation

### Radio Astronomy Dependencies

- **codex-africanus**: Radio astronomy algorithms
- **dask-ms**: Measurement set I/O
- **katbeam**: Beam models
- **python-casacore**: CASA table access

### Optional Dependencies

- **matplotlib**: Plotting
- **bokeh**: Interactive visualization
- **jupyter**: Notebook support

## Next Steps

After installation, check out the [Quick Start](quickstart.md) guide to begin processing your data.