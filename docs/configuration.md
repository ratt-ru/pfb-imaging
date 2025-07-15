# Configuration

PFB-Imaging uses a flexible configuration system based on YAML schemas. Each worker has its own configuration schema that automatically generates CLI parameters.

## Configuration Methods

### 1. Command Line Arguments

The simplest way to configure PFB-Imaging:

```bash
pfb kclean --niter 1000 --threshold 0.01 --gain 0.1
```

### 2. Configuration Files

Create a YAML configuration file:

```yaml
# config.yaml
niter: 1000
threshold: 0.01
gain: 0.1
nworkers: 4
output_filename: my_image
```

Use with:

```bash
pfb kclean --config config.yaml
```

### 3. Environment Variables

Set configuration via environment variables:

```bash
export PFB_NITER=1000
export PFB_THRESHOLD=0.01
pfb kclean --output-filename my_image
```

## Common Configuration Options

### Global Options

These options are available for all workers:

```yaml
# Processing
nworkers: 4                    # Number of parallel workers
nthreads_per_worker: 2         # Threads per worker
scheduler: threads             # Dask scheduler type

# Output
output_filename: my_image      # Base name for outputs
fits_output_folder: outputs/   # Directory for FITS files
log_level: INFO               # Logging level

# Performance
chunks: 4096                  # Chunk size for arrays
memory_limit: 8GB             # Memory limit per worker
```

### Gridding Options

Configure the gridding process:

```yaml
# Image parameters
nx: 2048                      # Image width
ny: 2048                      # Image height
cell_size: 2.0                # Cell size in arcseconds
field_of_view: 2.0            # Field of view in degrees

# Gridding
epsilon: 1e-7                 # Gridding accuracy
do_psf: true                  # Compute PSF
do_residual: true             # Compute residual
do_dirty: true                # Compute dirty image

# Weighting
robust: 0.0                   # Robust weighting parameter
natural: false                # Use natural weighting
```

### Deconvolution Options

#### Classical CLEAN (kclean)

```yaml
# Basic parameters
niter: 1000                   # Maximum iterations
threshold: 0.01               # Cleaning threshold
gain: 0.1                     # CLEAN gain

# Advanced
cyclefactor: 2.5              # Major cycle trigger
nchan: 1                      # Number of channels
nband: 1                      # Number of bands
```

#### SARA Deconvolution

```yaml
# Regularization
gamma: 0.1                    # Regularization parameter
l1_reweight_from: 100         # Start reweighting iteration
reweight_alpha: 0.8           # Reweighting parameter

# Constraints
positivity: true              # Enforce positivity
flux_constraint: false        # Enforce flux constraint
peak_constraint: false        # Enforce peak constraint

# Optimization
tol: 1e-5                     # Convergence tolerance
maxit: 500                    # Maximum iterations
```

## Schema-Based Configuration

PFB-Imaging uses YAML schemas to define configuration parameters. Schemas are located in `pfb/parser/` and automatically generate CLI interfaces.

### Example Schema

```yaml
# pfb/parser/kclean.yaml
niter:
  dtype: int
  default: 1000
  help: Maximum number of iterations

threshold:
  dtype: float
  default: 0.01
  help: Cleaning threshold as fraction of peak

gain:
  dtype: float
  default: 0.1
  help: CLEAN gain factor
```

### Custom Configuration

Create custom configurations for specific use cases:

```yaml
# high_dynamic_range.yaml
niter: 10000
threshold: 0.001
gain: 0.05
cyclefactor: 1.5
```

```yaml
# fast_processing.yaml
niter: 500
threshold: 0.05
gain: 0.2
nworkers: 8
```

## Environment-Specific Configuration

### Development

```yaml
# dev_config.yaml
log_level: DEBUG
nworkers: 1
chunks: 1024
output_filename: test_image
```

### Production

```yaml
# prod_config.yaml
log_level: INFO
nworkers: 16
nthreads_per_worker: 2
memory_limit: 16GB
chunks: 8192
```

### Cluster Computing

```yaml
# cluster_config.yaml
scheduler: distributed
nworkers: 32
memory_limit: 32GB
chunks: 16384
```

## Performance Tuning

### Memory Management

```yaml
# Memory optimization
chunks: 2048                  # Smaller chunks for less memory
memory_limit: 4GB             # Per-worker memory limit
spill_to_disk: true          # Enable disk spilling
```

### CPU Optimization

```yaml
# CPU optimization
nthreads_per_worker: 4        # Match CPU cores
scheduler: threads            # Use threaded scheduler
```

### I/O Optimization

```yaml
# I/O optimization
rechunk: true                 # Rechunk arrays for I/O
compression: lz4              # Use compression
```

## Advanced Configuration

### Dask Configuration

Create a Dask configuration file:

```yaml
# dask.yaml
distributed:
  worker:
    memory:
      target: 0.8
      spill: 0.9
      pause: 0.95
      terminate: 0.98
  scheduler:
    allowed-failures: 10
    work-stealing: true
```

Set with:

```bash
export DASK_CONFIG=/path/to/dask.yaml
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  default:
    class: logging.StreamHandler
    formatter: standard
    level: INFO
loggers:
  pfb:
    level: DEBUG
    handlers: [default]
```

## Configuration Validation

PFB-Imaging validates configuration parameters:

```bash
# Check configuration
pfb kclean --config config.yaml --validate-only

# Show default configuration
pfb kclean --show-config
```

## Best Practices

### 1. Use Version Control

Store configuration files in version control:

```bash
git add config.yaml
git commit -m "Add imaging configuration"
```

### 2. Environment-Specific Configs

Maintain separate configs for different environments:

```
configs/
├── development.yaml
├── staging.yaml
└── production.yaml
```

### 3. Parameter Sweeps

Use configuration files for parameter studies:

```bash
# Different regularization parameters
for gamma in 0.01 0.1 1.0; do
    sed "s/gamma: .*/gamma: $gamma/" config.yaml > config_$gamma.yaml
    pfb sara --config config_$gamma.yaml --output-filename image_$gamma
done
```

### 4. Documentation

Document your configuration choices:

```yaml
# config.yaml
# Configuration for high dynamic range imaging
# Optimized for point source recovery

niter: 10000      # High iteration count for deep cleaning
threshold: 0.001  # Low threshold for faint sources
gain: 0.05        # Conservative gain for stability
```

## Troubleshooting Configuration

### Common Issues

#### Invalid Parameters

```bash
# Check parameter validity
pfb kclean --config config.yaml --validate-only
```

#### Performance Issues

```bash
# Profile configuration
pfb kclean --config config.yaml --profile
```

#### Memory Problems

```bash
# Reduce memory usage
chunks: 1024
memory_limit: 2GB
spill_to_disk: true
```

### Configuration Debugging

```bash
# Show effective configuration
pfb kclean --config config.yaml --show-effective-config

# Dry run
pfb kclean --config config.yaml --dry-run
```