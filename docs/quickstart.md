# Quick Start

This guide will walk you through your first PFB-Imaging workflow, from raw measurement set to final image.

## Prerequisites

Ensure you have:
- PFB-Imaging installed ([Installation Guide](installation.md))
- A measurement set (MS) file
- Basic familiarity with radio interferometry concepts

## Basic Workflow

The PFB-Imaging pipeline follows a sequential workflow:

```mermaid
graph LR
    A[MS File] --> B[pfb init]
    B --> C[XDS Files]
    C --> D[pfb grid]
    D --> E[DDS Files]
    E --> F[pfb kclean]
    F --> G[pfb restore]
    G --> H[FITS Image]
```

### Step 1: Initialize Data

Convert your measurement set to PFB-Imaging format:

```bash
pfb init --ms my_data.ms --output-filename my_output
```

This creates xarray datasets (`.xds` files) containing:
- Visibility data
- UVW coordinates
- Antenna information
- Observation metadata

### Step 2: Grid Visibilities

Create the dirty image, PSF, and weights:

```bash
pfb grid --output-filename my_output
```

This produces:
- `my_output_dirty.dds`: Dirty image
- `my_output_psf.dds`: Point spread function
- `my_output_psfhat.dds`: PSF in Fourier domain
- `my_output_residual.dds`: Residual visibilities

### Step 3: Deconvolve (Classical)

Apply classical deconvolution using Hogbom CLEAN:

```bash
pfb kclean --output-filename my_output --niter 1000 --threshold 0.01
```

Options:
- `--niter`: Maximum number of iterations
- `--threshold`: Cleaning threshold (fraction of peak)
- `--gain`: CLEAN gain factor
- `--cyclefactor`: Major cycle trigger

### Step 4: Restore Image

Restore clean components to create the final image:

```bash
pfb restore --output-filename my_output
```

This produces the final FITS image: `my_output_restored.fits`

## Advanced Workflow

### Using SARA Deconvolution

For sparsity-based deconvolution with wavelets:

```bash
pfb sara --output-filename my_output --niter 500 --l1-reweight-from 100
```

Key parameters:
- `--l1-reweight-from`: Iteration to start reweighting
- `--gamma`: Regularization parameter
- `--positivity`: Enforce positivity constraint

### Parallel Processing

Scale up with multiple workers:

```bash
pfb grid --output-filename my_output --nworkers 4 --nthreads-per-worker 2
```

### Custom Configuration

Create a YAML configuration file:

```yaml
# config.yaml
niter: 1000
threshold: 0.01
gain: 0.1
nworkers: 4
```

Then run:

```bash
pfb kclean --config config.yaml --output-filename my_output
```

## Example: Complete Pipeline

Here's a complete example processing a measurement set:

```bash
#!/bin/bash

# Configuration
MS_FILE="my_observations.ms"
OUTPUT_NAME="my_image"
NWORKERS=4

# Step 1: Initialize
pfb init \
    --ms $MS_FILE \
    --output-filename $OUTPUT_NAME \
    --nworkers $NWORKERS

# Step 2: Grid
pfb grid \
    --output-filename $OUTPUT_NAME \
    --nworkers $NWORKERS

# Step 3: Deconvolve with SARA
pfb sara \
    --output-filename $OUTPUT_NAME \
    --niter 500 \
    --l1-reweight-from 100 \
    --gamma 0.1 \
    --positivity \
    --nworkers $NWORKERS

# Step 4: Restore
pfb restore \
    --output-filename $OUTPUT_NAME \
    --nworkers $NWORKERS

echo "Processing complete! Final image: ${OUTPUT_NAME}_restored.fits"
```

## Output Files

After running the pipeline, you'll have:

```
my_output_main.xds          # Main dataset
my_output_dirty.dds         # Dirty image
my_output_psf.dds           # Point spread function
my_output_model.mds         # Model image
my_output_residual.dds      # Residual image
my_output_restored.fits     # Final restored image
```

## Monitoring Progress

### Logging

Enable verbose logging:

```bash
pfb kclean --output-filename my_output --log-level DEBUG
```

### Performance Monitoring

Monitor resource usage:

```bash
# System resources
htop

# Disk usage
df -h

# Memory usage
free -h
```

### Dask Dashboard

For distributed processing monitoring:

```bash
# In Python
import dask
dask.config.set({"distributed.dashboard.link": "http://localhost:8787"})
```

## Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Reduce chunk size
pfb grid --output-filename my_output --chunks 2048

# Use fewer workers
pfb grid --output-filename my_output --nworkers 2
```

#### Slow Performance
```bash
# Use SSD for temporary files
export TMPDIR=/path/to/ssd/tmp

# Increase thread count
export OMP_NUM_THREADS=4
```

#### Convergence Issues
```bash
# Reduce threshold
pfb kclean --threshold 0.001

# Increase iterations
pfb kclean --niter 5000
```

## Next Steps

- Learn about [configuration options](configuration.md)
- Explore [advanced deconvolution techniques](user-guide/deconvolution.md)
- Understand the [mathematical background](theory/index.md)
- See more [examples](examples/basic-imaging.md)