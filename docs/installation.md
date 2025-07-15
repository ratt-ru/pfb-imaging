# Installation

## Requirements

pfb-imaging requires Python 3.10 or later and is tested on Linux systems. The package depends on several scientific Python libraries and specialized radio astronomy tools.

### System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)

## Installation Methods

### 1. PyPI Installation (Recommended)

The easiest way to install pfb-imaging is from PyPI:

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

## CASA Measures Data

pfb-imaging requires CASA measures data for coordinate transformations. This is automatically downloaded during the first run, but you can pre-install it:

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

You should see the pfb-imaging command-line interface with available workers.

## Docker Installation

For containerized deployment:

```bash
# Build Docker image
docker build -t pfb-imaging .

# Run with data mounted
docker run -v /path/to/data:/data pfb-imaging pfb init --ms /data/my_data.ms
```