# python:3.11-slim is a multi-arch manifest (linux/amd64 + linux/arm64), so
# this line is unchanged -- buildx selects the right base per --platform.
FROM python:3.11-slim

# Set automatically by buildx: "amd64" or "arm64".
ARG TARGETARCH

# Which extras to install. Defaults to the full cross-platform stack plus the
# casacore/distributed extras, i.e. the pre-split behaviour. Override to build
# a lean arm image:  --build-arg PFB_EXTRAS=".[full]"
ARG PFB_EXTRAS=".[all]"

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ src/

# install git in case we need to install a package from a git repo.
#
# On arm64 several dependencies have no manylinux aarch64 wheel and are
# compiled from sdist during the install below:
#   ducc0            scikit-build-core + nanobind/pybind11 -> cmake, C++17
#   python-casacore  scikit-build-core -> cmake, casacore-dev, boost-python,
#                    cfitsio, wcslib                        ([casacore] extra)
#   numcodecs        setuptools + Cython -> C compiler      ([casacore] extra,
#                    via dask-ms's numcodecs<0.16 pin)
#   python-lzf       tiny C extension -> C compiler         ([casacore] extra)
# amd64 gets all of these as wheels and needs only git + gcc, so the heavy
# toolchain is installed conditionally and the amd64 image is unchanged.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc build-essential curl \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         apt-get install -y --no-install-recommends \
           cmake ninja-build pkg-config python3-dev \
           casacore-dev libboost-python-dev libcfitsio-dev wcslib-dev ; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# ducc0's CMakeLists defaults DUCC0_ARCH_FLAGS to -march=native for every
# non-MSVC target, which in a container bakes the *builder's* microarchitecture
# into the image. On GB10 (Cortex-X925 + Cortex-A725 big.LITTLE) `native` also
# resolves against whichever core the compiler happens to land on. Pin a
# portable baseline -- ducc0 keeps its NEON paths at armv8.2-a. Override with
# --build-arg DUCC0_ARCH_FLAGS=-mcpu=cortex-x925 for a machine-specific image.
ARG DUCC0_ARCH_FLAGS=-march=armv8.2-a
ENV CMAKE_ARGS="-DDUCC0_ARCH_FLAGS=${DUCC0_ARCH_FLAGS}"

# Install package with the requested extras using uv
RUN uv pip install --system --no-cache "${PFB_EXTRAS}"

# CASA measures data. python-casacore needs it at runtime and, unlike the
# x86_64 wheel, the arm source build bundles no copy. Skipped when the image
# was built without the casacore extra.
RUN if python -c "import casacore" 2>/dev/null; then \
      mkdir -p /opt/measures \
      && curl -sSL --disable-epsv --max-time 300 \
           ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar \
           -o /tmp/WSRT_Measures.ztar \
      && tar xzf /tmp/WSRT_Measures.ztar -C /opt/measures \
      && rm /tmp/WSRT_Measures.ztar \
      && echo "measures.directory: /opt/measures" > /root/.casarc ; \
    fi

# So that TBB is visible to numba. Only meaningful on amd64: the tbb package is
# x86_64-only (no aarch64 wheel, no sdist), so on arm64 it is not installed and
# pfb_imaging selects numba's OpenMP layer instead. Harmless either way.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages:$LD_LIBRARY_PATH

# Build-time smoke test. Importing pfb_imaging is where the threading-layer
# selection and the TBB ctypes load happen, so a broken arm64 image fails here
# rather than on a user's first run.
RUN pfb --help > /dev/null \
    && python -c "import os, platform, pfb_imaging; \
print(platform.machine(), 'NUMBA_THREADING_LAYER =', os.environ['NUMBA_THREADING_LAYER'])"

# Make CLI available
CMD ["pfb", "--help"]
