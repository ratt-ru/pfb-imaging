FROM python:3.11-slim

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ src/

# install git in case we need to install a package from a git repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install package with full dependencies using uv
RUN uv pip install --system --no-cache ".[full]"

# So that TBB is visible to numba
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages:$LD_LIBRARY_PATH

# So that the numba cache dir exists and gets mounted
ENV NUMBA_CACHE_DIR=/tmp/numba
RUN mkdir -p /tmp/numba

# Make CLI available
CMD ["pfb", "--help"]
