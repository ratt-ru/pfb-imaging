FROM python:3.11-slim

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy package files
COPY pyproject.toml README.rst ./
COPY src/ src/

# install git in case we need to install a package from a git repo
RUN apt-get update && apt-get install -y git

# Install package with full dependencies using uv
RUN uv pip install --system --no-cache ".[full]"

# Make CLI available
CMD ["pfb", "--help"]
