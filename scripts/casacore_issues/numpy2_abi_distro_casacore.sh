#!/usr/bin/env bash
# Reproduce: python-casacore built from sdist against a distro casacore that was
# compiled for NumPy 1.x cannot run under NumPy 2.
#
#   RuntimeError: PycArray: failed to load the numpy API
#   A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
#
# This is why pfb-imaging's informational aarch64 `--extra all` CI leg fails.
# It is NOT arm-specific: it reproduces on x86_64, as below. aarch64 only hits
# it in practice because python-casacore has never published an aarch64 wheel,
# so sdist is the only route there. The x86_64 wheel bundles its own casacore
# 3.8 (libcasa_*.so.8) built against NumPy 2 and is unaffected.
#
# python-casacore does not compile against NumPy at all -- its CMakeLists asks
# only for `Python COMPONENTS Interpreter Development.Module` -- so the NumPy
# ABI comes entirely from casacore's libcasa_python3. No pip/uv build flag
# (including --no-build-isolation) changes that.
#
# Usage:
#   docker run --rm -v "$PWD/scripts/casacore_issues:/s:ro" ubuntu:24.04 \
#       bash /s/numpy2_abi_distro_casacore.sh          # -> FAILS  (numpy 2)
#   NUMPY_SPEC='numpy<2' docker run ... same command   # -> WORKS  (numpy 1)
#
# Measured on ubuntu:24.04, casacore-dev 3.5.0-4.1ubuntu2, python-casacore 3.7.1:
#   numpy 2.5.3  -> RuntimeError: PycArray: failed to load the numpy API
#   numpy 1.26.4 -> OK: wrote and read [1 2]
set -eux

NUMPY_SPEC="${NUMPY_SPEC:-numpy>=2}"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    casacore-dev libboost-python-dev libcfitsio-dev wcslib-dev cmake \
    libblas-dev liblapack-dev build-essential python3-dev python3-venv \
    python3-pip ca-certificates >/dev/null

# the offending binary: a prebuilt distro .so, never touched by pip
dpkg -s casacore-dev | sed -n 's/^Version: /casacore-dev /p'
ls -la /usr/lib/*/libcasa_python3.so.* || true

python3 -m venv /venv
/venv/bin/pip install -q --upgrade pip
/venv/bin/pip install -q "$NUMPY_SPEC"
/venv/bin/python -c "import numpy; print('runtime numpy', numpy.__version__)"

# --no-binary forces the sdist path that aarch64 has no choice about
/venv/bin/pip install -q --no-binary python-casacore 'python-casacore==3.7.1'

/venv/bin/python -c "
import os, tempfile
from casacore.tables import table, maketabdesc, makescacoldesc
d = tempfile.mkdtemp()
t = table(os.path.join(d, 't.tab'), maketabdesc([makescacoldesc('X', 0)]), nrow=2, ack=False)
t.putcol('X', [1, 2])
print('OK: wrote and read', t.getcol('X'))
"
