"""pfb_imaging.ZARR_ENV: zarr writes v2 and fsspec makes parent dirs, whichever is imported first (D62).

Subprocesses, because zarr 3's own pytest plugin imports zarr and fsspec before conftest, so
in-process the suite only ever exercises the already-imported fallback. Slow (D45): each
subprocess import costs ~2 s once conftest pins the BLAS/OpenMP pools to one thread. The fast
set still covers both branches indirectly -- the fallback in-process, and the env-var branch in
every Ray worker that writes a nested .dt group.
"""

import os
import subprocess
import sys

import pytest

zarr = pytest.importorskip("zarr")

PROBE = """
import fsspec, zarr
print(zarr.config.get("default_zarr_format"), fsspec.config.conf.get("file", {}).get("auto_mkdir"))
"""


@pytest.mark.slow  # ~2.1 s per subprocess under conftest's single-thread env (D45)
@pytest.mark.skipif(not hasattr(zarr, "config"), reason="zarr 2 has no config and writes v2 anyway")
@pytest.mark.parametrize("first", ["pfb_imaging", "zarr"])
def test_zarr_env_applies_in_either_import_order(first):
    env = {k: v for k, v in os.environ.items() if k not in ("ZARR_DEFAULT_ZARR_FORMAT", "FSSPEC_FILE_AUTO_MKDIR")}
    imports = (
        "import pfb_imaging\nimport zarr, fsspec\n"
        if first == "pfb_imaging"
        else "import zarr, fsspec\nimport pfb_imaging\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", imports + PROBE], env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["2", "True"]
