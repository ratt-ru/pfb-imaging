"""Cross-process numba cache regression test for issue #273.

weight_data's compile chain must be servable from the on-disk numba cache:
a second process compiling the same signature must LOAD, not recompile and
re-save. The old sympy-lambdified implementation failed this (closure cells
held dispatchers whose pickle embeds a per-process UUID), making the cache
write-only and growing /tmp/numba without bound.
"""

import os
import subprocess
import sys

SNIPPET = """
import os
os.environ["NUMBA_CACHE_DIR"] = {cache_dir!r}
os.environ["NUMBA_DEBUG_CACHE"] = "1"
import numpy as np
import pfb_imaging  # noqa: F401  (forces NUMBA_THREADING_LAYER=tbb)
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
from pfb_imaging.utils.weighting import weight_data

nrow, nchan, ncorr = 6, 4, 2
data = np.zeros((nrow, nchan, ncorr), np.complex64)
weight = np.ones((nrow, nchan, ncorr), np.float32)
flag = np.zeros((nrow, nchan, ncorr), bool)
jones = np.ones((1, 3, nchan, 1, 2), np.complex64)
tbin_idx = np.array([0], np.int32)
tbin_counts = np.array([nrow], np.int32)
ant1 = np.array([0, 0, 0, 1, 1, 2], np.int32)
ant2 = np.array([1, 2, 2, 2, 0, 0], np.int32)
weight_data(data, weight, flag, jones, tbin_idx, tbin_counts,
            ant1, ant2, "linear", "I", "2", "minvar")
"""


def _run(cache_dir):
    env = os.environ.copy()
    env.pop("NUMBA_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET.format(cache_dir=cache_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def _nbc_count(cache_dir):
    return sum(1 for _, _, files in os.walk(cache_dir) for f in files if f.endswith(".nbc"))


def test_weight_data_cross_process_cache(tmp_path):
    cache_dir = str(tmp_path / "numba_cache")

    out1 = _run(cache_dir)
    n1 = _nbc_count(cache_dir)
    assert n1 > 0, f"first process saved nothing to the cache:\n{out1}"

    out2 = _run(cache_dir)
    n2 = _nbc_count(cache_dir)
    assert "data loaded from" in out2, f"second process never loaded from cache:\n{out2}"
    assert n2 == n1, f"second process wrote new cache entries ({n1} -> {n2}):\n{out2}"
