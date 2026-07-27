"""The per-user cache-dir defaults under /tmp (issue #270, wiki D18).

Covers NUMBA_CACHE_DIR and MBEAMS_CACHE_DIR. Subprocess-based: the env vars
must be set at pfb_imaging import time, before numba can be imported, so each
case needs a fresh interpreter with a controlled environment.
"""

import os
import subprocess
import sys

SNIPPET = (
    "import os, tempfile, pfb_imaging; "
    "print(os.environ['NUMBA_CACHE_DIR']); "
    "print(os.environ['MBEAMS_CACHE_DIR']); "
    "print(tempfile.gettempdir())"
)


def _import_env(**overrides):
    """os.environ minus cache/tempdir vars, plus overrides."""
    env = os.environ.copy()
    for key in ("NUMBA_CACHE_DIR", "MBEAMS_CACHE_DIR", "TMPDIR", "TEMP", "TMP"):
        env.pop(key, None)
    env.update(overrides)
    return env


def _run(env):
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    numba_dir, mbeams_dir, tempdir = result.stdout.strip().splitlines()
    return numba_dir, mbeams_dir, tempdir


def test_default_is_per_user_under_tmp():
    numba_dir, mbeams_dir, _ = _run(_import_env())
    assert numba_dir == f"/tmp/numba-cache-{os.getuid()}"
    assert mbeams_dir == f"/tmp/mbeams-cache-{os.getuid()}"


def test_explicit_env_var_wins():
    numba_dir, mbeams_dir, _ = _run(_import_env(NUMBA_CACHE_DIR="/custom/numba", MBEAMS_CACHE_DIR="/custom/mbeams"))
    assert numba_dir == "/custom/numba"
    assert mbeams_dir == "/custom/mbeams"


def test_default_ignores_tmpdir(tmp_path):
    # The defaults are deliberately hard-coded to /tmp so they always match the
    # static implicit /tmp mount hints in the stimela cabs — apptainer leaks the
    # host TMPDIR into the container, where a TMPDIR-derived path may not be
    # mounted. Explicit cache env vars remain the override mechanism.
    numba_dir, mbeams_dir, tempdir = _run(_import_env(TMPDIR=str(tmp_path)))
    assert tempdir == str(tmp_path)  # TMPDIR did take effect for gettempdir()
    assert numba_dir == f"/tmp/numba-cache-{os.getuid()}"
    assert mbeams_dir == f"/tmp/mbeams-cache-{os.getuid()}"
