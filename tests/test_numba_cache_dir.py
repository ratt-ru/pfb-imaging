"""The per-user NUMBA_CACHE_DIR default (issue #270).

Subprocess-based: the env var must be set at pfb_imaging import time,
before numba can be imported, so each case needs a fresh interpreter
with a controlled environment.
"""

import os
import subprocess
import sys

SNIPPET = "import os, tempfile, pfb_imaging; print(os.environ['NUMBA_CACHE_DIR']); print(tempfile.gettempdir())"


def _import_env(**overrides):
    """os.environ minus cache/tempdir vars, plus overrides."""
    env = os.environ.copy()
    for key in ("NUMBA_CACHE_DIR", "TMPDIR", "TEMP", "TMP"):
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
    cache_dir, tempdir = result.stdout.strip().splitlines()
    return cache_dir, tempdir


def test_default_is_per_user_under_tempdir():
    cache_dir, tempdir = _run(_import_env())
    assert cache_dir == os.path.join(tempdir, f"numba-cache-{os.getuid()}")


def test_explicit_env_var_wins():
    cache_dir, _ = _run(_import_env(NUMBA_CACHE_DIR="/custom/cache"))
    assert cache_dir == "/custom/cache"


def test_tmpdir_is_respected(tmp_path):
    cache_dir, tempdir = _run(_import_env(TMPDIR=str(tmp_path)))
    assert tempdir == str(tmp_path)
    assert cache_dir == str(tmp_path / f"numba-cache-{os.getuid()}")
