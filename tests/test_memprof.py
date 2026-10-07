import numpy as np
import pytest

from pfb_imaging.utils.memprof import MEMRAY_DIR_ENV, MEMRAY_NATIVE_ENV, format_memory, memray_task, task_memory


def test_task_memory_splits_rss():
    mem = task_memory()
    assert set(mem) == {"pid", "rss_gb", "anon_gb", "shmem_gb", "peak_gb"}
    assert mem["rss_gb"] >= mem["anon_gb"] > 0
    line = format_memory(mem)
    assert "anon" in line and "shm" in line


def test_memray_task_is_a_noop_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(MEMRAY_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    with memray_task("noop"):
        np.ones(10)
    assert not any(tmp_path.iterdir())


def test_memray_task_writes_one_capture(tmp_path, monkeypatch):
    memray = pytest.importorskip("memray")
    monkeypatch.setenv(MEMRAY_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(MEMRAY_NATIVE_ENV, "0")
    with memray_task("unit"):
        a = np.ones(2**20)  # 8 MiB
        del a
    (cap,) = tmp_path.glob("unit-*.bin")
    reader = memray.FileReader(str(cap))
    assert reader.metadata.peak_memory >= 8 * 2**20


def test_memray_task_survives_a_rerun_into_the_same_directory(tmp_path, monkeypatch):
    """Pids repeat across container nodes and reruns; a capture must never collide.

    memray refuses to overwrite an existing capture, and that OSError would
    escape the Ray task and kill the run.
    """
    pytest.importorskip("memray")
    import itertools

    from pfb_imaging.utils import memprof

    monkeypatch.setenv(MEMRAY_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(MEMRAY_NATIVE_ENV, "0")
    for _ in range(2):
        # a fresh worker process with the same pid restarts its sequence
        monkeypatch.setattr(memprof, "_task_seq", itertools.count())
        with memray_task("rerun"):
            np.ones(1000)
    assert len(list(tmp_path.glob("rerun-*.bin"))) == 2


def test_memray_env_fails_fast_without_memray(monkeypatch, tmp_path):
    """Asking for captures where memray is not installed fails in the driver, not per task."""
    import sys

    from pfb_imaging.utils.memprof import memray_env

    monkeypatch.setenv(MEMRAY_DIR_ENV, str(tmp_path))
    monkeypatch.setitem(sys.modules, "memray", None)
    with pytest.raises(ImportError, match="memray"):
        memray_env()
