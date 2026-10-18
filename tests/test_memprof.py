import json
import pstats
import time

import numpy as np
import pytest
import ray

from pfb_imaging.utils.memprof import (
    CPROFILE_DIR_ENV,
    MEMRAY_DIR_ENV,
    MEMRAY_NATIVE_ENV,
    TIMELINE_ENV,
    PhaseClock,
    cprofile_task,
    dump_timeline,
    format_memory,
    memray_task,
    profile_env,
    profile_task,
    task_memory,
)


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


def _spin(seconds):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


def test_cprofile_task_is_a_noop_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(CPROFILE_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    with cprofile_task("noop"):
        _spin(0.01)
    assert not any(tmp_path.iterdir())


def test_cprofile_task_writes_one_capture_naming_the_hot_function(tmp_path, monkeypatch):
    monkeypatch.setenv(CPROFILE_DIR_ENV, str(tmp_path))
    with cprofile_task("unit"):
        _spin(0.05)
    (cap,) = tmp_path.glob("unit-*.pstats")
    st = pstats.Stats(str(cap))
    assert any(func[2] == "_spin" for func in st.stats)


def test_profile_task_clock_measures_wall_and_cpu(monkeypatch):
    monkeypatch.delenv(CPROFILE_DIR_ENV, raising=False)
    monkeypatch.delenv(MEMRAY_DIR_ENV, raising=False)
    with profile_task("clock") as clock:
        _spin(0.1)
    assert clock["wall_s"] >= 0.1
    assert clock["cpu_s"] >= 0.05  # a busy loop is CPU-bound
    line = format_memory({**task_memory(), **clock})
    assert "wall" in line and "cores" in line


def test_profile_task_clock_survives_an_exception(monkeypatch):
    monkeypatch.delenv(CPROFILE_DIR_ENV, raising=False)
    monkeypatch.delenv(MEMRAY_DIR_ENV, raising=False)
    with pytest.raises(ValueError):
        with profile_task("boom") as clock:
            raise ValueError
    assert {"wall_s", "cpu_s"} <= set(clock)


def test_profile_env_forwards_the_cprofile_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(MEMRAY_DIR_ENV, raising=False)
    monkeypatch.setenv(CPROFILE_DIR_ENV, str(tmp_path))
    assert profile_env() == {CPROFILE_DIR_ENV: str(tmp_path)}


def test_phase_clock_logs_each_lap_and_a_summary():
    lines = []

    class Log:
        def info(self, msg):
            lines.append(msg)

    phases = PhaseClock(Log())
    phases.lap("A")
    phases.lap("B")
    phases.summary()
    assert lines[0].startswith("A took") and lines[1].startswith("B took")
    assert lines[2].startswith("Time by phase: A") and "B" in lines[2]


def test_dump_timeline_writes_the_task_events(tmp_path, monkeypatch):
    @ray.remote
    def tiny():
        return 1

    ray.get(tiny.remote())
    path = tmp_path / "sub" / "timeline.json"
    monkeypatch.setenv(TIMELINE_ENV, str(path))
    dump_timeline()
    events = json.loads(path.read_text())
    assert any(e.get("cat", "").startswith("task::") for e in events)


def test_dump_timeline_warns_instead_of_failing_the_run(tmp_path, monkeypatch):
    """The run's work is done when the timeline is written; a bad path must not fail it."""
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(TIMELINE_ENV, str(blocker / "timeline.json"))  # parent is a file
    with pytest.warns(UserWarning, match="Could not write the Ray task timeline"):
        dump_timeline()
