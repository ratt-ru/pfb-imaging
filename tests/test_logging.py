"""Exceptions reach the log file, not just the terminal (#348)."""

import logging

import pytest
import ray
from rich.logging import RichHandler

from pfb_imaging.utils import logging as pfb_logging


@pytest.fixture
def logfile(tmp_path):
    """A log file on the app logger, detached again afterwards."""
    root = logging.getLogger("pfb")
    before = list(root.handlers)
    path = tmp_path / "run.log"
    pfb_logging.log_to_file(path)
    yield path
    for handler in root.handlers[:]:
        if handler not in before:
            handler.close()
            root.removeHandler(handler)


@ray.remote
def _fails_remotely():
    raise ValueError("raised inside a ray task")


def test_uncaught_exception_is_written_to_the_log_file(logfile):
    @pfb_logging.log_exceptions
    def entry():
        raise KeyError("not raised through error_and_raise")

    with pytest.raises(KeyError):
        entry()

    text = logfile.read_text()
    assert "entry failed" in text
    assert "Traceback" in text
    assert "not raised through error_and_raise" in text


def test_ray_task_failure_carries_the_remote_traceback_into_the_file(logfile):
    @pfb_logging.log_exceptions
    def entry():
        ray.get(_fails_remotely.remote())

    with pytest.raises(ValueError):
        entry()

    text = logfile.read_text()
    assert "raised inside a ray task" in text
    assert "_fails_remotely" in text  # the remote frame, not just the driver's


def test_nested_entry_points_log_the_exception_once(logfile):
    @pfb_logging.log_exceptions
    def inner():
        raise RuntimeError("boom")

    @pfb_logging.log_exceptions
    def outer():
        inner()

    with pytest.raises(RuntimeError):
        outer()

    assert logfile.read_text().count("RuntimeError: boom") == 1


def test_exception_record_is_kept_off_the_console():
    # the interpreter prints the traceback when it propagates; a second copy
    # from the console handler would be noise
    console = [h for h in logging.getLogger("pfb").handlers if isinstance(h, RichHandler)]
    assert console
    record = logging.LogRecord("pfb", logging.ERROR, __file__, 0, "x", None, None)
    assert console[0].filter(record)
    setattr(record, pfb_logging.FILE_ONLY, True)
    assert not console[0].filter(record)
