"""Tests for the CLI deprecation banner helper and its wiring into the app."""

import pytest
from typer.testing import CliRunner

from pfb_imaging.cli import DEPRECATED_COMMANDS, app
from pfb_imaging.cli._deprecation import REMOVED_IN, warn_deprecated

runner = CliRunner()


def _normalise(text: str) -> str:
    """Collapse rich's line-wrapping/box characters into a flat string."""
    return " ".join(text.split())


# --- the helper itself -----------------------------------------------------


def test_warn_deprecated_names_command_version_and_replacement(capsys):
    warn_deprecated("init", replacement="pfb imager")
    err = _normalise(capsys.readouterr().err)

    assert "DEPRECATION WARNING" in err
    assert "pfb init" in err
    assert f"v{REMOVED_IN}" in err
    assert "REMOVED" in err
    assert "pfb imager" in err


def test_warn_deprecated_without_replacement_omits_migrate_line(capsys):
    warn_deprecated("model2comps")
    err = _normalise(capsys.readouterr().err)

    assert "pfb model2comps" in err
    assert "Migrate to" not in err


def test_warn_deprecated_writes_to_stderr_not_stdout(capsys):
    warn_deprecated("grid", replacement="pfb imager")
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "grid" in captured.err


# --- wiring into the app ----------------------------------------------------
# `<cmd> --help` runs the group callback (which emits the banner) but short
# circuits before the command body, so these stay light and never touch data.


@pytest.mark.parametrize("command", sorted(DEPRECATED_COMMANDS))
def test_deprecated_command_shows_banner(command):
    result = runner.invoke(app, [command, "--help"])
    out = _normalise(result.output)

    assert result.exit_code == 0
    assert "DEPRECATION WARNING" in out
    assert f"pfb {command}" in out
    assert f"v{REMOVED_IN}" in out


@pytest.mark.parametrize("command", ["hci", "imager", "deconv", "degrid", "restore"])
def test_supported_command_has_no_banner(command):
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0
    assert "DEPRECATION WARNING" not in result.output
