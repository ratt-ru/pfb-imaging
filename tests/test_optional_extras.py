"""Guards for the optional-extra split that keeps pfb-imaging installable on arm.

python-casacore has never published a linux-aarch64 wheel and dask-ms depends
on it unconditionally, so both sit behind the ``[casacore]`` extra; ``tbb`` is
x86_64-only and sits behind ``[x86]``. These tests pin the two invariants that
split depends on:

1. the deconv/restore/CLI import graph never reaches an optional dependency at
   module scope -- a stray top-level ``from daskms import ...`` in a shared
   helper is the easy way to silently re-break an arm install;
2. a missing optional dependency reports which extra to install.

Neither test needs the optional packages installed, so both run everywhere.
"""

import ast
import platform
from pathlib import Path

import pytest

import pfb_imaging
from pfb_imaging.utils.optional import optional_dependency

SRC = Path(__file__).resolve().parent.parent / "src"
OPTIONAL_ROOTS = {"daskms", "casacore", "distributed"}

# Entry points that must work with only the [full] extra installed.
CASACORE_FREE_ENTRY_POINTS = [
    "pfb_imaging.core.deconv",
    "pfb_imaging.core.restore",
    # degrid reaches tables through arcae, never dask-ms. It is here because
    # the MSv4 front end arrived carrying a module-scope
    # `from daskms.fsspec_store import DaskMSStore` for a glob that plain
    # fsspec does -- exactly the silent re-break this guard exists to catch,
    # and it was invisible while the entry point was unlisted (#330).
    "pfb_imaging.core.degrid",
    "pfb_imaging.cli",
]


def _module_path(module):
    as_file = SRC / (module.replace(".", "/") + ".py")
    if as_file.exists():
        return as_file
    as_pkg = SRC / module.replace(".", "/") / "__init__.py"
    return as_pkg if as_pkg.exists() else None


def _module_scope_imports(path):
    """Imports executed when the module is imported, ignoring function bodies.

    Deferred imports inside a function are exactly the mechanism this split
    relies on, so they must not count as violations.
    """
    tree = ast.parse(path.read_text())
    found = []
    for node in tree.body:
        candidates = [node]
        if isinstance(node, (ast.If, ast.Try)):
            candidates = list(ast.walk(node))
        for sub in candidates:
            if isinstance(sub, ast.Import):
                found += [alias.name for alias in sub.names]
            elif isinstance(sub, ast.ImportFrom) and sub.level == 0 and sub.module:
                found.append(sub.module)
    return found


def _reachable_optional_imports(entry):
    seen, stack, violations = set(), [(entry, [entry])], []
    while stack:
        module, chain = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        for imported in _module_scope_imports(path):
            root = imported.split(".")[0]
            if root in OPTIONAL_ROOTS:
                violations.append(" -> ".join(chain + [imported]))
            elif root == "pfb_imaging":
                stack.append((imported, chain + [imported]))
    return violations


@pytest.mark.parametrize("entry", CASACORE_FREE_ENTRY_POINTS)
def test_entry_point_does_not_import_optional_deps(entry):
    violations = _reachable_optional_imports(entry)
    assert not violations, (
        f"{entry} reaches an optional dependency at module scope, which breaks "
        f"`uv sync --extra full` on linux-aarch64. Move the import into the "
        f"function that needs it:\n  " + "\n  ".join(violations)
    )


def test_optional_dependency_names_the_extra():
    with pytest.raises(ImportError) as excinfo:
        with optional_dependency("The degrid command"):
            import daskms  # noqa: F401

            pytest.skip("dask-ms is installed; nothing to assert about its absence")
    message = str(excinfo.value)
    assert "The degrid command" in message
    assert "casacore" in message
    assert "pfb-imaging[casacore]" in message


def test_optional_dependency_sees_through_a_reraised_import_error():
    """A shim re-raises with exc.name unset; the cause still names the module.

    dask-ms's own `requires_optional` does this for python-casacore
    (`daskms/utils.py`: `raise ImportError(msg) from import_errors[0]`), so the
    ImportError that reaches us carries a summary message and no `name` at all.
    Giving up on the first frame would report a missing extra as a mystery.
    """
    original = ModuleNotFoundError("No module named 'casacore'", name="casacore")
    with pytest.raises(ImportError) as excinfo:
        with optional_dependency("The hci command"):
            raise ImportError("Optional extras required by ... are missing") from original
    message = str(excinfo.value)
    assert "pfb-imaging[casacore]" in message


def test_optional_dependency_passes_through_unknown_import_errors():
    """A typo must not be reported as a missing extra."""
    with pytest.raises(ImportError) as excinfo:
        with optional_dependency("Something"):
            import pfb_imaging_not_a_real_module  # noqa: F401
    assert "extra" not in str(excinfo.value)


@pytest.mark.parametrize(
    "machine, tbb_installed, expected",
    [
        # TBB only when it is both plausible (x86_64) and actually installed.
        ("x86_64", True, "tbb"),
        ("AMD64", True, "tbb"),
        # x86_64 without the [x86] extra: must NOT demand tbb, or numba raises
        # "No threading layer could be loaded" at first parallel dispatch.
        ("x86_64", False, "default"),
        # No aarch64 tbb distribution exists, so the answer never depends on
        # the installed flag there.
        ("aarch64", False, "default"),
        ("aarch64", True, "default"),
        ("arm64", False, "default"),
    ],
)
def test_threading_layer_reflects_what_is_installed(monkeypatch, machine, tbb_installed, expected):
    monkeypatch.delenv("PFB_NUMBA_THREADING_LAYER", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(pfb_imaging, "_tbb_available", lambda: tbb_installed)
    assert pfb_imaging._default_threading_layer() == expected


def test_threading_layer_env_override(monkeypatch):
    monkeypatch.setenv("PFB_NUMBA_THREADING_LAYER", "workqueue")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert pfb_imaging._default_threading_layer() == "workqueue"


def test_load_tbb_is_a_noop_when_tbb_is_absent(monkeypatch):
    """Must return None rather than raising PackageNotFoundError."""
    monkeypatch.delenv("PFB_NUMBA_THREADING_LAYER", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert pfb_imaging._load_tbb() is None


def test_numba_actually_gets_a_threading_layer():
    """The end-to-end invariant: parallel dispatch must not raise.

    This is the regression that the tbb split introduced and that the unit
    tests above cannot catch -- numba only raises at first dispatch, in
    whichever process hits it.
    """
    import numba
    import numpy as np
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def _sum(a):
        total = 0.0
        for i in prange(a.size):
            total += a[i]
        return total

    assert _sum(np.ones(128)) == 128.0
    assert numba.threading_layer() in ("tbb", "omp", "workqueue")
