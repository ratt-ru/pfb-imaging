"""Helpers for dependencies that live behind an optional extra.

pfb-imaging's dependency set is split so that the cross-platform scientific
stack (`[full]`) installs cleanly on linux-aarch64, where several historically
assumed packages have no wheels:

* ``python-casacore`` has never published an aarch64 wheel, and ``dask-ms``
  depends on it unconditionally -- both live behind ``[casacore]``;
* ``tbb`` is x86_64/Windows only and has no sdist -- ``[x86]``.

Commands that need one of those import it lazily, so ``pfb --help`` and the
whole ``deconv``/``restore`` path keep working without them. This module turns
the resulting bare ``ModuleNotFoundError`` into a message that names the extra
to install.
"""

from contextlib import contextmanager

# Top-level module name -> the extra that provides it.
_EXTRA_FOR_MODULE = {
    "casacore": "casacore",
    "daskms": "casacore",
    "tbb": "x86",
}


def _missing_module(exc):
    """Top-level name of the module that was actually missing.

    ``exc.name`` is unset when a shim re-raises: ``dask.distributed`` catches
    ``ImportError`` from ``import distributed`` and raises its own advisory
    ``ImportError`` with no ``name``. The original is preserved on
    ``__cause__``, so walk the chain rather than giving up on the first frame.
    dask-ms does the same for python-casacore, which is why this matters after
    the distributed extra went away with the MSv2 degrid (#330).
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name = (getattr(exc, "name", None) or "").split(".")[0]
        if name:
            return name
        exc = exc.__cause__
    return ""


@contextmanager
def optional_dependency(feature):
    """Re-raise an optional dependency's ImportError with install instructions.

    Args:
        feature: Human-readable description of what the caller was doing, used
            to open the error message, e.g. ``"The degrid command"``.

    Raises:
        ImportError: With guidance, when the missing module is one of the known
            optional ones. Any other ImportError propagates untouched -- a typo
            or a genuinely broken install must not be reported as a missing
            extra.
    """
    try:
        yield
    except ImportError as exc:
        module = _missing_module(exc)
        extra = _EXTRA_FOR_MODULE.get(module)
        if extra is None:
            raise
        message = (
            f"{feature} requires the optional '{extra}' extra, which is not installed "
            f"(missing module: {module}).\n"
            f"    pip install 'pfb-imaging[{extra}]'\n"
            f"    uv sync --extra {extra}"
        )
        if extra == "casacore":
            message += (
                "\nOn linux-aarch64 this compiles python-casacore from source "
                "(no aarch64 wheel has ever been published) and additionally needs:\n"
                "    apt install casacore-dev libboost-python-dev libcfitsio-dev wcslib-dev cmake"
            )
        raise ImportError(message) from exc
