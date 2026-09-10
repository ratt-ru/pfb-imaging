"""Unit tests for the MSv4 degrid front end (`pfb degrid-msv4`, issue #278)."""

import numpy as np
import pytest


def test_write_support_and_kernel_available():
    """The three moving parts degrid-msv4 rests on are importable and armed.

    This is a dependency-pin regression test, not a smoke test: each import
    below fails for a *different* and easily-reintroduced reason -- a
    pfb-model-spec release older than the degrid kernel, an xarray-ms
    resolved off the write-support alpha line, or `ray` without `[serve]`.
    """
    import xarray_ms
    from pfb_model_spec.utils.degrid import (  # noqa: F401
        model_geometry,
        model_to_apparent_vis_for_region,
        render_model_region,
        stokes_vis_to_corr,
    )
    from rarg_python_patterns.multiton import Multiton  # noqa: F401
    from ray import serve  # noqa: F401

    assert xarray_ms.multithreaded_writes(), (
        "xarray-ms resolved without write support; the pin must stay on the "
        "0.4.0-alpha line (see plan Global Constraints)"
    )


def test_ducc_mask_must_be_uint8():
    """A bool mask is rejected by ducc even though its layout is identical.

    Regression for the trap in `degrid_region`: `~np.isnan(...)` yields bool,
    which raises `RuntimeError: incorrect data type` from ducc's pybind layer.
    """
    from ducc0.wgridder.experimental import dirty2vis

    rng = np.random.default_rng(0)
    uvw = rng.normal(0.0, 300.0, (16, 3))
    freq = np.linspace(1e9, 1.1e9, 4)
    dirty = np.zeros((32, 16))
    dirty[20, 5] = 1.0
    kw = dict(
        pixsize_x=1e-5,
        pixsize_y=1e-5,
        center_x=0.0,
        center_y=0.0,
        flip_u=False,
        flip_v=True,
        flip_w=False,
        epsilon=1e-7,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    good = np.ones((16, 4), dtype=np.uint8)
    dirty2vis(uvw=uvw, freq=freq, dirty=dirty, mask=good, **kw)
    with pytest.raises(RuntimeError):
        dirty2vis(uvw=uvw, freq=freq, dirty=dirty, mask=good.astype(bool), **kw)
