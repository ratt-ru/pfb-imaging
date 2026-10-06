"""The imager's ``--precision`` must hold for every array it stores.

ducc0's wgridder templates all of its real/complex arrays on a single type, so a
complex64 ``VIS`` demands float32 ``WEIGHT``/``dirty`` buffers -- mixing them
trips an "incorrect data type" assertion rather than silently upcasting. Keeping
the whole tree at the requested precision is therefore load-bearing, not just a
memory optimisation, and ``--double-accum`` is a wgridder-internal control that
must not leak into the stored dtypes.

``UVW``/``FREQ`` are the documented exceptions: ducc takes those as float64
regardless of the visibility precision.
"""

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.core.imager import imager as imager_core
from pfb_imaging.utils.misc import check_gridder_epsilon

# ducc takes these as f8 whatever the vis precision; MASK is a uint8 flag array
DTYPE_EXCEPTIONS = {"UVW": np.float64, "FREQ": np.float64, "MASK": np.uint8}

PRECISIONS = {"single": (np.float32, np.complex64), "double": (np.float64, np.complex128)}


@pytest.fixture(scope="module")
def precision_trees(ms_name, sky_truth, tmp_path_factory):
    """Image the same data at each precision; returns {precision: output basename}.

    `sky_truth` is requested because these tests assert the image is not dead,
    and the sky they image is *written into the shared MS* by that fixture. It
    used to be omitted, and the tests passed only because `test_imager.py` runs
    earlier in the session and injects the sky as a side effect. On a fresh
    checkout where that fixture was skipped, the imager correctly gridded an
    empty DATA column and these assertions failed on an all-zero DIRTY -- which
    looked like an aarch64 gridder bug for a while (#330).
    """
    out = {}
    tmp_path = tmp_path_factory.mktemp("precision")
    for precision in PRECISIONS:
        outname = str(tmp_path / f"prec_{precision}")
        imager_core(
            [Path(ms_name)],
            outname,
            integrations_per_image=-1,
            channels_per_image=2,
            product="I",
            field_of_view=1.0,
            robustness=0.0,
            precision=precision,
            # ducc0's single-precision kernels need epsilon >~ 1e-5
            epsilon=1e-5,
            fits_mfs=False,
            fits_cubes=False,
            overwrite=True,
            keep_scratch=True,
            keep_ray_alive=True,
        )
        out[precision] = outname
    return out


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_imager_precision_is_uniform(precision_trees, precision):
    """Every stored array follows --precision, in both the .dt and the .scratch."""
    real_type, complex_type = PRECISIONS[precision]
    outname = precision_trees[precision]

    for store in (outname + "_I.dt", outname + "_I.scratch"):
        dt = xr.open_datatree(store, engine="zarr", chunks=None)
        seen = set()
        for node in dt.subtree:
            for name, var in node.ds.data_vars.items():
                expected = DTYPE_EXCEPTIONS.get(
                    name, complex_type if np.issubdtype(var.dtype, np.complexfloating) else real_type
                )
                assert var.dtype == expected, (
                    f"{store.rsplit('/', 1)[-1]}:{node.path}/{name} is {var.dtype}, expected {expected.__name__}"
                )
                seen.add(name)
        # guard against the assertions above passing vacuously on a short tree
        assert {"DIRTY", "VIS", "WEIGHT", "WSUM"} & seen, f"no data variables found in {store}"

    # single precision is not an excuse for NaNs or a dead image
    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    band = dt[sorted(n for n in dt.children if n.startswith("band"))[0]]
    assert np.isfinite(band.ds.DIRTY.values).all()
    assert np.any(band.ds.DIRTY.values != 0)
    assert (band.ds.WSUM.values > 0).all()


def test_single_precision_matches_double(precision_trees):
    """Single-precision products agree with double to the requested gridding accuracy.

    Guards the other half of the contract: that the products are *right*, not
    merely stored in the right dtype.
    """
    trees = {p: xr.open_datatree(precision_trees[p] + "_I.dt", engine="zarr", chunks=None) for p in PRECISIONS}
    names = sorted(n for n in trees["double"].children if n.startswith("band"))
    assert names

    for name in names:
        bands = {p: trees[p][name].ds for p in PRECISIONS}
        for var in ("DIRTY", "BDIRTY", "PSF", "BEAM"):
            # DIRTY/PSF are stored un-normalised; compare in Jy/beam
            imgs = {
                p: (b[var].values / b.WSUM.values[:, None, None] if var != "BEAM" else b[var].values.astype(np.float64))
                for p, b in bands.items()
            }
            peak = np.abs(imgs["double"]).max()
            err = np.abs(imgs["single"] - imgs["double"]).max() / peak
            assert err < 1e-4, f"{name}/{var}: single differs from double by {err:.2e} of peak"
        wsum_err = np.abs(bands["single"].WSUM.values / bands["double"].WSUM.values - 1.0).max()
        assert wsum_err < 1e-5, f"{name}: WSUM differs by {wsum_err:.2e}"


# ---------------------------------------------------------------------------
# --epsilon / --precision coupling (#340)
#
# ducc0's wgridder picks its gridding kernel from (epsilon, dtype). There is no
# float32 kernel below ~1e-6, so a float32 run with the old 1e-7 default died on
# a C++ assertion -- in pass 2, after pass 1 had written the whole scratch store.
# Measured floors, by bisecting vis2dirty over epsilon (ducc0 0.39):
#
#     float32: 1e-6 works, 5e-7 fails
#     float64: 1e-12 works, 1e-14 fails
# ---------------------------------------------------------------------------


def test_epsilon_floor_refuses_single_below_ducc_kernel_limit():
    """--precision single with an unreachable --epsilon is refused, naming the option."""
    with pytest.raises(ValueError, match="--epsilon"):
        check_gridder_epsilon("single", 1e-7)


def test_epsilon_floor_allows_double_at_the_old_default():
    """1e-7 is perfectly reachable in float64; the guard must not touch it."""
    check_gridder_epsilon("double", 1e-7)


@pytest.mark.parametrize("precision,epsilon", [("single", 1e-6), ("double", 1e-12)])
def test_epsilon_floor_admits_what_ducc_admits(precision, epsilon):
    """The floors are the measured ducc limits, not a round number above them."""
    check_gridder_epsilon(precision, epsilon)


def test_epsilon_floor_refuses_double_below_float64_limit():
    with pytest.raises(ValueError, match="--epsilon"):
        check_gridder_epsilon("double", 1e-14)


@pytest.mark.parametrize("command", ["imager", "deconv", "degrid", "hci"])
def test_cli_epsilon_default_is_reachable_in_single_precision(command):
    """Every command's --epsilon default must work at either --precision.

    This is the regression gate for #340: the default used to be 1e-7, which is
    below ducc0's float32 kernel limit, so `--precision single` could not run at
    all without also passing --epsilon.
    """
    module = importlib.import_module(f"pfb_imaging.cli.{command}")
    default = inspect.signature(getattr(module, command)).parameters["epsilon"].default
    check_gridder_epsilon("single", default)
