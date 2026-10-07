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
from pfb_imaging.utils.misc import GRIDDER_EPSILON_FLOOR, check_gridder_epsilon

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
# ducc0's wgridder picks its gridding kernel from (epsilon, dtype, kernel
# dimension), where the dimension is 3 with w-gridding on and 2 with it off.
# `utils.misc.GRIDDER_EPSILON_FLOOR` holds the exact boundary for each of the
# four combinations; the first test below is what makes them trustworthy, by
# asserting each value is accepted and the next representable double beneath it
# is not. A ducc upgrade that moves the table fails there, with the real number
# in the failure, rather than silently making the guard wrong in one direction.
#
# Getting this wrong in the *refusing* direction is what the old 1e-7 default
# did: float32 has no kernel there, so --precision single always died on a C++
# assertion -- in pass 2, after pass 1 had written the whole scratch store.
# Getting it wrong the other way, as the first cut of this guard did by treating
# 1e-6/1e-12 as the floor, refuses accuracy ducc would have delivered.
# ---------------------------------------------------------------------------


def _can_grid(epsilon, precision, do_wgridding):
    """True if ducc0 accepts this (epsilon, precision, kernel) combination."""
    from ducc0.wgridder.experimental import vis2dirty

    complex_type = PRECISIONS[precision][1]
    rng = np.random.default_rng(0)
    try:
        vis2dirty(
            uvw=rng.normal(size=(64, 3)) * 100.0,
            freq=np.linspace(1e9, 1.1e9, 2),
            vis=np.ones((64, 2), dtype=complex_type),
            npix_x=32,
            npix_y=32,
            pixsize_x=1e-5,
            pixsize_y=1e-5,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            nthreads=1,
        )
        return True
    except RuntimeError:
        return False


@pytest.mark.parametrize("key", sorted(GRIDDER_EPSILON_FLOOR))
def test_gridder_epsilon_floor_is_duccs_boundary(key):
    """Each floor is exactly ducc's limit: it works, one representable step below does not.

    The floors are measured constants, so this is the test that keeps them
    honest across a ducc0 upgrade. It is deliberately two-sided -- a floor that
    is merely *safe* would pass a one-sided check while refusing valid work.
    """
    precision, do_wgridding = key
    floor = GRIDDER_EPSILON_FLOOR[key]

    assert _can_grid(floor, precision, do_wgridding), f"ducc0 refuses the floor {floor!r} for {key}"
    below = np.nextafter(floor, 0.0)
    assert not _can_grid(below, precision, do_wgridding), f"ducc0 accepts {below!r}, below the floor for {key}"


def test_the_2d_kernel_reaches_lower_than_the_3d_one():
    """--do-wgridding off selects a 2-D kernel with its own, lower limit.

    Keying the guard on precision alone would refuse this epsilon outright,
    although ducc serves it perfectly well with w-gridding off.
    """
    for precision in PRECISIONS:
        floor_2d = GRIDDER_EPSILON_FLOOR[(precision, False)]
        floor_3d = GRIDDER_EPSILON_FLOOR[(precision, True)]
        assert floor_2d < floor_3d

        check_gridder_epsilon(precision, floor_2d, do_wgridding=False)
        with pytest.raises(ValueError, match="--epsilon"):
            check_gridder_epsilon(precision, floor_2d, do_wgridding=True)


def test_epsilon_floor_refuses_single_below_ducc_kernel_limit():
    """--precision single with an unreachable --epsilon is refused, naming the option."""
    with pytest.raises(ValueError, match="--epsilon"):
        check_gridder_epsilon("single", 1e-7)


def test_epsilon_floor_allows_double_at_the_old_default():
    """1e-7 is perfectly reachable in float64; the guard must not touch it."""
    check_gridder_epsilon("double", 1e-7)


def test_epsilon_floor_refuses_double_below_float64_limit():
    with pytest.raises(ValueError, match="--epsilon"):
        check_gridder_epsilon("double", 1e-14)


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_epsilon_is_refused(epsilon):
    """NaN must be rejected before the floor comparison, not by it.

    Every comparison against NaN is False, so `epsilon < floor` waves a NaN
    through and ducc fails late -- the exact failure this guard replaces.
    """
    with pytest.raises(ValueError, match="finite"):
        check_gridder_epsilon("single", epsilon)


def test_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="--precision"):
        check_gridder_epsilon("quadruple", 1e-5)


@pytest.mark.parametrize("command", ["imager", "deconv", "degrid", "hci"])
@pytest.mark.parametrize("do_wgridding", [True, False])
def test_cli_epsilon_default_is_reachable_in_single_precision(command, do_wgridding):
    """Every command's --epsilon default must work at either --precision.

    This is the regression gate for #340: the default used to be 1e-7, which is
    below ducc0's float32 kernel limit, so `--precision single` could not run at
    all without also passing --epsilon.
    """
    module = importlib.import_module(f"pfb_imaging.cli.{command}")
    default = inspect.signature(getattr(module, command)).parameters["epsilon"].default
    check_gridder_epsilon("single", default, do_wgridding=do_wgridding)
