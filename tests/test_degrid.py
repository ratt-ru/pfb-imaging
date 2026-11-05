"""Tests for the MSv4 degrid front end (`pfb degrid`, issue #278).

Mostly unit tests of the seam in `utils/degrid.py` and the driver's guards,
plus the end-to-end null at the bottom -- which was the whole-pipeline
acceptance test for the port and outlived the MSv2 command it replaced
(#330).
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.utils.beam import BEAM_PA_TOLERANCE_DEG


def test_write_support_and_kernel_available():
    """The three moving parts degrid rests on are importable and armed.

    This is a dependency-pin regression test, not a smoke test: each import
    below fails for a *different* and easily-reintroduced reason -- a
    pfb-model-spec release older than the degrid kernel, an xarray-ms
    resolved off the write-support alpha line, or `ray` without `[serve]`.
    """
    import xarray_ms
    from pfb_model_spec.utils.degrid import (  # noqa: F401
        model_geometry,
        model_to_apparent_vis_for_regions,
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


def test_wrapped_angle_diff_handles_the_ra_zero_straddle():
    """RA differences must be compared as wrapped magnitudes.

    A naive `abs(a - b)` reads two angles either side of RA=0 as ~2*pi apart
    and would refuse a perfectly matched tangent point; a naive signed
    difference silently accepts half of all genuine mismatches.
    """
    from pfb_imaging.utils.msv4 import wrapped_angle_diff

    eps = 1e-9
    assert wrapped_angle_diff(2 * np.pi - eps, eps) == pytest.approx(2 * eps, abs=1e-12)
    assert wrapped_angle_diff(0.0, np.pi) == pytest.approx(np.pi)
    assert wrapped_angle_diff(0.1, 0.1) == pytest.approx(0.0)
    # elementwise over an (ra, dec) pair
    got = wrapped_angle_diff(np.array([2 * np.pi - eps, -0.5]), np.array([eps, -0.5]))
    assert got.shape == (2,)
    assert np.all(got < 1e-8)


def test_select_vis_nodes_filters_and_reports_geometry(ms_name):
    """Selection is by name, and chan0 is a full-node channel offset.

    `--freq-range` trims the frequency axis for compute, but the write region
    must be expressed in unsliced-node channel indices, so `chan0` is the
    offset the caller adds. Verified against the shipped test MS, which has a
    single partition of 60 times x 351 baselines x 8 channels x 4 correlations.
    """

    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    try:
        nodes = select_vis_nodes(dt)
        assert len(nodes) == 1
        node = nodes[0]
        assert node.ntime == 60
        assert node.nchan == 8
        assert node.chan0 == 0
        assert node.corr_types == ("XX", "XY", "YX", "YY")
        assert node.field_radec.shape == (2,)

        # a selection that matches nothing yields nothing
        assert select_vis_nodes(dt, field_names=["definitely-not-a-field"]) == []

        # a frequency range trims the axis and shifts chan0
        freqs = dt[node.path].ds.frequency.values
        trimmed = select_vis_nodes(dt, freq_min=float(freqs[2]), freq_max=float(freqs[5]))
        assert len(trimmed) == 1
        assert trimmed[0].chan0 == 2
        assert trimmed[0].nchan == 4
    finally:
        dt.close()


def test_ensure_model_columns_creates_a_canonical_column(degrid_ms):
    """MODEL_DATA must be created even though `sync_msv2` declines to.

    `MODEL_DATA` is in casacore's canonical MAIN descriptor, so xarray-ms's
    `generate_column_descriptor` validates the name and then emits no
    descriptor for it -- `addcols` is never asked. Without the fallback the
    default `--model-column MODEL_DATA` silently fails on any MS lacking it.
    """
    import arcae

    from pfb_imaging.utils.degrid import ensure_model_columns
    from pfb_imaging.utils.msv4 import get_engine
    from tests.conftest import drop_column

    drop_column(degrid_ms, "MODEL_DATA")
    with arcae.table(degrid_ms) as tab:
        assert "MODEL_DATA" not in tab.columns()

    dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms, main_ninstances=1))
    try:
        ensure_model_columns(degrid_ms, dt, ["MODEL_DATA"])
    finally:
        dt.close()

    with arcae.table(degrid_ms) as tab:
        assert "MODEL_DATA" in tab.columns()
        col = np.asarray(tab.getcol("MODEL_DATA"))
    assert col.shape == (21060, 8, 4)
    assert col.dtype == np.complex64
    assert np.all(col == 0)


def test_ensure_model_columns_creates_a_non_canonical_column(degrid_ms):
    """`--region-file` names columns MODEL_DATA1, MODEL_DATA2, ...

    Those are not canonical, so `sync_msv2` creates them itself; this asserts
    the wrapper does not get in its way.
    """
    import arcae

    from pfb_imaging.utils.degrid import ensure_model_columns
    from pfb_imaging.utils.msv4 import get_engine

    dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms, main_ninstances=1))
    try:
        ensure_model_columns(degrid_ms, dt, ["MODEL_DATA1"])
    finally:
        dt.close()

    with arcae.table(degrid_ms) as tab:
        assert "MODEL_DATA1" in tab.columns()
        assert np.asarray(tab.getcol("MODEL_DATA1")).shape == (21060, 8, 4)


def test_handles_open_before_a_column_is_added_still_read(degrid_ms):
    """A tree opened before ensure_model_columns keeps reading afterwards.

    Adding a column used to leave every table handle already cached in the
    process unable to resync ("another process changed the number of
    columns", ska-sa/arcae#241), so degrid wiped xarray-ms's whole Multiton
    cache after creating its columns. arcae >= 0.4.0a13 refreshes those
    handles itself, which is what let that eviction go (#325).
    """

    from pfb_imaging.utils.degrid import ensure_model_columns
    from pfb_imaging.utils.msv4 import get_engine

    before = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
    try:
        node = next(iter(before.children.values()))
        first = node.ds.VISIBILITY.isel(time=0).values

        writer = xr.open_datatree(degrid_ms, **get_engine(degrid_ms, main_ninstances=1))
        try:
            ensure_model_columns(degrid_ms, writer, ["MODEL_DATA1"])
        finally:
            writer.close()

        again = node.ds.VISIBILITY.isel(time=0).values
        np.testing.assert_array_equal(again, first)
    finally:
        before.close()

    after = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
    try:
        assert next(iter(after.children.values())).ds.VISIBILITY.isel(time=1).values.size
    finally:
        after.close()


def test_get_engine_can_pin_a_single_main_instance(degrid_ms):
    """Column creation needs a tree with one MAIN instance (see get_engine).

    With several, arcae adds the column on instance 0 but answers
    `sync_msv2`'s follow-up `columns()` from the least busy instance, which
    intermittently fails to resync or returns a stale list. That race is
    timing dependent, so pin the plumbing instead: the override must reach
    xarray-ms and must not drop its default `cache_size` bound.
    """
    from xarray_ms.backend.msv2.entrypoint_utils import DEFAULT_DRIVER_KWARGS

    from pfb_imaging.utils.msv4 import get_engine

    assert "driver_kwargs" not in get_engine(degrid_ms)

    kwargs = get_engine(degrid_ms, main_ninstances=1)
    assert kwargs["driver_kwargs"]["table_overrides"] == {"MAIN": {"ninstances": 1}}
    for key, value in DEFAULT_DRIVER_KWARGS.items():
        assert kwargs["driver_kwargs"][key] == value

    dt = xr.open_datatree(degrid_ms, **kwargs)
    dt.close()


def test_ensure_model_columns_is_idempotent(degrid_ms, pctable):
    """The common case is an MS that already has MODEL_DATA. Must be a no-op."""
    import arcae

    from pfb_imaging.utils.degrid import ensure_model_columns
    from pfb_imaging.utils.msv4 import get_engine

    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.full((21060, 8, 4), 7 + 3j, np.complex64))

    for _ in range(2):
        dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms, main_ninstances=1))
        try:
            ensure_model_columns(degrid_ms, dt, ["MODEL_DATA"])
        finally:
            dt.close()

    with arcae.table(degrid_ms) as tab:
        assert np.all(np.asarray(tab.getcol("MODEL_DATA")) == 7 + 3j)


def test_ensure_model_columns_does_not_read_the_visibility_column():
    """Declaring a column must not materialise one.

    `sync_msv2` reads only dims/shape/dtype. The obvious placeholder --
    `xr.zeros_like(node.VISIBILITY)`, which the upstream test uses -- forces a
    full read of the correlated-data column purely to declare a name. The
    implementation uses a zero-strided `np.broadcast_to` view instead; this
    asserts the placeholder really is one.
    """

    from pfb_imaging.utils.degrid import MODEL_DIMS, make_column_placeholder

    ph = make_column_placeholder((60, 351, 8, 4))
    assert ph.shape == (60, 351, 8, 4)
    assert ph.dtype == np.complex64
    assert ph.strides == (0, 0, 0, 0)
    assert ph.base.nbytes <= 8

    ds = xr.Dataset().assign({"X": (MODEL_DIMS, ph)})
    assert ds.X.shape == (60, 351, 8, 4)
    assert ds.X.dtype == np.complex64


def test_build_region_masks_without_a_region_file(simple_mds):
    """No region file means one all-ones mask on the model's own grid."""
    from pfb_imaging.utils.degrid import build_region_masks

    _, ds = simple_mds
    masks = build_region_masks(ds, None)
    assert len(masks) == 1
    assert masks[0].mask.shape == (32, 64)  # (ny, nx)
    assert (masks[0].i0, masks[0].j0) == (0, 0)
    assert np.all(masks[0].mask == 1.0)


def test_build_region_masks_is_y_major_on_a_non_square_grid(simple_mds, tmp_path):
    """The mask must come back (ny, nx), matching the spec-0.1 model.

    astropy `regions` renders to `(Y, X)`, and so is the model. On a square
    grid a stray transpose is invisible, which is why this grid is 64 x 32:
    an x-major mask would not even have the right shape, and a mask that had
    the right shape but the wrong orientation would select the wrong pixel.
    The region is placed on the component at (x=40, y=9).
    """
    from pfb_imaging.utils.degrid import build_region_masks

    _, ds = simple_mds
    nx, ny = 64, 32
    # ds9 image coordinates are 1-based and (x, y) ordered
    region_file = tmp_path / "one.reg"
    region_file.write_text("image\nbox(41,10,3,3,0)\n")

    masks = build_region_masks(ds, str(region_file))
    assert len(masks) == 2, "remainder first, then one mask per region"
    remainder, region = masks
    # the remainder is everything but a 3x3 box, so its bbox is the full grid
    assert remainder.mask.shape == (ny, nx)
    assert (remainder.i0, remainder.j0) == (0, 0)
    # the region is cropped to its own 3x3 box, centred on (x=40, y=9), then
    # grown by one pixel per axis because ducc refuses an odd grid
    assert region.mask.shape == (4, 4)
    assert (region.i0, region.j0) == (8, 39)  # (y, x) origin

    # the region covers the component's pixel and the remainder does not
    assert region.mask[9 - region.i0, 40 - region.j0] == 1.0
    assert remainder.mask[9, 40] == 0.0
    # and they partition the grid
    full = np.zeros((ny, nx))
    full[region.i0 : region.i0 + 4, region.j0 : region.j0 + 4] = region.mask
    np.testing.assert_array_equal(remainder.mask + full, np.ones((ny, nx)))
    # a 3x3 box, so exactly 9 pixels
    assert region.mask.sum() == 9.0


def test_build_region_masks_refuses_overlapping_regions(simple_mds, tmp_path):
    """Overlaps would double-count flux across columns; refuse, as today."""
    from pfb_imaging.utils.degrid import build_region_masks

    _, ds = simple_mds
    region_file = tmp_path / "two.reg"
    region_file.write_text("image\nbox(41,10,5,5,0)\nbox(42,11,5,5,0)\n")

    with pytest.raises(ValueError, match="[Oo]verlapping"):
        build_region_masks(ds, str(region_file))


def test_degrid_region_selects_the_requested_planes(degrid_ms, simple_mds):
    """Planes are picked by name from a multi-Stokes model, not by position.

    The model carries I and V = 0.5 I on the same component. Degridded as
    `["I"]` it must match the I-only model exactly; degridded as `["V"]` it
    must put 0.5x the I visibilities into the V-carrying correlations and
    nothing into the I combination. Taking the wrong plane, or labelling the
    right plane with the model's full Stokes list, fails one of the two.
    """
    from pfb_model_spec.utils.degrid import stokes_vis_to_corr
    from pfb_model_spec.utils.io import build_mds_dataset

    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    dt, node_ds, node, ds_i = _node_and_mds(degrid_ms, simple_mds)
    try:
        coeffs = ds_i.coefficients.values  # (1, npar, ncomps)
        two = build_mds_dataset(
            np.concatenate([coeffs, 0.5 * coeffs]),
            ds_i.location_y.values,
            ds_i.location_x.values,
            ds_i.attrs["parametrisation"],
            list(ds_i.params.values),
            ds_i.attrs["texpr"],
            ds_i.attrs["fexpr"],
            ds_i.times.values,
            ds_i.freqs.values,
            float(ds_i.cell_rad_x),
            int(ds_i.npix_x),
            int(ds_i.npix_y),
            float(ds_i.center_x),
            float(ds_i.center_y),
            bool(ds_i.flip_u),
            bool(ds_i.flip_v),
            bool(ds_i.flip_w),
            (float(ds_i.ra), float(ds_i.dec)),
            ["I", "V"],
            "test",
            # the same bands, so flat mode holds both models at the same frequency
            freq_bounds=ds_i.freq_bounds.values,
        )
        corr = tuple(node.corr_types)
        kw = dict(
            region={"time": slice(0, 4), "frequency": slice(0, 4)},
            columns=["MODEL_DATA"],
            corr_types=corr,
        )

        def run(model_ds, stokes):
            masks = build_region_masks(model_ds, None)
            return degrid_region(node_ds, model_ds=model_ds, masks=masks, stokes=stokes, **kw).MODEL_DATA.values

        i_only = run(ds_i, ["I"])
        np.testing.assert_array_equal(run(two, ["I"]), i_only)

        # which correlations carry I and V, from the same mapping degrid uses
        unit = np.ones((1, 1, 1))
        carries_i = np.abs(stokes_vis_to_corr(unit, ["I"], corr)[0, 0]) > 0
        carries_v = np.abs(stokes_vis_to_corr(unit, ["V"], corr)[0, 0]) > 0
        assert carries_i.any() and carries_v.any()

        # recover the I Stokes visibilities from a correlation with unit I weight
        c_i = int(np.flatnonzero(carries_i)[0])
        ntime, nbl, nchan, ncorr = i_only.shape
        vis_i = i_only[..., c_i].reshape(1, ntime * nbl, nchan)
        assert np.abs(vis_i).max() > 0
        expected = stokes_vis_to_corr(0.5 * vis_i, ["V"], corr).reshape(ntime, nbl, nchan, ncorr)

        v_only = run(two, ["V"])
        np.testing.assert_allclose(v_only, expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max())
        assert np.abs(v_only[..., carries_v]).max() > 0
        # the I combination (mean over the I-carrying correlations) is empty
        np.testing.assert_allclose(v_only[..., carries_i].mean(axis=-1), 0, atol=1e-6 * np.abs(expected).max())
    finally:
        dt.close()


def _node_and_mds(ms_path, mds_tuple):
    """Open the MS's single visibility node and the model dataset."""

    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    dt = xr.open_datatree(ms_path, **get_engine(ms_path))
    nodes = select_vis_nodes(dt)
    assert len(nodes) == 1
    return dt, dt[nodes[0].path].ds, nodes[0], mds_tuple[1]


def test_degrid_region_returns_only_the_model_columns(degrid_ms, simple_mds):
    """`to_msv2` writes every data variable it is given.

    Returning the isel'd node would rewrite UVW, DATA, FLAG and WEIGHT along
    with the model, so the returned dataset must carry the model column and
    nothing else -- while keeping `ds.encoding`, which the MSv2 store needs to
    find the table it came from.
    """
    from pfb_imaging.utils.degrid import MODEL_DIMS, build_region_masks, degrid_region

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        region = {"time": slice(0, 10), "frequency": slice(0, 4)}
        out = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, None),
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            stokes=["I"],
        )
        assert list(out.data_vars) == ["MODEL_DATA"]
        assert out.MODEL_DATA.dims == MODEL_DIMS
        assert out.MODEL_DATA.dtype == np.complex64
        assert out.MODEL_DATA.shape == (10, 351, 4, 4)
        assert out.encoding == node_ds.encoding
    finally:
        dt.close()


def test_degrid_region_matches_a_direct_kernel_call(degrid_ms, simple_mds):
    """The MSv4 wrapper must be faithful: same answer as calling the kernel.

    Numerical correctness of the kernel is pfb-model-spec's business (it has
    analytic point-source tests). What is this repo's business is the
    `(time, baseline_id, uvw_label)` -> `(nrow, 3)` reshape, the mask, the
    dtype cast and the reshape back -- so compare against the kernel driven
    with independently constructed arguments.
    """
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        region = {"time": slice(5, 15), "frequency": slice(2, 6)}
        expected = _direct_reference(node_ds, node, mds_ds, region, True).astype(np.complex64)
        freq = node_ds.isel(**region).frequency.values

        out = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, None),
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            stokes=["I"],
            epsilon=1e-9,
        )
        got = out.MODEL_DATA.values.reshape(-1, freq.size, len(node.corr_types))
        np.testing.assert_array_equal(got, expected)
    finally:
        dt.close()


def _direct_reference(node_ds, node, model_ds, region, divide_by_n, freq_out=None):
    """Call the plural kernel directly with independently built arguments."""
    from pfb_model_spec.utils.degrid import model_to_apparent_vis_for_regions

    from pfb_imaging.utils.degrid import flat_freq

    sub = node_ds.isel(**region)
    uvw = np.nan_to_num(sub.UVW.values.reshape(-1, 3))
    valid = ~np.isnan(sub.UVW.values.reshape(-1, 3)).any(axis=-1)
    freq = sub.frequency.values
    mask = np.ascontiguousarray(np.broadcast_to(valid[:, None], (uvw.shape[0], freq.size)).astype(np.uint8))
    (vis,) = model_to_apparent_vis_for_regions(
        model_ds,
        divide_by_n=divide_by_n,
        uvw=uvw,
        freq=freq,
        corr_types=node.corr_types,
        time=float(sub.time.values.mean()),
        freq_out=flat_freq(model_ds, freq) if freq_out is None else freq_out,
        stokes_out=["I"],
        mask=mask,
        epsilon=1e-9,
    )
    return vis


def test_degrid_region_divides_by_n_without_a_beam(degrid_ms, simple_mds):
    """No beam still means ducc folds 1/n in (D55). A wide cell makes n matter."""
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        wide = mds_ds.assign_attrs(cell_rad_x=2e-3, cell_rad_y=2e-3)
        region = {"time": slice(0, 6), "frequency": slice(0, 4)}
        got = degrid_region(
            node_ds,
            region=region,
            model_ds=wide,
            masks=build_region_masks(wide, None),
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            stokes=["I"],
            epsilon=1e-9,
        ).MODEL_DATA.values
        with_n = _direct_reference(node_ds, node, wide, region, True)
        without_n = _direct_reference(node_ds, node, wide, region, False)
        got = got.reshape(with_n.shape)
        scale = np.abs(with_n).max()
        np.testing.assert_allclose(got, with_n.astype(np.complex64), rtol=0, atol=1e-6 * scale)
        assert np.abs(got - without_n).max() > 1e-5 * scale
    finally:
        dt.close()


def test_degrid_region_samples_the_model_spectrum_per_chunk(degrid_ms, simple_mds):
    """Each chunk re-renders the model, at the frequency flat mode holds it at.

    A single point source degrids to a visibility of constant magnitude equal
    to its flux (ducc divides by n, which is 1 to ~3e-9 this close to the
    phase centre), which makes the assertion direct: |vis| per chunk == the
    model's flux at `flat_freq` of that chunk. The model's bands sit at 1.0 and
    1.1 GHz, so one-channel chunks there see the two band values.
    """
    from pfb_model_spec.utils.degrid import render_model_region

    from pfb_imaging.utils.degrid import build_region_masks, degrid_region, flat_freq

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        masks = build_region_masks(mds_ds, None)
        amplitudes = []
        for chan in (slice(0, 1), slice(1, 2)):
            region = {"time": slice(0, 5), "frequency": chan}
            out = degrid_region(
                node_ds,
                region=region,
                model_ds=mds_ds,
                masks=masks,
                columns=["MODEL_DATA"],
                corr_types=node.corr_types,
                stokes=["I"],
            )
            freq_out = flat_freq(mds_ds, node_ds.frequency.values[chan])
            expected_flux = float(
                render_model_region(mds_ds, time=float(node_ds.time.values[0:5].mean()), freq_out=freq_out).max()
            )
            # XX == I for a Stokes-I model with no beam
            got = np.abs(out.MODEL_DATA.values[..., 0])
            np.testing.assert_allclose(got, expected_flux, rtol=1e-5)
            amplitudes.append(expected_flux)

        assert amplitudes[0] != amplitudes[1], (
            "the two chunks must see different model fluxes, otherwise this "
            "test would pass with the frequency axis ignored"
        )
    finally:
        dt.close()


def test_degrid_region_tolerates_nan_padded_rows(degrid_ms, simple_mds):
    """Absent (time, baseline) cells arrive as NaN UVW and must stay inert.

    ducc derives its w range from every row it is handed, so an unmasked NaN
    row gives a NaN w extent and either a `too many w planes` assertion or
    heap corruption (issue #287). Padded rows must come back exactly zero and
    must not perturb the rest.
    """
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        region = {"time": slice(0, 6), "frequency": slice(0, 4)}
        masks = build_region_masks(mds_ds, None)
        clean = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=masks,
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            stokes=["I"],
        ).MODEL_DATA.values

        uvw = node_ds.UVW.values.copy()
        uvw[2, ::5, :] = np.nan  # pad a scattered set of cells
        padded_ds = node_ds.assign(UVW=(node_ds.UVW.dims, uvw))
        padded = degrid_region(
            padded_ds,
            region=region,
            model_ds=mds_ds,
            masks=masks,
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            stokes=["I"],
        ).MODEL_DATA.values

        assert np.isfinite(padded).all(), "NaN UVW leaked into the model column"
        assert np.all(padded[2, ::5] == 0), "padded cells must degrid to zero"
        untouched = np.ones(padded.shape[1], bool)
        untouched[::5] = False
        np.testing.assert_array_equal(padded[2, untouched], clean[2, untouched])
    finally:
        dt.close()


def test_degrid_region_accumulates_onto_the_existing_column(degrid_ms, simple_mds, pctable):
    """`--accumulate` adds to what is already in the column, within the region."""

    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.full((21060, 8, 4), 1 + 1j, np.complex64))

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        region = {"time": slice(0, 4), "frequency": slice(0, 2)}
        masks = build_region_masks(mds_ds, None)
        kw = dict(
            region=region,
            model_ds=mds_ds,
            masks=masks,
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            stokes=["I"],
        )
        plain = degrid_region(node_ds, **kw).MODEL_DATA.values
        acc = degrid_region(node_ds, accumulate=True, **kw).MODEL_DATA.values
        np.testing.assert_allclose(acc, plain + np.complex64(1 + 1j), rtol=1e-6)
    finally:
        dt.close()


def test_degrid_region_masks_split_flux_across_columns(degrid_ms, simple_mds, tmp_path):
    """Region masks partition the model, so the columns must sum to the whole."""
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    region_file = tmp_path / "one.reg"
    region_file.write_text("image\nbox(41,10,3,3,0)\n")

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        region = {"time": slice(0, 4), "frequency": slice(0, 4)}
        whole = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, None),
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            stokes=["I"],
        ).MODEL_DATA.values
        split = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, str(region_file)),
            columns=["MODEL_DATA", "MODEL_DATA1"],
            corr_types=node.corr_types,
            stokes=["I"],
        )
        assert set(split.data_vars) == {"MODEL_DATA", "MODEL_DATA1"}
        np.testing.assert_allclose(split.MODEL_DATA.values + split.MODEL_DATA1.values, whole, atol=1e-6)
        # the only component lives inside the region, so the remainder is empty
        assert np.all(split.MODEL_DATA.values == 0)
    finally:
        dt.close()


def test_assert_writable_rejects_a_dataset_without_store_encoding():
    """A dataset that lost its encoding cannot be written back to the MS.

    `msv2_store_from_dataset` needs `common_store_args` and `partition_key`;
    without them `to_msv2` raises `MissingEncodingError` at write time, after
    the work is done. Fail earlier and say what happened.
    """

    from pfb_imaging.utils.degrid import assert_writable

    ds = xr.Dataset({"MODEL_DATA": (("time",), np.zeros(4, np.complex64))})
    with pytest.raises(ValueError, match="encoding"):
        assert_writable(ds)

    ds.encoding = {"common_store_args": {}, "partition_key": ()}
    assert_writable(ds)  # must not raise


def test_check_model_accepts_a_well_formed_mds(simple_mds):
    from pfb_imaging.core.degrid import check_model

    _, ds = simple_mds
    geom = check_model(ds, "I")
    assert geom["nx"] == 64 and geom["ny"] == 32
    assert geom["stokes"] == ["I"]
    assert geom["cell_rad"] == pytest.approx(1e-5)


def test_check_model_refuses_a_product_outside_iquv(simple_mds):
    from pfb_imaging.core.degrid import check_model

    _, ds = simple_mds
    with pytest.raises(ValueError, match="not yet supported"):
        check_model(ds, "XX")


def test_check_model_refuses_a_product_the_model_is_not(simple_mds):
    """The .mds records what it was made from; degridding it as something else
    would produce confidently wrong correlations."""
    from pfb_imaging.core.degrid import check_model

    _, ds = simple_mds
    with pytest.raises(ValueError, match="missing"):
        check_model(ds, "Q")


def test_check_model_refuses_a_product_beyond_the_models_planes(simple_mds):
    """A multi-Stokes product must be a subset of the model's planes.

    The old `degrid` set `nstokes_out = len(product)` and degridded the same
    single plane into every slot, so `--product IQ` produced XX = 2I, YY = 0 --
    silently wrong. An I-only model asked for IQ is refused, not zero-filled.
    """
    from pfb_imaging.core.degrid import check_model

    _, ds = simple_mds
    with pytest.raises(ValueError, match="missing"):
        check_model(ds, "IQ")


def test_check_model_lets_a_beam_predict_planes_the_model_lacks(simple_mds):
    """With a beam, --product names apparent planes: an I-only model can be
    asked for IQUV, and the beam's leakage fills Q, U and V (opt-in, D55)."""
    from pfb_imaging.core.degrid import check_model
    from pfb_imaging.utils.degrid import check_stokes_subset

    _, ds = simple_mds
    assert check_model(ds, "IQUV", beam=True)["stokes"] == ["I"]
    assert check_stokes_subset("QI", ["I"], beam=True) == ["Q", "I"]
    with pytest.raises(ValueError, match="repeats"):
        check_stokes_subset("II", ["I"], beam=True)


@pytest.mark.parametrize(
    "product, model_stokes, expected",
    [("I", ["I"], ["I"]), ("I", ["I", "Q", "U", "V"], ["I"]), ("VI", ["I", "V"], ["V", "I"])],
)
def test_check_stokes_subset_accepts(product, model_stokes, expected):
    from pfb_imaging.utils.degrid import check_stokes_subset

    assert check_stokes_subset(product, model_stokes) == expected


@pytest.mark.parametrize(
    "product, model_stokes, match",
    [("Q", ["I"], "missing"), ("II", ["I"], "repeats"), ("XX", ["I"], "not yet supported"), ("", ["I"], "empty")],
)
def test_check_stokes_subset_refuses(product, model_stokes, match):
    from pfb_imaging.utils.degrid import check_stokes_subset

    with pytest.raises(ValueError, match=match):
        check_stokes_subset(product, model_stokes)


def _as_genesis(ds: xr.Dataset) -> xr.Dataset:
    """Hand-build the genesis form of a single-Stokes spec-0.1 `.mds`.

    Frozen on purpose: written from the genesis schema (pfb-model-spec's
    tests/_genesis.py), not via the library, so it cannot follow the library.
    """
    attrs = {k: v for k, v in ds.attrs.items() if k not in ("spec", "writer-version")}
    attrs.update({"spec": "genesis", "pfb-imaging-version": "0.0.9", "stokes": str(ds.stokes.values[0])})
    return xr.Dataset(
        data_vars={"coefficients": (("par", "comps"), ds.coefficients.values[0])},
        coords={
            "location_x": (("x",), ds.location_x.values),
            "location_y": (("y",), ds.location_y.values),
            "params": (("par",), ds.params.values),
            "times": (("t",), ds.times.values),
            "freqs": (("f",), ds.freqs.values),
        },
        attrs=attrs,
    )


def test_a_genesis_mds_renders_like_its_upgrade(simple_mds):
    from pfb_model_spec.utils.degrid import render_model_region

    from pfb_imaging.core.degrid import check_model

    _, ds = simple_mds
    gen = _as_genesis(ds.drop_vars("freq_bounds"))
    assert check_model(gen, "I")["stokes"] == ["I"]
    a = render_model_region(gen, time=float(ds.times[0]), freq_out=1.05e9)
    b = render_model_region(ds, time=float(ds.times[0]), freq_out=1.05e9)
    np.testing.assert_array_equal(a, b)
    assert a[0, 9, 40] == pytest.approx(1.5)


@pytest.mark.slow
def test_a_genesis_mds_degrids_like_its_upgrade(degrid_ms, simple_mds, tmp_path, pctable):
    from pfb_imaging.core.degrid import degrid

    _, ds = simple_mds
    gen_path = str(tmp_path / "genesis.mds")
    _as_genesis(ds.drop_vars("freq_bounds")).to_zarr(gen_path, mode="w")
    # a genesis .mds has no freq_bounds, which steer the chunking and flat_freq (D59), so the
    # upgrade it is compared with must lack them too
    mds_path = str(tmp_path / "upgraded.mds")
    ds.drop_vars("freq_bounds").to_zarr(mds_path, mode="w")
    other = str(tmp_path / "other.ms")
    shutil.copytree(degrid_ms, other)
    for ms, model in ((degrid_ms, mds_path), (other, gen_path)):
        degrid(
            [ms],
            str(tmp_path / "out"),
            8,
            mds=model,
            product="I",
            nworkers=1,
            nthreads=1,
            progressbar=False,
            log_directory=str(tmp_path / "logs"),
        )
    with pctable(degrid_ms, ack=False) as tab:
        want = np.asarray(tab.getcol("MODEL_DATA"))
    with pctable(other, ack=False) as tab:
        got = np.asarray(tab.getcol("MODEL_DATA"))
    assert np.abs(want).max() > 0
    np.testing.assert_array_equal(got, want)


def test_check_model_refuses_an_unknown_spec(simple_mds):
    from pfb_imaging.core.degrid import check_model

    _, ds = simple_mds
    bad = ds.copy()
    bad.attrs["spec"] = "exodus"
    with pytest.raises(ValueError, match="[Ss]pec"):
        check_model(bad, "I")


def test_check_model_refuses_non_square_pixels(simple_mds):
    from pfb_imaging.core.degrid import check_model

    _, ds = simple_mds
    bad = ds.copy()
    bad.attrs["cell_rad_y"] = 2.0 * bad.attrs["cell_rad_x"]
    with pytest.raises(ValueError, match="[Nn]on-square"):
        check_model(bad, "I")


def test_check_tangent_point_compares_wrapped_magnitudes(simple_mds):
    """A model made for one field must not be degridded against another.

    Mosaics need a per-row inverse w-phase (wiki D21) which v1 does not do, so
    a tangent-point mismatch is a refusal, not a warning. The comparison is a
    wrapped magnitude: a field at RA=2*pi-eps and a model at RA=+eps are the
    same direction and must pass.
    """
    from pfb_imaging.core.degrid import check_tangent_point
    from pfb_imaging.utils.msv4 import SelectedNode

    def node(ra, dec):
        return SelectedNode(
            path="/p",
            chan0=0,
            nchan=8,
            chan_width=1.0e6,
            ntime=60,
            field_name="f",
            spw_name="s",
            scan_name="0",
            field_radec=np.array([ra, dec]),
            corr_types=("XX", "XY", "YX", "YY"),
            freq=1.0e9 + 1.0e6 * np.arange(8),
        )

    _, ds = simple_mds
    ra, dec = float(ds.ra), float(ds.dec)
    check_tangent_point(ds, [node(ra, dec)])
    # the same direction expressed on the far side of the RA=0 wrap
    check_tangent_point(ds, [node(ra + 2 * np.pi - 1e-12, dec)])

    with pytest.raises(ValueError, match="[Tt]angent"):
        check_tangent_point(ds, [node(ra + 0.1, dec)])
    with pytest.raises(ValueError, match="[Tt]angent"):
        check_tangent_point(ds, [node(ra, dec + 0.1)])


@pytest.mark.slow
def test_degrid_writes_the_whole_ms(degrid_ms, simple_mds, tmp_path, pctable):
    """A full driver run must fill MODEL_DATA with what `degrid_region` computes.

    This is the region-write test: the driver chunks the node into several
    (time, frequency) regions and each replica writes its own, so if `region`
    were ever dropped or mis-offset the chunks would land on top of each other
    at the start of the array. Comparing against a chunk-matched
    `degrid_region` over the whole node catches exactly that.
    """

    from pfb_imaging.core.degrid import degrid
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region, frequency_chunks
    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    mds_path, mds_ds = simple_mds

    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.zeros((21060, 8, 4), np.complex64))

    degrid(
        [degrid_ms],
        str(tmp_path / "out"),
        mds=mds_path,
        product="I",
        integrations_per_chunk=20,
        channels_per_chunk=4,
        nworkers=2,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
        # Match `degrid_region`'s own default below: the reference and the
        # driver must grid to the same accuracy or the rtol=1e-6 comparison is
        # measuring the epsilon difference. The CLI default moved to 1e-5 (#340)
        # while the library helper's default stayed at 1e-7.
        epsilon=1e-7,
    )

    dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
    try:
        node = select_vis_nodes(dt)[0]
        node_ds = dt[node.path].ds
        got = node_ds.MODEL_DATA.values
        # the reference is deliberately chunked the same way: the model is
        # re-rendered per chunk, so a single whole-node call would evaluate it
        # at a different frequency and legitimately disagree. Chunks follow the
        # model's bands (D59), so take them from the same helper the driver uses
        fchunks = frequency_chunks(node.freq, 4, mds_ds.freq_bounds.values)
        assert len(fchunks) > 2, "several frequency chunks, or a mis-offset write would go unseen"
        expected = np.zeros_like(got)
        for t0 in range(0, 60, 20):
            for fs in fchunks:
                region = {"time": slice(t0, t0 + 20), "frequency": fs}
                expected[t0 : t0 + 20, :, fs, :] = degrid_region(
                    node_ds,
                    region=region,
                    model_ds=mds_ds,
                    masks=build_region_masks(mds_ds, None),
                    columns=["MODEL_DATA"],
                    corr_types=node.corr_types,
                    stokes=["I"],
                ).MODEL_DATA.values
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)
        assert np.any(got != 0), "nothing was written"
    finally:
        dt.close()


@pytest.mark.slow
def test_degrid_with_a_real_beam_matches_the_in_process_region(degrid_ms, simple_mds, tmp_path, pctable):
    """The Serve path with a real L-band wizard writes what degrid_region computes in-process.

    Skips unless PFB_TEST_BEAM_L=1: building the wizard reads (or downloads) the
    legacy L product from the beam cache, which CI does not stage.
    """
    import os

    if os.environ.get("PFB_TEST_BEAM_L") != "1":
        pytest.skip("set PFB_TEST_BEAM_L=1 with the L product in the beam cache")
    from meerkat_beams.utils import BeamWizard

    from pfb_imaging.core.degrid import degrid
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region
    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    mds_path, mds_ds = simple_mds
    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.zeros((21060, 8, 4), np.complex64))

    degrid(
        [degrid_ms],
        str(tmp_path / "out"),
        mds=mds_path,
        product="I",
        beam_model="L",
        integrations_per_chunk=20,
        channels_per_chunk=4,
        nworkers=2,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
        epsilon=1e-7,
    )

    wizards = {"all": BeamWizard(band="L")}
    dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
    try:
        node = select_vis_nodes(dt)[0]
        node_ds = dt[node.path].ds
        got = node_ds.MODEL_DATA.values
        expected = np.zeros_like(got)
        for t0 in range(0, 60, 20):
            for f0 in range(0, 8, 4):
                region = {"time": slice(t0, t0 + 20), "frequency": slice(f0, f0 + 4)}
                expected[t0 : t0 + 20, :, f0 : f0 + 4, :] = degrid_region(
                    node_ds,
                    region=region,
                    model_ds=mds_ds,
                    masks=build_region_masks(mds_ds, None),
                    columns=["MODEL_DATA"],
                    corr_types=node.corr_types,
                    stokes=["I"],
                    wizards=wizards,
                ).MODEL_DATA.values
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)
        assert np.any(got != 0), "nothing was written"
    finally:
        dt.close()


@pytest.mark.slow
def test_degrid_creates_the_column_when_it_is_absent(degrid_ms, simple_mds, tmp_path, pctable):
    """The default --model-column on an MS that has no MODEL_DATA."""

    from pfb_imaging.core.degrid import degrid
    from tests.conftest import drop_column

    drop_column(degrid_ms, "MODEL_DATA")
    mds_path, _ = simple_mds

    degrid(
        [degrid_ms],
        str(tmp_path / "out"),
        mds=mds_path,
        product="I",
        integrations_per_chunk=-1,
        channels_per_chunk=8,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )

    with pctable(degrid_ms, ack=False) as tab:
        assert "MODEL_DATA" in tab.colnames()
        col = np.asarray(tab.getcol("MODEL_DATA"))
    assert col.shape == (21060, 8, 4)
    assert np.any(col != 0)


@pytest.mark.slow
def test_degrid_runs_with_the_derived_channel_chunk(degrid_ms, simple_mds, tmp_path, pctable, monkeypatch):
    """Omitting --channels-per-chunk derives one band's worth of channels from the model's freq_bounds (#327).

    The unit test pins `default_channels_per_chunk`; this pins that the driver
    actually feeds it to the work list and runs to completion with it. On the
    test MS (100 MHz channels, 100 MHz model bands) the derived chunk is 1.
    """
    import arcae

    import pfb_imaging.core.degrid as degrid_mod
    from pfb_imaging.utils.degrid import default_channels_per_chunk

    mds_path, mds_ds = simple_mds
    assert "freq_bounds" in mds_ds

    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.zeros((21060, 8, 4), np.complex64))
    with arcae.table(f"{degrid_ms}::SPECTRAL_WINDOW") as spw:
        chan_width = float(np.asarray(spw.getcol("CHAN_WIDTH")).squeeze()[0])
    expected = default_channels_per_chunk(mds_ds.freq_bounds.values, chan_width)

    fsteps = []
    work_items = degrid_mod._work_items

    def spy(ms_index, node, integrations_per_chunk, channels_per_chunk, **kw):
        fsteps.append(channels_per_chunk)
        return work_items(ms_index, node, integrations_per_chunk, channels_per_chunk, **kw)

    monkeypatch.setattr(degrid_mod, "_work_items", spy)

    degrid_mod.degrid(
        [degrid_ms],
        str(tmp_path / "out"),
        mds=mds_path,
        product="I",
        integrations_per_chunk=-1,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )

    assert fsteps and set(fsteps) == {expected}
    with pctable(degrid_ms, ack=False) as tab:
        col = np.asarray(tab.getcol("MODEL_DATA"))
    assert np.any(col != 0), "nothing was written"


@pytest.mark.slow
def test_degrid_honours_freq_range(degrid_ms, simple_mds, tmp_path, pctable):
    """`--freq-range` must write to the selected channels and no others.

    The highest-risk arithmetic in the driver: the frequency axis is trimmed
    for compute but write regions index the *unsliced* node, so every slice is
    shifted by `SelectedNode.chan0`. Drop the shift and the right numbers land
    on the wrong channels -- which looks entirely plausible in a FITS image.
    """

    from pfb_imaging.core.degrid import degrid
    from pfb_imaging.utils.msv4 import get_engine

    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.zeros((21060, 8, 4), np.complex64))

    dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
    try:
        freqs = dt[next(iter(dt.children))].ds.frequency.values
    finally:
        dt.close()
    # channels 3, 4 and 5 only
    lo = float(freqs[3]) - 1.0
    hi = float(freqs[5]) + 1.0

    mds_path, _ = simple_mds
    degrid(
        [degrid_ms],
        str(tmp_path / "out"),
        mds=mds_path,
        product="I",
        freq_range=f"{lo}:{hi}",
        integrations_per_chunk=-1,
        channels_per_chunk=1,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )

    with pctable(degrid_ms, ack=False) as tab:
        col = np.asarray(tab.getcol("MODEL_DATA"))
    written = np.abs(col).sum(axis=(0, 2)) > 0
    np.testing.assert_array_equal(written, np.array([False, False, False, True, True, True, False, False]))


@pytest.mark.slow
def test_degrid_handles_multiple_measurement_sets(degrid_ms, simple_mds, tmp_path, pctable):
    """`WorkItem.ms_index` must route each region to the MS it came from."""
    import shutil

    from pfb_imaging.core.degrid import degrid

    second = str(tmp_path / "second.ms")
    shutil.copytree(degrid_ms, second)
    for path in (degrid_ms, second):
        with pctable(path, readonly=False, ack=False) as tab:
            tab.putcol("MODEL_DATA", np.zeros((21060, 8, 4), np.complex64))

    mds_path, _ = simple_mds
    degrid(
        [degrid_ms, second],
        str(tmp_path / "out"),
        mds=mds_path,
        product="I",
        integrations_per_chunk=30,
        channels_per_chunk=8,
        nworkers=2,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )

    with pctable(degrid_ms, ack=False) as tab:
        first_col = np.asarray(tab.getcol("MODEL_DATA"))
    with pctable(second, ack=False) as tab:
        second_col = np.asarray(tab.getcol("MODEL_DATA"))

    assert np.any(first_col != 0) and np.any(second_col != 0)
    # identical inputs, so identical outputs -- a routing bug leaves one empty
    np.testing.assert_array_equal(first_col, second_col)


def test_degrid_refuses_a_mismatched_tangent_point(degrid_ms, tmp_path):
    """The guard must fire before Ray starts and before a column is created."""
    from pfb_model_spec.utils.io import build_mds_dataset
    from pfb_model_spec.utils.modelspec import fit_image_cube

    from pfb_imaging.core.degrid import degrid

    image = np.zeros((1, 2, 1, 32, 32))
    image[:, :, 0, 5, 20] = 1.0
    times = np.array([1.62393461e9])
    freqs = np.array([1.0e9, 1.1e9])
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
        times, freqs, image, wgt=np.ones((1, 2, 1)), method="Legendre"
    )
    # a tangent point nowhere near the test MS's field
    ds = build_mds_dataset(
        coeffs,
        yi,
        xi,
        expr,
        params,
        texpr,
        fexpr,
        times,
        freqs,
        1e-5,
        32,
        32,
        0.0,
        0.0,
        False,
        True,
        False,
        (1.234, 0.567),
        ["I"],
        "test",
    )
    mds_path = tmp_path / "wrong.mds"
    ds.to_zarr(str(mds_path), mode="w")

    with pytest.raises(ValueError, match="[Tt]angent point mismatch"):
        degrid(
            [degrid_ms],
            str(tmp_path / "out"),
            mds=str(mds_path),
            product="I",
            integrations_per_chunk=-1,
            channels_per_chunk=8,
            nworkers=1,
            nthreads=1,
            progressbar=False,
            log_directory=str(tmp_path / "logs"),
        )


def _vis_node(nchan, ncorr=4, *, name="spw0", freqs=None):
    """A minimal correlated-data node: just what the code under test reads."""

    if freqs is None:
        freqs = np.linspace(1.0e9, 1.1e9, nchan)
    ds = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.zeros((2, 3, nchan, ncorr), np.complex64),
            ),
            "field_name": (("time",), np.array(["f0", "f0"])),
            "scan_name": (("time",), np.array(["0", "0"])),
        },
        coords={
            "time": np.array([1.62e9, 1.62e9 + 8.0]),
            "baseline_id": np.arange(3),
            "frequency": (
                "frequency",
                np.asarray(freqs),
                {"spectral_window_name": name, "channel_width": {"data": np.full(nchan, 1.0e6)}},
            ),
            "polarization": np.array(["XX", "XY", "YX", "YY"][:ncorr]),
        },
        attrs={
            "type": "visibility",
            "data_groups": {"base": {"correlated_data": "VISIBILITY", "field_and_source": "x/field_and_source"}},
        },
    )
    fns = xr.Dataset(
        {"FIELD_PHASE_CENTER_DIRECTION": (("field_name", "sky_dir_label"), np.array([[0.0, 0.5]]))},
        coords={"field_name": np.array(["f0"]), "sky_dir_label": np.array(["ra", "dec"])},
    )
    return xr.DataTree(dataset=ds, children={"field_and_source": xr.DataTree(fns)})


def test_ensure_model_columns_refuses_heterogeneous_partition_shapes(tmp_path):
    """A CASA column spans the whole MAIN table, so one cell shape must fit all.

    Heterogeneous spectral windows would need variably-shaped cells, and those
    cannot take a partial region write at all. The refusal must land *before*
    `sync_msv2`, so this passes a path that does not exist: if the guard ever
    moved below the write, the failure would be about the missing table
    instead of the shapes.
    """

    from pfb_imaging.utils.degrid import ensure_model_columns

    dt = xr.DataTree(children={"p0": _vis_node(8), "p1": _vis_node(4, name="spw1")})
    with pytest.raises(ValueError, match="differing"):
        ensure_model_columns(str(tmp_path / "does-not-exist.ms"), dt, ["MODEL_DATA"])


def test_select_vis_nodes_handles_a_descending_spectral_window():
    """Channel selection must not assume ascending frequency.

    `sel(frequency=slice(lo, hi))` returns nothing for a descending spectral
    window, and `searchsorted` carries the same assumption. Selection is by
    matching index instead, so both orderings work and `chan0` stays a
    full-node index.
    """

    from pfb_imaging.utils.msv4 import select_vis_nodes

    freqs = np.linspace(1.0e9, 1.1e9, 8)
    dt = xr.DataTree(children={"down": _vis_node(8, freqs=freqs[::-1])})

    got = select_vis_nodes(dt)
    assert len(got) == 1, "a descending SPW was dropped entirely"
    assert got[0].nchan == 8
    assert got[0].chan0 == 0

    # the three highest frequencies. On a descending axis those are channels
    # 0..2, so chan0 must be 0 -- searchsorted would have said 5.
    trimmed = select_vis_nodes(dt, freq_min=float(freqs[-3]) - 1.0, freq_max=float(freqs[-1]) + 1.0)
    assert len(trimmed) == 1
    assert trimmed[0].chan0 == 0
    assert trimmed[0].nchan == 3

    # and the ascending case still behaves
    up = xr.DataTree(children={"up": _vis_node(8, freqs=freqs)})
    asc = select_vis_nodes(up, freq_min=float(freqs[5]) - 1.0, freq_max=float(freqs[7]) + 1.0)
    assert asc[0].chan0 == 5
    assert asc[0].nchan == 3


def test_default_mds_prefers_deconv_then_model2comps(tmp_path):
    """Two producers write a `.mds` under different names; both are valid input.

    `deconv` writes `{base}_{suffix}.mds`; `pfbspec model2comps` writes
    `{base}_{suffix}_model.mds`. The legacy `degrid` only ever looked for the
    second, so `imager -> deconv -> degrid` never worked without `--mds`.
    """
    from pfb_imaging.core.degrid import _default_mds

    base = str(tmp_path / "out_I")
    deconv_name = f"{base}_main.mds"
    m2c_name = f"{base}_main_model.mds"

    with pytest.raises(ValueError) as excinfo:
        _default_mds(base, "main")
    # the error names both candidates, so the user knows what to pass
    assert deconv_name in str(excinfo.value)
    assert m2c_name in str(excinfo.value)

    Path(m2c_name).mkdir()
    assert _default_mds(base, "main") == m2c_name  # model2comps alone

    Path(deconv_name).mkdir()
    assert _default_mds(base, "main") == deconv_name  # deconv wins when both exist


def test_degrid_refuses_a_negative_integrations_per_chunk(degrid_ms, simple_mds, tmp_path):
    """A negative step other than -1 silently produces zero work items.

    `range(0, ntime, -2)` is empty, so the command would start Serve, write
    nothing, and exit successfully.
    """
    from pfb_imaging.core.degrid import degrid

    mds_path, _ = simple_mds
    with pytest.raises(ValueError, match="integrations-per-chunk"):
        degrid(
            [degrid_ms],
            str(tmp_path / "out"),
            8,
            mds=mds_path,
            product="I",
            integrations_per_chunk=-2,
            nworkers=1,
            nthreads=1,
            progressbar=False,
            log_directory=str(tmp_path / "logs"),
        )


def test_select_vis_nodes_handles_multiple_spectral_windows(multi_spw_ms):
    """Each SPW is its own node with its own channel axis, so chan0 is per-node.

    A frequency window that clips both bands must give each node an offset
    measured on its own axis -- a single global offset, or a label slice
    resolved against the wrong axis, silently writes the right numbers to the
    wrong channels.
    """

    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    dt = xr.open_datatree(multi_spw_ms, **get_engine(multi_spw_ms))
    try:
        nodes = select_vis_nodes(dt)
        assert len(nodes) == 2, f"expected one node per SPW, got {[n.path for n in nodes]}"
        assert {n.spw_name for n in nodes} == {"00", "spw-upper"}
        assert all(n.chan0 == 0 and n.nchan == 8 for n in nodes)

        lower, upper = (
            dt[nodes[0].path].ds.frequency.values,
            dt[nodes[1].path].ds.frequency.values,
        )
        assert lower[0] != upper[0], "the two SPWs should occupy different bands"

        # A window both bands overlap, landing at a DIFFERENT channel index in
        # each: that is the whole point. The upper SPW is offset by 2e8 with a
        # 1e8 channel width, so the same frequencies sit two channels lower in
        # it than in the lower SPW.
        lo, hi = float(lower[5]), float(lower[7])
        clipped = select_vis_nodes(dt, freq_min=lo, freq_max=hi)
        assert len(clipped) == 2
        by_spw = {n.spw_name: n for n in clipped}
        expected = {
            spw: (int(np.searchsorted(f, lo)), int(((f >= lo) & (f <= hi)).sum()))
            for spw, f in (("00", lower), ("spw-upper", upper))
        }
        assert expected["00"][0] != expected["spw-upper"][0], "test window is not discriminating"
        for spw, (chan0, nchan) in expected.items():
            assert (by_spw[spw].chan0, by_spw[spw].nchan) == (chan0, nchan), spw

        # --spw-names addresses one of them
        only_upper = select_vis_nodes(dt, spw_names=["spw-upper"])
        assert [n.spw_name for n in only_upper] == ["spw-upper"]
    finally:
        dt.close()


@pytest.mark.slow
def test_degrid_writes_every_spectral_window(multi_spw_ms, simple_mds, tmp_path, pctable):
    """A full run over two SPWs must fill both, each on its own channel axis."""

    from pfb_imaging.core.degrid import degrid
    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    with pctable(multi_spw_ms, readonly=False, ack=False) as tab:
        nrow = tab.nrows()
        tab.putcol("MODEL_DATA", np.zeros((nrow, 8, 4), np.complex64))

    mds_path, _ = simple_mds
    degrid(
        [multi_spw_ms],
        str(tmp_path / "out"),
        4,
        mds=mds_path,
        product="I",
        integrations_per_chunk=-1,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )

    dt = xr.open_datatree(multi_spw_ms, **get_engine(multi_spw_ms))
    try:
        nodes = select_vis_nodes(dt)
        assert len(nodes) == 2
        for node in nodes:
            written = dt[node.path].ds.MODEL_DATA.values
            assert np.any(written != 0), f"{node.spw_name} was never written"
            assert np.isfinite(written).all()
        # the two SPWs sit in different bands, so the model differs between them
        a = dt[nodes[0].path].ds.MODEL_DATA.values
        b = dt[nodes[1].path].ds.MODEL_DATA.values
        assert not np.allclose(a, b), "both SPWs got identical visibilities"
    finally:
        dt.close()


def test_workitem_is_hashable_with_slice_regions():
    """`region` holds slices, which are unhashable before Python 3.12.

    `WorkItem` is frozen and defines `__hash__`, so it advertises itself as
    hashable; a `frozenset` of the region's items made that a lie on 3.11,
    which is what the container ships and the floor for core code.
    """
    from pfb_imaging.core.degrid import WorkItem

    a = WorkItem(0, "/node", {"time": slice(0, 4), "frequency": slice(0, 128)})
    b = WorkItem(0, "/node", {"frequency": slice(0, 128), "time": slice(0, 4)})
    c = WorkItem(0, "/node", {"time": slice(0, 4), "frequency": slice(128, 256)})

    assert hash(a) == hash(b), "hash must not depend on the mapping's order"
    assert a == b
    assert hash(a) != hash(c)
    assert len({a, b, c}) == 2


def test_build_region_masks_names_a_region_that_misses_the_grid(simple_mds, tmp_path):
    """A region off the grid must say so, not die on `None.T`."""
    from pfb_imaging.utils.degrid import build_region_masks

    _, ds = simple_mds
    region_file = tmp_path / "off.reg"
    # the grid is 64 x 32, so this box lies entirely beyond it
    region_file.write_text("image\nbox(5000,5000,3,3,0)\n")

    with pytest.raises(ValueError, match="does not overlap the model grid"):
        build_region_masks(ds, str(region_file))


def test_check_writable_backend_refuses_a_non_casa_store(tmp_path):
    """degrid writes; only the CASA backend can be written back to."""
    from pfb_imaging.utils.degrid import check_writable_backend

    store = tmp_path / "some.zarr"
    store.mkdir()
    (store / ".zgroup").write_text('{"zarr_format": 2}')

    with pytest.raises(ValueError, match="can only write to a CASA measurement set"):
        check_writable_backend(str(store))


def test_check_writable_backend_accepts_a_measurement_set(ms_name):
    """The happy path stays open, including through a file:// prefix."""
    from pfb_imaging.utils.degrid import check_writable_backend

    check_writable_backend(ms_name)
    check_writable_backend(f"file://{ms_name}")


@pytest.mark.slow
def test_imager_degrid_nulls_the_residual(sky_truth, ms_meta, ms_name, tmp_path):
    """The full loop must subtract the sky it degridded.

    `imager --psf` -> real `.mds` -> `degrid` -> `imager` on
    `DATA-MODEL_DATA`. The model here is the injected truth rather than a
    deconvolved estimate, written through pfb-model-spec's own `model_to_ds`,
    so the `.mds` schema, the component fit and the re-render are all the
    production ones and the null is strict: no beam anywhere on this path, and
    a perfect model, so what is left is gridder error.

    **Why `deconv` is not in this chain.** It cannot be, in this test session.
    `fit_image_cube` handles one band since pfb-model-spec 0.1, so a
    single-band `deconv` would run, but this test keeps 2 bands so the `.mds`
    carries a real spectral fit -- and more than one band makes `deconv` stand
    up one Ray actor per band, which starves on the session cluster's
    `RAY_NUM_CPUS=2` (the same constraint `test_deconv_groundtruth` documents
    when it pins itself to `channels_per_image=-1`). Using the truth as the
    model also tests strictly more of degrid than a deconvolved model would,
    and the `.mds` is still produced by the real writer.
    """
    from pfb_model_spec.utils.io import model_to_ds

    from pfb_imaging.core.degrid import degrid
    from pfb_imaging.core.imager import imager as imager_core
    from pfb_imaging.operators.gridder import wgridder_conventions

    work_ms = tmp_path / "e2e.ms"
    shutil.copytree(ms_name, work_ms)

    outname = str(tmp_path / "e2e")
    imager_core(
        [Path(work_ms)],
        outname,
        channels_per_image=4,  # 2 bands
        integrations_per_image=-1,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=0.0,
        psf=True,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        assert len(nodes) == 2, f"expected 2 bands, got {nodes}"
        freq_out = np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes])
        time_out = np.array([float(dt[nodes[0]].ds.attrs["time_out"])])
        radec = (float(dt[nodes[0]].ds.attrs["ra"]), float(dt[nodes[0]].ds.attrs["dec"]))
        dirty = sum(dt[n].ds.DIRTY[0] for n in nodes).values
        wsum_total = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)
        dirty_peak = float(np.abs(dirty).max() / wsum_total)
    finally:
        dt.close()

    # the injected truth on the imager's own grid, (nband, nstokes, ny, nx)
    nx, ny = sky_truth.nx, sky_truth.ny
    model = np.zeros((freq_out.size, 1, ny, nx))
    for b, f in enumerate(freq_out):
        for s in range(sky_truth.lpix.size):
            ix = nx // 2 - int(sky_truth.lpix[s])
            iy = ny // 2 + int(sky_truth.mpix[s])
            model[b, 0, iy, ix] = sky_truth.ref_flux[s] * (f / ms_meta.freq0) ** sky_truth.alpha[s]

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    mds_path = str(tmp_path / "truth.mds")
    model_to_ds(
        time=time_out,
        freq=freq_out,
        fsel=np.ones(freq_out.size, dtype=bool),
        model=model,
        wgt=np.ones((freq_out.size, 1)),
        mds_name=mds_path,
        cell_rad=sky_truth.cell_rad,
        nx=nx,
        ny=ny,
        x0=x0,
        y0=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        radec=radec,
        stokes=["I"],
        writer_version="test",
    )
    assert Path(mds_path).exists()

    degrid(
        [Path(work_ms)],
        outname,
        4,
        mds=mds_path,
        product="I",
        integrations_per_chunk=-1,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )

    resname = str(tmp_path / "e2e_res")
    imager_core(
        [Path(work_ms)],
        resname,
        data_column="DATA-MODEL_DATA",
        channels_per_image=4,
        integrations_per_image=-1,
        product="I",
        nx=nx,
        ny=ny,
        cell_size=sky_truth.cell_size,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = xr.open_datatree(resname + "_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        residual = sum(dt[n].ds.DIRTY[0] for n in nodes).values
        wsum = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)
        peak = float(np.abs(residual).max() / wsum)
    finally:
        dt.close()

    # Stated against the pre-subtraction dirty peak, so an all-zero
    # MODEL_DATA (peak == dirty_peak) cannot pass and the threshold does not
    # silently encode a model-quality assumption.
    assert peak < 0.05 * dirty_peak, (
        f"residual peak {peak:.3e} vs dirty peak {dirty_peak:.3e} -- the degridded model did not subtract"
    )


@pytest.mark.parametrize(
    "chan_width, expected",
    [(1.0e7, 10), (-5.0e6, 20), (2.0e8, 1), (3.0e7, 3)],  # descending SPW; channel wider than a band
)
def test_default_channels_per_chunk(chan_width, expected):
    from pfb_imaging.utils.degrid import default_channels_per_chunk

    fb = np.array([[1.0e9, 1.1e9], [1.1e9, 1.2e9]])
    assert default_channels_per_chunk(fb, chan_width) == expected


def test_flux_scale_warning():
    from pfb_imaging.utils.degrid import flux_scale_warning

    assert "primary beam" in flux_scale_warning("intrinsic")
    assert "--beam-model" in flux_scale_warning("intrinsic")
    assert flux_scale_warning("apparent") is None
    assert flux_scale_warning(None) is None


def test_ignore_xarray_ms_warnings():
    import warnings

    from xarray_ms.errors import (
        ColumnShapeImputationWarning,
        FrameConversionWarning,
        IrregularGridWarning,
        MissingMetadataWarning,
    )

    from pfb_imaging.utils.msv4 import ignore_xarray_ms_warnings

    noisy = (IrregularGridWarning, MissingMetadataWarning, FrameConversionWarning, ColumnShapeImputationWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ignore_xarray_ms_warnings()
        for category in noisy:
            warnings.warn("noise", category, stacklevel=1)
        warnings.warn("signal", UserWarning, stacklevel=1)
    assert [w.category for w in caught] == [UserWarning]


def test_degrid_refuses_a_model_without_freq_bounds_before_ray(degrid_ms, simple_mds, tmp_path, monkeypatch):
    import pfb_imaging.core.degrid as degrid_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(degrid_mod, "init_ray", _boom)
    _, ds = simple_mds
    bare = str(tmp_path / "bare.mds")
    ds.drop_vars("freq_bounds").to_zarr(bare, mode="w")
    with pytest.raises(ValueError, match="freq_bounds"):
        degrid_mod.degrid(
            [degrid_ms],
            str(tmp_path / "out"),
            mds=bare,
            product="I",
            nworkers=1,
            nthreads=1,
            progressbar=False,
            log_directory=str(tmp_path / "logs"),
        )


def test_selected_nodes_carry_a_positive_channel_width(ms_name):
    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    dt = xr.open_datatree(ms_name, **get_engine(ms_name, None))
    try:
        nodes = select_vis_nodes(dt)
    finally:
        dt.close()
    assert nodes and all(n.chan_width > 0 for n in nodes)


class _FakeWizard:
    """Stands in for meerkat_beams' BeamWizard: a diagonal Mueller block
    value * (1 + slope * l_deg), constant in m, so an E-W mirror changes the answer."""

    def __init__(self, value=1.0, slope_deg=0.0, leak_q=0.0, freq_coeff=0.0):
        from types import SimpleNamespace

        self.value, self.slope, self.leak_q = value, slope_deg, leak_q
        self.freq_coeff = freq_coeff  # fractional change per Hz from 1 GHz
        self.bds = SimpleNamespace(X=SimpleNamespace(values=np.linspace(-2.0, 2.0, 401)))  # 0.01 deg pixels
        self.calls = []

    def partition_mueller(
        self,
        *,
        field_centre,
        times,
        freq,
        l,  # noqa: E741
        m,
        stokes_out,
        stokes_in,
        nthreads=1,
        pa_tolerance=0.0,
    ):
        self.calls.append(
            {
                "freq": freq,
                "ntime": times.size,
                "stokes": (stokes_out, stokes_in),
                "nthreads": nthreads,
                "pa_tolerance": pa_tolerance,
            }
        )
        ll, _ = np.meshgrid(l, m)  # (Y, X), like the real one
        field = self.value * (1.0 + self.slope * ll) * (1.0 + self.freq_coeff * (freq - 1.0e9))
        out = np.zeros((len(stokes_out), len(stokes_in)) + ll.shape, dtype=np.result_type(self.value, np.float32))
        for o, s in enumerate(stokes_out):
            if s in stokes_in:
                out[o, stokes_in.index(s)] = field
        if "Q" in stokes_out and "I" in stokes_in:
            out[stokes_out.index("Q"), stokes_in.index("I")] += self.leak_q * field
        return out


def _region_vis(node_ds, node, mds_ds, **kw):
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    out = degrid_region(
        node_ds,
        model_ds=mds_ds,
        masks=build_region_masks(mds_ds, None),
        columns=["MODEL_DATA"],
        corr_types=node.corr_types,
        epsilon=1e-9,
        **{"region": {"time": slice(0, 6), "frequency": slice(0, 4)}, "stokes": ["I"], **kw},
    )
    return out.MODEL_DATA.values


def test_degrid_region_attenuates_by_the_beam_at_the_source(degrid_ms, simple_mds):
    """One component at (x=40, y=9): the beam scales it by the beam's value there,
    and by the mirrored position's value only if l has the wrong sign."""
    from pfb_model_spec.utils.degrid import model_geometry, pixel_lm

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        wizard = _FakeWizard(slope_deg=30.0)
        beam = _region_vis(node_ds, node, mds_ds, wizards={"all": wizard})
        bare = _region_vis(node_ds, node, mds_ds)
        l_px, _ = pixel_lm(model_geometry(mds_ds))
        gain = 1.0 + 30.0 * np.rad2deg(l_px[40])
        mirrored = 1.0 + 30.0 * np.rad2deg(-l_px[40])
        assert abs(gain - mirrored) > 1e-3  # the test can tell the two apart
        np.testing.assert_allclose(beam, gain * bare, rtol=1e-6, atol=1e-6 * np.abs(bare).max())
        assert wizard.calls and wizard.calls[0]["stokes"] == ("I", "I")
        # the rotation average is threaded and merges angles (meerkat-beams#39)
        assert wizard.calls[0]["pa_tolerance"] == BEAM_PA_TOLERANCE_DEG
    finally:
        dt.close()


def test_degrid_region_predicts_leakage_when_the_product_asks_for_it(degrid_ms, simple_mds):
    """An I-only model, --product IQ and an I->Q leak of 0.1: XX = I + Q and
    YY = I - Q, so the two correlations differ by exactly the leak."""
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        assert list(node.corr_types) == ["XX", "XY", "YX", "YY"]
        bare = _region_vis(node_ds, node, mds_ds)
        leaky = _region_vis(node_ds, node, mds_ds, stokes=["I", "Q"], wizards={"all": _FakeWizard(leak_q=0.1)})
        atol = 1e-6 * np.abs(bare).max()
        np.testing.assert_allclose(leaky[..., 0], 1.1 * bare[..., 0], rtol=1e-6, atol=atol)
        np.testing.assert_allclose(leaky[..., 3], 0.9 * bare[..., 3], rtol=1e-6, atol=atol)
        np.testing.assert_allclose(leaky[..., 1:3], 0, atol=atol)
    finally:
        dt.close()


def test_degrid_region_with_a_beam_lets_ducc_divide_by_n(degrid_ms, simple_mds):
    """Pins the 1/n convention on the beam path (D55): the wizard's B is used bare
    and ducc divides. Fails if the beam path folds 1/n itself or the flag flips."""
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        wide = mds_ds.assign_attrs(cell_rad_x=2e-3, cell_rad_y=2e-3)
        beam = _region_vis(node_ds, node, wide, wizards={"all": _FakeWizard()})
        region = {"time": slice(0, 6), "frequency": slice(0, 4)}
        with_n = _direct_reference(node_ds, node, wide, region, True)
        without_n = _direct_reference(node_ds, node, wide, region, False)
        got = beam.reshape(with_n.shape)
        scale = np.abs(with_n).max()
        np.testing.assert_allclose(got, with_n.astype(np.complex64), rtol=0, atol=1e-6 * scale)
        assert np.abs(got - without_n).max() > 1e-5 * scale
    finally:
        dt.close()


def test_degrid_region_groups_take_their_own_beam_and_conjugate(degrid_ms, simple_mds):
    """MM rows get 0.5, MPM rows c and their reversed rows conj(c), exactly."""
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        nbl = int(node_ds.sizes["baseline_id"])
        idx = np.arange(nbl)
        conj = (idx % 4) == 1
        groups = ({"MM": idx[idx % 2 == 0], "MPM": idx[idx % 2 == 1]}, conj)
        c = 0.8 + 0.2j
        wizards = {"MM": _FakeWizard(0.5), "MPM": _FakeWizard(c)}
        beam = _region_vis(node_ds, node, mds_ds, wizards=wizards, groups=groups)
        bare = _region_vis(node_ds, node, mds_ds)
        factor = np.where(idx % 2 == 0, 0.5, np.where(conj, np.conj(c), c))
        expected = bare * factor[None, :, None, None]
        np.testing.assert_allclose(beam, expected, rtol=1e-6, atol=1e-6 * np.abs(bare).max())
    finally:
        dt.close()


def test_beam_terms_must_cover_every_baseline(degrid_ms, simple_mds):
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        nbl = int(node_ds.sizes["baseline_id"])
        groups = ({"MM": np.arange(nbl - 1)}, np.zeros(nbl, dtype=bool))
        with pytest.raises(ValueError, match="baseline"):
            _region_vis(node_ds, node, mds_ds, wizards={"MM": _FakeWizard()}, groups=groups)
    finally:
        dt.close()


def _no_ray(monkeypatch):
    import pfb_imaging.core.degrid as degrid_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(degrid_mod, "init_ray", _boom)
    return degrid_mod


def test_degrid_refuses_katbeam_before_ray(degrid_ms, simple_mds, tmp_path, monkeypatch):
    degrid_mod = _no_ray(monkeypatch)
    with pytest.raises(ValueError, match="katbeam"):
        degrid_mod.degrid([degrid_ms], str(tmp_path / "out"), mds=simple_mds[0], product="I", beam_model="katbeam")


def test_degrid_beam_failure_leaves_the_ms_untouched(degrid_ms, simple_mds, tmp_path, monkeypatch):
    """A beam that cannot be built must fail before any column is added."""
    import arcae

    degrid_mod = _no_ray(monkeypatch)

    def _unstaged(**kw):
        raise RuntimeError("beam products not staged")

    monkeypatch.setattr(degrid_mod, "BeamWizard", _unstaged)
    with pytest.raises(RuntimeError, match="not staged"):
        degrid_mod.degrid(
            [degrid_ms],
            str(tmp_path / "out"),
            mds=simple_mds[0],
            product="I",
            beam_model="L",
            model_column="BEAM_MODEL_DATA",
        )
    with arcae.table(degrid_ms) as tab:
        assert "BEAM_MODEL_DATA" not in tab.columns()


def test_degrid_refuses_a_beam_on_an_apparent_model(degrid_ms, simple_mds, tmp_path, monkeypatch):
    degrid_mod = _no_ray(monkeypatch)
    apparent = str(tmp_path / "apparent.mds")
    simple_mds[1].assign_attrs(flux_scale="apparent").to_zarr(apparent, mode="w")
    with pytest.raises(ValueError, match="twice"):
        degrid_mod.degrid([degrid_ms], str(tmp_path / "out"), mds=apparent, product="I", beam_model="L")


def test_degrid_group_beams_refuse_a_non_meerkat_array(degrid_ms, simple_mds, tmp_path, monkeypatch):
    degrid_mod = _no_ray(monkeypatch)
    with pytest.raises(ValueError, match="MeerKAT"):
        degrid_mod.degrid(
            [degrid_ms], str(tmp_path / "out"), mds=simple_mds[0], product="I", beam_model="L", baseline_groups=True
        )


def test_degrid_refuses_antenna_groups_without_baseline_groups(degrid_ms, simple_mds, tmp_path, monkeypatch):
    degrid_mod = _no_ray(monkeypatch)
    with pytest.raises(ValueError, match="only has an effect with --baseline-groups"):
        degrid_mod.degrid(
            [degrid_ms], str(tmp_path / "out"), mds=simple_mds[0], product="I", beam_model="L", antenna_groups="m*,e*"
        )


def test_degrid_refuses_malformed_antenna_groups_without_a_beam(degrid_ms, simple_mds, tmp_path, monkeypatch):
    degrid_mod = _no_ray(monkeypatch)
    with pytest.raises(ValueError, match="exactly two comma-separated patterns"):
        degrid_mod.degrid(
            [degrid_ms],
            str(tmp_path / "out"),
            mds=simple_mds[0],
            product="I",
            baseline_groups=True,
            antenna_groups="m*",
        )


def _spectral_mds(ms_name, path, method, bounds=None):
    """`simple_mds`'s component and grid, histopolated over two bands spanning the test MS.

    Bands [0.95, 1.35) and [1.35, 1.75) GHz hold the MS's channels 1.0-1.3 and
    1.4-1.7 GHz; the anchors sit off the band centres, as the imager's weighted
    `freq_out` does (D28).
    """
    import arcae
    from pfb_model_spec.utils.io import build_mds_dataset
    from pfb_model_spec.utils.modelspec import fit_image_cube

    with arcae.table(f"{ms_name}::FIELD") as tab:
        radec = np.asarray(tab.getcol("PHASE_DIR")).squeeze()
    nx, ny = 64, 32
    image = np.zeros((1, 2, 1, ny, nx))
    image[0, 0, 0, 9, 40] = 1.0
    image[0, 1, 0, 9, 40] = 0.7
    times = np.array([1.62393461e9])
    freqs = np.array([1.12e9, 1.53e9])
    bounds = np.array([[0.95e9, 1.35e9], [1.35e9, 1.75e9]]) if bounds is None else bounds
    fit = fit_image_cube(times, freqs, image, wgt=np.ones((1, 2, 1)), method=method, freq_bounds=bounds)
    ds = build_mds_dataset(
        *fit,
        times,
        freqs,
        1.0e-5,
        nx,
        ny,
        0.0,
        0.0,
        False,
        True,
        False,
        (float(radec[0]), float(radec[1])),
        ["I"],
        "test",
        freq_bounds=bounds,
    )
    ds.to_zarr(str(path), mode="w")
    return str(path), ds


def _per_channel(node_ds, node, mds_ds, nchan=4, **kw):
    """Exact reference: one chunk per channel, at that channel's own frequency.

    A one-channel linear chunk is evaluated at its channel; a flat one would be
    held at its band's weighted frequency."""
    from pfb_imaging.utils.degrid import build_region_masks, degrid_region

    planes = [
        degrid_region(
            node_ds,
            region={"time": slice(0, 6), "frequency": slice(c, c + 1)},
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, None),
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
            epsilon=1e-9,
            **{"stokes": ["I"], "spectral": "linear", **kw},
        ).MODEL_DATA.values
        for c in range(nchan)
    ]
    return np.concatenate(planes, axis=2)


def test_degrid_region_linear_matches_per_channel_inside_a_piece(degrid_ms, tmp_path):
    """Inside one tent piece the model is linear in frequency, so linear mode is exact."""
    tent = _spectral_mds(degrid_ms, tmp_path / "tent.mds", "tent")
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, tent)
    try:
        got = _region_vis(node_ds, node, mds_ds, spectral="linear")
        ref = _per_channel(node_ds, node, mds_ds)
        flat = _region_vis(node_ds, node, mds_ds)
    finally:
        dt.close()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6 * np.abs(ref).max())
    assert np.abs(flat - ref).max() > 1e-3 * np.abs(ref).max()  # flat really is wrong here


def _beam_terms(node_ds, mds_ds, wizard, **kw):
    from pfb_model_spec.utils.degrid import model_geometry

    from pfb_imaging.utils.degrid import chunk_beam_terms

    return chunk_beam_terms(
        {"all": wizard},
        None,
        nbl=int(node_ds.sizes["baseline_id"]),
        ntime=6,
        geom=model_geometry(mds_ds),
        radec=(float(mds_ds.ra), float(mds_ds.dec)),
        times=node_ds.time.values[:6],
        stokes_out=["I"],
        **kw,
    )


def test_degrid_region_linear_beam_matches_per_channel(degrid_ms, tmp_path):
    """A box model is constant inside a piece, so under a beam linear in frequency the
    apparent image is linear and interpolating between the end channels is exact."""
    box = _spectral_mds(degrid_ms, tmp_path / "box.mds", "box")
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, box)
    try:
        wizards = {"all": _FakeWizard(slope_deg=30.0, freq_coeff=2.0e-9)}
        got = _region_vis(node_ds, node, mds_ds, spectral="linear", wizards=wizards)
        ref = _per_channel(node_ds, node, mds_ds, wizards=wizards)
        flat = _region_vis(node_ds, node, mds_ds, wizards=wizards)
    finally:
        dt.close()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6 * np.abs(ref).max())
    assert np.abs(flat - ref).max() > 1e-3 * np.abs(ref).max()


def test_degrid_refuses_an_unknown_spectral_mode_before_ray(degrid_ms, simple_mds, tmp_path, monkeypatch):
    degrid_mod = _no_ray(monkeypatch)
    with pytest.raises(ValueError, match="spectral"):
        degrid_mod.degrid([degrid_ms], str(tmp_path / "out"), mds=simple_mds[0], product="I", spectral="cubic")


def test_degrid_checks_beam_frequencies_before_ray(degrid_ms, simple_mds, tmp_path, monkeypatch):
    """Every beam frequency is validated on the driver, before Ray and before any column exists."""
    import arcae

    degrid_mod = _no_ray(monkeypatch)
    built = []

    class _Recorder:
        def __init__(self, **kw):
            self.kw, self.checked = kw, []
            built.append(self)

        def check_freqs(self, freq):
            self.checked.append(np.asarray(freq))
            raise ValueError("requested frequencies [...] MHz fall outside the BDS frequency range")

    monkeypatch.setattr(degrid_mod, "BeamWizard", _Recorder)
    with pytest.raises(ValueError, match="fall outside"):
        degrid_mod.degrid(
            [degrid_ms],
            str(tmp_path / "out"),
            mds=simple_mds[0],
            product="I",
            beam_model="L",
            model_column="BEAM_MODEL_DATA",
            log_directory=str(tmp_path / "logs"),
        )
    assert built and built[0].kw["freq_extrapolation"] == "scale"
    checked = np.concatenate(built[0].checked)
    assert checked.min() == pytest.approx(1.0e9) and checked.max() == pytest.approx(1.7e9)
    with arcae.table(degrid_ms) as tab:
        assert "BEAM_MODEL_DATA" not in tab.columns()


@pytest.mark.slow
def test_linear_degrid_beats_flat_on_a_tent_model(sky_truth, ms_meta, ms_name, tmp_path):
    """imager -> tent `.mds` of the truth -> degrid flat and linear -> per-channel residuals.

    The construction of `test_imager_degrid_nulls_the_residual`, with the truth
    fitted by the `tent` histopolation over the imager's own band edges. One
    4-channel chunk per band holds the model flat across 400 MHz in flat mode;
    linear mode carries the tent's slope, so its worst single-channel residual
    must be smaller (D59). Residuals are judged per channel, because a flat
    chunk's error is antisymmetric about its mean and partly cancels in a band
    sum.
    """
    from pfb_model_spec.utils.io import model_to_ds

    from pfb_imaging.core.degrid import degrid
    from pfb_imaging.core.imager import imager as imager_core
    from pfb_imaging.operators.gridder import wgridder_conventions

    work_ms = tmp_path / "e2e.ms"
    shutil.copytree(ms_name, work_ms)
    imaging = dict(
        integrations_per_image=-1,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )
    outname = str(tmp_path / "e2e")
    imager_core([Path(work_ms)], outname, channels_per_image=4, psf=True, **imaging)

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        assert len(nodes) == 2, f"expected 2 bands, got {nodes}"
        freq_out = np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes])
        freq_bounds = np.array([dt[n].ds.attrs["freq_bounds"] for n in nodes], dtype=float)
        time_out = np.array([float(dt[nodes[0]].ds.attrs["time_out"])])
        radec = (float(dt[nodes[0]].ds.attrs["ra"]), float(dt[nodes[0]].ds.attrs["dec"]))
    finally:
        dt.close()

    nx, ny = sky_truth.nx, sky_truth.ny
    model = np.zeros((freq_out.size, 1, ny, nx))
    for b, f in enumerate(freq_out):
        for s in range(sky_truth.lpix.size):
            ix = nx // 2 - int(sky_truth.lpix[s])
            iy = ny // 2 + int(sky_truth.mpix[s])
            model[b, 0, iy, ix] = sky_truth.ref_flux[s] * (f / ms_meta.freq0) ** sky_truth.alpha[s]

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    mds_path = str(tmp_path / "truth.mds")
    model_to_ds(
        time=time_out,
        freq=freq_out,
        fsel=np.ones(freq_out.size, dtype=bool),
        model=model,
        wgt=np.ones((freq_out.size, 1)),
        mds_name=mds_path,
        cell_rad=sky_truth.cell_rad,
        nx=nx,
        ny=ny,
        x0=x0,
        y0=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        radec=radec,
        stokes=["I"],
        writer_version="test",
        method="tent",
        freq_bounds=freq_bounds,
    )

    # both modes write MODEL_DATA in turn, each overwriting the last: the
    # imager's column arithmetic reads the canonical name without any schema help
    peaks = {}
    for spectral in ("flat", "linear"):
        degrid(
            [Path(work_ms)],
            outname,
            4,
            mds=mds_path,
            product="I",
            spectral=spectral,
            integrations_per_chunk=-1,
            nworkers=1,
            nthreads=1,
            progressbar=False,
            log_directory=str(tmp_path / "logs"),
        )
        resname = str(tmp_path / f"res_{spectral}")
        imager_core([Path(work_ms)], resname, data_column="DATA-MODEL_DATA", channels_per_image=1, **imaging)
        dt = xr.open_datatree(resname + "_I.dt", engine="zarr", chunks=None)
        try:
            bands = sorted(n for n in dt.children if n.startswith("band"))
            peaks[spectral] = max(
                float(np.abs(dt[n].ds.DIRTY[0].values).max() / dt[n].ds.WSUM.values[0]) for n in bands
            )
        finally:
            dt.close()

    assert peaks["linear"] < peaks["flat"], f"per-channel residual peaks {peaks}"


def test_degrid_flat_checks_only_the_frequencies_it_evaluates(degrid_ms, simple_mds, tmp_path, monkeypatch):
    """Flat mode evaluates the beam at chunk means only, so it must not refuse a run over an
    edge channel it never evaluates."""
    degrid_mod = _no_ray(monkeypatch)
    checked = []

    class _Recorder:
        def __init__(self, **kw):
            pass

        def check_freqs(self, freq):
            checked.append(np.asarray(freq))
            raise ValueError("stop after the check")

    monkeypatch.setattr(degrid_mod, "BeamWizard", _Recorder)
    with pytest.raises(ValueError, match="stop after"):
        degrid_mod.degrid(
            [degrid_ms],
            str(tmp_path / "out"),
            channels_per_chunk=8,
            mds=simple_mds[0],
            product="I",
            beam_model="L",
            log_directory=str(tmp_path / "logs"),
        )
    # chunks follow the model's bands at 1.0 and 1.1 GHz, then one out-of-band run held at its
    # mean: never the 1.7 GHz edge channel as such
    np.testing.assert_allclose(np.unique(np.concatenate(checked)), [1.0e9, 1.1e9, 1.45e9])


def test_degrid_region_flat_holds_the_band_frequency(degrid_ms, tmp_path):
    """Flat mode renders the model, and evaluates the beam, at the weighted frequency of the band
    holding the chunk: the `freq_out` deconv used for that band, where the tent returns the band
    value exactly (D28, D59). Over one band's channels it is deconv's own forward model."""
    tent = _spectral_mds(degrid_ms, tmp_path / "tent.mds", "tent")
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, tent)
    try:
        region = {"time": slice(0, 6), "frequency": slice(0, 4)}
        expected = _direct_reference(node_ds, node, mds_ds, region, True, freq_out=1.12e9).astype(np.complex64)
        got = _region_vis(node_ds, node, mds_ds)
        wizard = _FakeWizard()
        _region_vis(node_ds, node, mds_ds, wizards={"all": wizard})
    finally:
        dt.close()
    np.testing.assert_array_equal(got.reshape(expected.shape), expected)
    assert [c["freq"] for c in wizard.calls] == [pytest.approx(1.12e9)]


@pytest.mark.parametrize(
    "chan, expected",
    [
        (slice(0, 2), 1.12e9),  # centre 1.05 GHz, first band
        (slice(2, 4), 1.12e9),  # a narrower chunk of the same band holds the same frequency
        (slice(4, 6), 1.53e9),  # second band
        (slice(6, 8), 1.65e9),  # centred beyond every band: the chunk's own mean
    ],
)
def test_flat_freq(degrid_ms, tmp_path, chan, expected):
    from pfb_imaging.utils.degrid import flat_freq

    bounds = np.array([[0.95e9, 1.35e9], [1.35e9, 1.55e9]])
    _, mds_ds = _spectral_mds(degrid_ms, tmp_path / "tent.mds", "tent", bounds=bounds)
    freq = 1.0e9 + 1.0e8 * np.arange(8)
    assert flat_freq(mds_ds, freq[chan]) == pytest.approx(expected)
    assert flat_freq(mds_ds, freq[chan][::-1]) == pytest.approx(expected)


def test_flat_freq_without_band_bounds(simple_mds):
    """A model that records no band edges falls back to the chunk's unweighted mean (D39)."""
    from pfb_imaging.utils.degrid import flat_freq

    ds = simple_mds[1].drop_vars("freq_bounds")
    assert flat_freq(ds, np.array([1.0e9, 1.1e9])) == pytest.approx(1.05e9)


@pytest.mark.parametrize(
    "spectral, chan, expected",
    [
        ("flat", slice(0, 4), [1.12e9]),
        ("linear", slice(0, 4), [1.0e9, 1.3e9]),
        ("linear", slice(3, 4), [1.3e9]),  # one channel has nothing to interpolate
    ],
)
def test_chunk_eval_freqs(degrid_ms, tmp_path, spectral, chan, expected):
    from pfb_imaging.utils.degrid import chunk_eval_freqs

    _, mds_ds = _spectral_mds(degrid_ms, tmp_path / "tent.mds", "tent")
    freq = 1.0e9 + 1.0e8 * np.arange(8)
    np.testing.assert_allclose(chunk_eval_freqs(mds_ds, freq[chan], spectral), expected)
    np.testing.assert_allclose(chunk_eval_freqs(mds_ds, freq[chan][::-1], spectral), expected)


def test_degrid_region_linear_is_exact_for_a_linear_spectrum(degrid_ms, simple_mds, monkeypatch):
    """`simple_mds` is linear in frequency everywhere, so interpolating between the chunk's
    end channels reproduces every channel, and the model is rendered at the two ends only."""
    from pfb_model_spec.utils import degrid as kernel

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        ref = _per_channel(node_ds, node, mds_ds)
        rendered = []
        render = kernel.render_model_region

        def spy(model_ds, **kw):
            rendered.append(kw["freq_out"])
            return render(model_ds, **kw)

        monkeypatch.setattr(kernel, "render_model_region", spy)
        got = _region_vis(node_ds, node, mds_ds, spectral="linear")
    finally:
        dt.close()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6 * np.abs(ref).max())
    np.testing.assert_allclose(sorted(rendered), [1.0e9, 1.3e9])


def test_degrid_region_linear_across_a_kink_beats_flat(degrid_ms, tmp_path):
    """A chunk spanning two tent pieces is no longer refused: the chord between its end
    channels misses the kink, but by far less than holding the chunk flat."""
    tent = _spectral_mds(degrid_ms, tmp_path / "tent.mds", "tent")
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, tent)
    try:
        region = {"time": slice(0, 6), "frequency": slice(0, 8)}
        ref = _per_channel(node_ds, node, mds_ds, nchan=8)
        got = _region_vis(node_ds, node, mds_ds, region=region, spectral="linear")
        flat = _region_vis(node_ds, node, mds_ds, region=region)
    finally:
        dt.close()
    assert np.abs(got - ref).max() < 0.5 * np.abs(flat - ref).max()


def test_degrid_region_linear_evaluates_the_beam_at_the_end_channels(degrid_ms, simple_mds):
    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        wizard = _FakeWizard()
        _region_vis(node_ds, node, mds_ds, spectral="linear", wizards={"all": wizard})
    finally:
        dt.close()
    assert sorted(c["freq"] for c in wizard.calls) == [pytest.approx(1.0e9), pytest.approx(1.3e9)]


def test_degrid_linear_runs_through_the_driver(degrid_ms, tmp_path, pctable):
    import pfb_imaging.core.degrid as degrid_mod

    mds_path, _ = _spectral_mds(degrid_ms, tmp_path / "tent.mds", "tent")
    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.zeros((21060, 8, 4), np.complex64))
    degrid_mod.degrid(
        [degrid_ms],
        str(tmp_path / "out"),
        channels_per_chunk=8,
        mds=mds_path,
        product="I",
        spectral="linear",
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )
    with pctable(degrid_ms, ack=False) as tab:
        assert np.any(np.asarray(tab.getcol("MODEL_DATA")) != 0)


@pytest.fixture(scope="module")
def deconv_chain(sky_truth, ms_name, tmp_path_factory):
    """imager (3 uneven bands) -> deconv -> `.mds` -> degrid into `MODEL_DATA`, flat, default chunks.

    No beam anywhere, so degrid predicts exactly deconv's forward model (D59). Several bands,
    so the `.mds` has a real spectrum: a single-band fit is constant in frequency, which
    would make holding the band's weighted frequency vacuous. Eight channels at three per
    image give bands of 3, 3 and 2 channels, which equal-width band edges would misplace,
    and degrid's default chunk (the narrowest band, 2 channels) splits the 3-channel bands
    in two: every chunk must still be held at its own band's frequency. Tests only read the
    MS and the deconvolved tree.
    """
    from pfb_imaging.core.deconv import deconv as deconv_core
    from pfb_imaging.core.degrid import degrid
    from pfb_imaging.core.imager import imager as imager_core
    from tests.conftest import GT_DECONV_KW, GT_IMAGER_KW

    tmp = tmp_path_factory.mktemp("deconv_chain")
    work_ms = tmp / "resid.ms"
    shutil.copytree(ms_name, work_ms)
    imaging = {
        **GT_IMAGER_KW,
        "channels_per_image": 3,
        "nx": sky_truth.nx,
        "ny": sky_truth.ny,
        "cell_size": sky_truth.cell_size,
    }
    base = str(tmp / "img")
    imager_core([Path(work_ms)], base, **imaging)
    deconv_core(base, **{**GT_DECONV_KW, "nworkers": 1, "niter": 2})
    (mds_path,) = tmp.glob("*_main.mds")
    degrid(
        [Path(work_ms)],
        str(tmp / "pred"),
        None,  # the default: the narrowest band's width, cut at the band edges
        mds=str(mds_path),
        product="I",
        spectral="flat",
        integrations_per_chunk=-1,
        epsilon=1e-7,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp / "logs"),
    )
    return SimpleNamespace(
        tmp=tmp,
        ms=work_ms,
        base=base,
        mds=mds_path,
        imaging=imaging,
    )


def _band_vars(base, names):
    """`{name: (nband, ny, nx)}` for each band-node variable, plus `WSUM` and `freq_out`."""
    dt = xr.open_datatree(f"{base}_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        out = {name: np.stack([dt[n].ds[name].values[0] for n in nodes]) for name in names}
        out["WSUM"] = np.array([float(dt[n].ds.WSUM.values[0]) for n in nodes])
        out["freq_out"] = np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes])
    finally:
        dt.close()
    return out


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_imaged_residual_matches_deconv(deconv_chain, ms_meta):
    """Imaging `DATA-MODEL_DATA` reproduces the residual deconv computes (D59).

    deconv's `RESIDUAL` is `DIRTY - sum_p G^T W G (B_p * MODEL)`, with each band's model held
    at the band's weighted `freq_out` over all of the band's channels. degrid predicts the
    same thing when there is no beam, one chunk spans each band, and flat mode holds the
    chunk at the band's weighted frequency, which the `.mds` records. Imaging the difference
    with the same weighting then grids the same residual visibilities, so the two images
    agree to gridder precision, which makes this a direct check of degrid's accuracy.
    """
    from pfb_model_spec.utils.degrid import render_model_region

    from pfb_imaging.core.imager import imager as imager_core

    chain = deconv_chain
    ref = _band_vars(chain.base, ["DIRTY", "RESIDUAL"])
    assert ref["DIRTY"].shape[0] == 3
    res = str(chain.tmp / "res")
    imager_core([Path(chain.ms)], res, data_column="DATA-MODEL_DATA", **chain.imaging)
    got = _band_vars(res, ["DIRTY"])
    np.testing.assert_allclose(got["WSUM"], ref["WSUM"], rtol=1e-6)

    tol = 1e-5 * np.abs(ref["DIRTY"]).max()
    # not vacuous: the model removed most of the dirty image ...
    assert np.abs(ref["RESIDUAL"]).max() < 0.2 * np.abs(ref["DIRTY"]).max()
    np.testing.assert_allclose(got["DIRTY"], ref["RESIDUAL"], rtol=0, atol=tol)

    # ... and holding a band at its channels' plain mean instead would shift its residual by
    # far more than the tolerance (the flagged channel moves the first band's weighted
    # frequency well off its plain mean), so the frequency condition is not vacuous
    model_ds = xr.open_zarr(chain.mds)
    time = float(model_ds.times.values[0])
    shift = []
    bounds = model_ds.freq_bounds.values
    for b, f in enumerate(ref["freq_out"]):
        mean = float(ms_meta.freq[(bounds[b, 0] < ms_meta.freq) & (ms_meta.freq < bounds[b, 1])].mean())
        held = render_model_region(model_ds, time=time, freq_out=f)
        at_mean = render_model_region(model_ds, time=time, freq_out=mean)
        # a point source of flux d puts d * wsum into the unnormalised dirty peak
        shift.append(np.abs(held - at_mean).max() * ref["WSUM"][b])
    assert max(shift) > 100 * tol


_UNEVEN = np.array([[0.95e9, 1.25e9], [1.25e9, 1.75e9]])  # 3 and 5 of the MS's 8 channels


@pytest.mark.parametrize(
    "fstep, bounds, expected",
    [
        (4, None, [slice(0, 4), slice(4, 8)]),  # no bounds: fixed blocks from channel 0
        (3, None, [slice(0, 3), slice(3, 6), slice(6, 8)]),
        (4, _UNEVEN, [slice(0, 3), slice(3, 6), slice(6, 8)]),  # cut at the band edge, 5 split 3 + 2
        (8, _UNEVEN, [slice(0, 3), slice(3, 8)]),  # never across a band edge
        (8, np.array([[1.15e9, 1.45e9]]), [slice(0, 2), slice(2, 5), slice(5, 8)]),  # out-of-band runs
    ],
)
def test_frequency_chunks(fstep, bounds, expected):
    from pfb_imaging.utils.degrid import frequency_chunks

    freq = 1.0e9 + 1.0e8 * np.arange(8)
    assert frequency_chunks(freq, fstep, bounds) == expected


def test_frequency_chunks_descending_axis():
    from pfb_imaging.utils.degrid import frequency_chunks

    freq = (1.0e9 + 1.0e8 * np.arange(8))[::-1]
    assert frequency_chunks(freq, 8, _UNEVEN) == [slice(0, 5), slice(5, 8)]


def test_band_labels():
    from pfb_imaging.utils.degrid import band_labels

    freq = 1.0e9 + 1.0e8 * np.arange(8)
    np.testing.assert_array_equal(band_labels(freq, _UNEVEN), [0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(band_labels(freq, _UNEVEN[:1]), [0, 0, 0, -1, -1, -1, -1, -1])


def test_work_items_follow_the_model_bands_and_shift_by_chan0():
    """A --freq-range selection is cut at the bands of the selected channels, then shifted by chan0."""
    from pfb_imaging.core.degrid import _work_items
    from pfb_imaging.utils.msv4 import SelectedNode

    node = SelectedNode(
        path="/p",
        chan0=2,
        nchan=6,
        chan_width=1.0e8,
        ntime=10,
        field_name="f",
        spw_name="s",
        scan_name="0",
        field_radec=np.zeros(2),
        corr_types=("XX", "YY"),
        freq=1.2e9 + 1.0e8 * np.arange(6),
    )
    regions = [w.region["frequency"] for w in _work_items(0, node, -1, 8, freq_bounds=_UNEVEN)]
    assert regions == [slice(2, 3), slice(3, 8)]


def test_degrid_chunks_follow_the_model_bands(degrid_ms, tmp_path, pctable, monkeypatch):
    """The driver hands the model's band edges to the chunking, so no chunk straddles two bands."""
    import pfb_imaging.core.degrid as degrid_mod

    mds_path, _ = _spectral_mds(degrid_ms, tmp_path / "tent.mds", "tent", bounds=_UNEVEN)
    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.zeros((21060, 8, 4), np.complex64))
    recorded = []
    chunks = degrid_mod.frequency_chunks

    def spy(*args, **kw):
        out = chunks(*args, **kw)
        recorded.append(out)
        return out

    monkeypatch.setattr(degrid_mod, "frequency_chunks", spy)
    degrid_mod.degrid(
        [degrid_ms],
        str(tmp_path / "out"),
        channels_per_chunk=4,
        mds=mds_path,
        product="I",
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )
    assert recorded and all(r == [slice(0, 3), slice(3, 6), slice(6, 8)] for r in recorded)
