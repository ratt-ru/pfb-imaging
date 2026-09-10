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
    import xarray as xr

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
    import xarray as xr

    from pfb_imaging.utils.degrid_msv4 import ensure_model_columns
    from pfb_imaging.utils.msv4 import get_engine
    from tests.conftest import drop_column

    drop_column(degrid_ms, "MODEL_DATA")
    with arcae.table(degrid_ms) as tab:
        assert "MODEL_DATA" not in tab.columns()

    dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
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
    import xarray as xr

    from pfb_imaging.utils.degrid_msv4 import ensure_model_columns
    from pfb_imaging.utils.msv4 import get_engine

    dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
    try:
        ensure_model_columns(degrid_ms, dt, ["MODEL_DATA1"])
    finally:
        dt.close()

    with arcae.table(degrid_ms) as tab:
        assert "MODEL_DATA1" in tab.columns()
        assert np.asarray(tab.getcol("MODEL_DATA1")).shape == (21060, 8, 4)


def test_ensure_model_columns_is_idempotent(degrid_ms):
    """The common case is an MS that already has MODEL_DATA. Must be a no-op."""
    import arcae
    import xarray as xr
    from casacore.tables import table as pctable

    from pfb_imaging.utils.degrid_msv4 import ensure_model_columns
    from pfb_imaging.utils.msv4 import get_engine

    with pctable(degrid_ms, readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.full((21060, 8, 4), 7 + 3j, np.complex64))

    for _ in range(2):
        dt = xr.open_datatree(degrid_ms, **get_engine(degrid_ms))
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
    import xarray as xr

    from pfb_imaging.utils.degrid_msv4 import MODEL_DIMS, make_column_placeholder

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
    from pfb_imaging.utils.degrid_msv4 import build_region_masks

    _, ds = simple_mds
    masks = build_region_masks(ds, None)
    assert len(masks) == 1
    assert masks[0].shape == (64, 32)
    assert np.all(masks[0] == 1.0)


def test_build_region_masks_is_x_major_on_a_non_square_grid(simple_mds, tmp_path):
    """The mask must come back (nx, ny), matching the x-major model.

    astropy `regions` renders to `(Y, X)`; the model is x-major. On a square
    grid a missing transpose is invisible, which is why this grid is 64 x 32:
    a `(Y, X)` mask would not even have the right shape, and a mask that had
    the right shape but the wrong orientation would select the wrong pixel.
    The region is placed on the component at (x=40, y=9).
    """
    from pfb_imaging.utils.degrid_msv4 import build_region_masks

    _, ds = simple_mds
    nx, ny = 64, 32
    # ds9 image coordinates are 1-based and (x, y) ordered
    region_file = tmp_path / "one.reg"
    region_file.write_text("image\nbox(41,10,3,3,0)\n")

    masks = build_region_masks(ds, str(region_file))
    assert len(masks) == 2, "remainder first, then one mask per region"
    remainder, region = masks
    assert remainder.shape == (nx, ny)
    assert region.shape == (nx, ny)

    # the region covers the component's pixel and the remainder does not
    assert region[40, 9] == 1.0
    assert remainder[40, 9] == 0.0
    # and they partition the grid
    np.testing.assert_array_equal(remainder + region, np.ones((nx, ny)))
    # a 3x3 box, so exactly 9 pixels
    assert region.sum() == 9.0


def test_build_region_masks_refuses_overlapping_regions(simple_mds, tmp_path):
    """Overlaps would double-count flux across columns; refuse, as today."""
    from pfb_imaging.utils.degrid_msv4 import build_region_masks

    _, ds = simple_mds
    region_file = tmp_path / "two.reg"
    region_file.write_text("image\nbox(41,10,5,5,0)\nbox(42,11,5,5,0)\n")

    with pytest.raises(ValueError, match="[Oo]verlapping"):
        build_region_masks(ds, str(region_file))


def _node_and_mds(ms_path, mds_tuple):
    """Open the MS's single visibility node and the model dataset."""
    import xarray as xr

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
    from pfb_imaging.utils.degrid_msv4 import MODEL_DIMS, build_region_masks, degrid_region

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
    from pfb_model_spec.utils.degrid import model_to_apparent_vis_for_region

    from pfb_imaging.utils.degrid_msv4 import build_region_masks, degrid_region

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        region = {"time": slice(5, 15), "frequency": slice(2, 6)}
        sub = node_ds.isel(**region)
        uvw = np.nan_to_num(sub.UVW.values.reshape(-1, 3))
        freq = sub.frequency.values
        valid = ~np.isnan(sub.UVW.values.reshape(-1, 3)).any(axis=-1)
        mask = np.ascontiguousarray(np.broadcast_to(valid[:, None], (uvw.shape[0], freq.size)).astype(np.uint8))
        expected = model_to_apparent_vis_for_region(
            mds_ds,
            uvw=uvw,
            freq=freq,
            corr_types=node.corr_types,
            time=float(sub.time.values.mean()),
            freq_out=float(freq.mean()),
            mask=mask,
            region_mask=None,
            divide_by_n=False,
        ).astype(np.complex64)

        out = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, None),
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
        )
        got = out.MODEL_DATA.values.reshape(-1, freq.size, len(node.corr_types))
        np.testing.assert_array_equal(got, expected)
    finally:
        dt.close()


def test_degrid_region_samples_the_model_spectrum_per_chunk(degrid_ms, simple_mds):
    """Narrower frequency chunks sample the model's spectrum more finely.

    The model is a continuous function of frequency, re-rendered once per
    chunk, so this is the feature `--channels-per-chunk` exists for. A single
    point source degrids to a visibility of constant magnitude equal to its
    flux (`divide_by_n=False`, so no n-term scaling), which makes the assertion
    direct: |vis| per chunk == the model's flux at that chunk's mean frequency.
    """
    from pfb_model_spec.utils.degrid import render_model_region

    from pfb_imaging.utils.degrid_msv4 import build_region_masks, degrid_region

    dt, node_ds, node, mds_ds = _node_and_mds(degrid_ms, simple_mds)
    try:
        masks = build_region_masks(mds_ds, None)
        amplitudes = []
        for chan in (slice(0, 4), slice(4, 8)):
            region = {"time": slice(0, 5), "frequency": chan}
            out = degrid_region(
                node_ds,
                region=region,
                model_ds=mds_ds,
                masks=masks,
                columns=["MODEL_DATA"],
                corr_types=node.corr_types,
            )
            freq_out = float(node_ds.frequency.values[chan].mean())
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
    from pfb_imaging.utils.degrid_msv4 import build_region_masks, degrid_region

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
        ).MODEL_DATA.values

        assert np.isfinite(padded).all(), "NaN UVW leaked into the model column"
        assert np.all(padded[2, ::5] == 0), "padded cells must degrid to zero"
        untouched = np.ones(padded.shape[1], bool)
        untouched[::5] = False
        np.testing.assert_array_equal(padded[2, untouched], clean[2, untouched])
    finally:
        dt.close()


def test_degrid_region_accumulates_onto_the_existing_column(degrid_ms, simple_mds):
    """`--accumulate` adds to what is already in the column, within the region."""
    from casacore.tables import table as pctable

    from pfb_imaging.utils.degrid_msv4 import build_region_masks, degrid_region

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
        )
        plain = degrid_region(node_ds, **kw).MODEL_DATA.values
        acc = degrid_region(node_ds, accumulate=True, **kw).MODEL_DATA.values
        np.testing.assert_allclose(acc, plain + np.complex64(1 + 1j), rtol=1e-6)
    finally:
        dt.close()


def test_degrid_region_masks_split_flux_across_columns(degrid_ms, simple_mds, tmp_path):
    """Region masks partition the model, so the columns must sum to the whole."""
    from pfb_imaging.utils.degrid_msv4 import build_region_masks, degrid_region

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
        ).MODEL_DATA.values
        split = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, str(region_file)),
            columns=["MODEL_DATA", "MODEL_DATA1"],
            corr_types=node.corr_types,
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
    import xarray as xr

    from pfb_imaging.utils.degrid_msv4 import assert_writable

    ds = xr.Dataset({"MODEL_DATA": (("time",), np.zeros(4, np.complex64))})
    with pytest.raises(ValueError, match="encoding"):
        assert_writable(ds)

    ds.encoding = {"common_store_args": {}, "partition_key": ()}
    assert_writable(ds)  # must not raise
