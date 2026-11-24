"""Pass-2 per-partition gridding (casacore-free, synthetic data)."""

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.core.imager import _concat_pieces
from pfb_imaging.operators.gridder import grid_partition, residual_from_partitions
from pfb_imaging.utils.stokes2vis import partition_beam
from pfb_imaging.utils.weighting import _compute_counts, sparse_counts


def _synth_partition(nrow=200, seed=0, nx=16, ny=16):
    """A synthetic single-correlation partition with spread-out uvw.

    BEAM is on the output image grid (placed in pass 2, #365), so callers
    must pass the same nx/ny to grid_partition.
    """
    rng = np.random.default_rng(seed)
    uvw = rng.standard_normal((nrow, 3)) * 100.0
    freq = np.array([1.0e9])
    vis = rng.standard_normal((1, nrow, 1)) + 1j * rng.standard_normal((1, nrow, 1))
    wgt = np.abs(rng.standard_normal((1, nrow, 1))) + 0.1
    mask = np.ones((nrow, 1), dtype=np.uint8)
    beam = np.ones((1, ny, nx))
    return xr.Dataset(
        {
            "VIS": (("corr", "row", "chan"), vis),
            "WEIGHT": (("corr", "row", "chan"), wgt),
            "MASK": (("row", "chan"), mask),
            "UVW": (("row", "three"), uvw),
            "FREQ": (("chan",), freq),
            "BEAM": (("corr", "y", "x"), beam),
        },
        coords={"corr": ["I"]},
    )


def test_grid_partition_shapes_and_wsum():
    """Non-square on purpose: output arrays are (Y, X)-ordered (wiki D19)."""
    part = _synth_partition(nx=16, ny=12)
    out = grid_partition(part, None, nx=16, ny=12, nx_psf=32, ny_psf=24, cell_rad=1.0e-6, robustness=None)
    assert out["DIRTY"].shape == (1, 12, 16)
    assert out["PSF"].shape == (1, 24, 32)
    assert out["PSFHAT"].shape == (1, 24, 32 // 2 + 1)
    assert out["BEAM"].shape == (1, 12, 16)
    assert out["WSUM"].shape == (1,)
    expected = (part.WEIGHT.values[0] * part.MASK.values).sum()
    np.testing.assert_allclose(out["WSUM"][0], expected, rtol=1e-6)
    assert np.isfinite(out["DIRTY"]).all()


def test_grid_partition_no_psf():
    """do_psf=False skips the PSF products entirely but keeps DIRTY/BEAM/WSUM."""
    part = _synth_partition(nx=16, ny=12)
    out = grid_partition(part, None, nx=16, ny=12, nx_psf=32, ny_psf=24, cell_rad=1.0e-6, robustness=None, do_psf=False)
    for k in ("PSF", "PSFHAT", "PSFPARSN"):
        assert k not in out
    for k in ("DIRTY", "BEAM", "WSUM", "WEIGHT"):
        assert k in out
    # DIRTY identical to the default path
    ref = grid_partition(part, None, nx=16, ny=12, nx_psf=32, ny_psf=24, cell_rad=1.0e-6, robustness=None)
    np.testing.assert_array_equal(out["DIRTY"], ref["DIRTY"])


def test_grid_partition_row_additivity():
    """Gridding is linear over rows: cat(p0, p1) dirty == dirty(p0) + dirty(p1).

    This is exactly the property that makes the sum-over-partitions Hessian
    correct for partitions sharing a phase centre and beam.
    """
    p0 = _synth_partition(nrow=120, seed=0)
    p1 = _synth_partition(nrow=80, seed=1)
    cat = xr.concat([p0[["VIS", "WEIGHT", "MASK", "UVW"]], p1[["VIS", "WEIGHT", "MASK", "UVW"]]], dim="row")
    cat = cat.assign(FREQ=p0.FREQ, BEAM=p0.BEAM)

    kw = dict(nx=16, ny=16, nx_psf=32, ny_psf=32, cell_rad=1.0e-6, robustness=None)
    o_cat = grid_partition(cat, None, **kw)
    o0 = grid_partition(p0, None, **kw)
    o1 = grid_partition(p1, None, **kw)

    np.testing.assert_allclose(o_cat["DIRTY"], o0["DIRTY"] + o1["DIRTY"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(o_cat["PSF"], o0["PSF"] + o1["PSF"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(o_cat["WSUM"], o0["WSUM"] + o1["WSUM"], rtol=1e-6)


def test_grid_partition_robust_reweights():
    part = _synth_partition()
    nx_pad = ny_pad = 32
    # counts must match the convention used inside grid_partition:
    # wgridder_conventions(0,0) -> flip_u=False (usign=-1), flip_v=True (vsign=+1)
    counts = _compute_counts(
        part.UVW.values,
        part.FREQ.values,
        part.MASK.values,
        part.WEIGHT.values,
        nx_pad,
        ny_pad,
        1.0e-6,
        1.0e-6,
        part.WEIGHT.values.dtype,
        usign=-1.0,
        vsign=1.0,
    )
    out = grid_partition(
        part,
        counts,
        nx=16,
        ny=16,
        nx_psf=32,
        ny_psf=32,
        cell_rad=1.0e-6,
        robustness=-2.0,
        counts_nx=nx_pad,
        counts_ny=ny_pad,
    )
    assert out["WEIGHT"].shape == part.WEIGHT.shape
    # robust/uniform weighting downweights dense cells -> never exceeds the natural max
    assert out["WEIGHT"].max() <= part.WEIGHT.values.max() + 1e-9
    # natural-weight call leaves weights untouched
    nat = grid_partition(part, None, nx=16, ny=16, nx_psf=32, ny_psf=32, cell_rad=1.0e-6, robustness=None)
    np.testing.assert_allclose(nat["WEIGHT"], part.WEIGHT.values)


def _image_beam_partition(nx, ny, nrow=200, seed=0, beam_val=1.0, l0=0.0, m0=0.0):
    """A partition with the beam already on the image grid (as stored in pass 2)."""
    rng = np.random.default_rng(seed)
    uvw = rng.standard_normal((nrow, 3)) * 100.0
    freq = np.array([1.0e9])
    wgt = np.abs(rng.standard_normal((1, nrow, 1))) + 0.1
    mask = np.ones((nrow, 1), dtype=np.uint8)
    beam = np.full((1, ny, nx), float(beam_val))
    return xr.Dataset(
        {
            "WEIGHT": (("corr", "row", "chan"), wgt),
            "MASK": (("row", "chan"), mask),
            "UVW": (("row", "three"), uvw),
            "FREQ": (("chan",), freq),
            "BEAM": (("corr", "y", "x"), beam),
        },
        coords={"corr": ["I"]},
        attrs={"l0": l0, "m0": m0},
    )


def test_residual_zero_model_returns_dirty():
    nx, ny = 16, 12  # non-square: residual path is (Y, X)-ordered
    part = _image_beam_partition(nx, ny, seed=0)
    dirty = np.random.default_rng(5).standard_normal((1, ny, nx))
    model = np.zeros((1, ny, nx))
    res = residual_from_partitions(dirty, [part], model, cell_rad=1.0e-6)
    np.testing.assert_allclose(res, dirty, atol=1e-12)


def test_residual_partition_additivity():
    """convim summed over [p0, p1] equals convim(p0) + convim(p1)."""
    nx = ny = 16
    p0 = _image_beam_partition(nx, ny, nrow=120, seed=0)
    p1 = _image_beam_partition(nx, ny, nrow=80, seed=1)
    rng = np.random.default_rng(7)
    dirty = rng.standard_normal((1, ny, nx))
    model = rng.standard_normal((1, ny, nx))
    c01 = dirty - residual_from_partitions(dirty, [p0, p1], model, 1.0e-6)
    c0 = dirty - residual_from_partitions(dirty, [p0], model, 1.0e-6)
    c1 = dirty - residual_from_partitions(dirty, [p1], model, 1.0e-6)
    np.testing.assert_allclose(c01, c0 + c1, rtol=1e-5, atol=1e-8)


def test_residual_beam_applied_once():
    """Doubling the beam doubles the model term (beam applied once on degrid side)."""
    nx = ny = 16
    p1 = _image_beam_partition(nx, ny, seed=0, beam_val=1.0)
    p2 = _image_beam_partition(nx, ny, seed=0, beam_val=2.0)
    rng = np.random.default_rng(9)
    dirty = np.zeros((1, ny, nx))
    model = rng.standard_normal((1, ny, nx))
    c1 = dirty - residual_from_partitions(dirty, [p1], model, 1.0e-6)
    c2 = dirty - residual_from_partitions(dirty, [p2], model, 1.0e-6)
    np.testing.assert_allclose(c2, 2.0 * c1, rtol=1e-5, atol=1e-8)


def test_residual_gradient_beam_applied_twice():
    """The gradient residual applies the partition beam again on the regrid
    side: with a constant beam b the model term scales as b^2 (vs b for the
    apparent residual), and with distinct per-partition beams the gradient is
    the exact per-partition sum, not a band-average approximation."""
    nx = ny = 16
    rng = np.random.default_rng(11)
    model = rng.standard_normal((1, ny, nx))
    zero = np.zeros((1, ny, nx))
    p1 = _image_beam_partition(nx, ny, seed=0, beam_val=1.0)
    p2 = _image_beam_partition(nx, ny, seed=0, beam_val=2.0)

    # constant beam: apparent model term ~ b, gradient model term ~ b^2
    r1, g1 = residual_from_partitions(zero, [p1], model, 1.0e-6, bdirty=zero)
    r2, g2 = residual_from_partitions(zero, [p2], model, 1.0e-6, bdirty=zero)
    np.testing.assert_allclose(r2, 2.0 * r1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(g2, 4.0 * g1, rtol=1e-5, atol=1e-8)

    # distinct beams: both outputs are per-partition sums over the terms above
    dirty = rng.standard_normal((1, ny, nx))
    bdirty = rng.standard_normal((1, ny, nx))
    r12, g12 = residual_from_partitions(dirty, [p1, p2], model, 1.0e-6, bdirty=bdirty)
    np.testing.assert_allclose(r12, dirty + r1 + r2, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(g12, bdirty + g1 + g2, rtol=1e-5, atol=1e-8)

    # bdirty=None keeps the legacy single-return signature
    r_only = residual_from_partitions(dirty, [p1, p2], model, 1.0e-6)
    np.testing.assert_allclose(r_only, r12, rtol=0, atol=0)


def _synth_piece(freq_out, wsum_nat, beam_val, nrow=10, nbeam=6, seed=0, beam=None):
    """A minimal scratch piece carrying only what _concat_pieces touches.

    The beam is field-centred (BEAM_SMALL on its own l/m grid, deg), as pass 1
    stores it; pass 2 places it on the image grid (partition_beam).
    """
    rng = np.random.default_rng(seed)
    lm = np.linspace(-1.0, 1.0, nbeam)
    small = np.full((1, nbeam, nbeam), float(beam_val)) if beam is None else beam
    return xr.Dataset(
        {
            "VIS": (("corr", "row", "chan"), rng.standard_normal((1, nrow, 1)) + 0j),
            "WEIGHT": (("corr", "row", "chan"), np.ones((1, nrow, 1))),
            "MASK": (("row", "chan"), np.ones((nrow, 1), dtype=np.uint8)),
            "UVW": (("row", "three"), rng.standard_normal((nrow, 3))),
            "FREQ": (("chan",), np.array([1.0e9])),
            "BEAM_SMALL": (("corr", "m_beam", "l_beam"), small),
        },
        coords={"corr": ["I"], "l_beam": lm, "m_beam": lm},
        attrs={"freq_out": float(freq_out), "wsum_img": float(wsum_nat), "ra0": 1.0, "dec0": -0.5},
    )


def test_concat_pieces_single_is_identity():
    p = _synth_piece(1.0e9, 5.0, 0.5)
    assert _concat_pieces([p]) is p


def test_concat_pieces_weighted_beam_and_freq():
    """Pieces of a partition may differ in beam and effective frequency; both
    reduce as wsum_img-weighted means. Taking piece 0's beam (the old
    behaviour) silently discarded the rest -- issue #296, wiki D28.
    """
    a = _synth_piece(1.0e9, 3.0, 1.0, nrow=10, seed=1)
    b = _synth_piece(1.4e9, 1.0, 5.0, nrow=6, seed=2)

    out = _concat_pieces([a, b])

    assert out.sizes["row"] == 16
    np.testing.assert_allclose(out.attrs["freq_out"], (3.0 * 1.0e9 + 1.0 * 1.4e9) / 4.0)
    np.testing.assert_allclose(out.attrs["wsum_img"], 4.0)
    np.testing.assert_allclose(out.BEAM_SMALL.values, (3.0 * 1.0 + 1.0 * 5.0) / 4.0)


def test_concat_pieces_rejects_mismatched_beam_grids():
    a = _synth_piece(1.0e9, 1.0, 1.0)
    b = _synth_piece(1.0e9, 1.0, 1.0).assign_coords(l_beam=np.linspace(-2.0, 2.0, 6))
    with pytest.raises(AssertionError):
        _concat_pieces([a, b])


@pytest.mark.parametrize("offset", [False, True], ids=["single-field", "reprojected"])
def test_placing_the_mean_small_beam_equals_the_mean_of_placed_beams(offset):
    """Pass 2 averages a partition's small beams, then places the mean once (#365).

    Both placements -- bilinear for a single field and SIN->SIN reprojection
    for a pointing off the tangent point -- are linear in the beam on a shared
    grid, so this equals placing each piece and averaging, which is what pass 1
    used to store. Pinned on distinct, asymmetric beams so the weights matter.
    """
    rng = np.random.default_rng(42)
    nb = 33
    lm = np.linspace(-1.0, 1.0, nb)
    mm, ll = np.meshgrid(lm, lm, indexing="ij")
    beams = [
        np.exp(-((ll - 0.1 * k) ** 2 + (mm + 0.05 * k) ** 2) / 0.3)[None] + 0.01 * rng.random((1, nb, nb))
        for k in range(3)
    ]
    w = [3.0, 1.0, 2.0]
    pieces = [_synth_piece(1.0e9, wi, 0.0, nbeam=nb, seed=k, beam=b) for k, (wi, b) in enumerate(zip(w, beams))]
    radec_grid = np.array([1.0, -0.5 + (np.deg2rad(0.2) if offset else 0.0)])
    cell, nx, ny = np.deg2rad(0.02), 48, 40

    placed = partition_beam(_concat_pieces(pieces), radec_grid, cell, nx, ny)
    expected = sum(wi * partition_beam(p, radec_grid, cell, nx, ny) for wi, p in zip(w, pieces)) / sum(w)

    assert placed.shape == (1, ny, nx)
    assert np.abs(expected).max() > 0.5  # the beam lands on the grid
    np.testing.assert_allclose(placed, expected, rtol=1e-12, atol=1e-14)


def test_concat_pieces_rejects_mismatched_freq():
    a = _synth_piece(1.0e9, 1.0, 1.0)
    b = _synth_piece(1.0e9, 1.0, 1.0).assign(FREQ=(("chan",), np.array([2.0e9])))
    with pytest.raises(AssertionError):
        _concat_pieces([a, b])


def test_loaded_piece_does_not_pin_data_on_the_scratch_tree(tmp_path):
    """Dropping a loaded piece frees it while the scratch tree is still open (#339).

    `.load()` fills the Variables a Dataset shares with its tree, so a piece
    loaded straight off the tree stayed reachable from it -- and the tree sits
    in a reference cycle that survives into the worker's next task, carrying
    vis-sized pieces with it.
    """
    import gc
    import tracemalloc

    import xarray as xr

    from pfb_imaging.core.imager import _load_piece

    store = str(tmp_path / "s.scratch")
    nrow = 2**18  # 4 MiB of complex128 VIS
    xr.Dataset(
        {
            "VIS": (("corr", "row", "chan"), np.ones((1, nrow, 2), dtype=np.complex128)),
            **sparse_counts(np.ones((1, 8, 8))),
        }
    ).to_zarr(store, group="band0000_time0000/p0", consolidated=False)
    dt = xr.open_datatree(store, engine="zarr", chunks=None, cache=False, consolidated=False)

    tracemalloc.start()
    ds = _load_piece(dt["band0000_time0000/p0"])
    assert "COUNTS_INDEX" not in ds and "COUNTS_VALUE" not in ds
    loaded = ds.VIS.nbytes
    del ds
    gc.collect()
    held, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert held < loaded / 4, f"{held / loaded:.2f} of the piece still held by the open tree"


def test_pass1_load_does_not_pin_data_on_the_task_argument(ms_name):
    """Pass 1's selective load leaves its Ray-argument node unloaded (#339).

    `node.ds[needed].load()` fills the Variables the Dataset shares with the
    node Ray deserialised as the task argument. On real data that kept ~1.7 GiB
    of read buffers per worker alive into the next task.
    """
    import gc
    import tracemalloc

    import xarray as xr

    from pfb_imaging.utils.msv4 import get_engine, load_detached

    dt = xr.open_datatree(ms_name, **get_engine(ms_name))
    node = next(iter(dt.children.values()))
    try:
        tracemalloc.start()
        ds = load_detached(node.ds[["VISIBILITY", "FLAG", "UVW"]])
        loaded = ds.VISIBILITY.nbytes
        del ds
        gc.collect()
        held, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        dt.close()
    assert held < loaded / 4, f"{held / loaded:.2f} of the read still held by the node"


@pytest.mark.parametrize("robustness", [-2.0, None])
def test_grid_partition_can_overwrite_the_weights_it_owns(robustness):
    """With overwrite_weight the imaging weights reuse part.WEIGHT's buffer (#339).

    The copy is a vis-sized array (2.1 GiB at the pass-2 peak on real data);
    pass 2 owns the partition it grids, so it can let the weights be written
    in place. The result must not depend on the choice.
    """
    part = _synth_partition()
    nx_pad = ny_pad = 32
    counts = _compute_counts(
        part.UVW.values,
        part.FREQ.values,
        part.MASK.values,
        part.WEIGHT.values,
        nx_pad,
        ny_pad,
        1.0e-6,
        1.0e-6,
        part.WEIGHT.values.dtype,
        usign=-1.0,
        vsign=1.0,
    )
    kw = dict(
        nx=16, ny=16, nx_psf=32, ny_psf=32, cell_rad=1.0e-6, robustness=robustness, counts_nx=nx_pad, counts_ny=ny_pad
    )
    natural = part.WEIGHT.values.copy()
    ref = grid_partition(part, counts, **kw)
    np.testing.assert_array_equal(part.WEIGHT.values, natural)  # the default never touches the input

    out = grid_partition(part, counts, overwrite_weight=True, **kw)
    assert np.shares_memory(out["WEIGHT"], part.WEIGHT.values)
    for key in ("WEIGHT", "DIRTY", "PSF", "WSUM"):
        np.testing.assert_array_equal(out[key], ref[key])


def _vis_partition(nrow, nchan=16, seed=0):
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "VIS": (
                ("corr", "row", "chan"),
                rng.normal(size=(1, nrow, nchan)) + 1j * rng.normal(size=(1, nrow, nchan)),
            ),
            "WEIGHT": (("corr", "row", "chan"), rng.random((1, nrow, nchan))),
            "MASK": (("row", "chan"), (rng.random((nrow, nchan)) > 0.1).astype(np.uint8)),
            "UVW": (("row", "three"), rng.normal(size=(nrow, 3))),
            "FREQ": (("chan",), np.linspace(1e9, 1.1e9, nchan)),
            "BEAM": (("corr", "y", "x"), rng.random((1, 32, 32))),
        },
        coords={"corr": ["I"]},
        attrs={"wsum": [1.0], "field_name": "f"},
    )


@pytest.mark.parametrize("nrow", [1, 999, 4096])
def test_write_partition_reads_back_identically(tmp_path, nrow):
    from pfb_imaging.core.imager import _write_partition

    ds = _vis_partition(nrow)
    store = f"file://{tmp_path}/x.dt"  # an fsspec store, as uri_and_fs gives the imager
    # a slab smaller than one row's worth still writes every row exactly once
    _write_partition(ds, store, "band0000_time0000/part0000", slab_bytes=64 * 1024)
    # zarr v2 on disk under zarr-python 3 too (pfb_imaging.ZARR_ENV)
    node = tmp_path / "x.dt" / "band0000_time0000" / "part0000"
    assert (node / ".zgroup").exists() and not (node / "zarr.json").exists()
    got = xr.open_datatree(store, engine="zarr", chunks=None, consolidated=False)["band0000_time0000/part0000"].ds
    for v in ds.data_vars:
        np.testing.assert_array_equal(got[v].values, ds[v].values)
    assert got.attrs == ds.attrs


def test_write_partition_bounds_the_encode_transient(tmp_path):
    """Writing a partition costs about one slab of encode buffers, not the vis (#339).

    zarr encodes every chunk of a to_zarr call before storing any; on a 5120^2
    UHF run writing part#### held 7.4 GiB of encode copies at the pass-2 peak.
    """
    import tracemalloc

    from pfb_imaging.core.imager import _write_partition

    ds = _vis_partition(2**17)  # VIS 32 MiB, ~50 MiB of row variables in all
    data = sum(ds[v].nbytes for v in ("VIS", "WEIGHT", "MASK", "UVW"))
    store = f"file://{tmp_path}/x.dt"  # an fsspec store, as uri_and_fs gives the imager
    tracemalloc.start()
    _write_partition(ds, store, "band0000_time0000/part0000", slab_bytes=4 * 2**20)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < data / 4, f"write peak {peak / data:.2f} x the row data"


@pytest.mark.parametrize("l0, m0", [(0.0, 0.0), (3e-4, -2e-4)])
def test_pass2_residual_matches_the_image_space_residual(l0, m0):
    """grid_partition's vis-space RESIDUAL equals deconv's DIRTY - G^T W G(B x) (#365).

    Pass 2 forms r = V - G(B x) while VIS is in memory; deconv's major cycles
    subtract in image space from the stored DIRTY (residual_from_partitions).
    The two must agree, beams, robust weights and a --target offset included.
    A flagged row carries NaN VIS, which must not reach either image.
    """
    rng = np.random.default_rng(7)
    nx = ny = 32
    cell = 2e-5
    nrow = 300
    uvw = rng.uniform(-2e3, 2e3, (nrow, 3))
    uvw[:, 2] = rng.uniform(-30, 30, nrow)
    freq = np.array([1.0e9, 1.2e9])
    mask = np.ones((nrow, 2), np.uint8)
    mask[0] = 0
    vis = rng.standard_normal((1, nrow, 2)) + 1j * rng.standard_normal((1, nrow, 2))
    vis[:, 0] = np.nan
    wgt = rng.uniform(0.5, 2.0, (1, nrow, 2))
    yy, xx = np.mgrid[:ny, :nx]
    beam = (1.0 - 0.02 * np.hypot(yy - ny / 2, xx - nx / 2))[None]
    part = xr.Dataset(
        {
            "VIS": (("corr", "row", "chan"), vis),
            "WEIGHT": (("corr", "row", "chan"), wgt),
            "MASK": (("row", "chan"), mask),
            "UVW": (("row", "three"), uvw),
            "FREQ": (("chan",), freq),
            "BEAM": (("corr", "y", "x"), beam),
        },
        coords={"corr": ["I"]},
        attrs={"l0": l0, "m0": m0},
    )
    nx_pad = ny_pad = 64
    counts = _compute_counts(uvw, freq, mask, wgt, nx_pad, ny_pad, cell, cell, wgt.dtype, usign=-1.0, vsign=1.0)
    model = np.zeros((1, ny, nx))
    model[0, 10, 20] = 1.0
    model[0, 25, 6] = 0.5
    kw = dict(
        nx=nx,
        ny=ny,
        nx_psf=2 * nx,
        ny_psf=2 * ny,
        cell_rad=cell,
        robustness=0.0,
        counts_nx=nx_pad,
        counts_ny=ny_pad,
        l0=l0,
        m0=m0,
        epsilon=1e-10,
        do_psf=False,
    )
    out = grid_partition(part, counts, model=model, **kw)
    assert np.isfinite(out["RESIDUAL"]).all()
    # the reference DIRTY comes from the same data with zeros under the mask, as
    # pass 1 stores it (weight_data skips flagged samples): ducc's mask does not
    # stop a NaN on aarch64, which is what the residual path guards against
    clean = part.assign(VIS=(("corr", "row", "chan"), np.where(mask, vis, 0)))
    plain = grid_partition(clean, counts, **kw)
    assert "RESIDUAL" not in plain

    stored = part.assign(WEIGHT=(("corr", "row", "chan"), out["WEIGHT"]))
    ref = residual_from_partitions(plain["DIRTY"], [stored], model, cell, epsilon=1e-10)
    assert np.abs(ref).max() > 0 and np.abs(plain["DIRTY"] - ref).max() > 1e-3 * np.abs(ref).max()
    np.testing.assert_allclose(out["RESIDUAL"], ref, rtol=0, atol=1e-8 * np.abs(ref).max())


def test_counts_grid_cell_and_reach():
    """The uv cell is 1/(min_padding*sqrt(2)*R) and the grid reaches past the longest baseline (#354)."""
    from scipy.constants import c as lightspeed

    from pfb_imaging.core.imager import counts_grid

    r = np.deg2rad(2.0)
    n, ny, cell = counts_grid(8000.0, 1.7e9, r, 1.7)
    assert n == ny and n % 2 == 0
    np.testing.assert_allclose(1.0 / (n * cell), 1.0 / (1.7 * np.sqrt(2.0) * r), rtol=1e-12)
    umax = 8000.0 * 1.7e9 / lightspeed
    # the kernels' grid spans |u| < 1 / (2 cell); one spare cell beyond umax
    assert umax < 1.0 / (2 * cell) - 1.0 / (n * cell) * (1 - 1e-9)


def test_weights_on_the_max_fov_grid_match_the_image_grid_of_the_largest_image():
    """An image of side sqrt(2)*R gets the same robust weights from either counts grid (#354)."""
    from pfb_imaging.core.imager import counts_grid

    part = _synth_partition(nrow=2000)
    nx = ny = 64
    cell = 1.0e-4
    nx_pad = int(np.ceil(1.7 * nx))
    nx_pad += nx_pad % 2
    uvw, freq, mask, wgt = part.UVW.values, part.FREQ.values, part.MASK.values, part.WEIGHT.values
    max_uvw = float(np.linalg.norm(uvw, axis=-1).max())

    def weights(cnx, cny, ccell):
        counts = _compute_counts(uvw, freq, mask, wgt, cnx, cny, ccell, ccell, wgt.dtype, usign=-1.0, vsign=1.0)
        kw = dict(nx=nx, ny=ny, nx_psf=2 * nx, ny_psf=2 * ny, cell_rad=cell, robustness=0.0, do_psf=False)
        return grid_partition(part, counts, counts_nx=cnx, counts_ny=cny, counts_cell=ccell, **kw)

    on_image = weights(nx_pad, nx_pad, cell)
    r = nx_pad * cell / (1.7 * np.sqrt(2.0))
    cnx, cny, ccell = counts_grid(max_uvw, float(freq.max()), r, 1.7)
    assert cnx < nx_pad  # a different extent, the same cell edges
    on_fov = weights(cnx, cny, ccell)
    # the reference must actually reweight, else equality says nothing
    assert not np.allclose(on_image["WEIGHT"], wgt)
    np.testing.assert_allclose(on_fov["WEIGHT"], on_image["WEIGHT"], rtol=1e-12)
    np.testing.assert_allclose(on_fov["DIRTY"], on_image["DIRTY"], rtol=1e-12, atol=0)

    # a larger R makes the cells finer, so the weights move towards natural
    finer = weights(*counts_grid(max_uvw, float(freq.max()), 2 * r, 1.7))
    assert not np.allclose(finer["WEIGHT"], on_image["WEIGHT"])


def test_counts_grid_holds_every_rephased_baseline():
    """Rephasing rotates uvw, so the grid is sized on |uvw|: no rotated sample lands off it (#354)."""
    from scipy.constants import c as lightspeed
    from scipy.spatial.transform import Rotation

    from pfb_imaging.core.imager import counts_grid

    rng = np.random.default_rng(3)
    uvw = rng.standard_normal((5000, 3)) * np.array([4000.0, 4000.0, 2000.0])
    freq = 1.7e9
    n, _, cell = counts_grid(float(np.linalg.norm(uvw, axis=-1).max()), freq, np.deg2rad(3.0), 1.7)
    half = 1.0 / (2 * cell) - 1.0 / (n * cell)  # one cell inside the edge
    for rot in Rotation.random(20, random_state=4):
        uv = rot.apply(uvw)[:, :2] * freq / lightspeed
        assert np.abs(uv).max() < half
    # the 2D length alone would not have held them
    n2, _, cell2 = counts_grid(float(np.hypot(uvw[:, 0], uvw[:, 1]).max()), freq, np.deg2rad(3.0), 1.7)
    worst = max(np.abs(rot.apply(uvw)[:, :2]).max() for rot in Rotation.random(20, random_state=4))
    assert worst * freq / lightspeed > 1.0 / (2 * cell2)
