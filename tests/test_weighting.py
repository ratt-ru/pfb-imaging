import os
import subprocess
import sys
import textwrap
import tracemalloc

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.weighting import (
    _compute_counts,
    box_sum_counts,
    counts_to_weights,
    filter_extreme_counts,
    reduce_counts,
    write_group_counts,
)

pmp = pytest.mark.parametrize


def _counts_grid(value):
    return np.full((1, 4, 4), float(value))


def test_reduce_counts_per_band_time_identity():
    counts = {(0, 0): _counts_grid(1), (1, 0): _counts_grid(2)}
    out = reduce_counts(counts, "per-band-time")
    assert set(out) == {(0, 0), (1, 0)}
    np.testing.assert_array_equal(out[(0, 0)], _counts_grid(1))
    np.testing.assert_array_equal(out[(1, 0)], _counts_grid(2))


def test_reduce_counts_mfs_sums_bands_within_time():
    counts = {(0, 0): _counts_grid(1), (1, 0): _counts_grid(2), (0, 1): _counts_grid(10), (1, 1): _counts_grid(20)}
    out = reduce_counts(counts, "mfs")
    # time 0 -> 1+2=3 shared; time 1 -> 10+20=30 shared
    np.testing.assert_array_equal(out[(0, 0)], _counts_grid(3))
    np.testing.assert_array_equal(out[(1, 0)], _counts_grid(3))
    np.testing.assert_array_equal(out[(0, 1)], _counts_grid(30))
    np.testing.assert_array_equal(out[(1, 1)], _counts_grid(30))
    # per-time matches mfs over the band axis
    np.testing.assert_array_equal(reduce_counts(counts, "per-time")[(0, 0)], _counts_grid(3))


def test_reduce_counts_per_band_sums_time_within_band():
    counts = {(0, 0): _counts_grid(1), (0, 1): _counts_grid(4), (1, 0): _counts_grid(2)}
    out = reduce_counts(counts, "per-band")
    np.testing.assert_array_equal(out[(0, 0)], _counts_grid(5))
    np.testing.assert_array_equal(out[(0, 1)], _counts_grid(5))
    np.testing.assert_array_equal(out[(1, 0)], _counts_grid(2))


def test_reduce_counts_unknown_raises():
    with pytest.raises(ValueError, match="nonsense"):
        reduce_counts({}, "nonsense")


@pmp("srf", [1.0, 2.0, 3.2])
@pmp("fov", [0.1, 0.33, 1.0])
def test_counts(ms_meta, srf, fov):
    """
    Compares _compute_counts to memory greedy numpy implementation
    """

    import numpy as np

    np.random.seed(420)
    from africanus.constants import c as lightspeed

    freq = ms_meta.freq
    nchan = ms_meta.nchan
    ncorr = ms_meta.ncorr
    uvw = ms_meta.uvw
    nrow = ms_meta.nrow
    max_blength = ms_meta.max_blength

    # image size
    cell_n = 1.0 / (2 * max_blength * freq.max() / lightspeed)
    cell_rad = cell_n / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600
    print("Cell size set to %5.5e arcseconds" % cell_size)

    from ducc0.fft import good_size

    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    print("Image size set to (%i, %i, %i)" % (ncorr, nx, ny))
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    usign = 1.0 if not flip_u else -1.0
    vsign = 1.0 if not flip_v else -1.0
    mask = np.ones((nrow, nchan), dtype=np.uint8)
    # wgt = np.ones((ncorr, nrow, nchan), dtype=uvw.dtype)
    wgt = np.exp(np.random.randn(ncorr, nrow, nchan))

    counts = _compute_counts(
        uvw, freq, mask, wgt, nx, ny, cell_rad, cell_rad, dtype=np.float64, usign=usign, vsign=vsign
    )

    # convert counts to imaging weights
    imwgt = counts_to_weights(
        counts, uvw, freq, np.ones_like(wgt), mask, nx, ny, cell_rad, cell_rad, -3, usign=usign, vsign=vsign
    )

    # computing counts with uniform weights should yield
    # ones everywhere
    counts2 = _compute_counts(
        uvw,
        freq,
        mask,
        wgt * imwgt,
        nx,
        ny,
        cell_rad,
        cell_rad,
        dtype=np.float64,
        usign=usign,
        vsign=vsign,
    )

    ic, ix, iy = np.where(counts2 > 0)
    assert np.allclose(counts2[ic, ix, iy], 1.0, rtol=1e-8, atol=1e-8)


@pmp("nx", [128, 1034, 44, 10000])
@pmp("cellx", [1.0, 0.01, 100, 1e-5])
def test_uv2xy(nx, cellx):
    import numpy as np

    np.random.seed(42)
    x = np.arange(0, nx)

    ucell = 1.0 / (nx * cellx)  # 1/fov
    umax = np.abs(1 / cellx / 2)
    u = (-(nx // 2) + np.arange(nx)) * ucell

    utmp = u + np.random.random(nx) * ucell

    ug = (utmp + umax) / ucell

    assert ((np.floor(ug) - x) == 0).all()


def test_box_sum_counts_identity():
    """npix_super <= 0 returns counts unchanged (super-uniform disabled)."""
    import numpy as np

    from pfb_imaging.utils.weighting import box_sum_counts

    rng = np.random.default_rng(0)
    counts = rng.random((2, 16, 16))

    out0 = box_sum_counts(counts, 0)
    out_neg = box_sum_counts(counts, -3)
    out_none = box_sum_counts(counts, None)

    # Identity function for all "disabled" inputs
    assert out0 is counts
    assert out_neg is counts
    assert out_none is counts


def test_box_sum_counts_3x3():
    """npix_super=1 replaces each cell with the sum of its 3x3 neighbourhood
    with zero-padding at image edges.
    """
    import numpy as np

    from pfb_imaging.utils.weighting import box_sum_counts

    counts = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
            ]
        ]
    )

    out = box_sum_counts(counts, 1)

    # Interior cell (2,2): sum of rows 1..3 and cols 1..3 = 7+8+9+12+13+14+17+18+19 = 117
    assert out[0, 2, 2] == pytest.approx(117.0)
    # Corner cell (0,0): sum of rows 0..1 and cols 0..1 = 1+2+6+7 = 16 (zero-padded outside)
    assert out[0, 0, 0] == pytest.approx(16.0)
    # Edge cell (0,2): sum of rows 0..1 and cols 1..3 = 2+3+4+7+8+9 = 33
    assert out[0, 0, 2] == pytest.approx(33.0)
    # Shape preserved and dtype preserved
    assert out.shape == counts.shape
    assert out.dtype == counts.dtype


def test_as_contiguous_readonly_view_flags():
    from pfb_imaging.utils.weighting import as_contiguous_readonly_view

    # contiguous input: no copy, readonly view
    a = np.ones((4, 3, 2), dtype=np.float32)
    v = as_contiguous_readonly_view(a)
    assert v.flags["C_CONTIGUOUS"] and not v.flags["WRITEABLE"]
    assert np.shares_memory(a, v)
    assert a.flags["WRITEABLE"]  # original untouched

    # non-contiguous input (like the jones swapaxes view): copied contiguous
    b = np.ones((4, 3, 2), dtype=np.complex64).swapaxes(0, 1)
    assert not b.flags["C_CONTIGUOUS"]
    w = as_contiguous_readonly_view(b)
    assert w.flags["C_CONTIGUOUS"] and not w.flags["WRITEABLE"]
    assert not np.shares_memory(b, w)


def _write_piece(store, path, grid):
    xr.Dataset({"COUNTS": (("corr", "u", "v"), grid)}).to_zarr(store, group=path, mode="a", consolidated=False)


def _sparse_grid(rng, n, fill):
    # uv counts are mostly empty cells: a few occupied ones, as in pass 1
    g = np.zeros((1, n, n))
    idx = rng.integers(0, n, size=(2, n * 4))
    g[0, idx[0], idx[1]] = fill + rng.random(idx.shape[1])
    return g


@pmp("npix_super", [0, 1])
def test_write_group_counts_matches_in_memory_reduction(tmp_path, npix_super):
    rng = np.random.default_rng(0)
    store = str(tmp_path / "s.scratch")
    grids = {f"band{b:04d}_time0000/p{i}": _sparse_grid(rng, 64, b + 1) for b in range(2) for i in range(3)}
    for path, g in grids.items():
        _write_piece(store, path, g)
    sources = {(b,): [p for p in grids if p.startswith(f"band{b:04d}")] for b in range(2)}

    out = write_group_counts(store, sources, filter_level=5.0, npix_super=npix_super)

    assert set(out) == set(sources)
    dt = xr.open_datatree(store, engine="zarr", chunks=None, consolidated=False)
    for key, paths in sources.items():
        ref = box_sum_counts(filter_extreme_counts(sum(grids[p] for p in paths), level=5.0), npix_super)
        np.testing.assert_array_equal(dt[out[key]].ds.COUNTS.values, ref)


def test_write_group_counts_streams(tmp_path):
    """The driver holds O(1) counts grids, not one per piece (#339).

    xarray's default cache=True memoises every .values read on an open tree,
    which kept every piece's grid alive for the rest of the run.
    """
    rng = np.random.default_rng(1)
    store = str(tmp_path / "s.scratch")
    n, npiece = 512, 12
    paths = [f"band0000_time{t:04d}/p0" for t in range(npiece)]
    for p in paths:
        _write_piece(store, p, _sparse_grid(rng, n, 1.0))
    grid_bytes = n * n * 8

    tracemalloc.start()
    write_group_counts(store, {(0,): paths}, filter_level=5.0, npix_super=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 4 * grid_bytes, f"peak {peak / grid_bytes:.1f} grids for {npiece} pieces"


def _counts_reference(uvw, freq, mask, wgt, nx, ny, cell, usign, vsign):
    """Plain-python nearest-cell accumulation with _compute_counts' conventions."""
    from scipy.constants import c as lightspeed

    u_cell, umax = 1 / (nx * cell), abs(1 / cell / 2)
    v_cell, vmax = 1 / (ny * cell), abs(1 / cell / 2)
    out = np.zeros((wgt.shape[0], nx, ny))
    for r in range(uvw.shape[0]):
        for f in range(freq.size):
            if not mask[r, f]:
                continue
            u = uvw[r, 0] * freq[f] / lightspeed * usign
            v = uvw[r, 1] * freq[f] / lightspeed * vsign
            if v < 0:
                u, v = -u, -v
            iu = int(np.floor((u + umax) / u_cell))
            iv = int(np.floor((v + vmax) / v_cell))
            if 0 <= iu < nx and 0 <= iv < ny:
                out[:, iu, iv] += wgt[:, r, f]
    return out


@pmp("usign,vsign", [(1.0, -1.0), (-1.0, 1.0)])
def test_compute_counts_matches_reference(usign, vsign):
    rng = np.random.default_rng(3)
    nrow, nchan, n = 300, 3, 64
    cell = np.deg2rad(20 / 3600)
    # some baselines fall off the grid on purpose; a third of the samples are flagged
    uvw = rng.uniform(-6000, 6000, (nrow, 3))
    freq = np.linspace(1.0e9, 1.4e9, nchan)
    mask = (rng.random((nrow, nchan)) > 0.33).astype(np.uint8)
    wgt = rng.random((2, nrow, nchan))
    got = _compute_counts(uvw, freq, mask, wgt, n, n, cell, cell, np.float64, usign=usign, vsign=vsign)
    np.testing.assert_allclose(got, _counts_reference(uvw, freq, mask, wgt, n, n, cell, usign, vsign), rtol=1e-12)


@pytest.mark.slow
def test_compute_counts_holds_one_grid():
    """Counts accumulate into the returned grid alone (#339).

    Per-thread grids plus their sum held nthreads + 1 padded grids (1.6 GiB at
    3840^2 with 4 threads) and were slower than one grid at realistic sizes.
    Numba's allocations are invisible to tracemalloc, so measure peak RSS in a
    fresh process.
    """
    script = textwrap.dedent(
        """
        import numpy as np
        from pfb_imaging.utils.weighting import _compute_counts

        def hwm():
            for line in open("/proc/self/status"):
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024

        rng = np.random.default_rng(0)
        nrow, nchan, n = 20000, 4, 4096
        uvw = rng.uniform(-2000, 2000, (nrow, 3))
        freq = np.linspace(1.0e9, 1.4e9, nchan)
        mask = np.ones((nrow, nchan), np.uint8)
        wgt = np.ones((1, nrow, nchan))
        cell = np.deg2rad(1 / 3600)
        _compute_counts(uvw[:8], freq, mask[:8], wgt[:, :8], 16, 16, cell, cell, np.float64)
        # reset the high-water mark to current RSS, so import/JIT peaks do not mask the call
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        base = hwm()
        c = _compute_counts(uvw, freq, mask, wgt, n, n, cell, cell, np.float64)
        print((hwm() - base) / c.nbytes)
        """
    )
    env = {**os.environ, "PFB_NUMBA_THREADING_LAYER": "workqueue"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    grids = float(out.stdout.strip().splitlines()[-1])
    assert grids < 1.5, f"peak {grids:.2f} grids"
