import pytest

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.weighting import _compute_counts, counts_to_weights

pmp = pytest.mark.parametrize


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
        uvw, freq, mask, wgt, nx, ny, cell_rad, cell_rad, dtype=np.float64, ngrid=1, usign=usign, vsign=vsign
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
        ngrid=1,
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
