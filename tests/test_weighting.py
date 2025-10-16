import pytest
from pathlib import Path
from pfb_imaging.utils.weighting import _compute_counts, counts_to_weights
from pfb_imaging.operators.gridder import wgridder_conventions

pmp = pytest.mark.parametrize

@pmp("srf", [1.0, 2.0, 3.2])
@pmp("fov", [0.1, 0.33, 1.0])
def test_counts(ms_name, srf, fov):
    '''
    Compares _compute_counts to memory greedy numpy implementation
    '''

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from daskms import xds_from_ms, xds_from_table
    from africanus.constants import c as lightspeed

    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]

    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    freq0 = np.mean(freq)

    ntime = utime.size
    nchan = freq.size
    nant = np.maximum(xds.ANTENNA1.values.max(), xds.ANTENNA1.values.max()) + 1

    ncorr = xds.corr.size

    uvw = xds.UVW.values
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)
    cell_rad = cell_N / srf
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

    counts = _compute_counts(uvw, freq, mask, wgt, nx, ny, cell_rad, cell_rad,
                             dtype=np.float64, k=0, ngrid=1,
                             usign=usign, vsign=vsign)

    # convert counts to imaging weights
    imwgt = counts_to_weights(counts, uvw, freq,
                              np.ones_like(wgt),
                              mask, nx, ny,
                              cell_rad, cell_rad,
                              -3,
                              usign=usign, vsign=vsign)

    # computing counts with uniform weights should yield
    # ones everywhere
    counts2 = _compute_counts(uvw, freq, mask,
                              wgt*imwgt,
                              nx, ny, cell_rad, cell_rad,
                              dtype=np.float64, k=0, ngrid=1,
                              usign=usign, vsign=vsign)

    ic, ix, iy = np.where(counts2 > 0)
    assert np.allclose(counts2[ic, ix, iy], 1.0, rtol=1e-8, atol=1e-8)


@pmp("nx", [128, 1034, 44, 10000])
@pmp("cellx", [1.0, 0.01, 100, 1e-5])
def test_uv2xy(nx, cellx):
    import numpy as np
    np.random.seed(42)
    x = np.arange(0, nx)

    ucell = 1.0/(nx*cellx)  # 1/fov
    umax = np.abs(1/cellx/2)
    u = (-(nx//2) + np.arange(nx)) * ucell

    utmp = u + np.random.random(nx)*ucell

    ug = (utmp + umax)/ucell

    assert ((np.floor(ug) - x) == 0).all()
