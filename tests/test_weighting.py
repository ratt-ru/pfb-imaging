import pytest
from pathlib import Path
from pfb.utils.weighting import _compute_counts
from pfb.operators.gridder import wgridder_conventions

pmp = pytest.mark.parametrize

def test_counts(ms_name):
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

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600
    print("Cell size set to %5.5e arcseconds" % cell_size)

    from ducc0.fft import good_size
    # the test will fail in intrinsic if sources fall near beam sidelobes
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    print("Image size set to (%i, %i, %i)" % (nchan, nx, ny))
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    usign = 1.0 if not flip_u else -1.0
    vsign = 1.0 if not flip_v else -1.0
    mask = np.ones((nrow, nchan), dtype=np.uint8)
    wgt = np.ones((ncorr, nrow, nchan), dtype=uvw.dtype)

    counts = _compute_counts(uvw, freq, mask, wgt, nx, ny, cell_rad, cell_rad,
                             dtype=np.float64, k=0, ngrid=2,
                             usign=usign, vsign=vsign)
    ku = np.sort(np.fft.fftfreq(nx, cell_rad))
    # shift by half a pixel to get bin edges
    kucell = ku[1] - ku[0]
    ku -= kucell/2
    # add upper edge
    ku = np.append(ku, ku.max() + kucell)
    kv = np.sort(np.fft.fftfreq(ny, cell_rad))
    kvcell = kv[1] - kv[0]
    kv -= kvcell/2
    kv = np.append(kv, kv.max() + kvcell)
    weights = np.ones((nrow*nchan), dtype=np.float64)
    u = (usign*uvw[:, 0:1] * freq[None, :]/lightspeed).ravel()
    v = (vsign*uvw[:, 1:2] * freq[None, :]/lightspeed).ravel()
    counts2, _, _ = np.histogram2d(u, v, bins=[ku, kv], weights=weights)

    for c in range(ncorr):
        # import ipdb; ipdb.set_trace()
        assert_allclose(counts[c], counts2)
