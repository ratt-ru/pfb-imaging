import pytest
from pathlib import Path
from pfb.utils.weighting import (_compute_counts, _counts_to_weights,
                                 compute_counts, counts_to_weights)

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


    mask = np.ones((nrow, nchan), dtype=bool)
    counts = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                             dtype=np.float64, k=0).squeeze()
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
    u = (uvw[:, 0:1] * freq[None, :]/lightspeed).ravel()
    v = (uvw[:, 1:2] * freq[None, :]/lightspeed).ravel()
    counts2, _, _ = np.histogram2d(u, v, bins=[ku, kv], weights=weights)

    assert_allclose(counts, counts2)


def test_counts_dask(ms_name):
    '''
    Compares dask compute_counts to numba implementation
    '''

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from daskms import xds_from_ms, xds_from_table
    from africanus.constants import c as lightspeed
    import dask.array as da
    import dask
    dask.config.set(scheduler='sync')

    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]

    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    freq0 = np.mean(freq)

    ntime = utime.size
    nchan = freq.size
    nant = np.maximum(xds.ANTENNA1.values.max(), xds.ANTENNA2.values.max()) + 1

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
    mask = np.ones((nrow, nchan), dtype=bool)
    counts = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                             np.float64, k=0).squeeze()

    rc = 5000
    uvw = da.from_array(uvw, chunks=(rc, 3))
    freq = da.from_array(freq, chunks=-1)
    mask = da.from_array(mask, chunks=(rc, -1))
    counts_dask = compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                                 np.float64, k=0)

    assert_allclose(counts, counts_dask.compute())


# deprecated
# def test_uniform(tmp_path_factory):
#     '''
#     Tests that the grid is uniformly weighted after doing weighting
#     '''
#     test_dir = tmp_path_factory.mktemp("test_weighting")
#     # test_dir = Path('/home/landman/data/')
#     packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

#     import numpy as np
#     np.random.seed(420)
#     from numpy.testing import assert_allclose
#     from pyrap.tables import table

#     ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
#     freq = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')).getcol('CHAN_FREQ').squeeze()
#     nchan = freq.size
#     uvw = ms.getcol('UVW')
#     nrow = uvw.shape[0]
#     u_max = abs(uvw[:, 0]).max()
#     v_max = abs(uvw[:, 1]).max()
#     uv_max = np.maximum(u_max, v_max)

#     # image size
#     from africanus.constants import c as lightspeed
#     cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

#     srf = 2.0
#     cell_rad = cell_N / srf
#     cell_deg = cell_rad * 180 / np.pi
#     cell_size = cell_deg * 3600

#     from ducc0.fft import good_size
#     fov = 1.0
#     npix = good_size(int(fov / cell_deg))
#     while npix % 2:
#         npix += 1
#         npix = good_size(npix)

#     nx = npix
#     ny = npix

#     mask = np.ones((nrow, nchan), dtype=bool)
#     counts = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
#                              dtype=np.float64, k=0).squeeze()

#     weights = _counts_to_weights(counts, uvw, freq, nx, ny,
#                                  cell_rad, cell_rad, -3)


#     counts2 = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
#                               np.float64, wgt=weights, k=0).squeeze()

#     assert_allclose(counts2[counts2>0], 1)


# def test_uniform_dask(tmp_path_factory):
#     test_dir = tmp_path_factory.mktemp("test_weighting")
#     # test_dir = Path('/home/landman/data/')
#     packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

#     import numpy as np
#     np.random.seed(420)
#     from numpy.testing import assert_allclose
#     from pyrap.tables import table
#     import dask.array as da
#     import dask

#     ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
#     freq = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')).getcol('CHAN_FREQ').squeeze()
#     nchan = freq.size
#     uvw = ms.getcol('UVW')
#     nrow = uvw.shape[0]
#     u_max = abs(uvw[:, 0]).max()
#     v_max = abs(uvw[:, 1]).max()
#     uv_max = np.maximum(u_max, v_max)

#     # image size
#     from africanus.constants import c as lightspeed
#     cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

#     srf = 2.0
#     cell_rad = cell_N / srf
#     cell_deg = cell_rad * 180 / np.pi
#     cell_size = cell_deg * 3600

#     from ducc0.fft import good_size
#     fov = 1.0
#     npix = good_size(int(fov / cell_deg))
#     while npix % 2:
#         npix += 1
#         npix = good_size(npix)

#     nx = npix
#     ny = npix

#     rc = 5000
#     uvw = da.from_array(uvw, chunks=(rc, 3))
#     freq = da.from_array(freq, chunks=-1)
#     mask = da.ones((nrow, nchan), chunks=(rc, -1), dtype=bool)
#     counts = compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
#                             np.float64, k=0)

#     weights = counts_to_weights(counts, uvw, freq, nx, ny,
#                                 cell_rad, cell_rad, -3)

#     counts2 = compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
#                              np.float64, wgt=weights, k=0)

#     counts2 = counts2.compute()
#     assert_allclose(counts2[counts2>0], 1)
