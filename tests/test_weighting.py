import packratt
import pytest
from pathlib import Path
from pfb.utils.weighting import (_compute_counts, _counts_to_weights,
                                 compute_counts, counts_to_weights)

pmp = pytest.mark.parametrize

def test_counts(tmp_path_factory):
    '''
    Compares _compute_counts to memory greedy numpy implementation
    '''
    test_dir = tmp_path_factory.mktemp("test_weighting")
    # test_dir = Path('/home/landman/data/')
    packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from pyrap.tables import table

    ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
    freq = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')).getcol('CHAN_FREQ').squeeze()
    nchan = freq.size
    uvw = ms.getcol('UVW')
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    from africanus.constants import c as lightspeed
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600

    from ducc0.fft import good_size
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    mask = np.ones((nrow, nchan), dtype=bool)
    counts = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                             dtype=np.float64).squeeze()
    # we need to take npix+1 to get the bin edges
    ku = np.sort(np.fft.fftfreq(nx+1, cell_rad))
    kv = np.sort(np.fft.fftfreq(ny+1, cell_rad))
    weights = np.ones((nrow*nchan), dtype=np.float64)
    u = (uvw[:, 0:1] * freq[None, :]/lightspeed).ravel()
    v = (uvw[:, 1:2] * freq[None, :]/lightspeed).ravel()
    counts2, _, _ = np.histogram2d(u, v, bins=[ku, kv], weights=weights)

    assert_allclose(counts, counts2)


def test_counts_dask(tmp_path_factory):
    '''
    Compares dask compute_counts to numba implementation
    '''
    test_dir = tmp_path_factory.mktemp("test_weighting")
    # test_dir = Path('/home/landman/data/')
    packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from pyrap.tables import table
    import dask.array as da
    import dask
    dask.config.set(scheduler='sync')

    ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
    freq = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')).getcol('CHAN_FREQ').squeeze()
    nchan = freq.size
    uvw = ms.getcol('UVW')
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    from africanus.constants import c as lightspeed
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600

    from ducc0.fft import good_size
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    weight = np.abs(np.random.randn(nrow, nchan))

    mask = np.ones((nrow, nchan), dtype=bool)
    counts = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                             np.float64, wgt=weight).squeeze()

    rc = 5000
    uvw = da.from_array(uvw, chunks=(rc, 3))
    freq = da.from_array(freq, chunks=-1)
    mask = da.from_array(mask, chunks=(rc, -1))
    weight = da.from_array(weight, chunks=(rc, -1))
    counts_dask = compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                                 np.float64, wgt=weight)

    assert_allclose(counts, counts_dask.compute())



def test_uniform(tmp_path_factory):
    '''
    Tests that the grid is uniformly weighted after doing weighting
    '''
    test_dir = tmp_path_factory.mktemp("test_weighting")
    # test_dir = Path('/home/landman/data/')
    packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from pyrap.tables import table

    ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
    freq = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')).getcol('CHAN_FREQ').squeeze()
    nchan = freq.size
    uvw = ms.getcol('UVW')
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    from africanus.constants import c as lightspeed
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600

    from ducc0.fft import good_size
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    mask = np.ones((nrow, nchan), dtype=bool)
    counts = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                             dtype=np.float64).squeeze()

    weights = _counts_to_weights(counts, uvw, freq, nx, ny,
                                 cell_rad, cell_rad, -3)


    counts2 = _compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                              np.float64, wgt=weights).squeeze()

    assert_allclose(counts2[counts2>0], 1)


def test_uniform_dask(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("test_weighting")
    # test_dir = Path('/home/landman/data/')
    packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from pyrap.tables import table
    import dask.array as da
    import dask

    ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
    freq = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')).getcol('CHAN_FREQ').squeeze()
    nchan = freq.size
    uvw = ms.getcol('UVW')
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    from africanus.constants import c as lightspeed
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600

    from ducc0.fft import good_size
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    rc = 5000
    uvw = da.from_array(uvw, chunks=(rc, 3))
    freq = da.from_array(freq, chunks=-1)
    mask = da.ones((nrow, nchan), chunks=(rc, -1), dtype=bool)
    counts = compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                            np.float64)

    weights = counts_to_weights(counts, uvw, freq, nx, ny,
                                cell_rad, cell_rad, -3)

    counts2 = compute_counts(uvw, freq, mask, nx, ny, cell_rad, cell_rad,
                             np.float64, wgt=weights)

    counts2 = counts2.compute()
    assert_allclose(counts2[counts2>0], 1)


# test_uniform_dask()
