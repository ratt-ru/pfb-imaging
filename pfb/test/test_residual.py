import numpy as np
from pfb.operators import Gridder, PSF
from africanus.constants import c as lightspeed
import packratt
from pyrap.tables import table
from numpy.testing import assert_allclose
import pytest
import os
import traceback

np.random.seed(420)

# pmp = pytest.mark.parametrize

# @pmp("srf", [1.2, 2.0])
# @pmp("fov", [0.5, 2.5])
# @pmp("nband", [1, 2])
def test_convolve(tmp_path_factory):  #, srf, fov, nband):
    srf = 2.0
    fov = 1.5
    nband = 2
    test_dir = tmp_path_factory.mktemp("test_convolve")
    packratt.get('/test/ms/2020-06-04/elwood/smallest_ms.tar.gz', test_dir)
    msname = [str(test_dir / 'smallest_ms.ms_p0')]

    uvw = table(msname[0]).getcol('UVW')

    freq = table(msname[0]+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ').squeeze()

    u_max = uvw[:, 0].max()
    v_max = uvw[:, 1].max()
    uv_max = np.maximum(u_max, v_max)

    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)
    cell_rad = cell_N/srf
    cell_size = cell_rad*60*60*180/np.pi

    nx = int(fov*3600/cell_size)
    if nx%2:
        nx += 1
    ny = int(1.5*nx)
    if ny%2:
        ny += 1

    R = Gridder(msname, nx, ny, cell_size, nband=nband, nthreads=1,
                do_wstacking=False, row_chunks=-1, psf_oversize=2.0,
                data_column='DATA', weight_column='WEIGHT', epsilon=1e-7,
                real_type='f4')
    
    psf = R.make_psf()

    wsums = np.amax(psf.reshape(nband, 2*nx*2*ny), axis=1)
    wsum = np.sum(wsums)
    psf /= wsum

    dirty = R.make_dirty()/wsum

    model = np.random.randn(nband, nx, ny).astype(np.float32)

    res1 = R.make_residual(model)/wsum
    res2 = dirty - R.convolve(model)/wsum

    assert_allclose(res1, res2, rtol=0.1, atol=1e-5)
