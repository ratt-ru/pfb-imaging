import numpy as np
from numpy.testing import assert_array_equal
from scipy.linalg import norm
import scipy.misc
from pfb.utils import prox_21
from pfb.operators import PSI, DaskPSI
import pywt

import pytest

def test_dask_psi_operator():
    da = pytest.importorskip('dask.array')
    nx = 512
    ny = 512
    nchan = 8
    nlevels = 3

    # construct operator (frequency axis dealt with in dot and hdot)
    # psi = PSI(nchan, nx, ny, nlevels=nlevels)
    # nbasis = psi.nbasis
    # ntot = psi.ntot  # number of wavelet coefficients

    # p = np.ones((nchan, nx, ny), dtype=np.float64)
    # weights_21 = np.empty(nbasis, dtype=object)
    # weights_21[0] = np.ones(nx*ny, dtype=np.float64)  # for dirac basis
    # for m in range(1, psi.nbasis):
    #     weights_21[m] = np.ones(psi.ntot, dtype=np.float64)  # all other wavelet bases

    # sig_21 = 1e-5

    # x = prox_21(p, sig_21, weights_21, psi=psi)

    img = da.from_array(scipy.misc.face(gray=True))
    dask_p = da.stack([img]*nchan, axis=0)
    nchan, nx, ny = dask_p.shape
    p = dask_p.reshape(nchan, -1).compute()
    assert_array_equal(dask_p, p.reshape(nchan, nx, ny))

    psi = PSI(nchan, nx, ny, nlevels=nlevels)
    dask_psi = DaskPSI(nchan, nx, ny, nlevels=nlevels)
    dask_weights_21 = da.ones((dask_psi.nbasis, nx, ny), dtype=dask_psi.real_type)
    sig_21 = 1e-5
    dask_dot = dask_psi.dot(dask_p)
    dask_hdot = dask_psi.hdot(dask_dot)
    assert dask_psi.nbasis == psi.nbasis


    for m in range(1, dask_psi.nbasis):
        mdbasis = dask_dot[m].compute()
        mbasis = psi.dot(p, m)
        assert_array_equal(mdbasis, mbasis)

        mdbasis = dask_hdot[m].compute()
        #mbasis = psi.hdot(p.reshape(nchan, nx, ny), m)
        mbasis = psi.hdot(mbasis, m)
        assert_array_equal(mdbasis, mbasis.reshape(nchan, nx, ny))


    x = prox_21(dask_p, sig_21, dask_weights_21, psi=dask_psi)
    x.compute()

