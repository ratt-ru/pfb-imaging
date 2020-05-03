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

    # x = prox_21(p, sig_21, weights_21, psi=psi)

    img = da.from_array(scipy.misc.face(gray=True))
    dask_p = da.stack([img]*nchan, axis=0)
    nchan, nx, ny = dask_p.shape
    p = dask_p.reshape(nchan, -1).compute()
    assert_array_equal(dask_p, p.reshape(nchan, nx, ny))

    psi = PSI(nchan, nx, ny, nlevels=nlevels)
    dask_psi = DaskPSI(nchan, nx, ny, nlevels=nlevels)
    dask_dot = dask_psi.dot(dask_p)
    dask_hdot = dask_psi.hdot(dask_p)
    assert dask_psi.nbasis == psi.nbasis

    for m in range(1, psi.nbasis):
        # Test dot
        mdbasis = dask_dot[m].compute()
        mbasis = psi.dot(p, m)
        assert_array_equal(mdbasis, mbasis)

        # Test hdot
        mdbasis = dask_hdot[m].compute()
        mbasis = psi.hdot(p.reshape(nchan, nx, ny), m)
        assert_array_equal(mdbasis, mbasis.reshape(nchan, nx, ny))


