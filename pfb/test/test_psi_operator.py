import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.linalg import norm
import scipy.misc
from pfb.utils import prox_21
from pfb.operators import PSI, DaskPSI
import pywt
from time import time as timeit
import pytest

def test_dask_psi_operator():
    da = pytest.importorskip('dask.array')
    nx = 2050
    ny = 2050
    nband = 8
    nlevels = 5

    # random test image
    x = np.random.randn(nband, nx, ny)
    
    # initialise serial operator 
    psi = PSI(nband, nx, ny, nlevels)

    # decompose
    ti = timeit()
    alpha = psi.hdot(x)
    print("Serial decomp took ", timeit() - ti)
    # reconstruct
    ti = timeit()
    xrec = psi.dot(alpha)
    print("Serial rec took ", timeit() - ti)

    assert_array_almost_equal(x, xrec, decimal=13)

    # initialise parallel operator
    dask_psi = DaskPSI(nband, nx, ny, nlevels)
    # decompose
    ti = timeit()
    alphad = dask_psi.hdot(x)
    print("Parallel decomp took ", timeit() - ti)
    # reconstruct
    ti = timeit()
    xrecd = dask_psi.dot(alphad)
    print("Parallel rec took ", timeit() - ti)

    assert_array_almost_equal(x, xrecd, decimal=13)

    assert_array_almost_equal(alphad, alpha, decimal=13)

    # # compare prox
    # for m in range(1, dask_psi.nbasis):
    #     mdbasis = dask_dot[m].compute()
    #     mbasis = psi.dot(p, m)
    #     assert_array_equal(mdbasis, mbasis)

    #     mdbasis = dask_hdot[m].compute()
    #     #mbasis = psi.hdot(p.reshape(nchan, nx, ny), m)
    #     mbasis = psi.hdot(mbasis, m)
    #     assert_array_equal(mdbasis, mbasis.reshape(nchan, nx, ny))


    # x = prox_21(dask_p, sig_21, dask_weights_21, psi=dask_psi)
    # x.compute()

    # weight_21 = dask_weights_21.compute().reshape(psi.nbasis, nx*ny)
    # x2 = prox_21(image_cube, sig_21, weight_21, psi=psi)

    # assert_array_equal(x, x2)

if __name__=="__main__":
    test_dask_psi_operator()