from time import time as timeit

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.linalg import norm
import scipy.misc
import pywt
import pytest
import yappi

from pfb.utils import prox_21
from pfb.operators import PSI, DaskPSI


def test_dask_psi_operator():
    da = pytest.importorskip('dask.array')
    nx = 1024
    ny = 1024
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
    yappi.set_clock_type("cpu")
    yappi.start()
    alphad = dask_psi.hdot(x)
    yappi.stop()
    # import pdb; pdb.set_trace()
    yappi.get_func_stats().print_all()
    print("Parallel decomp took ", timeit() - ti)
    # reconstruct
    ti = timeit()
    xrecd = dask_psi.dot(alphad)
    print("Parallel rec took ", timeit() - ti)

    assert_array_almost_equal(x, xrecd, decimal=13)

    assert_array_almost_equal(alphad, alpha, decimal=13)

def test_prox():
    nx = 2050
    ny = 2050
    nband = 8
    nlevels = 5

    # random test image
    x = np.random.randn(nband, nx, ny)

    # initialise serial operator
    psi = PSI(nband, nx, ny, nlevels)
    ntot = psi.ntot
    nbasis = psi.nbasis

    weights_21 = np.ones((nbasis, ntot))
    sig_21 = 0.0

    y = prox_21(x, sig_21, weights_21, psi=psi)

    assert_array_almost_equal(x, y, decimal=13)

    # weight_21 = dask_weights_21.compute().reshape(psi.nbasis, nx*ny)
    # x2 = prox_21(image_cube, sig_21, weight_21, psi=psi)

    # assert_array_equal(x, x2)

if __name__=="__main__":
    # test_dask_psi_operator()
    test_prox()