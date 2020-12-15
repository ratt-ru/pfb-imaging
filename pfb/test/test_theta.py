import numpy as np
from numpy.testing import assert_array_almost_equal
from pfb.operators import Theta, DaskTheta
from pfb.opt import power_method
import pytest

pmp = pytest.mark.parametrize

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
def test_theta_norm(nx, ny, nband):
    theta = Theta(nband, nx, ny)

    x = np.random.randn(theta.nbasis+1, nband, theta.nmax)

    c = theta.dot(x)

    x = theta.hdot(c)

    op = lambda x: theta.hdot(theta.dot(x))

    L = power_method(op, x.shape, tol=1e-5, maxit=50)

    assert(np.allclose(L, 1))

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
def test_theta_adjoint(nx, ny, nband):
    theta = Theta(nband, nx, ny)


    x = np.random.randn(theta.nbasis+1, nband, theta.nmax)
    y = np.random.randn(2, nband, nx, ny)

    res1 = np.vdot(y, theta.dot(x))

    res2 = np.vdot(theta.hdot(y), x)

    assert(np.allclose(res1, res2))

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
def test_theta_dask(nx, ny, nband):
    theta = Theta(nband, nx, ny)

    x = np.random.randn(theta.nbasis+1, nband, theta.nmax)
    y = np.random.randn(2, nband, nx, ny)

    # from time import time

    # ti = time()
    res1 = theta.dot(x)
    res2 = theta.hdot(y)
    # print(time()-ti)

    theta_dask = DaskTheta(nband, nx, ny, 8)

    # ti = time()
    res1_dask = theta_dask.dot(x)
    res2_dask = theta_dask.hdot(y)
    # print(time()-ti)

    assert_array_almost_equal(res1, res1_dask, decimal=10)
    assert_array_almost_equal(res2, res2_dask, decimal=10)



# if __name__=="__main__":
#     test_theta_dask(1500, 1500, 8)