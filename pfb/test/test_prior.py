import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pfb.operators import Prior 
import pytest
pmp = pytest.mark.parametrize


@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 6])
@pmp("sigma0", [0.1, 1.0])
def test_dot_idot_explicit(nx, ny, nband, sigma0):
    # Initialise operator
    Kop = Prior(sigma0, nband, nx, ny)

    # generate random vector and act on it
    xi = np.random.randn(nband, nx, ny)
    res = Kop.dot(xi)

    # rec with idot and compare
    rec = Kop.idot(res)

    assert_array_almost_equal(xi, rec, decimal=6)


@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 6])
@pmp("sigma0", [0.1, 1.0])
def test_dot_convolve(nx, ny, nband, sigma0):
    # Initialise operator
    Kop = Prior(sigma0, nband, nx, ny)

    # generate random vector and act on it
    xi = np.random.randn(nband, nx, ny)
    res = Kop.dot(xi)
    res2 = Kop.convolve(xi)

    assert_array_almost_equal(res, res2, decimal=12)