import numpy as np
from pfb.operators import Dirac
from numpy.testing import assert_array_almost_equal
import pytest

pmp = pytest.mark.parametrize

@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 3, 6])
@pmp("ncomp", [27, 69])
def test_adjoint(nband, nx, ny, ncomp):
    Ix = np.random.randint(0, nx, ncomp)
    Iy = np.random.randint(0, ny, ncomp)
    mask = np.zeros((nx, ny), dtype=np.bool)
    mask[Ix, Iy] = 1

    H = Dirac(nband, nx, ny, mask=mask)

    beta = np.random.randn(nband, nx, ny)
    x = np.random.randn(nband, nx, ny)

    tmp1 = H.dot(beta)
    tmp2 = H.hdot(x)

    lhs = np.vdot(x, tmp1)
    rhs = np.vdot(tmp2, beta)

    assert np.abs(lhs - rhs) < 1e-10

@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 3, 6])
@pmp("ncomp", [27, 69])
def test_concatenate(nband, nx, ny, ncomp):
    Ix = np.random.randint(0, nx, ncomp)
    Iy = np.random.randint(0, ny, ncomp)
    mask = np.zeros((nx, ny), dtype=np.bool)
    mask[Ix, Iy] = 1

    H = Dirac(nband, nx, ny, mask=mask)

    def psi(x):
        return x[0] + H.dot(x[1])
    
    def psih(x):
        return np.concatenate((x[None], H.hdot(x)[None]), axis=0)

    x = np.random.randn(2, nband, nx, ny)
    y = np.random.randn(nband, nx, ny)

    tmp1 = psi(x)
    tmp2 = psih(y)

    lhs = np.vdot(y, tmp1)
    rhs = np.vdot(tmp2, x)

    assert np.abs(lhs - rhs) < 1e-10