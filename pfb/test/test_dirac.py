import numpy as np
from pfb.operators import Dirac
from numpy.testing import assert_array_almost_equal
import pytest

pmp = pytest.mark.parametrize

@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 3, 6])
@pmp("ncomps", [27, 69])
def test_adjoint(nband, nx, ny, ncomp):
    Ix = np.random.randint(0, nx, ncomp)
    Iy = np.random.randint(0, ny, ncomp)
    mask = np.zeros((nx, ny), dtype=np.bool)
    mask[Ix, Iy] = 1

    H = Dirac(nband, nx, ny, mask=mask)

    beta = np.random.randn(nband, ncomp)
    x = np.random.randn(nband, nx, ny)

    tmp1 = H.dot(beta)
    tmp2 = H.hdot(x)

    lhs = np.vdot(x, tmp1)
    rhs = np.vdot(tmp2, beta)

    print(np.abs(lhs-rhs))
    assert np.abs(lhs - rhs) < 1e-13

def test_concatenate(nband, nx, ny, ncomp):
    Ix = np.random.randint(0, nx, ncomp)
    Iy = np.random.randint(0, ny, ncomp)
    mask = np.zeros((nx, ny), dtype=np.bool)
    mask[Ix, Iy] = 1

    H = Dirac(nband, nx, ny, mask=mask)

    def psi(x):
        alpha = x[:, 0:nx*ny].reshape(nband, nx, ny)
        beta = x[:, nx*ny::] 
        return alpha + H.dot(beta)
    
    def psih(x):
        y = x.reshape(nband, nx*ny)
        y = np.append(y, H.hdot(x), axis=1)
        return y

    x = np.random.randn(nband, nx*ny + ncomp)
    y = np.random.randn(nband, nx, ny)

    tmp1 = psi(x)
    tmp2 = psih(y)

    lhs = np.vdot(y, tmp1)
    rhs = np.vdot(tmp2, x)

    print(np.abs(lhs-rhs))
    assert np.abs(lhs - rhs) < 1e-13

if __name__=="__main__":
    test_concatenate(8,32, 64, 10)