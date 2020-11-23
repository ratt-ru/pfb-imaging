import numpy as np
from pfb.operators import Theta
from pfb.opt import power_method
import pytest

pmp = pytest.mark.parametrize

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
@pmp("nsource", [5, 20])
def test_theta_norm(nx, ny, nband, nsource):
    Ix = np.random.randint(0, nx, nsource)
    Iy = np.random.randint(0, ny, nsource)

    mask=np.zeros((1, nx, ny))
    mask[:, Ix, Iy] = 1


    theta = Theta(nband, nx, ny, mask)


    x = np.random.randn(theta.nbasis+1, nband, theta.nmax)

    c = theta.dot(x)

    x = theta.hdot(c)

    op = lambda x: theta.hdot(theta.dot(x))

    L = power_method(op, x.shape, tol=1e-5, maxit=50)

    assert(np.allclose(L, 1))

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
@pmp("nsource", [5, 20])
def test_theta_adjoint(nx, ny, nband, nsource):
    Ix = np.random.randint(0, nx, nsource)
    Iy = np.random.randint(0, ny, nsource)

    mask=np.zeros((1, nx, ny))
    mask[:, Ix, Iy] = 1


    theta = Theta(nband, nx, ny, mask)


    x = np.random.randn(theta.nbasis+1, nband, theta.nmax)
    y = np.random.randn(2, nband, nx, ny)

    res1 = np.vdot(y, theta.dot(x))

    res2 = np.vdot(theta.hdot(y), x)

    assert(np.allclose(res1, res2))


# if __name__=="__main__":
#     test_theta_adjoint(512, 256, 8, 5)