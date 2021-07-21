import numpy as np
from numpy.testing import assert_array_almost_equal
from pfb.utils.misc import kron_matvec
from africanus.gps.kernels import exponential_squared as expsq
import pytest
pmp = pytest.mark.parametrize

@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [2, 6])
@pmp("sigma0", [0.1, 1.0])
@pmp("length_scale", [1.0, 1.5])
def test_matvec(nx, ny, nband, sigma0, length_scale):
    v = np.arange(-(nband//2), nband//2)
    x = np.arange(-(nx//2), nx//2)
    y = np.arange(-(ny//2), ny//2)

    A1 = expsq(v, v, sigma0, length_scale)
    A2 = expsq(x, x, 1.0, length_scale)
    A3 = expsq(y, y, 1.0, length_scale)

    sigma = 1e-13
    C1 = np.linalg.pinv(A1, hermitian=True, rcond=sigma)
    C2 = np.linalg.pinv(A2, hermitian=True, rcond=sigma)
    C3 = np.linalg.pinv(A3, hermitian=True, rcond=sigma)

    A = (A1, A2, A3)
    C = (C1, C2, C3)

    xi = np.random.randn(nband*nx*ny)
    res = kron_matvec(A, xi)

    rec = kron_matvec(C, res)

    assert_array_almost_equal(rec, xi, decimal=5)