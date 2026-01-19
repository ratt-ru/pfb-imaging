import numpy as np
import pytest
from africanus.gps.kernels import exponential_squared as expsq
from numpy.testing import assert_array_almost_equal

from pfb_imaging.utils.misc import kron_matvec

pmp = pytest.mark.parametrize


@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [2, 6])
@pmp("sigma0", [0.1, 1.0])
@pmp("length_scale", [1.0, 1.5])
def test_matvec(nx, ny, nband, sigma0, length_scale):
    v = np.arange(-(nband // 2), nband // 2)
    x = np.arange(-(nx // 2), nx // 2)
    y = np.arange(-(ny // 2), ny // 2)

    a1 = expsq(v, v, sigma0, length_scale)
    a2 = expsq(x, x, 1.0, length_scale)
    a3 = expsq(y, y, 1.0, length_scale)

    sigma = 1e-13
    c1 = np.linalg.pinv(a1, hermitian=True, rcond=sigma)
    c2 = np.linalg.pinv(a2, hermitian=True, rcond=sigma)
    c3 = np.linalg.pinv(a3, hermitian=True, rcond=sigma)

    amat = (a1, a2, a3)
    cmat = (c1, c2, c3)

    xi = np.random.randn(nband * nx * ny)
    res = kron_matvec(amat, xi)

    rec = kron_matvec(cmat, res)

    assert_array_almost_equal(rec, xi, decimal=5)
