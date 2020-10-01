

# import numpy as np
# from numpy.testing import assert_array_almost_equal, assert_array_equal
# import pytest

# def test_dot_idot_explicit():
#     nx = 32
#     ny = 32
#     nchan = 4
#     N = nchan*nx*ny

#     v = np.linspace(0.5, 1.5, nchan)
#     Kop = Prior(v, 1.0, 0.1, nx, ny, 1.0)
    
#     K = np.kron(Kop.Kv, np.kron(Kop.Kl, Kop.Km))

#     xi = np.random.randn(N)
#     res1 = Kop.dot(xi)
#     res2 = K.dot(xi)

#     print(np.abs(res1 - res2).max())

#     Kinv = np.linalg.inv(K + 1e-12*np.eye(N))

#     xi = np.random.randn(N)

#     res1 = Kop.idot(xi)
#     res2 = Kinv.dot(xi)

#     print(np.abs(res1 - res2).max())
