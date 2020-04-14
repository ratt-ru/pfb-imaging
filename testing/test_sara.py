import numpy as np
from numpy.testing import assert_array_almost_equal
from pfb.sara import set_Psi
from pfb.operators import PSI

def compare_old():
    nx = 1024
    ny = 1024
    nchan = 8

    nlevels = 3

    psi = PSI(nchan, nx, ny, nlevels=nlevels)
    ntot = psi.ntot
    
    psi_old, psit_old = set_Psi(nx, ny, nlevels=nlevels)

    x = np.ones((nchan, nx, ny), dtype=np.float64)
    basis_k = 4
    res_old = np.zeros((nchan, ntot), dtype=np.float64)
    for i in range(nchan):
        res_old[i] = psit_old[basis_k](x[i])

    res_new = psi.hdot(x, basis_k)

    assert_array_almost_equal(res_old, res_new, decimal=10)

    y_old = np.zeros((nchan, nx, ny), dtype=np.float64)

    for i in range(nchan):
        y_old[i] = psi_old[basis_k](res_old[i])

    y_new = psi.dot(res_new, basis_k)

    assert_array_almost_equal(y_old, y_new, decimal=10)

def check_ortho():
    nx = 4050
    ny = 4050
    nchan = 8

    nlevels = 3

    psi = PSI(nchan, nx, ny, nlevels=nlevels)
    ntot = psi.ntot

    x = np.ones((nchan, nx, ny), dtype=np.float64)
    y = np.zeros((nchan, nx, ny), dtype=np.float64)
    for i in range(len(psi.basis)):
        alpha = psi.hdot(x, i)
        y += psi.dot(alpha, i)

    assert(np.abs(x - y).max() < 1e-15)


if __name__=="__main__":
    compare_old()
    check_ortho()