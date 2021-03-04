import numpy as np
from numpy.testing import assert_array_almost_equal
from pfb.utils import prox_21
from pfb.operators import PSI, DaskPSI
import pywt
import pytest
from pfb.wavelets.wavelets import wavedecn, waverecn

pmp = pytest.mark.parametrize

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_dask_psi_operator(nx, ny, nband, nlevels):
    """
    Check that dask version produces the same result as np version
    """
    da = pytest.importorskip('dask.array')

    # get test image
    image = pywt.data.aero()

    nu = 1.0  + 0.1 * np.arange(nband)

    # take subset and add freq axis
    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)
    
    # initialise serial operator 
    psi = PSI(imsize=(nband, nx, ny), nlevels=nlevels)

    # decompose
    alpha = psi.hdot(x)
    # reconstruct
    xrec = psi.dot(alpha)

    # initialise parallel operator
    dask_psi = DaskPSI(imsize=(nband, nx, ny), nlevels=nlevels)
    # decompose
    alphad = dask_psi.hdot(x)
    xrecd = dask_psi.dot(alphad)

    assert_array_almost_equal(x, xrecd, decimal=12)
    assert_array_almost_equal(alphad, alpha, decimal=12)


@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_psi(nx, ny, nband, nlevels):
    """
    Check that decomposition + reconstruction is the identity
    """
    # get test image
    image = pywt.data.aero()
    nu = 1.0  + 0.1 * np.arange(nband)

    # take subset and add freq axis
    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)

    # initialise serial operator 
    psi = PSI(imsize=(nband, nx, ny), nlevels=nlevels)

    # decompose
    alpha = psi.hdot(x)
    # reconstruct
    xrec = psi.dot(alpha)

    assert_array_almost_equal(x, xrec, decimal=12)

@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_prox(nx, ny, nband, nlevels):
    # get test image
    image = pywt.data.aero()
    nu = 1.0  + 0.1 * np.arange(nband)

    # take subset and add freq axis
    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)
    
    # initialise serial operator 
    psi = PSI(imsize=(nband, nx, ny), nlevels=nlevels)
    nmax = psi.nmax
    nbasis = psi.nbasis

    weights_21 = np.ones((nbasis, nmax))
    sig_21 = 0.0

    alpha = psi.hdot(x)

    y = prox_21(alpha, sig_21, weights_21)

    xrec = psi.dot(y)

    assert_array_almost_equal(x, xrec, decimal=12)
