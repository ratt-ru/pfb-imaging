import numpy as np
from functools import partial
import dask.array as da
from numpy.testing import assert_array_almost_equal
from pfb.prox.prox_21 import prox_21
from pfb.prox.prox_21m import prox_21m
from pfb.prox.prox_21m import prox_21m_numba, dual_update, dual_update_numba
import pywt
import pytest
from pfb.operators.psi import Psi

pmp = pytest.mark.parametrize

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_psi(nx, ny, nband, nlevels):
    """
    Check that decomposition + reconstruction is the identity
    """
    np.random.seed(420)
    # image = pywt.data.aero()
    image = np.random.randn(nx, ny)
    nu = 1.0  + 0.1 * np.arange(nband)

    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)

    # set up dictionary
    bases = ['self','db1','db2','db3','db4','db5']
    nbasis = len(bases)
    psi = Psi(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.Nxmax
    nymax = psi.Nymax

    # make sure this works even when output arrays are randomly populated
    alpha = np.random.randn(nband, nbasis, nymax, nxmax)  #, dtype=x.dtype)
    xrec = np.random.randn(nband, nx, ny)  #, dtype=x.dtype)

    # decompose
    psi.dot(x, alpha)
    # reconstruct
    psi.hdot(alpha, xrec)

    # the nbasis is required here because the operator is not normalised
    assert_array_almost_equal(nbasis*x, xrec, decimal=12)

@pmp("nx", [120, 240])
@pmp("ny", [64, 150])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_prox21(nx, ny, nband, nlevels):
    """
    Check that applying the prox with zero step size is the identity
    """
    np.random.seed(420)
    image = pywt.data.aero()

    nu = 1.0  + 0.1 * np.arange(nband)

    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)

    # set up dictionary
    bases = ['self','db1','db2','db3','db4','db5']
    nbasis = len(bases)
    psi = Psi(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.Nxmax
    nymax = psi.Nymax

    weights_21 = np.random.random(nbasis*nymax*nxmax).reshape(nbasis, nymax, nxmax)
    sig_21 = 0.0

    alpha = np.zeros((nband, nbasis, nymax, nxmax), dtype=x.dtype)
    xrec = np.zeros((nband, nx, ny), dtype=x.dtype)

    psi.dot(x, alpha)

    y = prox_21(alpha, sig_21, weights_21)

    psi.hdot(y, xrec)

    # the nbasis is required here because the operator is not normalised
    assert_array_almost_equal(nbasis*x, xrec, decimal=12)


@pmp("nx", [1202, 240])
@pmp("ny", [324, 1506])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_prox21m(nx, ny, nband, nlevels):
    """
    Check that applying the prox with zero step size is the identity
    """
    np.random.seed(420)
    image = np.random.randn(nx, ny)

    nu = 1.0  + 0.1 * np.arange(nband)

    x = image[None, :, :] * nu[:, None, None] ** (-0.7)

    # set up dictionary
    bases = ['self','db1','db2','db3','db4','db5']
    nbasis = len(bases)
    psi = Psi(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.Nxmax
    nymax = psi.Nymax

    weights_21 = np.random.random(nbasis*nymax*nxmax).reshape(nbasis, nymax, nxmax)
    sig_21 = 0.0

    alpha = np.zeros((nband, nbasis, nymax, nxmax), dtype=x.dtype)
    xrec = np.zeros((nband, nx, ny), dtype=x.dtype)

    psi.dot(x, alpha)

    y = prox_21m(alpha, sig_21, weights_21)

    psi.hdot(y, xrec)

    # the nbasis is required here because the operator is not normalised
    assert_array_almost_equal(nbasis*x, xrec, decimal=12)


@pmp("nymax", [1234, 240])
@pmp("nxmax", [134, 896])
@pmp("nbasis", [1, 5])
@pmp("nband", [1, 3, 6])
@pmp("lam", [1.0, 1e-1, 1e-3])
@pmp("sigma", [75.0, 1.0, 1e-3])
def test_prox21m_numba(nband, nbasis, nymax, nxmax, lam, sigma):
    np.random.seed(420)
    # check numba implementation matches numpy even when output contains random
    # numbers initially
    v = np.random.randn(nband, nbasis, nymax, nxmax)
    vout = np.random.randn(nband, nbasis, nymax, nxmax)
    l1weight = np.random.random(nbasis*nymax*nxmax).reshape(nbasis, nymax, nxmax)
    res = prox_21m(v, lam, weight=l1weight)
    prox_21m_numba(v, vout, lam, weight=l1weight)

    assert_array_almost_equal(res, vout, decimal=12)

    res = prox_21m(v/sigma, lam/sigma, weight=l1weight)
    prox_21m_numba(v, vout, lam, sigma=sigma, weight=l1weight)

    assert_array_almost_equal(res, vout, decimal=8)


@pmp("nx", [120, 240])
@pmp("ny", [324, 150])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
@pmp("lam", [1.0, 1e-1, 1e-3])
@pmp("sigma", [75.0, 1.0, 1e-3])
def test_dual_update(nx, ny, nband, nlevels, lam, sigma):
    """
    Compare numpy to numba optimised dual update
    """
    np.random.seed(420)
    image = np.random.randn(nx, ny)

    nu = 1.0  + 0.1 * np.arange(nband)

    x = image[None, :, :] * nu[:, None, None] ** (-0.7)

    xp = x.copy()

    # set up dictionary
    bases = ['self','db1','db2','db3','db4','db5']
    nbasis = len(bases)
    psi = Psi(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.Nxmax
    nymax = psi.Nymax

    weight21 = np.random.random(nbasis*nymax*nxmax).reshape(nbasis, nymax, nxmax)
    # lam21 = 0.1
    # sigma = 1.75

    # can't initialise v randomly because edges won't agree
    # vout in dual_update is initialsed with zeros
    # v = np.random.randn(nband, nbasis, nymax, nxmax)
    v = np.zeros((nband, nbasis, nymax, nxmax))
    x2 = np. random.randn(nband, nx, ny)
    psi.dot(x2, v)
    vp = v.copy()
    res1 = dual_update(v, x, psi.dot, lam, sigma=sigma, weight=weight21)

    # initialise v with psiH(x)
    psi.dot(xp, v)
    dual_update_numba(vp, v, lam, sigma=sigma, weight=weight21)
    # TODO - why the low accuracy?
    assert_array_almost_equal(1 + res1,1 + v, decimal=9)
