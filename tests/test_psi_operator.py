import numpy as np
from functools import partial
import dask.array as da
from numpy.testing import assert_array_almost_equal
from pfb.prox.prox_21 import prox_21
from pfb.prox.prox_21m import prox_21m
from pfb.prox.prox_21m import prox_21m_numba
from pfb.operators.psi import im2coef
from pfb.operators.psi import coef2im
import pywt
import pytest
from numba.typed import Dict
from pfb.wavelets.wavelets import (wavedecn, waverecn, unravel_coeffs,
                                   ravel_coeffs, wavelet_setup)

pmp = pytest.mark.parametrize

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_psi(nx, ny, nband, nlevels):
    """
    Check that decomposition + reconstruction is the identity
    """
    # image = pywt.data.aero()
    image = np.random.randn(nx, ny)
    nu = 1.0  + 0.1 * np.arange(nband)

    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)

    # set up dictionary
    bases = ('self','db1','db2','db3','db4','db5')
    nbasis = len(bases)
    iys, sys, ntot, nmax = wavelet_setup(x[0:1], bases, nlevels)
    ntot = tuple(ntot)

    psiH = partial(im2coef, bases=bases, ntot=ntot, nmax=nmax,
                   nlevels=nlevels)
    psi = partial(coef2im, bases=bases, ntot=ntot,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    alpha = np.zeros((nband, nbasis, nmax), dtype=x.dtype)
    xrec = np.zeros((nband, nx, ny), dtype=x.dtype)

    # decompose
    psiH(x, alpha)
    # reconstruct
    psi(alpha, xrec)

    # the nbasis is required here because the operator is not normalised
    assert_array_almost_equal(nbasis*x, xrec, decimal=12)

@pmp("nx", [120, 240])
@pmp("ny", [32, 150])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_prox21(nx, ny, nband, nlevels):
    """
    Check that applying the prox with zero step size is the identity
    """
    image = pywt.data.aero()

    nu = 1.0  + 0.1 * np.arange(nband)

    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)

    # set up dictionary info
    bases = ('self','db1','db2','db3','db4','db5')
    nbasis = len(bases)
    iys, sys, ntot, nmax = wavelet_setup(x, bases, nlevels)
    ntot = tuple(ntot)
    psiH = partial(im2coef, bases=bases, ntot=ntot, nmax=nmax,
                   nlevels=nlevels)
    psi = partial(coef2im, bases=bases, ntot=ntot,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    weights_21 = np.ones((nbasis, nmax))
    sig_21 = 0.0

    alpha = np.zeros((nband, nbasis, nmax), dtype=x.dtype)
    xrec = np.zeros((nband, nx, ny), dtype=x.dtype)

    psiH(x, alpha)

    y = prox_21(alpha, sig_21, weights_21)

    psi(y, xrec)

    # the nbasis is required here because the operator is not normalised
    assert_array_almost_equal(nbasis*x, xrec, decimal=12)

@pmp("nmax", [1234, 240])
@pmp("nbasis", [1, 5])
@pmp("nband", [1, 3, 6])
def test_prox21m(nband, nbasis, nmax):
    # check numba implementation matches numpy even when output contains random
    # numbers initially
    v = np.random.randn(nband, nbasis, nmax)
    vout = np.random.randn(nband, nbasis, nmax)
    l1weight = np.ones((nbasis, nmax))
    sigma = 1e-3
    res = prox_21m(v, sigma, weight=l1weight)
    prox_21m_numba(v, vout, sigma, weight=l1weight)

    assert_array_almost_equal(res, vout, decimal=12)
