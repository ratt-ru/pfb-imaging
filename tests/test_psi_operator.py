import numpy as np
from functools import partial
import dask.array as da
from numpy.testing import assert_array_almost_equal
from pfb.prox.prox_21 import prox_21
from pfb.operators.psi import im2coef, coef2im
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
    iys, sys, ntots, nmax = wavelet_setup(x, bases, nlevels)
    ntots = tuple(ntots)
    psiH = partial(im2coef, bases=bases, ntot=ntots, nmax=nmax,
                   nlevels=nlevels)
    psi = partial(coef2im, bases=bases, ntot=ntots,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    # decompose
    alpha = psiH(x)
    # reconstruct
    xrec = psi(alpha)

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
    iys, sys, ntots, nmax = wavelet_setup(x, bases, nlevels)
    ntots = tuple(ntots)
    psiH = partial(im2coef, bases=bases, ntot=ntots, nmax=nmax,
                   nlevels=nlevels)
    psi = partial(coef2im, bases=bases, ntot=ntots,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    weights_21 = np.ones((nbasis, nmax))
    sig_21 = 0.0

    alpha = psiH(x)

    y = prox_21(alpha, sig_21, weights_21)

    xrec = psi(y)

    # the nbasis is required here because the operator is not normalised
    assert_array_almost_equal(nbasis*x, xrec, decimal=12)
