import numpy as np
from functools import partial
import dask.array as da
from numpy.testing import assert_array_almost_equal
from pfb.prox.prox_21 import prox_21, prox_21m
from pfb.operators.psi import im2coef, coef2im
import pywt
import pytest

pmp = pytest.mark.parametrize

@pmp("nx", [128, 250])
@pmp("ny", [64, 78])
@pmp("nband", [1, 3, 6])
@pmp("nlevels", [1, 2])
def test_psi(nx, ny, nband, nlevels):
    """
    Check that decomposition + reconstruction is the identity
    """
    image = pywt.data.aero()

    nu = 1.0  + 0.1 * np.arange(nband)

    x = image[None, 0:nx, 0:ny] * nu[:, None, None] ** (-0.7)

    # set up dictionary info
    bases = ['self','db1']
    ntots = []
    iys = {}
    sys = {}
    for base in bases:
        if base == 'self':
            y, iy, sy = x[0].ravel(), 0, 0
        else:
            alpha = pywt.wavedecn(x[0], base, mode='zero',
                                  level=nlevels)
            y, iy, sy = pywt.ravel_coeffs(alpha)
        iys[base] = iy
        sys[base] = sy
        ntots.append(y.size)

    # get padding info
    nmax = np.asarray(ntots).max()
    padding = []
    nbasis = len(ntots)
    for i in range(nbasis):
        padding.append(slice(0, ntots[i]))

    bases = da.from_array(np.array(bases, dtype=object), chunks=-1)
    ntots = da.from_array(np.array(ntots, dtype=object), chunks=-1)
    padding = da.from_array(np.array(padding, dtype=object), chunks=-1)
    psiH = partial(im2coef, bases=bases, ntot=ntots, nmax=nmax,
                   nlevels=nlevels)
    psi = partial(coef2im, bases=bases, padding=padding,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    # decompose
    alpha = psiH(x)
    # reconstruct
    xrec = psi(alpha)

    # the two is required here because the operator is not normalised
    # to have a spectral norm of one and there are two bases
    assert_array_almost_equal(2*x, xrec, decimal=12)

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
    bases = ['self','db1']
    ntots = []
    iys = {}
    sys = {}
    for base in bases:
        if base == 'self':
            y, iy, sy = x[0].ravel(), 0, 0
        else:
            alpha = pywt.wavedecn(x[0], base, mode='zero',
                                  level=nlevels)
            y, iy, sy = pywt.ravel_coeffs(alpha)
        iys[base] = iy
        sys[base] = sy
        ntots.append(y.size)

    # get padding info
    nmax = np.asarray(ntots).max()
    padding = []
    nbasis = len(ntots)
    for i in range(nbasis):
        padding.append(slice(0, ntots[i]))

    bases = da.from_array(np.array(bases, dtype=object), chunks=-1)
    ntots = da.from_array(np.array(ntots, dtype=object), chunks=-1)
    padding = da.from_array(np.array(padding, dtype=object), chunks=-1)
    psiH = partial(im2coef, bases=bases, ntot=ntots, nmax=nmax,
                   nlevels=nlevels)
    psi = partial(coef2im, bases=bases, padding=padding,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    weights_21 = np.ones((nbasis, nmax))
    sig_21 = 0.0

    alpha = psiH(x)

    y = prox_21(alpha, sig_21, weights_21)

    xrec = psi(y)

    # the two is required here because the operator is not normalised
    # to have a spectral norm of one and there are two bases
    assert_array_almost_equal(2*x, xrec, decimal=12)
