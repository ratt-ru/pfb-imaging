import numpy as np
from scipy.special import digamma, polygamma
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import norm
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
from pyrap.tables import table
from numpy.testing import assert_array_equal
from time import time
from astropy.io import fits
from astropy.wcs import WCS

from pfb.operators import PSI, DaskPSI





# def prox_21(p, sig_21, weights_21, psi=None, positivity=False):
#     nchan, nx, ny = p.shape
#     if psi is None:
#         # l21 norm
#         # meanp = norm(p.reshape(nchan, nx*ny), axis=0)
#         meanp = np.mean(p.reshape(nchan, nx*ny), axis=0)
#         l2_soft = np.maximum(meanp - sig_21 * weights_21, 0.0)
#         indices = np.nonzero(meanp)
#         ratio = np.zeros(meanp.shape, dtype=np.float64)
#         ratio[indices] = l2_soft[indices]/meanp[indices]
#         x = (p.reshape(nchan, nx*ny) * ratio[None, :]).reshape(nchan, nx, ny)
        
#     elif type(psi) is PSI:
#         nchan, nx, ny = p.shape
#         x = np.zeros(p.shape, p.dtype)
#         v = psi.hdot(p)
#         l2_norm = norm(v, axis=1)  # drops freq axis
#         l2_soft = np.maximum(np.abs(l2_norm) - sig_21 * weights_21, 0.0) * np.sign(l2_norm)
#         mask = l2_norm[:, :] != 0
#         ratio = np.zeros(mask.shape, dtype=psi.real_type)
#         ratio[mask] = l2_soft[mask] / l2_norm[mask]
#         v *= ratio[:, None, :]  # restore freq axis
#         x = psi.dot(v)

#     elif type(psi) is DaskPSI:
#         v = psi.hdot(p)
#         # 2-norm along spectral axis
#         l2_norm = da.linalg.norm(v, axis=1)

#         def safe_ratio(l2_norm, weights, sig_21):
#             l2_soft = np.maximum(np.abs(l2_norm) - weights*sig_21, 0.0)*np.sign(l2_norm)
#             result = np.zeros_like(l2_norm)
#             mask = l2_norm != 0
#             result[mask] = l2_soft[mask] / l2_norm[mask]
#             return result

#         r = da.blockwise(safe_ratio, ("basis", "nx", "ny"),
#                          l2_norm, ("basis", "nx", "ny"),
#                          weights_21, ("basis", "nx", "ny"),
#                          sig_21, None,
#                          dtype=l2_norm.dtype)

#         # apply inverse operator
#         x = psi.dot(v * r[:, None, :, :])
#         # Sum over bases
#         x = x.sum(axis=0)

#         def ensure_positivity(x):
#             x = x.copy()
#             x[x < 0] = 0.0
#             return x

#         x = x.map_blocks(ensure_positivity, dtype=x.dtype)

#     if positivity:
#         x[x<0] = 0.0

#     return x

def prox_21(v, sigma, weights, axis=1):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape nbasis x nband x ntot where

    nbasis  - number of orthogonal bases
    nband   - number of imaging bands
    ntot    - total number of coefficients for each basis (must be equal)
    """
    l2_norm = norm(v, axis=axis)  # drops freq axis
    l2_soft = np.maximum(l2_norm - sigma * weights, 0.0)  # norm is always positive
    mask = l2_norm != 0
    ratio = np.zeros(mask.shape, dtype=v.dtype)
    ratio[mask] = l2_soft[mask] / l2_norm[mask]
    return v * np.expand_dims(ratio, axis=axis)  # restore freq axis


def robust_reweight(residuals, weights, v=None):
    """
    Find the robust weights corresponding to a soln that generated residuals

    residuals - residuals i.e. (data - model) (nrow, nchan, ncorr)
    weights - inverse of data covariance (nrow, nchan, ncorr)
    v - initial guess for degrees for freedom parameter (float)
    corrs - which correlation axes to compute new weights for (default is for LL and RR (or XX and)) 

    Correlation axis not currently supported
    """
    # elements of Mahalanobis distance (Delta^2_i's)
    nrow, nchan = residuals.shape
    ressq = (residuals.conj()*residuals).real

    # func to solve for degrees of freedom parameter
    def func(v, N, ressq):
        # expectation values
        tmp = v+ressq
        Etau = (v + 1)/tmp
        Elogtau = digamma(v + 1) - np.log(tmp)

        # derivatives of expectation value terms
        dEtau = 1.0/tmp - (v+1)/tmp**2
        dElogtau = polygamma(1, v + 1) - 1.0/(v + ressq)
        return N*np.log(v) - N*digamma(v) + np.sum(Elogtau) - np.sum(Etau), N/v - N*polygamma(1, v) + np.sum(dElogtau) - np.sum(dEtau)


    v, f, d = fmin_l_bfgs_b(func, v, args=(nrow, ressq), pgtol=1e-2, approx_grad=False, bounds=[(1e-3, 30)])
    Etau = (v + 1.0)/(v + ressq)  # used as new weights
    Lambda = np.mean(ressq*Etau, axis=0)
    return v, np.sqrt(Etau / Lambda[None, :])

def test_convolve(R, psf, args):
    x = np.random.randn(args.channels_out, args.nx, args.ny)

    res1 = R.convolve(x)
    res2 = psf.convolve(x)

    max_diff = np.abs(res1 - res2).max()/res1.max()

    print("Max frac diff is %5.5e and precision is %5.5e"%(max_diff, args.precision))

def test_adjoint(R):
    x = np.random.randn(R.nband, R.nx, R.ny)
    y = np.random.randn(R.nrow, R.nchan).astype(np.complex128)

    # y.H R x = x.H R.H y
    lhs = np.vdot(y, R.dot(x))
    rhs = np.vdot(x, R.hdot(y))
    print(" Natural = ", (lhs - rhs)/rhs)

    lhs = np.vdot(y, R.udot(x))
    rhs = np.vdot(x, R.uhdot(y))
    print(" Uniform = ", (lhs - rhs)/rhs)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')



def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.)):
    S0, S1, PA = GaussPar
    Smaj = np.maximum(S0, S1)
    Smin = np.minimum(S0, S1)
    A = np.array([[1. / Smin ** 2, 0],
                  [0, 1. / Smaj ** 2]])

    c, s, t = np.cos, np.sin, np.deg2rad(-PA)
    R = np.array([[c(t), -s(t)],
                  [s(t), c(t)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (5 * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    ind = np.argwhere(xflat**2 + yflat**2 <= extent).squeeze()
    idx = ind[:, 0]
    idy = ind[:, 1]
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2*np.sqrt(2*np.log(2))
    tmp = np.exp(-fwhm_conv*R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp
    return np.ascontiguousarray(gausskern.reshape(sOut),
                                dtype=np.float64)


def give_edges(p, q, nx, ny):
    # image overlap edges
    # left edge for x coordinate
    dxl = p - nx//2
    xl = np.maximum(dxl, 0)
    # right edge for x coordinate
    dxu = p + nx//2
    xu = np.minimum(dxu, nx)
    # left edge for y coordinate
    dyl = q - ny//2
    yl = np.maximum(dyl, 0)
    # right edge for y coordinate
    dyu = q + ny//2
    yu = np.minimum(dyu, ny)

    # PSF overlap edges
    
    xlpsf = np.maximum(nx//2 - p, 0)
    xupsf = np.minimum(3*nx//2 - p, nx)
    ylpsf = np.maximum(ny//2 - q, 0)
    yupsf = np.minimum(3*ny//2 - q, ny)


    return slice(xl, xu), slice(yl, yu), slice(xlpsf, xupsf), slice(ylpsf, yupsf)
