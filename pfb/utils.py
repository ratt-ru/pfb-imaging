import numpy as np
from numba import njit, prange
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

from .operators import PSI, DaskPSI


def data_from_header(hdr, axis=3):
    npix = hdr['NAXIS' + str(axis)]
    refpix = hdr['CRPIX' + str(axis)]
    delta = hdr['CDELT' + str(axis)]  # assumes units are Hz
    ref_val = hdr['CRVAL' + str(axis)]
    return ref_val + np.arange(1 - refpix, 1 + npix - refpix) * delta

def load_fits(name, dtype=np.float64):
    data = fits.getdata(name)
    if len(data.shape) == 3:
        return np.ascontiguousarray(np.transpose(data[:, ::-1].astype(dtype), axes=(0, 2, 1)))
    elif len(data.shape) == 2:
        return np.ascontiguousarray(data[::-1].T.astype(dtype))
    else:
        raise ValueError("Unsupported number of axes for fits file %s"%name)

def save_fits(name, data, hdr, overwrite=True, dtype=np.float32):
    hdu = fits.PrimaryHDU(header=hdr)
    if len(data.shape) == 3:
        hdu.data = np.transpose(data, axes=(0, 2, 1))[:, ::-1].astype(dtype)
    elif len(data.shape) == 2:
        hdu.data = data.T[::-1].astype(dtype)
    else:
        raise ValueError("Unsupported number of axes for fits file %s"%name)
    hdu.writeto(name, overwrite=overwrite)


def prox_21(p, sig_21, weights_21, psi=None):
    nchan, nx, ny = p.shape
    if psi is None:
        # l21 norm
        # meanp = norm(p.reshape(nchan, nx*ny), axis=0)
        meanp = np.mean(p.reshape(nchan, nx*ny), axis=0)
        l2_soft = np.maximum(meanp - sig_21 * weights_21, 0.0)
        indices = np.nonzero(meanp)
        ratio = np.zeros(meanp.shape, dtype=np.float64)
        ratio[indices] = l2_soft[indices]/meanp[indices]
        x = (p.reshape(nchan, nx*ny) * ratio[None, :]).reshape(nchan, nx, ny)
        x[x<0] = 0.0
    elif type(psi) is PSI:
        nchan, nx, ny = p.shape
        x = np.zeros(p.shape, p.dtype)
        for k in range(psi.nbasis):
            v = psi.hdot(p, k)
            # get 2-norm along spectral axis
            l2norm = norm(v, axis=0)
            # impose average sparsity
            l2_soft = np.maximum(np.abs(l2norm) - sig_21 * weights_21[k], 0.0) * np.sign(l2norm)
            indices = np.nonzero(l2norm)
            ratio = np.zeros(l2norm.shape, dtype=np.float64)
            ratio[indices] = l2_soft[indices]/l2norm[indices]
            v *= ratio[None]
            x += psi.dot(v, k)
        x[x<0] = 0.0

    elif type(psi) is DaskPSI:
        v = psi.dot(p)
        # 2-norm along spectral axis
        l2_norm = da.linalg.norm(v, axis=1)
        w = sig_21*weights_21
        l2_soft = da.maximum(da.absolute(l2_norm) - sig_21*w, 0.0)*da.sign(l2_norm)

        def safe_ratio(l2_norm, l2_soft):
            result = np.zeros_like(l2_norm)
            mask = l2_norm != 0
            result[mask] = l2_norm[mask] /  l2_soft[mask]
            return result

        r = da.blockwise(safe_ratio, ("basis", "nx", "ny"),
                         l2_norm, ("basis", "nx", "ny"),
                         l2_soft, ("basis", "nx", "ny"),
                         dtype=l2_norm.dtype)

        # apply inverse operator
        x = psi.hdot(v * r[:, None, :, :])
        # Sum over bases
        x = x.sum(axis=0)

        def ensure_positivity(x):
            x = x.copy()
            x[x < 0] = 0.0
            return x

        x = x.map_blocks(ensure_positivity, dtype=x.dtype)


    return x


def robust_reweight(v, residuals):
    """
    Find the robust weights corresponding to a soln that generated residuals

    v - initial guess for degrees of freedom parameter (float)
    residuals - array containing residuals of soln (N x 1)

    Note only one dimensional data points currently supported
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


    # v, f, d = fmin_l_bfgs_b(func, v, args=(nrow, ressq), pgtol=1e-2, approx_grad=False, bounds=[(1e-3, 30)])
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

def set_wcs(cell_x, cell_y, nx, ny, radec, freq):

    w = WCS(naxis=3)
    w.wcs.ctype = ['RA---SIN','DEC--SIN','FREQ']
    w.wcs.cdelt[0] = -cell_x/3600.0
    w.wcs.cdelt[1] = cell_y/3600.0
    # w.wcs.cdelt[2] = 1
    w.wcs.cunit = ['deg','deg','Hz']
    w.wcs.crval = [radec[0]*180.0/np.pi,radec[1]*180.0/np.pi, 0.0]
    w.wcs.crpix = [1 + nx//2,1 + ny//2, 1]

    if freq.size > 1:
        w.wcs.crval[2] = freq[0]
        df = freq[1]-freq[0]
        w.wcs.cdelt[2] = df
        fmean = np.mean(freq)
    else:
        fmean = freq

    header = w.to_header()
    header['RESTFRQ'] = fmean
    header['ORIGIN'] = 'pfb-clean'
    header['BTYPE'] = 'Intensity'
    header['BUNIT'] = 'Jy/beam'
    header['SPECSYS'] = 'TOPOCENT'

    return header

def compare_headers(hdr1, hdr2):
    for key in hdr1.keys():
        try:
            assert hdr1[key] == hdr2[key]
        except:
            raise ValueError("Headers do not match on key %s"%key)
