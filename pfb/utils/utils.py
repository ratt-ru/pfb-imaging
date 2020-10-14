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


def data_from_header(hdr, axis=3):
    npix = hdr['NAXIS' + str(axis)]
    refpix = hdr['CRPIX' + str(axis)]
    delta = hdr['CDELT' + str(axis)] 
    ref_val = hdr['CRVAL' + str(axis)]
    return ref_val + np.arange(1 - refpix, 1 + npix - refpix) * delta

def load_fits(name, dtype=np.float64):
    data = fits.getdata(name)
    if len(data.shape) == 4:
        return np.ascontiguousarray(np.transpose(data[:, :, ::-1].astype(dtype), axes=(0, 1, 3, 2)))
    elif len(data.shape) == 3:
        return np.ascontiguousarray(np.transpose(data[:, ::-1].astype(dtype), axes=(0, 2, 1)))
    elif len(data.shape) == 2:
        return np.ascontiguousarray(data[::-1].T.astype(dtype))
    else:
        raise ValueError("Unsupported number of axes for fits file %s"%name)

def save_fits(name, data, hdr, overwrite=True, dtype=np.float32):
    hdu = fits.PrimaryHDU(header=hdr)
    if len(data.shape) == 4:
        hdu.data = np.transpose(data, axes=(0, 1, 3, 2))[:, ::-1].astype(dtype)
    elif len(data.shape) == 3:
        hdu.data = np.transpose(data, axes=(0, 2, 1))[:, ::-1].astype(dtype)
    elif len(data.shape) == 2:
        hdu.data = data.T[::-1].astype(dtype)
    else:
        raise ValueError("Unsupported number of axes for fits file %s"%name)
    hdu.writeto(name, overwrite=overwrite)


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

def prox_21(v, sigma, weights):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape nbasis x nband x ntot where

    nbasis  - number of orthogonal bases
    nband   - number of imaging bands
    ntot    - total number of coefficients for each basis (must be equal)
    """
    l2_norm = norm(v, axis=1)  # drops freq axis
    l2_soft = np.maximum(l2_norm - sigma * weights, 0.0)  # norm is always positive
    mask = l2_norm != 0
    ratio = np.zeros(mask.shape, dtype=v.dtype)
    ratio[mask] = l2_soft[mask] / l2_norm[mask]
    return v * ratio[:, None, :]  # restore freq axis


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

def load_fits_contiguous(name):
    arr = fits.getdata(name)
    if arr.ndim == 4: 
        # figure out which axes to keep from header
        hdr = fits.getheader(name)
        if hdr["CTYPE4"].lower() == 'stokes':
            arr = arr[0]
        else:
            arr = arr[:, 0]
    # reverse last spatial axis then transpose (f -> c contiguous)
    arr = np.transpose(arr[:, :, ::-1], axes=(0, 2, 1))
    return np.ascontiguousarray(arr, dtype=np.float64)

def set_header_info(mhdr, ref_freq, freq_axis, args, beampars):
    hdr_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                'NAXIS4', 'CTYPE1', 'CTYPE2', 'CTYPE3', 'CTYPE4', 'CRPIX1',
                'CRPIX2', 'CRPIX3', 'CRPIX4', 'CRVAL1', 'CRVAL2', 'CRVAL3',
                'CRVAL4', 'CDELT1', 'CDELT2', 'CDELT3', 'CDELT4']

    new_hdr = {}
    for key in hdr_keys:
        new_hdr[key] = mhdr[key]

    if freq_axis == 3:
        new_hdr["NAXIS3"] = 1
        new_hdr["CRVAL3"] = ref_freq
    elif freq_axis == 4:
        new_hdr["NAXIS4"] = 1
        new_hdr["CRVAL4"] = ref_freq

    new_hdr['BMAJ'] = beampars[0]
    new_hdr['BMIN'] = beampars[1]
    new_hdr['BPA'] = beampars[2]

    new_hdr = fits.Header(new_hdr)

    return new_hdr

def get_fits_freq_space_info(hdr):
    if hdr['CUNIT1'].lower() != "deg":
        raise ValueError("Image coordinates must be in degrees")
    npix_l = hdr['NAXIS1']
    refpix_l = hdr['CRPIX1']
    delta_l = hdr['CDELT1']
    l_coord = np.arange(1 - refpix_l, 1 + npix_l - refpix_l)*delta_l

    if hdr['CUNIT2'].lower() != "deg":
        raise ValueError("Image coordinates must be in degrees")
    npix_m = hdr['NAXIS2']
    refpix_m = hdr['CRPIX2']
    delta_m = hdr['CDELT2']
    m_coord = np.arange(1 - refpix_m, 1 + npix_m - refpix_m)*delta_m

    # get frequencies
    if hdr["CTYPE4"].lower() == 'freq':
        freq_axis = 4
        nband = hdr['NAXIS4']
        refpix_nu = hdr['CRPIX4']
        delta_nu = hdr['CDELT4']  # assumes units are Hz
        ref_freq = hdr['CRVAL4']
        ncorr = hdr['NAXIS3']
    elif hdr["CTYPE3"].lower() == 'freq':
        freq_axis = 3
        nband = hdr['NAXIS3']
        refpix_nu = hdr['CRPIX3']
        delta_nu = hdr['CDELT3']  # assumes units are Hz
        ref_freq = hdr['CRVAL3']
        ncorr = hdr['NAXIS4']
    else:
        raise ValueError("Freq axis must be 3rd or 4th")

    if ncorr > 1:
        raise ValueError("Only Stokes I cubes supported")

    freqs = ref_freq + np.arange(1 - refpix_nu,
                                 1 + nband - refpix_nu) * delta_nu

    return l_coord, m_coord, freqs, ref_freq, freq_axis
