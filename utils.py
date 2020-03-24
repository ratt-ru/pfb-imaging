import numpy as np
from numba import njit, prange
from scipy.special import digamma, polygamma
from scipy.optimize import fmin_l_bfgs_b
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
from pyrap.tables import table
from numpy.testing import assert_array_equal
from astropy.wcs import WCS

@njit(parallel=True, nogil=True, cache=True, fastmath=True, inline='always')
def freqmul(A, x):
    nchan, npix = x.shape
    out = np.zeros((nchan, npix), dtype=x.dtype)
    for i in prange(npix):
        for j in range(nchan):
            for k in range(nchan):
                out[j, i] += A[j, k] * x[k, i]
    return out

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

def init_data(args):
    print("Reading data")
    if args.precision < 1e-7:
        complex_type = np.complex64
        real_type = np.float32
    else:
        complex_type = np.complex128
        real_type = np.float64

    ti = time()
    uvw_list = []
    data_list = []
    weight_list = []
    freqp = None
    nrow = 0
    nvis = 0
    for ms_name in args.ms:
        print("Loading data from ", ms_name)
        xds = xds_from_ms(ms_name,
                          columns=('UVW', 'FLAG', args.data_column, args.weight_column),
                          group_cols=["FIELD_ID"],
                          chunks={"row": args.row_chunks})[args.field]
        
        data_full = getattr(xds, args.data_column).data
        _, nchan, ncorr = data_full.shape
        weight_full = getattr(xds, args.weight_column).data
        if len(weight_full.shape) < 3:
            # print("Assuming weights were taken from less informative "
            #     "WEIGHT column. Tiling over frequency.")
            weight_full = da.tile(weight_full[:, None, :], (1, nchan, 1))
        flags_full = xds.FLAG
        
        # taking weighted sum to get Stokes I
        data = ((weight_full[:, :, 0] * data_full[:, :, 0] + weight_full[:, :, ncorr-1] * data_full[:, :, ncorr-1])/(weight_full[:, :, 0] + weight_full[:, :, ncorr-1])).compute()
        weight = (weight_full[:, :, 0] + weight_full[:, :, ncorr-1]).compute()
        flags = (flags_full[:, :, 0] | flags_full[:, :, ncorr-1]).compute()

        nrowtmp = np.sum(~flags)
        nrow += nrowtmp

        nvis += data.size

        print("Effective number of rows for ms = ", nrowtmp)
        print("Number of visibilities for ms = ", data.size)

        # only keep data where both correlations are unflagged
        data = np.where(~flags, data, 0.0j)
        weight = np.where(~flags, weight, 0.0)
        nrow = np.sum(~flags)
        freq = xds_from_table(ms_name + '::SPECTRAL_WINDOW')[0].CHAN_FREQ.data.compute()
        if freqp is not None:
            try:
                assert np.array_equal(freqp, freq)
            except:
                raise RuntimeError("Not all MS Freqs match")

        data_list.append(data)
        weight_list.append(weight)
        uvw_list.append(xds.UVW.data.compute())
        freqp = freq

    data = np.concatenate(data_list)
    uvw = np.concatenate(uvw_list)
    weight = np.concatenate(weight_list)
    sqrtW = np.sqrt(weight)
    print("Time to read data = ", time()- ti)

    print("Effective number of rows total = ", nrow)
    print("Total number of visibilities = ", nvis)
    return {'data':data.astype(complex_type), 'uvw':uvw.astype(real_type), 
            'sqrtW':sqrtW.astype(real_type), 'freq':freq.astype(real_type)}

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

def concat_ms_to_I_tbl(ms, outname, cols=["DATA", "WEIGHT", "UVW"]):
    """
    Concatenate list of measurement sets to single table holding
    Stokes I data and weights

    ms - list of measurement sets to concatenate
    outname - name of the table to write
    cols - list of column names to concatenate

    """
    # Currently MS's need to have the same frequencies and only a single spw
    freq = None
    radec = None
    for ims in ms:
        if freq is None:
            freq = xds_from_table(ims + '::SPECTRAL_WINDOW')[0].CHAN_FREQ.data.compute()[0]
        else:
            tmpfreq = xds_from_table(ims + '::SPECTRAL_WINDOW')[0].CHAN_FREQ.data.compute()[0]
            assert_array_equal(freq, tmpfreq)

        if radec is None:
            radec = xds_from_table(ims + '::FIELD')[0].PHASE_DIR.data.compute()[0].squeeze()
        else:
            tmpradec = xds_from_table(ims + '::FIELD')[0].PHASE_DIR.data.compute()[0].squeeze()
            assert_array_equal(radec, tmpradec)
    nchan = freq.size

    # convert to Stokes I vis and concatenate
    fid = 0
    concat_cols = ('FLAG', 'FLAG_ROW', 'UVW', 'DATA', 'WEIGHT')
    nrows = 0
    dataf = []
    uvwf = []
    weightf = []
    for ims in ms:
        xds = xds_from_ms(ims, 
                          columns=concat_cols,
                          group_cols=["FIELD_ID"],
                          chunks={"row": 10000})[fid]
        
        nrow, _, ncorr = xds.DATA.shape
        nrows += nrow

        data = xds.DATA.data
        weight = xds.WEIGHT.data
        weight = da.tile(weight[:, None, :], (1, nchan, 1))
        uvw = xds.UVW.data
        flag = xds.FLAG.data

        data_I = ((weight[:, :, 0] * data[:, :, 0] + weight[:, :, ncorr-1] * data[:, :, ncorr-1])/(weight[:, :, 0] + weight[:, :, ncorr-1]))
        weight_I = (weight[:, :, 0] + weight[:, :, ncorr-1])
        flag_I = (flag[:, :, 0] | flag[:, :, ncorr-1])

        
        weight_I = da.where(~flag_I, weight_I, 0.0)
        data_I = da.where(~flag_I, data_I, 0.0j)

        dataf.append(data_I)
        uvwf.append(uvw)
        weightf.append(weight_I)

    data = da.concatenate(dataf, axis=0)
    weight = da.concatenate(weightf, axis=0)
    uvw = da.concatenate(uvwf, axis=0)

    data_vars = {
        'DATA':(('row', 'chan'), data),
        'WEIGHT':(('row', 'chan'), weight),
        'UVW':(('row', 'uvw'), uvw)
    }

    writes = xds_to_table([Dataset(data_vars)], outname, "ALL")
    dask.compute(writes)

    return freq, radec


def set_wcs(cell_x, cell_y, nx, ny, radec, freq):

    w = WCS(naxis=4)
    w.wcs.ctype = ['RA---SIN','DEC--SIN','STOKES','FREQ']
    w.wcs.cdelt[0] = -cell_x/3600.0
    w.wcs.cdelt[1] = cell_y/3600.0
    w.wcs.cdelt[2] = 1
    w.wcs.cunit = ['deg','deg','','Hz']
    w.wcs.crval = [radec[0]*180.0/np.pi,radec[1]*180.0/np.pi,0.0, 0.0]
    w.wcs.crpix = [1 + nx//2,1 + ny//2,1,1]

    if freq.size > 1:
        w.wcs.crval[3] = freq[0]
        df = freq[1]-freq[0]
        w.wcs.cdelt[3] = df
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
