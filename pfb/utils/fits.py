import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

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

def set_wcs(cell_x, cell_y, nx, ny, radec, freq, unit='Jy/beam'):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in degrees
    freq - frequencies in Hz
    """

    w = WCS(naxis=4)
    w.wcs.ctype = ['RA---SIN','DEC--SIN','FREQ','STOKES']
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = 'deg'
    w.wcs.cunit[1] = 'deg'
    w.wcs.cunit[2] = 'Hz'
    w.wcs.crval = [radec[0]*180.0/np.pi,radec[1]*180.0/np.pi, 0.0, 1]
    w.wcs.crpix = [1 + nx//2,1 + ny//2, 1, 1]

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
    header['BUNIT'] = unit
    header['SPECSYS'] = 'TOPOCENT'

    return header

def compare_headers(hdr1, hdr2):
    for key in hdr1.keys():
        try:
            assert hdr1[key] == hdr2[key]
        except:
            raise ValueError("Headers do not match on key %s"%key)

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