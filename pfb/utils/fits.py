import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pfb.utils.misc import to4d


def data_from_header(hdr, axis=3):
    npix = hdr['NAXIS' + str(axis)]
    refpix = hdr['CRPIX' + str(axis)]
    delta = hdr['CDELT' + str(axis)]
    ref_val = hdr['CRVAL' + str(axis)]
    return ref_val + np.arange(1 - refpix, 1 + npix - refpix) * delta, ref_val


def load_fits(name, dtype=np.float32):
    data = fits.getdata(name)
    data = np.transpose(to4d(data)[:, :, ::-1], axes=(0, 1, 3, 2))
    return np.require(data, dtype=dtype, requirements='C')


def save_fits(name, data, hdr, overwrite=True, dtype=np.float32):
    hdu = fits.PrimaryHDU(header=hdr)
    data = np.transpose(to4d(data), axes=(0, 1, 3, 2))[:, :, ::-1]
    hdu.data = np.require(data, dtype=dtype, requirements='F')
    hdu.writeto(name, overwrite=overwrite)


def set_wcs(cell_x, cell_y, nx, ny, radec, freq,
            unit='Jy/beam', GuassPar=None):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in radians
    freq - frequencies in Hz
    """

    w = WCS(naxis=4)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = 'deg'
    w.wcs.cunit[1] = 'deg'
    w.wcs.cunit[2] = 'Hz'
    if np.size(freq) > 1:
        ref_freq = freq[0]
    else:
        ref_freq = freq
    w.wcs.crval = [radec[0]*180.0/np.pi, radec[1]*180.0/np.pi, ref_freq, 1]
    # LB - y axis treated differently because of stupid fits convention
    w.wcs.crpix = [1 + nx//2, ny//2, 1, 1]

    if np.size(freq) > 1:
        w.wcs.crval[2] = freq[0]
        df = freq[1]-freq[0]
        w.wcs.cdelt[2] = df
        fmean = np.mean(freq)
    else:
        if isinstance(freq, np.ndarray):
            fmean = freq[0]
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
    '''
    utility function to ensure that WCS's are compatible
    'NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4',
    '''
    keys = ['CTYPE1', 'CTYPE2', 'CTYPE3', 'CTYPE4',
            'CDELT1', 'CDELT2', 'CDELT3', 'CDELT4',
            'CRPIX1', 'CRPIX2', 'CRPIX3', 'CRPIX4',
            'CUNIT1', 'CUNIT2', 'CUNIT3']
    for key in keys:
        try:
            assert hdr1[key] == hdr2[key]
        except BaseException:
            raise ValueError("Headers do not match on key %s. " % key, hdr1[key], hdr2[key])


def add_beampars(hdr, GaussPar, GaussPars=None):
    """
    Add beam keywords to header.
    GaussPar - MFS beam pars
    GaussPars - beam pars for cube
    """
    hdr['BMAJ'] = GaussPar[0]
    hdr['BMIN'] = GaussPar[1]
    hdr['BPA'] = GaussPar[2]

    if GaussPars is not None:
        for i in range(len(GaussPars)):
            hdr['BMAJ' + str(i+1)] = GaussPars[i][0]
            hdr['BMIN' + str(i+1)] = GaussPars[i][1]
            hdr['PA' + str(i+1)] = GaussPars[i][2]

    return hdr


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
