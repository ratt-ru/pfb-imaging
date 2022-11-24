import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pfb.utils.misc import to4d
import dask.array as da
from dask import delayed
from datetime import datetime
from casacore.quanta import quantity


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
            unit='Jy/beam', GuassPar=None, unix_time=None):
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
    if unix_time is not None:
        header['UTC_TIME'] = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')

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

@delayed
def normwsum(data, wsum):
    if wsum > 0:
        data /= wsum
    return data

def dds2fits(dds, column, outname, norm_wsum=True, otype=np.float32):
    imsout = []
    basename = outname + '_' + column.lower()
    for ds in dds:
        t = ds.timeid
        b = ds.bandid
        name = basename + f'_time{t:04d}_band{b:04d}.fits'
        data = ds.get(column).data
        if norm_wsum:
            data = normwsum(data, ds.WSUM.data[0])
            unit = 'Jy/beam'
        else:
            unit = 'Jy/pixel'
        radec = (ds.ra, ds.dec)
        cell_deg = np.rad2deg(ds.cell_rad)
        nx, ny = ds.get(column).shape
        unix_time = quantity(f'{ds.time_out}s').to_unix_time()
        hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, ds.freq_out,
                      unit=unit, unix_time=unix_time)
        imout = delayed(save_fits)(name, data, hdr, overwrite=True,
                                   dtype=np.float32)
        imsout.append(imout)
    return imsout


def dds2fits_mfs(dds, column, outname, norm_wsum=True, otype=np.float32):
    times_out = []
    freqs = []
    for ds in dds:
        times_out.append(ds.time_out)
        freqs.append(ds.freq_out)
    times_out = np.unique(np.array(times_out))
    ntimes_out = times_out.size
    freq_out = np.mean(np.unique(np.array(freqs)))
    basename = outname + '_' + column.lower()
    imsout = []
    nx, ny = dds[0].get(column).shape
    datas = [da.zeros((nx, ny), chunks=(-1, -1),
                dtype=otype) for _ in range(ntimes_out)]
    wsums = [da.zeros(1) for _ in range(ntimes_out)]
    counts = [da.zeros(1) for _ in range(ntimes_out)]
    radecs = [[] for _ in range(ntimes_out)]
    cell_deg = np.rad2deg(dds[0].cell_rad)
    for ds in dds:
        t = ds.timeid
        datas[t] += ds.get(column).data
        wsums[t] += ds.WSUM.data[0]
        counts[t] += 1
        radecs[t] = (ds.ra, ds.dec)
    for t in range(ntimes_out):
        name = basename + f'_time{t:04d}_mfs.fits'
        if norm_wsum:
            data = normwsum(datas[t], wsums[t])
            unit = 'Jy/beam'
        else:
            data = normwsum(datas[t], counts[t])
            unit = 'Jy/pixel'
        unix_time = quantity(f'{times_out[t]}s').to_unix_time()
        hdr = set_wcs(cell_deg, cell_deg, nx, ny, radecs[t], freq_out,
                      unit=unit, unix_time=unix_time)
        imout = delayed(save_fits)(name, data, hdr, overwrite=True,
                                   dtype=np.float32)
        imsout.append(imout)

    return imsout
