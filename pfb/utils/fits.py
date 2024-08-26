import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import dask.array as da
from dask import delayed
from datetime import datetime
from casacore.quanta import quantity
from astropy.time import Time
from pfb.utils.naming import xds_from_list


def to4d(data):
    if data.ndim == 4:
        return data
    elif data.ndim == 2:
        return data[None, None]
    elif data.ndim == 3:
        return data[None]
    elif data.ndim == 1:
        return data[None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")


def data_from_header(hdr, axis=3):
    npix = hdr['NAXIS' + str(axis)]
    refpix = hdr['CRPIX' + str(axis)]
    delta = hdr['CDELT' + str(axis)]
    ref_val = hdr['CRVAL' + str(axis)]
    return ref_val + np.arange(1 - refpix, 1 + npix - refpix) * delta, ref_val


def load_fits(name, dtype=np.float32):
    data = fits.getdata(name)
    data = np.transpose(to4d(data), axes=(0, 1, 3, 2))
    return np.require(data, dtype=dtype, requirements='C')


def save_fits(data, name, hdr, overwrite=True, dtype=np.float32):
    hdu = fits.PrimaryHDU(header=hdr)
    data = np.transpose(to4d(data), axes=(0, 1, 3, 2))
    hdu.data = np.require(data, dtype=dtype, requirements='F')
    hdu.writeto(name, overwrite=overwrite)
    return


def set_wcs(cell_x, cell_y, nx, ny, radec, freq,
            unit='Jy/beam', GuassPar=None, ms_time=None):
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
        nchan = freq.size
        crpix3 = nchan//2+1
        ref_freq = freq[crpix3]
        df = freq[1]-freq[0]
        w.wcs.cdelt[2] = df
    else:
        if isinstance(freq, np.ndarray) and freq.size == 1:
            ref_freq = freq[0]
        else:
            ref_freq = freq
        crpix3 = 1
    w.wcs.crval = [radec[0]*180.0/np.pi, radec[1]*180.0/np.pi, ref_freq, 1]
    w.wcs.crpix = [1 + nx//2, 1 + ny//2, crpix3, 1]

    header = w.to_header()
    header['RESTFRQ'] = ref_freq
    header['ORIGIN'] = 'pfb-imaging'
    header['BTYPE'] = 'Intensity'
    header['BUNIT'] = unit
    header['SPECSYS'] = 'TOPOCENT'
    if ms_time is not None:
        # TODO - this is probably a bit of a round about way of doing this
        unix_time = quantity(f'{ms_time}s').to_unix_time()
        utc_iso = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
        header['UTC_TIME'] = utc_iso
        t = Time(utc_iso)
        t.format = 'fits'
        header['DATE-OBS'] = t.value

    # What are these used for?
    # if 'LONPOLE' in header:
    #     header.pop('LONPOLE')
    # if 'LATPOLE' in header:
    #     header.pop('LATPOLE')
    # if 'RADESYS' in header:
    #     header.pop('RADESYS')
    # if 'MJDREF' in header:
    #     header.pop('MJDREF')

    header['EQUINOX'] = '2000. / J2000'
    header['BSCALE'] = 1.0
    header['BZERO'] = 0.0

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


def add_beampars(hdr, GaussPar, GaussPars=None, unit2deg=1.0):
    """
    Add beam keywords to header.
    GaussPar - MFS beam pars
    GaussPars - beam pars for cube
    """
    if len(GaussPar) == 1:
        GaussPar = GaussPar[0]
    elif len(GaussPar) != 3:
        raise ValueError('Invalid value for GaussPar')

    if not np.isnan(GaussPar).any():
        hdr['BMAJ'] = GaussPar[0]*unit2deg
        hdr['BMIN'] = GaussPar[1]*unit2deg
        hdr['BPA'] = GaussPar[2]*unit2deg

    if GaussPars is not None:
        for i in range(len(GaussPars)):
            if not np.isnan(GaussPars[i]).any():
                hdr['BMAJ' + str(i+1)] = GaussPars[i][0]*unit2deg
                hdr['BMIN' + str(i+1)] = GaussPars[i][1]*unit2deg
                hdr['PA' + str(i+1)] = GaussPars[i][2]*unit2deg

    return hdr


def dds2fits(dsl, column, outname, norm_wsum=True,
             otype=np.float32, nthreads=1,
             do_mfs=True, do_cube=True):
    basename = outname + '_' + column.lower()
    if norm_wsum:
        unit = 'Jy/beam'
    else:
        unit = 'Jy/pixel'
    dds = xds_from_list(dsl, drop_all_but=column,
                        nthreads=nthreads,
                        order_freq=False)
    timeids = [ds.timeid for ds in dds]
    freqs = [ds.freq_out for ds in dds]
    freqs = np.unique(freqs)
    nband = freqs.size
    nx, ny = getattr(dds[0], column).shape
    for timeid in timeids:
        cube = np.zeros((nband, nx, ny))
        wsums = np.zeros(nband)
        wsum = 0.0
        for i, ds in enumerate(dds):
            if ds.timeid == timeid:
                b = int(ds.bandid)
                cube[b] = ds.get(column).values
                wsums[b] = ds.wsum
                wsum += ds.wsum
        radec = (ds.ra, ds.dec)
        cell_deg = np.rad2deg(ds.cell_rad)
        nx, ny = ds.get(column).shape
        unix_time = quantity(f'{ds.time_out}s').to_unix_time()
        fmask = wsums > 0

        if do_mfs:
            freq_mfs = np.sum(freqs*wsums)/wsum
            if norm_wsum:
                # already weighted by wsum
                cube_mfs = np.sum(cube, axis=0)/wsum
            else:
                # weigted sum
                cube_mfs = np.sum(cube[fmask]*wsums[fmask, None, None],
                                  axis=0)/wsum

            name = basename + f'_time{timeid}_mfs.fits'
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_mfs,
                        unit=unit, ms_time=ds.time_out)
            hdr['WSUM'] = wsum
            save_fits(cube_mfs, name, hdr, overwrite=True,
                      dtype=otype)

        if do_cube:
            if norm_wsum:
                cube[fmask] = cube[fmask]/wsums[fmask, None, None]
            name = basename + f'_time{timeid}.fits'
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freqs,
                          unit=unit, ms_time=ds.time_out)
            for b in range(fmask.size):
                hdr[f'WSUM{b}'] = wsums[b]
            save_fits(cube, name, hdr, overwrite=True,
                      dtype=otype)

    return column

