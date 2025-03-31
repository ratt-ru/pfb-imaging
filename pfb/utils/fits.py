import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime, timezone
from casacore.quanta import quantity
from astropy.time import Time
from pfb.utils.naming import xds_from_list
import xarray as xr


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
    data = np.transpose(to4d(data), axes=(1, 0, 3, 2))  # fits and beams table
    return np.require(data, dtype=dtype, requirements='C')


def save_fits(data, name, hdr, overwrite=True, dtype=np.float32, beams_hdu=None):
    hdu = fits.PrimaryHDU(header=hdr)
    data = np.transpose(to4d(data), axes=(1, 0, 3, 2))
    hdu.data = np.require(data, dtype=dtype, requirements='F')
    if beams_hdu is not None:
        hdul = fits.HDUList([hdu, beams_hdu])
        hdul.writeto(name, overwrite=overwrite)
    else:
        hdu.writeto(name, overwrite=overwrite)
    return


def set_wcs(cell_x, cell_y, nx, ny, radec, freq,
            unit='Jy/beam', GuassPar=None, ms_time=None,
            header=True, casambm=True):
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
    w.wcs.cunit[3] = ''
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
    w.wcs.equinox = 2000.0

    if header:
        header = w.to_header()
        header['RESTFRQ'] = ref_freq
        header['ORIGIN'] = 'pfb-imaging'
        header['BTYPE'] = 'Intensity'
        header['BUNIT'] = unit
        header['SPECSYS'] = 'TOPOCENT'
        if ms_time is not None:
            # TODO - probably a round about way of doing this
            unix_time = quantity(f'{ms_time}s').to_unix_time()
            utc_iso = datetime.fromtimestamp(unix_time,
                            tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
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

        # header['EQUINOX'] = 2000.0
        header['BSCALE'] = 1.0
        header['BZERO'] = 0.0
        header['CASAMBM'] = casambm  # we need this to pick up the beams table

        return header
    else:
        return w


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


def create_beams_table(beams_data, cell2deg):
    """
    Add a BEAMS subtable to a FITS file.
    
    Parameters:
    -----------
    filename : str
        The FITS file to modify
    beams_data : dict
        Dictionary containing arrays for:
        - chan_id: Channel indices
        - pol_id: Polarization indices
        - bmaj: Major axis values (in pixels)
        - bmin: Minor axis values (in pixels)
        - bpa: Position angles (in radians)
    """
    # Create the columns for the BEAMS table
    nband = beams_data.band.size
    npol = beams_data.corr.size
    band_id = []
    pol_id = []
    for b in range(nband):
        for p in range(npol):
            band_id.append(b)
            pol_id.append(p)

    # we need the transpose for C -> F ordering
    bmaj = beams_data.sel({'bpar': 'BMAJ'}).values.ravel() * cell2deg
    bmin = beams_data.sel({'bpar': 'BMIN'}).values.ravel() * cell2deg
    bpa = 90 - beams_data.sel({'bpar': 'BPA'}).values.ravel() * 180/np.pi
    col1 = fits.Column(name='BMAJ', format='1E', array=bmaj, unit='deg')
    col2 = fits.Column(name='BMIN', format='1E', array=bmin, unit='deg')
    col3 = fits.Column(name='BPA', format='1E', array=bpa, unit='deg')
    col4 = fits.Column(name='CHAN', format='1J', array=np.array(band_id))
    col5 = fits.Column(name='POL', format='1J', array=np.array(pol_id))
    
    # Create the BEAMS table HDU
    cols = fits.ColDefs([col1, col2, col3, col4, col5])
    beams_hdu = fits.BinTableHDU.from_columns(cols)
    beams_hdu.name = 'BEAMS'
    beams_hdu.header['EXTNAME'] = 'BEAMS'
    beams_hdu.header['EXTVER'] = 1
    beams_hdu.header['XTENSION'] = 'BINTABLE'
    beams_hdu.header.comments['XTENSION'] = 'Binary extension'
    beams_hdu.header['NCHAN'] = nband
    beams_hdu.header['NPOL'] = npol

    return beams_hdu


def dds2fits(dsl, column, outname, norm_wsum=True,
             otype=np.float32, nthreads=1,
             do_mfs=True, do_cube=True,
             psfpars_mfs=None):
    basename = outname + '_' + column.lower()
    if norm_wsum:
        unit = 'Jy/beam'
    else:
        unit = 'Jy/pixel'
    dds = xds_from_list(dsl, drop_all_but=[column, 'PSFPARSN', 'WSUM'],
                        nthreads=nthreads,
                        order_freq=False)
    timeids = np.unique(np.array([int(ds.timeid) for ds in dds]))
    freqs = [ds.freq_out for ds in dds]
    freqs = np.unique(freqs)
    nband = freqs.size
    for timeid in timeids:
        # filter by time ID
        dst = [ds for ds in dds if int(ds.timeid)==timeid]
        # concat creating a new band axis
        dsb = xr.concat(dst, dim='band')
        # LB - these seem to remain in the correct order
        dsb = dsb.assign_coords({'band': np.arange(nband)})
        wsums = dsb.WSUM.values
        wsum = dsb.WSUM.sum(dim='band').values
        cube = dsb.get(column).values
        _, _, nx, ny = cube.shape
        # these should be the same across band and corr axes
        radec = (dsb.ra, dsb.dec)
        cell_deg = np.rad2deg(dsb.cell_rad)

        if do_mfs:
            # we need a single freq_mfs for the cube
            freq_mfs = np.sum(freqs[:, None]*wsums)/wsum.sum()
            if norm_wsum:
                # already weighted by wsum
                cube_mfs = np.sum(cube, axis=0)/wsum[:, None, None]
            else:
                # weigted sum
                cube_mfs = np.sum(cube*wsums[:, :, None, None],
                                  axis=0)/wsum[:, None, None]

            name = basename + f'_time{dsb.timeid}_mfs.fits'
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_mfs,
                          unit=unit, ms_time=dsb.time_out)
            # hdr['WSUM'] = wsum
            da_mfs = xr.DataArray(data=np.array(psfpars_mfs[dsb.timeid])[None, :, :],
                                  coords={'band': np.arange(1),
                                          'corr': dsb.corr.values,
                                          'bpar': dsb.bpar.values})
            if psfpars_mfs is not None:
                beams_hdu = create_beams_table(da_mfs, cell2deg=cell_deg)
            else:
                beams_hdu = None

            save_fits(cube_mfs, name, hdr, overwrite=True,
                      dtype=otype, beams_hdu=beams_hdu)

        if do_cube:
            if norm_wsum:
                cube = cube/wsums[:, :, None, None]
            name = basename + f'_time{dsb.timeid}.fits'
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freqs,
                          unit=unit, ms_time=dsb.time_out)
            
            if 'PSFPARSN' in dsb:
                beams_hdu = create_beams_table(dsb.PSFPARSN, cell2deg=cell_deg)
            save_fits(cube, name, hdr, overwrite=True,
                      dtype=otype, beams_hdu=beams_hdu)

    return column

