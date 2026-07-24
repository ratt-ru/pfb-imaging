from datetime import datetime, timezone

import numpy as np
import ray
import xarray as xr
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from pfb_imaging import pfb_version
from pfb_imaging.utils.misc import to_unix_time
from pfb_imaging.utils.naming import xds_from_list


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
    npix = hdr["NAXIS" + str(axis)]
    refpix = hdr["CRPIX" + str(axis)]
    delta = hdr["CDELT" + str(axis)]
    ref_val = hdr["CRVAL" + str(axis)]
    return ref_val + np.arange(1 - refpix, 1 + npix - refpix) * delta, ref_val


def load_fits(name, dtype=np.float32):
    data = fits.getdata(name)
    data = np.transpose(to4d(data), axes=(1, 0, 3, 2))  # fits and beams table
    return np.require(data, dtype=dtype, requirements="C")


def save_fits(data, name, hdr, overwrite=True, dtype=np.float32, beams_hdu=None, yx_order=False):
    hdu = fits.PrimaryHDU(header=hdr)
    if yx_order:
        # data is already (..., ny, nx) = FITS row-major layout (wiki D19);
        # only move band/corr onto the FITS STOKES/FREQ axis order
        data = np.transpose(to4d(data), axes=(1, 0, 2, 3))
    else:
        # legacy x-major (..., nx, ny) input
        data = np.transpose(to4d(data), axes=(1, 0, 3, 2))
    hdu.data = np.require(data, dtype=dtype, requirements="F")
    if beams_hdu is not None:
        hdul = fits.HDUList([hdu, beams_hdu])
        hdul.writeto(name, overwrite=overwrite)
    else:
        hdu.writeto(name, overwrite=overwrite)
    return


def set_wcs(
    cell_x,
    cell_y,
    nx,
    ny,
    radec,
    freq,
    unit="Jy/beam",
    gausspar=None,
    gausspars=None,
    ms_time=None,
    time_is_unix=False,
    header=True,
    casambm=True,
    ncorr=1,
    l0=0.0,
    m0=0.0,
):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in radians
    freq - frequencies in Hz
    unit - Jy/beam or Jy/pixel
    gausspar - MFS beam parameters in degrees
    ms_time - measurement set time
    time_is_unix - if True, ms_time is already in unix seconds (MSv4/.dt
        convention); otherwise it is MSv2 MJD seconds and is shifted to unix
    header - if True, return a header, otherwise return a WCS object
    casambm - if True, add the CASAMBM keyword to the header
    l0/m0 - image-centre offset from the tangent point in radians (--target).
        CRVAL stays the tangent point; CRPIX shifts so the centre pixel lands
        on the target direction (facets share CRVAL and differ in CRPIX).
    """

    w = WCS(naxis=4)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = "deg"
    w.wcs.cunit[1] = "deg"
    w.wcs.cunit[2] = "Hz"
    w.wcs.cunit[3] = ""
    if np.size(freq) > 1:
        nchan = freq.size
        crpix3 = nchan // 2 + 1
        ref_freq = freq[crpix3 - 1]  # zero-based indexing
        df = freq[1] - freq[0]
        w.wcs.cdelt[2] = df
    else:
        if isinstance(freq, np.ndarray) and freq.size == 1:
            ref_freq = freq[0]
        else:
            ref_freq = freq
        nchan = 1
        crpix3 = 1
    w.wcs.crval = [radec[0] * 180.0 / np.pi, radec[1] * 180.0 / np.pi, ref_freq, 1]
    # CRVAL stays the gridding tangent point; a --target offset shifts CRPIX
    # so the image-centre pixel lands on the target. RA axis: cdelt = -cell_x,
    # so l(p) = -cell*(p - crpix) and centre-at-l0 gives crpix = 1 + nx//2 +
    # l0/cell; Dec axis has +cdelt, hence the opposite sign.
    crpix_x = 1 + nx // 2 + np.rad2deg(l0) / cell_x
    crpix_y = 1 + ny // 2 - np.rad2deg(m0) / cell_y
    w.wcs.crpix = [crpix_x, crpix_y, crpix3, 1]
    w.wcs.equinox = 2000.0

    if header:
        # the order does seem to matter here,
        # especially when using with StreamingHDU
        header = fits.Header()
        header["SIMPLE"] = (True, "conforms to FITS standard")
        header["BITPIX"] = (-32, "array data type")
        header["NAXIS"] = (4, "number of array dimensions")
        data_shape = (nx, ny, nchan, ncorr)
        for i, size in enumerate(data_shape, 1):
            header[f"NAXIS{i}"] = (size, f"length of data axis {i}")

        header["EXTEND"] = True
        header["BSCALE"] = 1.0
        header["BZERO"] = 0.0
        header["BUNIT"] = unit
        header["EQUINOX"] = 2000.0
        header["BTYPE"] = "Intensity"

        # add wcs keywords
        wcs_header = w.to_header()
        header.update(wcs_header)

        header["RESTFRQ"] = ref_freq
        header["ORIGIN"] = f"pfb-imaging: v{pfb_version}"
        header["SPECSYS"] = "TOPOCENT"
        if ms_time is not None:
            # MSv2 (.dds) carries MJD seconds; MSv4 (.dt) already carries unix
            unix_time = ms_time if time_is_unix else to_unix_time(ms_time)
            utc_iso = datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            header["UTC_TIME"] = utc_iso
            t = Time(utc_iso)
            t.format = "fits"
            header["DATE-OBS"] = t.value

        # What are these used for?
        # if 'LONPOLE' in header:
        #     header.pop('LONPOLE')
        # if 'LATPOLE' in header:
        #     header.pop('LATPOLE')
        # if 'RADESYS' in header:
        #     header.pop('RADESYS')
        # if 'MJDREF' in header:
        #     header.pop('MJDREF')

        #

        if casambm:
            header["CASAMBM"] = casambm  # we need this to pick up the beams table

        if gausspar is not None or gausspars is not None:
            header = add_beampars(header, gausspar, gausspars=gausspars, unit2deg=cell_x)

        return header
    else:
        return w


def add_beampars(hdr, gausspar, gausspars=None, unit2deg=1.0):
    """
    Add beam keywords to header.
    gausspar - MFS beam pars
    gausspars - beam pars for cube
    unit2deg - conversion factor to convert BMAJ/BMIN to degrees

    PA is passed in radians and follows the parametrisation in

    pfb/utils/misc/gaussian2d

    """
    if gausspar is not None:
        if not isinstance(gausspar, np.ndarray):
            gausspar = np.asarray(gausspar)
        if len(gausspar.shape) == 2:
            gausspar = gausspar[0]
        elif gausspar.shape[0] != 3:
            raise ValueError("Invalid value for gausspar")

        if not np.isnan(gausspar).any():
            hdr["BMAJ"] = gausspar[0] * unit2deg
            hdr["BMIN"] = gausspar[1] * unit2deg
            hdr["BPA"] = gausspar[2] * 180 / np.pi

    if gausspars is not None:
        gausspars = np.asarray(gausspars)
        if len(gausspars.shape) != 2:
            raise ValueError("gausspars should have shape (nband, 3)")
        nband = gausspars.shape[0]
        for i in range(nband):
            if not np.isnan(gausspars[i]).any():
                hdr["BMAJ" + str(i + 1)] = gausspars[i, 0] * unit2deg
                hdr["BMIN" + str(i + 1)] = gausspars[i, 1] * unit2deg
                hdr["BPA" + str(i + 1)] = gausspars[i, 2] * 180 / np.pi

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
    bmaj = beams_data.sel({"bpar": "BMAJ"}).values.ravel() * cell2deg
    bmin = beams_data.sel({"bpar": "BMIN"}).values.ravel() * cell2deg
    bpa = beams_data.sel({"bpar": "BPA"}).values.ravel() * 180 / np.pi
    col1 = fits.Column(name="BMAJ", format="1E", array=bmaj, unit="deg")
    col2 = fits.Column(name="BMIN", format="1E", array=bmin, unit="deg")
    col3 = fits.Column(name="BPA", format="1E", array=bpa, unit="deg")
    col4 = fits.Column(name="CHAN", format="1J", array=np.array(band_id))
    col5 = fits.Column(name="POL", format="1J", array=np.array(pol_id))

    # Create the BEAMS table HDU
    cols = fits.ColDefs([col1, col2, col3, col4, col5])
    beams_hdu = fits.BinTableHDU.from_columns(cols)
    beams_hdu.name = "BEAMS"
    beams_hdu.header["EXTNAME"] = "BEAMS"
    beams_hdu.header["EXTVER"] = 1
    beams_hdu.header["XTENSION"] = "BINTABLE"
    beams_hdu.header.comments["XTENSION"] = "Binary extension"
    beams_hdu.header["NCHAN"] = nband
    beams_hdu.header["NPOL"] = npol

    return beams_hdu


@ray.remote
def rdds2fits(*args, **kwargs):
    return dds2fits(*args, **kwargs)


def dds2fits(
    dsl,
    column,
    outname,
    norm_wsum=True,
    otype=np.float32,
    nthreads=1,
    do_mfs=True,
    do_cube=True,
    psfpars_mfs=None,
    psfparsf=None,
    force_unit=None,
):
    """Render data variables in dds to FITS files.

    Args:
        dsl: List of dataset names.
        column: Data variable to render.
        outname: Output name for FITS files. Will be appended with _time{timeid}.
        norm_wsum: If True, divide by wsum to get Jy/beam, otherwise output Jy/pixel.
        otype: Output data type for FITS file.
        nthreads: Number of threads to use for loading data.
        do_mfs: If True, render the MFS image.
        do_cube: If True, render the cube.
        psfpars_mfs: Dict mapping timeid to MFS beam parameters (emaj, emin, pa)
            in degrees with shape (ncorr, 3).
        psfparsf: Dict mapping timeid to beam parameters for cube with shape
            (nband, ncorr, 3) in degrees.
        force_unit: If not None, override the unit in the FITS header with this value.

    Returns:
        The column name that was rendered.
    """
    basename = outname + "_" + column.lower()
    if norm_wsum:
        unit = "Jy/beam"
    else:
        unit = "Jy/pixel"
    if force_unit is not None:
        unit = force_unit
    dds = xds_from_list(dsl, drop_all_but=[column, "PSFPARSN", "WSUM"], nthreads=nthreads, order_freq=False)
    timeids = np.unique(np.array([int(ds.timeid) for ds in dds]))
    freqs = [ds.freq_out for ds in dds]
    freqs = np.unique(freqs)
    nband = freqs.size
    for timeid in timeids:
        # filter by time ID
        dst = [ds for ds in dds if int(ds.timeid) == timeid]
        # concat creating a new band axis
        dsb = xr.concat(dst, dim="band")
        # LB - these seem to remain in the correct order
        dsb = dsb.assign_coords({"band": np.arange(nband)})
        wsums = dsb.WSUM.values
        wsum = dsb.WSUM.sum(dim="band").values
        cube = dsb.get(column).values
        _, _, nx, ny = cube.shape
        # these should be the same across band and corr axes
        radec = (dsb.ra, dsb.dec)
        cell_deg = np.rad2deg(dsb.cell_rad)
        nband = dsb.band.size
        ncorr = dsb.corr.size

        if psfpars_mfs is not None:
            psfpars_mfs_timeid = psfpars_mfs[dsb.timeid]
            assert psfpars_mfs_timeid.shape == (ncorr, 3), "psfpars_mfs should have shape (ncorr, 3)"
            da_mfs_timeid = xr.DataArray(
                data=np.array(psfpars_mfs_timeid)[None, :, :],
                coords={"band": np.arange(1), "corr": dsb.corr.values, "bpar": dsb.bpar.values},
            )
            beams_hdu = create_beams_table(da_mfs_timeid, cell2deg=cell_deg)
            psfpars_mfs_timeid = psfpars_mfs_timeid[0]  # always Stokes I in fits header
        else:
            psfpars_mfs_timeid = None
            beams_hdu = None

        if do_mfs:
            # we need a single freq_mfs for the cube
            freq_mfs = np.sum(freqs[:, None] * wsums) / wsum.sum()
            hdr = set_wcs(
                cell_deg,
                cell_deg,
                nx,
                ny,
                radec,
                freq_mfs,
                unit=unit,
                ms_time=dsb.time_out,
                gausspar=psfpars_mfs_timeid,
            )
            hdr["WSUM"] = wsum[0]  # always Stokes I in fits header
            if norm_wsum:
                # already weighted by wsum
                cube_mfs = np.sum(cube, axis=0) / wsum[:, None, None]
            else:
                # weigted sum
                cube_mfs = np.sum(cube * wsums[:, :, None, None], axis=0) / wsum[:, None, None]

            name = basename + f"_time{dsb.timeid}_mfs.fits"
            save_fits(cube_mfs, name, hdr, overwrite=True, dtype=otype, beams_hdu=beams_hdu)

        if do_cube:
            if norm_wsum:
                cube = cube / wsums[:, :, None, None]
            name = basename + f"_time{dsb.timeid}.fits"
            if psfparsf is not None:
                psfparsf_timeid = psfparsf[dsb.timeid]
                assert psfparsf_timeid.shape == (nband, ncorr, 3), "psfparsf should have shape (nband, ncorr, 3)"
                da_timeid = xr.DataArray(
                    data=psfparsf_timeid,
                    coords={"band": dsb.band.values, "corr": dsb.corr.values, "bpar": dsb.bpar.values},
                )
                psfparsf_timeid = psfparsf_timeid[:, 0]  # always Stokes I in fits header
                beams_hdu = create_beams_table(da_timeid, cell2deg=cell_deg)
            elif "PSFPARSN" in dsb:
                da_timeid = dsb.PSFPARSN
                psfparsf_timeid = da_timeid.values[:, 0]  # always Stokes I in fits header
                beams_hdu = create_beams_table(da_timeid, cell2deg=cell_deg)
            else:
                psfparsf_timeid = None
                beams_hdu = None

            hdr = set_wcs(
                cell_deg,
                cell_deg,
                nx,
                ny,
                radec,
                freqs,
                unit=unit,
                ms_time=dsb.time_out,
                gausspar=psfpars_mfs_timeid,
                gausspars=psfparsf_timeid,
            )
            for i in range(nband):
                hdr[f"WSUM{i + 1}"] = wsums[i, 0]  # always Stokes I value in fits header
            save_fits(cube, name, hdr, overwrite=True, dtype=otype, beams_hdu=beams_hdu)

    return column


@ray.remote
def rdt2fits(*args, **kwargs):
    return dt2fits(*args, **kwargs)


def dt2fits(
    store_url,
    column,
    outname,
    norm_wsum=True,
    otype=np.float32,
    nthreads=1,
    do_mfs=True,
    do_cube=True,
    psfpars_mfs=None,
    force_unit=None,
    extra_hdr=None,
):
    """Render a band-node variable from the imager ``.dt`` DataTree to FITS.

    The DataTree analogue of :func:`dds2fits`: it sources band-stacked data from
    the ``.dt`` store (one group per output image ``band{b}_time{t}``) instead of
    a flat ``.dds`` list, and reuses the access-agnostic helpers
    :func:`set_wcs`/:func:`save_fits`/:func:`create_beams_table`. :func:`dds2fits`
    is left untouched for the live ``.dds`` consumers.

    Args:
        store_url: path/URL of the ``.dt`` store.
        column: band-node data variable to render (e.g. ``"DIRTY"``).
        outname: output basename; files are ``<outname>_<column>_time{t}[_mfs].fits``.
        norm_wsum: divide by wsum (Jy/beam) vs weighted Jy/pixel.
        do_mfs, do_cube: which products to write.
        psfpars_mfs: optional ``{timeid: (ncorr, 3)}`` MFS beam params.
        force_unit: override the BUNIT header.
        extra_hdr: optional dict of extra FITS header cards stamped into every
            written header.

    Returns:
        The rendered ``column`` name.
    """
    basename = outname + "_" + column.lower()
    # explicit None check: force_unit="" (dimensionless, e.g. BEAM) is a valid override
    unit = force_unit if force_unit is not None else ("Jy/beam" if norm_wsum else "Jy/pixel")
    bpar = ["BMAJ", "BMIN", "BPA"]

    dt = xr.open_datatree(store_url, engine="zarr", chunks=None)
    nodes = [dt[name].ds for name in dt.children if name.startswith("band")]
    nodes = [ds for ds in nodes if column in ds]
    if not nodes:
        return column

    timeids = np.unique([int(ds.attrs["timeid"]) for ds in nodes])
    for timeid in timeids:
        # bands for this time chunk, ordered by frequency
        dst = sorted((ds for ds in nodes if int(ds.attrs["timeid"]) == timeid), key=lambda d: d.attrs["freq_out"])
        ref = dst[0]
        nband = len(dst)
        freqs = np.array([ds.attrs["freq_out"] for ds in dst])
        cube = np.stack([ds[column].values for ds in dst], axis=0)  # (band, corr, ny, nx)
        wsums = np.stack([ds.WSUM.values for ds in dst], axis=0)  # (band, corr)
        wsum = wsums.sum(axis=0)  # (corr,)
        _, ncorr, ny, nx = cube.shape
        radec = (ref.attrs["ra"], ref.attrs["dec"])
        l0 = float(ref.attrs.get("l0", 0.0))
        m0 = float(ref.attrs.get("m0", 0.0))
        cell_deg = np.rad2deg(ref.attrs["cell_rad"])
        time_out = ref.attrs["time_out"]

        beams_hdu = None
        psfpars_mfs_timeid = None
        if psfpars_mfs is not None:
            pp = np.asarray(psfpars_mfs[timeid])
            assert pp.shape == (ncorr, 3), "psfpars_mfs should have shape (ncorr, 3)"
            da = xr.DataArray(
                pp[None], dims=("band", "corr", "bpar"), coords={"band": [0], "corr": ref.corr.values, "bpar": bpar}
            )
            beams_hdu = create_beams_table(da, cell2deg=cell_deg)
            psfpars_mfs_timeid = pp[0]  # always Stokes I in fits header

        if do_mfs:
            freq_mfs = np.sum(freqs[:, None] * wsums) / wsum.sum()
            hdr = set_wcs(
                cell_deg,
                cell_deg,
                nx,
                ny,
                radec,
                freq_mfs,
                unit=unit,
                ms_time=time_out,
                time_is_unix=True,
                gausspar=psfpars_mfs_timeid,
                l0=l0,
                m0=m0,
            )
            if extra_hdr:
                for k, v in extra_hdr.items():
                    hdr[k] = v
            hdr["WSUM"] = float(wsum[0])
            if norm_wsum:
                cube_mfs = np.sum(cube, axis=0) / wsum[:, None, None]
            else:
                cube_mfs = np.sum(cube * wsums[:, :, None, None], axis=0) / wsum[:, None, None]
            save_fits(
                cube_mfs,
                basename + f"_time{timeid}_mfs.fits",
                hdr,
                overwrite=True,
                dtype=otype,
                beams_hdu=beams_hdu,
                yx_order=True,
            )

        if do_cube:
            cube_beams = None
            psfparsf_timeid = None
            if all("PSFPARSN" in ds for ds in dst):
                pp = np.stack([ds.PSFPARSN.values for ds in dst], axis=0)  # (band, corr, 3)
                da = xr.DataArray(
                    pp,
                    dims=("band", "corr", "bpar"),
                    coords={"band": np.arange(nband), "corr": ref.corr.values, "bpar": bpar},
                )
                psfparsf_timeid = pp[:, 0]  # always Stokes I in fits header
                cube_beams = create_beams_table(da, cell2deg=cell_deg)
            hdr = set_wcs(
                cell_deg,
                cell_deg,
                nx,
                ny,
                radec,
                freqs,
                unit=unit,
                ms_time=time_out,
                time_is_unix=True,
                gausspar=psfpars_mfs_timeid,
                gausspars=psfparsf_timeid,
                l0=l0,
                m0=m0,
            )
            if extra_hdr:
                for k, v in extra_hdr.items():
                    hdr[k] = v
            for i in range(nband):
                hdr[f"WSUM{i + 1}"] = float(wsums[i, 0])
            cube_out = cube / wsums[:, :, None, None] if norm_wsum else cube
            save_fits(
                cube_out,
                basename + f"_time{timeid}.fits",
                hdr,
                overwrite=True,
                dtype=otype,
                beams_hdu=cube_beams,
                yx_order=True,
            )

    return column
