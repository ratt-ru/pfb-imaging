from datetime import datetime, timezone

import numpy as np
import ray
import xarray as xr
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from pfb_imaging import pfb_version
from pfb_imaging.utils.misc import to_unix_time


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


def save_fits(data, name, hdr, overwrite=True, dtype=np.float32, beams_hdu=None, yx_order=False):
    hdu = fits.PrimaryHDU(header=hdr)
    if yx_order:
        # data is already (..., ny, nx) = FITS row-major layout (wiki D19);
        # only move band/corr onto the FITS STOKES/FREQ axis order
        data = np.transpose(to4d(data), axes=(1, 0, 2, 3))
    else:
        # legacy x-major (..., nx, ny) input
        data = np.transpose(to4d(data), axes=(1, 0, 3, 2))
    # C order: (corr, band, ny, nx) is already C-contiguous for one correlation,
    # so a cube built in the output dtype is written without a copy
    hdu.data = np.require(data, dtype=dtype, requirements="C")
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
    drop_bands=None,
    psfpars_var="PSFPARSN",
):
    """Render a band-node variable from the imager ``.dt`` DataTree to FITS.

    Sources band-stacked data from the ``.dt`` store (one group per output image
    ``band{b}_time{t}``) and reuses the access-agnostic helpers
    :func:`set_wcs`/:func:`save_fits`/:func:`create_beams_table`.

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
        drop_bands: optional list of ``bandid`` values to exclude from the cube
            and from every MFS reduction (image sum, wsum, ``freq_mfs``). Bands
            are omitted, not zeroed, so a non-edge drop leaves a non-uniform
            FITS frequency axis -- see issue #302.
        psfpars_var: band variable supplying the cube BEAMS table and
            ``BMAJ{i}`` cards. ``"PSFPARSF"`` publishes a restored image's final
            resolution instead of the native ``"PSFPARSN"``.

    Returns:
        The rendered ``column`` name.
    """
    basename = outname + "_" + column.lower()
    # explicit None check: force_unit="" (dimensionless, e.g. BEAM) is a valid override
    unit = force_unit if force_unit is not None else ("Jy/beam" if norm_wsum else "Jy/pixel")
    bpar = ["BMAJ", "BMIN", "BPA"]

    # cache=False: the tree would otherwise keep a float64 copy of the column
    dt = xr.open_datatree(store_url, engine="zarr", chunks=None, cache=False)
    nodes = [dt[name].ds for name in dt.children if name.startswith("band")]
    nodes = [ds for ds in nodes if column in ds]
    if not nodes:
        return column

    timeids = np.unique([int(ds.attrs["timeid"]) for ds in nodes])
    for timeid in timeids:
        # bands for this time chunk, ordered by bandid, not freq_out: effective
        # frequencies are data-dependent
        # (issue #296) and severe asymmetric flagging could invert two bands,
        # silently reordering cube planes. bandid is monotonic in frequency by
        # construction (the band_edges grid in core.imager).
        dst = sorted((ds for ds in nodes if int(ds.attrs["timeid"]) == timeid), key=lambda d: int(d.attrs["bandid"]))
        if drop_bands:
            dst = [ds for ds in dst if int(ds.attrs["bandid"]) not in set(drop_bands)]
            if not dst:
                continue
        drop_card = {"DROPBAND": ",".join(str(b) for b in sorted(drop_bands))} if drop_bands else {}
        ref = dst[0]
        nband = len(dst)
        freqs = np.array([ds.attrs["freq_out"] for ds in dst])
        wsums = np.stack([ds.WSUM.values for ds in dst], axis=0)  # (band, corr)
        wsum = wsums.sum(axis=0)  # (corr,)
        # one band at a time: the cube is built in the output dtype, already
        # normalised, and the MFS image accumulates in float64 alongside --
        # stacking the float64 column and then normalising and casting it held
        # three copies of the cube (D58)
        ncorr, ny, nx = dst[0][column].shape
        cube = np.empty((nband, ncorr, ny, nx), dtype=otype) if do_cube else None
        cube_mfs = np.zeros((ncorr, ny, nx)) if do_mfs else None
        for b, ds in enumerate(dst):
            band = ds[column].values
            if do_mfs:
                cube_mfs += band if norm_wsum else band * wsums[b][:, None, None]
            if do_cube:
                cube[b] = band / wsums[b][:, None, None] if norm_wsum else band
            del band
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
            for k, v in drop_card.items():
                hdr[k] = v
            hdr["WSUM"] = float(wsum[0])
            cube_mfs /= wsum[:, None, None]
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
            if all(psfpars_var in ds for ds in dst):
                pp = np.stack([ds[psfpars_var].values for ds in dst], axis=0)  # (band, corr, 3)
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
            for k, v in drop_card.items():
                hdr[k] = v
            for i in range(nband):
                hdr[f"WSUM{i + 1}"] = float(wsums[i, 0])
            save_fits(
                cube,
                basename + f"_time{timeid}.fits",
                hdr,
                overwrite=True,
                dtype=otype,
                beams_hdu=cube_beams,
                yx_order=True,
            )

    return column
