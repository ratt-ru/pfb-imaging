"""Pieces shared by pfb hci and pfb hci-msv4. Must not import dask-ms or pyrap."""

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import yaml
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from meerkat_beams.utils import BeamWizard

from pfb_imaging import pfb_version
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.beam import reproject_and_interp_scat_beam
from pfb_imaging.utils.misc import radec_barycentre
from pfb_imaging.utils.transients import generate_transient_spectra

log = pfb_logging.get_logger("HCI")

# FITS axis keywords that must be removed/renumbered when changing dimensionality
FITS_AXIS_KEYWORDS = ("CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT", "NAXIS")

# FITS Stokes parameter indices (FITS standard)
STOKES_FITS_INDEX = {"I": 1, "Q": 2, "U": 3, "V": 4}


def make_fits_header(base_hdr, data_shape, stokes_params, drop_axes):
    """Build an ordered FITS header from the stored 5-axis header.

    Drops the specified axes, renumbers the remaining ones, explicitly sets
    the STOKES axis from the coordinate values, and ensures mandatory
    keywords appear first (required by StreamingHDU).

    Args:
        base_hdr: dict of FITS header keyword-value pairs (from attrs["fits_header"]).
        data_shape: tuple of the target array shape in numpy order
            e.g. (STOKES, TIME, Y, X) or (STOKES, FREQ, Y, X).
        stokes_params: list of Stokes parameter strings e.g. ["I"] or ["I", "Q"].
        drop_axes: set of 1-based FITS axis numbers to drop from the 5-axis header.

    Returns:
        astropy.io.fits.Header with correct ordering and keywords.
    """
    hdr = base_hdr.copy()
    orig_naxis = hdr["NAXIS"]
    ndim = len(data_shape)
    # FITS shape is reversed from numpy shape
    fits_shape = data_shape[::-1]

    # remove WCSAXES early — we'll set it at the end
    hdr.pop("WCSAXES", None)

    # collect which original axes survive (1-based, in order)
    kept_axes = [ax for ax in range(1, orig_naxis + 1) if ax not in drop_axes]

    # build the new axis keywords: new axis i gets old axis kept_axes[i-1]
    # first, collect the new values
    new_axis_kws = {}
    for new_ax, old_ax in enumerate(kept_axes, 1):
        for kw in FITS_AXIS_KEYWORDS:
            old_key = f"{kw}{old_ax}"
            if old_key in hdr:
                new_axis_kws[f"{kw}{new_ax}"] = hdr[old_key]

    # remove all old axis keywords
    for ax in range(1, orig_naxis + 1):
        for kw in FITS_AXIS_KEYWORDS:
            hdr.pop(f"{kw}{ax}", None)

    # set NAXIS and NAXISn
    hdr["NAXIS"] = ndim
    for i, size in enumerate(fits_shape, 1):
        new_axis_kws[f"NAXIS{i}"] = size

    # find which new axis is STOKES and set it explicitly
    stokes_rvals = [STOKES_FITS_INDEX[s] for s in stokes_params]
    stokes_delta = stokes_rvals[1] - stokes_rvals[0] if len(stokes_rvals) > 1 else 1
    stokes_ax = ndim  # STOKES is always the last (slowest) FITS axis
    new_axis_kws[f"CTYPE{stokes_ax}"] = "STOKES"
    new_axis_kws[f"CRPIX{stokes_ax}"] = 1
    new_axis_kws[f"CRVAL{stokes_ax}"] = stokes_rvals[0]
    new_axis_kws[f"CDELT{stokes_ax}"] = stokes_delta
    new_axis_kws[f"CUNIT{stokes_ax}"] = ""

    # build ordered header: mandatory keywords first
    mandatory = ["SIMPLE", "BITPIX", "NAXIS"] + [f"NAXIS{i}" for i in range(1, ndim + 1)]
    ordered = {}
    for kw in mandatory:
        if kw in hdr:
            ordered[kw] = hdr.pop(kw)
        elif kw in new_axis_kws:
            ordered[kw] = new_axis_kws.pop(kw)

    # then the rest of the original header (EXTEND, BSCALE, BZERO, etc.)
    ordered.update(hdr)

    # then the WCS axis keywords
    ordered.update(new_axis_kws)

    # set WCSAXES at the end
    ordered["WCSAXES"] = ndim

    return fits.Header(ordered)


def default_phase_dir(out_ra: np.ndarray, out_dec: np.ndarray) -> str | None:
    """Barycentre of the distinct pointings, or None when there is only one.

    Pointings are de-duplicated as (ra, dec) pairs: fields sharing an RA or a
    Dec would otherwise leave the two arrays different lengths (#320).

    Args:
        out_ra: RA of every selected partition, radians.
        out_dec: Dec of every selected partition, radians.

    Returns:
        ``'HH:MM:SS,DD:MM:SS'`` for the workers to rephase to, or None for a
        single pointing, which keeps the MS phase centre.
    """
    radecs = np.unique(np.column_stack((out_ra, out_dec)), axis=0)
    if radecs.shape[0] == 1:
        return None
    # propagated as phase_dir so the workers rephase to the same reference;
    # otherwise the scaffold and the images would be centred on different skies
    log.info("Using barycenter as phase center since no phase-dir was specified")
    ra0, dec0 = radec_barycentre(radecs)
    bary = SkyCoord(ra=ra0 * units.rad, dec=dec0 * units.rad, frame="fk5")
    return f"{bary.ra.to_string(unit=units.hourangle, sep=':')},{bary.dec.to_string(unit=units.deg, sep=':')}"


def image_centre_deg(
    out_ra: np.ndarray,
    out_dec: np.ndarray,
    phase_dir: str | None,
    target: str | None,
    out_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Image-centre (ra, dec) in degrees from the phase direction and ``--target``.

    Args:
        out_ra: RA of every selected partition, radians.
        out_dec: Dec of every selected partition, radians.
        phase_dir: ``'HH:MM:SS,DD:MM:SS'`` or None to use the first partition's centre.
        target: ``--target`` string or None.
        out_times: Sorted unique output image times (MJD seconds).

    Returns:
        Length-1 (ra, dec) arrays in degrees.

    Raises:
        ValueError: If ``target`` is a moving body and there is more than one time bin.
    """
    if phase_dir is None:
        # single-field / single radec — use the MS phase centre directly
        out_ra_deg = np.rad2deg(out_ra[:1])
        out_dec_deg = np.rad2deg(out_dec[:1])
    else:
        ra_str, dec_str = phase_dir.split(",")
        coord = SkyCoord(ra_str, dec_str, frame="fk5", unit=(units.hourangle, units.deg))
        # match the deg->rad->deg path inside stokes_image so the rounded Y/X coords
        # agree bitwise with what the workers write back.
        out_ra_deg = np.array([np.rad2deg(np.deg2rad(coord.ra.value))])
        out_dec_deg = np.array([np.rad2deg(np.deg2rad(coord.dec.value))])

    # remove duplicates
    out_times = np.unique(out_times)

    # --target moves the image centre off the tangent point, so the scaffold's
    # X/Y coords (and the header's CRVAL) must move with it: stokes_image labels
    # its grid from the target, and its slab is written back with
    # to_zarr(region="auto"), which resolves the region by looking those values
    # up here. Same resolver on both sides so they cannot disagree.
    if target is not None:
        from pfb_imaging.utils.astrometry import (
            resolve_target_radec,
        )  # deferred: utils.astrometry imports pyrap (python-casacore); only --target runs pay it

        ra_t, dec_t, moving = resolve_target_radec(target, float(out_times[0]))
        if moving and out_times.size > 1:
            # an ephemeris target moves between images, but X/Y is one axis shared
            # across TIME, so no single labelling is correct for the whole cube
            refs = np.rad2deg([resolve_target_radec(target, float(t))[:2] for t in out_times])
            if not np.all(refs == refs[0]):
                raise ValueError(
                    f"target '{target}' is a solar-system body, so its position differs across the "
                    f"{out_times.size} output time bins ({np.ptp(refs[:, 0]):.4f} deg in RA, "
                    f"{np.ptp(refs[:, 1]):.4f} deg in Dec) and the cube's shared X/Y axis cannot "
                    "label all of them. Either ask for a single time bin "
                    "(--integrations-per-image >= the number of integrations in the scan), or pass "
                    "the position explicitly as --target 'HH:MM:SS,DD:MM:SS'."
                )
        out_ra_deg = np.array([np.rad2deg(ra_t)])
        out_dec_deg = np.array([np.rad2deg(dec_t)])

    return out_ra_deg, out_dec_deg


def build_scaffold(
    out_times: np.ndarray,
    out_freqs: np.ndarray,
    out_ra_deg: np.ndarray,
    out_dec_deg: np.ndarray,
    product: str,
    beam: bool,
    psf_out: bool,
    nx: int,
    ny: int,
    nx_psf: int,
    ny_psf: int,
    cell_deg: float,
    spatial_chunk: int = 128,
    images_per_chunk: int = 16,
    obs_label: str | None = None,
) -> tuple[xr.Dataset, dict]:
    """Lazy scaffold of the hci output cube and its attrs (no I/O).

    Args:
        out_times: Sorted unique output image times (MJD seconds).
        out_freqs: Sorted unique output band frequencies (Hz).
        out_ra_deg: Length-1 array, image-centre RA in degrees.
        out_dec_deg: Length-1 array, image-centre Dec in degrees.
        product: Stokes product string.
        beam: Whether a beam model is applied (adds ``beam_weight``).
        psf_out: Whether PSFs are written (adds ``psf``/``psf2`` and PSF coords).
        nx, ny, nx_psf, ny_psf: Image and PSF sizes.
        cell_deg: Cell size in degrees.
        spatial_chunk: Requested spatial chunk, clipped to the smallest image dimension.
        images_per_chunk: TIME chunk.
        obs_label: Optional OBSLABEL header card.

    Returns:
        (dataset, attrs). The dataset is dask-backed and empty; the caller writes it
        with ``to_zarr(..., compute=False)``.
    """
    n_stokes = len(set(product))
    n_times = out_times.size
    n_freqs = out_freqs.size

    # guard against chunk sizes larger than the image size
    spatial_chunk = np.minimum(spatial_chunk, min(nx, ny, nx_psf, ny_psf))

    cube_dims = (n_stokes, n_freqs, n_times, ny, nx)
    cube_chunks = (1, 1, images_per_chunk, spatial_chunk, spatial_chunk)

    mean_dims = (n_stokes, n_freqs, ny, nx)
    mean_chunks = (1, 1, spatial_chunk, spatial_chunk)

    rms_dims = (n_stokes, n_freqs, n_times)
    rms_chunks = (1, 1, images_per_chunk)

    ra_dim = "RA--SIN"
    dec_dim = "DEC--SIN"
    if out_freqs.size > 1:
        delta_freq = np.diff(out_freqs).min()
    else:
        delta_freq = 1
    if out_times.size > 1:
        delta_time = np.diff(out_times).min()
    else:
        delta_time = 1

    # construct reference header
    w = WCS(naxis=5)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "INTEGRATION", "FREQ", "STOKES"]
    w.wcs.cdelt = [-cell_deg, cell_deg, delta_time, delta_freq, 1]
    w.wcs.cunit = ["deg", "deg", "s", "Hz", ""]
    w.wcs.crval = [out_ra_deg[0], out_dec_deg[0], out_times[0], out_freqs[0], 1]
    w.wcs.crpix = [1 + nx // 2, 1 + ny // 2, 1, 1, 1]
    w.wcs.equinox = 2000.0
    wcs_hdr = w.to_header()

    # the order does seem to matter here,
    # especially when using with StreamingHDU
    hdr = fits.Header()
    hdr["SIMPLE"] = (True, "conforms to FITS standard")
    hdr["BITPIX"] = (-32, "array data type")
    hdr["NAXIS"] = (5, "number of array dimensions")
    data_shape = (nx, ny, n_times, n_freqs, n_stokes)
    for i, size in enumerate(data_shape, 1):
        hdr[f"NAXIS{i}"] = (size, f"length of data axis {i}")

    hdr["EXTEND"] = True
    hdr["BSCALE"] = 1.0
    hdr["BZERO"] = 0.0
    hdr["BUNIT"] = "Jy/beam"
    hdr["EQUINOX"] = 2000.0
    hdr["BTYPE"] = "Intensity"
    hdr.update(wcs_hdr)
    hdr["TIMESCAL"] = delta_time
    if obs_label is not None:
        hdr["OBSLABEL"] = obs_label

    # if we don't pass these into stokes2im they get overwritten
    attrs = {
        "pfb-imaging-version": pfb_version,
        # Store dicts as list since zarr doesn't maintain dict order.
        "fits_header": list(dict(hdr).items()),
        "radec_dims": (ra_dim, dec_dim),
        "fits_dims": (("X", ra_dim), ("Y", dec_dim), ("TIME", "TIME"), ("FREQ", "FREQ"), ("STOKES", "STOKES")),
    }

    # Quantise to 1e-12 deg (3.6 nanoarcsec) so the workers' independently computed
    # X/Y coords agree bitwise with this scaffold: they write back with
    # to_zarr(region="auto"), which looks the values up in the stored coords and
    # raises KeyError on a 1-ulp difference. Redundant with the deg->rad->deg
    # symmetry above -- either alone suffices; keep both.
    # See wiki design-decisions D36 (#155).
    out_ras = out_ra_deg + np.arange(nx // 2, -(nx // 2), -1) * cell_deg
    out_decs = out_dec_deg + np.arange(-(ny // 2), ny // 2) * cell_deg
    out_ras = np.round(out_ras, decimals=12)
    out_decs = np.round(out_decs, decimals=12)

    dummy_ds = xr.Dataset(
        data_vars={
            "cube": (
                ("STOKES", "FREQ", "TIME", "Y", "X"),
                da.empty(cube_dims, chunks=cube_chunks, dtype=np.float32),
            ),
            "cube_mean": (("STOKES", "FREQ", "Y", "X"), da.empty(mean_dims, chunks=mean_chunks, dtype=np.float32)),
            "rms": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "weight": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "nonzero": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.bool_),
            ),
            "flag": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.bool_),
            ),
            "psf_maj": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "psf_min": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "psf_pa": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "channel_width": (
                ("FREQ",),
                da.empty((n_freqs), chunks=(1), dtype=np.float32),
            ),
            # we can't add these as coordinates as they are not necessarily unique
            # simplest to set them inside stokes2im
            "orig_phase_dir_ra": (
                ("TIME",),
                da.empty((n_times), chunks=(images_per_chunk,), dtype=np.float64),
            ),
            "orig_phase_dir_dec": (
                ("TIME",),
                da.empty((n_times), chunks=(images_per_chunk,), dtype=np.float64),
            ),
        },
        coords={
            "TIME": (("TIME",), out_times),
            "STOKES": (("STOKES",), list(sorted(set(product)))),
            "FREQ": (("FREQ",), out_freqs),
            "X": (("X",), out_ras),
            "Y": (("Y",), out_decs),
        },
        attrs=attrs,
    )

    if beam:
        dummy_ds["beam_weight"] = (
            ("STOKES", "FREQ", "TIME", "Y", "X"),
            da.empty(cube_dims, chunks=cube_chunks, dtype=np.float32),
        )

    if psf_out:
        psf_dims = (n_stokes, n_freqs, n_times, ny_psf, nx_psf)
        dummy_ds = dummy_ds.assign_coords(
            {
                "Y_PSF": (("Y_PSF",), out_dec_deg + np.arange(-(ny_psf // 2), ny_psf // 2) * cell_deg),
                "X_PSF": (("X_PSF",), out_ra_deg + np.arange(nx_psf // 2, -(nx_psf // 2), -1) * cell_deg),
            }
        )
        dummy_ds["psf"] = (
            ("STOKES", "FREQ", "TIME", "Y_PSF", "X_PSF"),
            da.empty(psf_dims, chunks=cube_chunks, dtype=np.float32),
        )
        dummy_ds["psf2"] = (
            ("STOKES", "Y_PSF", "X_PSF"),
            da.empty((n_stokes, ny_psf, nx_psf), chunks=(1, spatial_chunk, spatial_chunk), dtype=np.float32),
        )

    return dummy_ds, attrs


def write_fits_vars(store: str, fits_vars: list[str], fits_output_folder: Path, images_per_chunk: int, log) -> None:
    """Stream the requested cube variables to per-band FITS files.

    Args:
        store: Output zarr store.
        fits_vars: Variables to export (``cube``, ``psf``, ``beam_weight``, ``weight_grid``, ``cube_mean``).
        fits_output_folder: Destination directory (must exist).
        images_per_chunk: TIME read granularity.
        log: Logger.
    """
    # reopen with chunking aligned to zarr layout for efficient streaming
    cds = xr.open_zarr(store, chunks={"TIME": images_per_chunk})
    stokes_params = list(cds.coords["STOKES"].values)
    # reconstruct the base header from stored attrs
    base_hdr = dict(cds.attrs["fits_header"])
    for var in fits_vars:
        match var:
            case "cube" | "psf" | "beam_weight" | "weight_grid":
                if cds.get(var, None) is None:
                    log.warning(f"{var} not found in dataset, skipping FITS export")
                    continue
                cube = cds.get(var).data  # (STOKES, FREQ, TIME, Y, X)
                n_stokes, n_freqs, n_time, ny, nx = cube.shape

                # --- per-band time cubes via StreamingHDU ---
                # target shape per band: (STOKES, TIME, Y, X)
                # FITS axes: NAXIS1=X, NAXIS2=Y, NAXIS3=TIME, NAXIS4=STOKES
                # drop FITS axis 4 (FREQ) from the 5-axis header
                band_shape = (n_stokes, n_time, ny, nx)
                for f in range(n_freqs):
                    band_hdr = make_fits_header(base_hdr, band_shape, stokes_params, drop_axes={4})

                    filename = fits_output_folder / f"{var}.band{f:04d}.fits"
                    if Path(filename).exists():
                        Path(filename).unlink()
                    log.info(f"Streaming {var} band {f} to {filename}")

                    shdu = fits.StreamingHDU(str(filename), band_hdr)
                    for s in range(n_stokes):
                        for t in range(0, n_time, images_per_chunk):
                            tend = min(t + images_per_chunk, n_time)
                            chunk = cube[s, f, t:tend, :, :].compute()
                            shdu.write(chunk)
                    shdu.close()
            case "cube_mean":
                # --- mean image (small enough to write at once) ---
                # target shape: (STOKES, FREQ, Y, X)
                # drop FITS axis 3 (TIME/INTEGRATION) from the 5-axis header
                cube_mean = cds.cube_mean.data  # (STOKES, FREQ, Y, X)
                mean_hdr = make_fits_header(base_hdr, cube_mean.shape, stokes_params, drop_axes={3})

                mean_filename = fits_output_folder / "cube_mean.fits"
                log.info(f"Writing mean image to {mean_filename}")
                mean_data = cube_mean.compute()
                hdu = fits.PrimaryHDU(mean_data, mean_hdr)
                hdu.writeto(str(mean_filename), overwrite=True)

        log.info("FITS export complete")


def write_transient_spectra(inject_transients: str, all_times: np.ndarray, all_freqs: np.ndarray) -> None:
    """Compute transient time/frequency profiles over the full domain and write them to zarr.

    Args:
        inject_transients: Path to the transients YAML; output goes to ``<stem>.zarr``.
        all_times: Sorted unique times of the full domain.
        all_freqs: Sorted unique frequencies of the full domain.
    """
    with open(inject_transients, "r") as f:
        config = yaml.safe_load(f)
    transients = config["transients"]
    coords = {
        "TIME": (("TIME",), all_times),
        "FREQ": (("FREQ",), all_freqs),
    }
    names = []
    ras = []
    decs = []
    transient_ds = xr.Dataset(coords=coords)
    for transient in transients:
        tprofile, fprofile = generate_transient_spectra(all_times, all_freqs, transient)
        name = transient["name"]
        transient_ds[f"{name}_time_profile"] = (("TIME",), tprofile)
        transient_ds[f"{name}_freq_profile"] = (("FREQ",), fprofile)
        names.append(name)
        ras.append(transient["position"]["ra"])
        decs.append(transient["position"]["dec"])
    transient_ds = transient_ds.assign_attrs({"names": names, "ras": ras, "decs": decs})
    transient_baseoname = inject_transients.removesuffix("yaml")
    transient_ds.to_zarr(f"{transient_baseoname}zarr", mode="a")


def beam_gain_for_source(beam_model, ra_deg, dec_deg, product, utime, time, freq):
    """Return the Stokes power beam at a fixed sky position on the (row, chan) grid.

    Used to attenuate injected transients by the primary beam in the original
    (un-rephased) frame, so the gain must be evaluated on the same grid as the
    dynamic spectrum it multiplies.

    Args:
        beam_model: BeamWizard instance.
        ra_deg, dec_deg: source position in degrees, as given in the transient config.
        product: Stokes product string. Transients are injected into the first
            (I, Q, U, V ordered) Stokes plane only, so that product's beam is used.
        utime: unique times of this chunk (MJD seconds).
        time: per-row times of this chunk (MJD seconds).
        freq: channel frequencies (Hz).

    Returns:
        (time.size, freq.size) power beam gain.
    """
    assert isinstance(beam_model, BeamWizard), "Only BeamWizard instances currently supported as beam_model."
    prod = sorted(set(product))[0]
    # returns (nchan, ntime) on this chunk's own grid, so no interpolation is needed
    gain = beam_model.get_time_variable_beamgain(
        SkyCoord(ra_deg * units.deg, dec_deg * units.deg, frame="fk5"),
        times=Time(utime / (24 * 3600), format="mjd"),
        freq=freq,
        var="nstokes",
        i=prod,
        j=prod,
    )
    # rows are grouped by time so expand the unique time axis onto rows
    return gain.T[np.searchsorted(utime, time)]


def beam_for_band(
    beam_model,
    utime,
    product,
    real_type,
    freq_out,
    radec,
    radec_new,
    cell_deg,
    nx,
    ny,
):
    """Return the Stokes power beam on the output image grid.

    Contract: (len(product), ny, nx) in cube/FITS (Y, X) order, matching the
    image-space arrays in stokes_image (docs/wiki/image-and-beam-orientation.md).
    """
    if beam_model is not None:
        assert isinstance(beam_model, BeamWizard), "Only BeamWizard instances currently supported as beam_model."
        # should we compute a weighted mean over freq instead of interpolating here?
        bds = beam_model.bds
        l_beam = bds.X.values
        m_beam = bds.Y.values
        t_beam = Time(utime / (24 * 3600), format="mjd")
        # get_rotation_averaged_beam returns (Y, X)-ordered maps
        # (see docs/wiki/image-and-beam-orientation.md)
        beam = np.zeros((len(product), m_beam.size, l_beam.size), dtype=real_type)
        if isinstance(freq_out, np.ndarray):
            freq_out = freq_out.item()
        # sorted to match the I, Q, U, V Stokes axis of the data (stokes_expr_funcs)
        for i, p in enumerate(sorted(set(product))):
            beam[i], _ = beam_model.get_rotation_averaged_beam(
                l=l_beam,
                m=m_beam,
                times=t_beam,
                freq=np.array([freq_out]),  # TODO - can we fold in the variance due to finite bandwidth?
                time_stepping=1,
                pixel_stepping=1,
                var="nstokes",
                i=p,
                j=p,
                verbose=0,
            )
        pbeam = reproject_and_interp_scat_beam(
            beam,
            l_beam,
            m_beam,
            radec,
            radec_new,
            cell_deg,
            nx,
            ny,
            product,
        ).astype(real_type)

    else:
        pbeam = np.ones((len(product), ny, nx), dtype=real_type)
    return pbeam
