"""Restoration helpers for the imager DataTree.

Pure array functions: no tree I/O, no Ray, no FITS. ``core/restore.py`` owns
reading the ``.dt``, writing products back and rendering FITS.

Flux-scale conventions (wiki D22/D23): the band ``MODEL`` is intrinsic flux and
the band ``RESIDUAL`` is apparent (once-attenuated) flux, so a restored image
must say which scale it is on. Three are offered, keyed by their CLI letter.
"""

import numpy as np
import xarray as xr

from pfb_imaging.utils.fits import create_beams_table, save_fits, set_wcs
from pfb_imaging.utils.misc import convolve2gaussres, fitcleanbeam

# CLI letter -> DataTree variable name. The B prefix follows the tree's
# convention for beam-attenuated quantities (BDIRTY, BRESIDUAL).
PRODUCT_VARS = {"a": "BIMAGE", "i": "IMAGE", "k": "KIMAGE"}


def clean_beam(psf, wsum):
    """Fit the restoring Gaussian to a wsum-normalised PSF.

    Args:
        psf: ``(ncorr, ny_psf, nx_psf)`` un-normalised PSF, as stored on a band
            node. Pass the sum over bands to obtain the MFS restoring beam.
        wsum: ``(ncorr,)`` matching weight sum. Pass the sum over the same bands.

    Returns:
        ``(ncorr, 3)`` array of ``(emaj, emin, pa)`` in pixels, pixels, radians.
        Rows are NaN where ``wsum`` is zero.
    """
    psf = np.asarray(psf)
    wsum = np.asarray(wsum)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(wsum[:, None, None] > 0, psf / wsum[:, None, None], 0.0)
    return np.array(fitcleanbeam(norm, yx_order=True))


def lowest_resolution(gausspars):
    """Reduce per-band resolutions to the lowest-resolution one.

    Args:
        gausspars: ``(nband, ncorr, 3)`` in pixels, pixels, radians. NaN rows
            (fully flagged bands) are skipped.

    Returns:
        ``(ncorr, 3)``: the largest major and minor axes over bands and the mean
        position angle.
    """
    gausspars = np.asarray(gausspars, dtype=float)
    out = np.empty(gausspars.shape[1:], dtype=float)
    out[:, 0] = np.nanmax(gausspars[:, :, 0], axis=0)
    out[:, 1] = np.nanmax(gausspars[:, :, 1], axis=0)
    out[:, 2] = np.nanmean(gausspars[:, :, 2], axis=0)
    return out


def restore_products(model, residual, beam, gaussparf, gausspari=None, products=("k",), pb_min=0.1, nthreads=1):
    """Build the restored images for one image grid.

    Args:
        model: ``(ncorr, ny, nx)`` intrinsic model, Jy/pixel, not wsum-scaled.
        residual: ``(ncorr, ny, nx)`` apparent residual already normalised to
            Jy/beam (the stored ``RESIDUAL`` divided by the matching ``WSUM``).
        beam: ``(ncorr, ny, nx)`` effective response ``B/n`` (band node ``BEAM``).
        gaussparf: ``(ncorr, 3)`` target resolution in pixels, pixels, radians.
        gausspari: ``(ncorr, 3)`` intrinsic resolution of ``residual``, or None
            to skip the residual reconvolution. None is correct whenever
            ``gaussparf`` is the residual's own resolution.
        products: iterable over ``"a"`` (apparent), ``"i"`` (intrinsic) and
            ``"k"`` (intrinsic model plus apparent residual).
        pb_min: beam floor below which the intrinsic image is zeroed, matching
            the cutoff convention in ``utils/spi.py``.
        nthreads: threads for the convolution FFTs.

    Returns:
        dict mapping each requested key to an ``(ncorr, ny, nx)`` array.
    """
    model = np.asarray(model)
    residual = np.asarray(residual)
    beam = np.asarray(beam)
    gaussparf = np.asarray(gaussparf, dtype=float)
    _, ny, nx = model.shape
    # grids stay x-major; convolve2gaussres(yx_order=True) transposes the
    # (corr, y, x) images onto them and back (wiki D19/D20)
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    kw = {"nthreads": nthreads, "pfrac": 0.2, "norm_kernel": False, "yx_order": True}

    if gausspari is None or np.allclose(gaussparf, np.asarray(gausspari, dtype=float)):
        rconv = residual
    else:
        rconv = convolve2gaussres(residual, xx, yy, gaussparf, gausspari=np.asarray(gausspari, dtype=float), **kw)

    out = {}
    if "i" in products or "k" in products:
        # the model is treated as having zero intrinsic resolution, which is
        # the standard clean-component assumption
        mconv = convolve2gaussres(model, xx, yy, gaussparf, **kw)
    if "k" in products:
        out["k"] = mconv + rconv
    if "i" in products:
        with np.errstate(invalid="ignore", divide="ignore"):
            rint = np.where(beam > pb_min, rconv / beam, 0.0)
        out["i"] = np.where(beam > pb_min, mconv + rint, 0.0)
    if "a" in products:
        # (B*m) (x) G, NOT B*(m (x) G) -- convolution does not commute with
        # multiplication by a spatially varying beam, so mconv cannot be reused
        out["a"] = convolve2gaussres(beam * model, xx, yy, gaussparf, **kw) + rconv
    return out


def beams_table(gausspars, corr, cell_deg):
    """Build a BEAMS BinTableHDU from resolutions in pixel units.

    Args:
        gausspars: ``(nband, ncorr, 3)`` in pixels, pixels, radians.
        corr: correlation labels, length ``ncorr``.
        cell_deg: cell size in degrees, converting the axes to degrees.

    Returns:
        An astropy ``BinTableHDU`` named BEAMS.
    """
    gausspars = np.asarray(gausspars, dtype=float)
    da = xr.DataArray(
        gausspars,
        dims=("band", "corr", "bpar"),
        coords={
            "band": np.arange(gausspars.shape[0]),
            "corr": list(corr),
            "bpar": ["BMAJ", "BMIN", "BPA"],
        },
    )
    return create_beams_table(da, cell2deg=cell_deg)


def write_fits(
    data, name, meta, unit="Jy/beam", gausspar=None, gausspars=None, beams_hdu=None, extra_hdr=None, otype=np.float32
):
    """Render a (Y, X)-ordered array to FITS with the ``.dt``'s WCS conventions.

    Used for the products the driver computes itself, which :func:`dt2fits`
    cannot source from a stored band variable -- principally the MFS restored
    images, which are not weighted sums of the per-band restored images.

    Args:
        data: ``(ncorr, ny, nx)`` or ``(nband, ncorr, ny, nx)``, (Y, X)-ordered.
        name: output path.
        meta: dict with ``cell_deg``, ``nx``, ``ny``, ``radec``, ``freq``,
            ``time_out``, ``l0`` and ``m0``. ``freq`` is scalar for an MFS image
            and an array for a cube.
        unit: BUNIT value.
        gausspar: ``(3,)`` MFS beam in pixel units for the BMAJ/BMIN/BPA cards.
        gausspars: ``(nband, 3)`` per-plane beams in pixel units.
        beams_hdu: optional BEAMS table from :func:`beams_table`.
        extra_hdr: extra header cards.
        otype: output dtype.
    """
    hdr = set_wcs(
        meta["cell_deg"],
        meta["cell_deg"],
        meta["nx"],
        meta["ny"],
        meta["radec"],
        meta["freq"],
        unit=unit,
        ms_time=meta["time_out"],
        time_is_unix=True,  # the .dt carries unix seconds (wiki D13)
        gausspar=gausspar,
        gausspars=gausspars,
        l0=meta["l0"],
        m0=meta["m0"],
    )
    for key, value in (extra_hdr or {}).items():
        hdr[key] = value
    save_fits(data, name, hdr, overwrite=True, dtype=otype, beams_hdu=beams_hdu, yx_order=True)
