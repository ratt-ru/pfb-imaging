"""Restoration helpers for the imager DataTree.

Pure array functions: no tree I/O, no Ray, no FITS. ``core/restore.py`` owns
reading the ``.dt``, writing products back and rendering FITS.

Flux-scale conventions (wiki D22/D23): the band ``MODEL`` is intrinsic flux and
the band ``RESIDUAL`` is apparent (once-attenuated) flux, so a restored image
must say which scale it is on. Three are offered, keyed by their CLI letter.
"""

import numpy as np

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
        out["a"] = convolve2gaussres(beam * model, xx, yy, gaussparf, **kw) + rconv
    return out
