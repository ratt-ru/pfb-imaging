"""Restoration helpers for the imager DataTree.

Two groups, both free of tree I/O and of Ray -- ``core/restore.py`` owns reading
the ``.dt`` and writing products back into it:

* **Pure array functions** -- :func:`clean_beam`, :func:`lowest_resolution` and
  :func:`restore_products` -- which take and return numpy arrays.
* **FITS rendering helpers** -- :func:`beams_table` and :func:`write_fits` --
  for the products the driver computes itself and so cannot source from a
  stored band variable via :func:`~pfb_imaging.utils.fits.dt2fits`. Principally
  the MFS restored images: the per-band restored images sit at different
  resolutions, so their weighted sum is not the MFS restored image.

Flux-scale conventions (wiki D22/D23): the band ``MODEL`` is intrinsic flux and
the band ``RESIDUAL`` is apparent (once-attenuated) flux, so a restored image
must say which scale it is on. Three are offered, keyed by their CLI letter.
"""

import numpy as np
import xarray as xr
from scipy.linalg import eigh

from pfb_imaging.utils.fits import create_beams_table, save_fits, set_wcs
from pfb_imaging.utils.misc import convolve2gaussres, fitcleanbeam, gauss_cov

# CLI letter -> DataTree variable name. The B prefix follows the tree's
# convention for beam-attenuated quantities (BDIRTY, BRESIDUAL); the C prefix on
# CRESIDUAL means convolved to the restoring resolution.
PRODUCT_VARS = {"a": "BIMAGE", "i": "IMAGE", "k": "KIMAGE", "s": "CRESIDUAL"}


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


def _mean_pa(pas):
    """Circular mean of position angles, which are defined modulo pi.

    ``np.nanmean`` is wrong for angles: PAs of 0.05 and pi - 0.05 describe two
    nearly identical orientations but average to pi/2, which is orthogonal to
    both. Doubling before averaging maps the modulo-pi circle onto a full one.
    """
    pas = np.asarray(pas, dtype=float)
    return 0.5 * np.arctan2(np.nanmean(np.sin(2 * pas)), np.nanmean(np.cos(2 * pas)))


def resolution_deficit(target, gausspars):
    """How far short of dominating every input the target resolution falls.

    Args:
        target: ``(ncorr, 3)`` candidate restoring resolution.
        gausspars: ``(n, ncorr, 3)`` resolutions it must be convolvable from.
            NaN rows are skipped.

    Returns:
        ``(ncorr,)`` largest factor by which ``target``'s axes must grow, as a
        ratio of areas. At most 1.0 means the target is valid; above 1.0 the
        target is sharper than some input along some direction, and each axis
        must grow by its square root. NaN where every input is NaN.
    """
    target = np.asarray(target, dtype=float)
    gausspars = np.asarray(gausspars, dtype=float)
    out = np.full(gausspars.shape[1], np.nan, dtype=float)
    for c in range(gausspars.shape[1]):
        rows = gausspars[:, c, :]
        rows = rows[~np.isnan(rows).any(axis=1)]
        if not len(rows):
            continue
        shape = gauss_cov(target[c])
        out[c] = max(eigh(gauss_cov(row), shape, eigvals_only=True).max() for row in rows)
    return out


def lowest_resolution(gausspars):
    """Smallest common resolution every input Gaussian can be convolved to.

    "At least as wide as every input" is a statement about the covariances in
    the Loewner order -- ``Sf - Sj`` positive semi-definite for every input j --
    not about the axes separately. Taking the max of each axis and the mean of
    the position angles satisfies the second and not the first: a rotated
    ellipse pokes out diagonally, and convolving to such a target is a
    deconvolution along some direction. ``convolve2gaussres`` now refuses it
    (wiki D31), where it used to silently amplify noise.

    So the max-axis, mean-PA ellipse is used only as a candidate *shape*. Each
    candidate is inflated by the smallest scalar that makes it dominate every
    input -- the largest generalised eigenvalue, exact in closed form -- and the
    smallest resulting ellipse wins. Candidates are the circular-mean PA plus
    each input's own PA, so when one input is the widest at its own angle the
    result is that input, with no inflation at all.

    Args:
        gausspars: ``(n, ncorr, 3)`` in pixels, pixels, radians, over whatever
            set must be dominated -- the per-band beams, and the MFS beam too
            when it is reconvolved to the same target. NaN rows (fully flagged
            bands) are skipped.

    Returns:
        ``(ncorr, 3)``: a resolution at least as low as every input, in every
        direction. Rows are NaN where every input is NaN.
    """
    gausspars = np.asarray(gausspars, dtype=float)
    out = np.full(gausspars.shape[1:], np.nan, dtype=float)
    for c in range(gausspars.shape[1]):
        rows = gausspars[:, c, :]
        rows = rows[~np.isnan(rows).any(axis=1)]
        if not len(rows):
            continue
        covs = [gauss_cov(row) for row in rows]
        emaj, emin = rows[:, 0].max(), rows[:, 1].max()
        best = None
        # candidate orientations, and candidate axis ratios between the widest
        # input's and circular. Neither alone is enough: when the inputs share a
        # PA the max-axis ellipse wins, and when they are spread over a wide
        # range of angles a rounder beam contains them more tightly.
        for pa in np.concatenate([[_mean_pa(rows[:, 2])], rows[:, 2]]):
            for ratio in np.linspace(emin / emaj, 1.0, 5):
                shape = gauss_cov((emaj, emaj * ratio, pa))
                # smallest s with s * shape >= cov, i.e. the largest generalised
                # eigenvalue of the pencil (cov, shape). A margin keeps the
                # binding input inside gauss_ratio_hat's definiteness check.
                scale = (1.0 + 1e-6) * max(eigh(cov, shape, eigvals_only=True).max() for cov in covs)
                area = emaj * emaj * ratio * scale
                if best is None or area < best[0]:
                    best = (area, np.sqrt(scale) * emaj, np.sqrt(scale) * emaj * ratio, pa)
        out[c] = best[1:]
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
        products: iterable over ``"a"`` (apparent), ``"i"`` (intrinsic),
            ``"k"`` (intrinsic model plus apparent residual) and ``"s"``, the
            residual convolved to ``gaussparf`` on its own. ``"s"`` is not a
            restored image; it is the term the other three add to the model, and
            it is stored because nothing else in the tree records what the
            resolution change actually did to the residual.
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
    if "s" in products:
        # apparent, pre-beam-division: BEAM is on the node, so the intrinsic form
        # is one divide away, and this is the scale image-plane noise is flat on
        out["s"] = rconv if rconv is not residual else residual.copy()
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
