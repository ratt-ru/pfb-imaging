"""Restore clean components onto residuals in the imager DataTree.

Consumes the ``.dt`` written by ``pfb imager`` and extended by ``pfb deconv``.
The band ``MODEL`` is intrinsic flux while the band ``RESIDUAL`` is apparent
(wiki D22/D23), so three restored products are offered rather than one: see
``utils/restoration.PRODUCT_VARS``.

No Ray: the work is FFT-bound with threaded ducc, and the driver needs the
image cubes resident anyway to form the MFS products.
"""

import time

import numpy as np
import psutil
import xarray as xr
from ducc0.misc import resize_thread_pool

from pfb_imaging import set_envs
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.naming import set_output_names
from pfb_imaging.utils.restoration import (
    PRODUCT_VARS,
    clean_beam,
    lowest_resolution,
    restore_products,
)

log = pfb_logging.get_logger("RESTORE")


def restore(
    output_filename: str,
    model_name: str = "MODEL",
    residual_name: str = "RESIDUAL",
    suffix: str = "main",
    outputs: str = "kK",
    gausspar: tuple[float, float, float] | None = None,
    drop_bands: list[int] | None = None,
    pb_min: float = 0.1,
    nthreads: int | None = None,
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
):
    """Restore clean components onto residuals and render FITS.

    Args:
        output_filename: basename; the tree read is ``<output>_<PRODUCT>.dt``.
        model_name: band variable holding the model image.
        residual_name: band variable holding the residual image.
        suffix: namespaces the FITS outputs only, never the tree path.
        outputs: product letters, lowercase for MFS and uppercase for cubes.
        gausspar: restoring resolution ``(emaj, emin, pa)`` in degrees, degrees
            and degrees. ``(0, 0, 0)`` selects the lowest-resolution band. None
            restores each band at its native resolution.
        drop_bands: band ids excluded from cubes and from every MFS reduction.
        pb_min: beam floor below which the intrinsic image is zeroed.
        nthreads: FFT threads; half the logical CPUs by default.
        log_directory: log destination.
        product: Stokes product, used to build the tree and FITS names.
        fits_output_folder: FITS destination.

    Raises:
        ValueError: the tree has no band nodes, was written without ``--psf``,
            lacks the named model or residual variable, or has no band left
            once dropped and fully flagged bands are excluded.
    """
    opts_dict = locals().copy()
    time_start = time.time()

    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )
    opts_dict["output_filename"] = output_filename
    opts_dict["fits_output_folder"] = fits_output_folder
    opts_dict["log_directory"] = log_directory

    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
    ncpu = int(np.minimum(nthreads, psutil.cpu_count(logical=False)))
    opts_dict["nthreads"] = nthreads
    resize_thread_pool(nthreads)
    set_envs(nthreads, ncpu, log=log)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pfb_logging.log_to_file(f"{log_directory}/restore_{timestamp}.log")
    log.log_options_dict(opts_dict, title="RESTORE options")

    # unlike the legacy .dds (where `suffix` came from), imager() never appends
    # a suffix to the tree name -- it namespaces the FITS outputs only
    basename = output_filename
    fits_oname = f"{fits_output_folder}/{oname}_{suffix}"
    dt_name = f"{basename}.dt"

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    band_nodes = [n for n in dt.children if n.startswith("band")]
    if not band_nodes:
        log.error_and_raise(f"No band nodes found in {dt_name}", ValueError)

    dropped = set(drop_bands or ())
    products = tuple(k for k in ("a", "i", "k") if k in outputs.lower())
    timeids = sorted({int(dt[n].ds.attrs["timeid"]) for n in band_nodes})
    log.info(f"Number of output times = {len(timeids)}")

    psfpars_by_time = {}

    for timeid in timeids:
        nodes = sorted(
            (n for n in band_nodes if int(dt[n].ds.attrs["timeid"]) == timeid),
            key=lambda n: int(dt[n].ds.attrs["bandid"]),
        )
        keep = [n for n in nodes if int(dt[n].ds.attrs["bandid"]) not in dropped]
        if dropped:
            log.info(f"time {timeid}: dropping bands {sorted(dropped)} from cubes and all MFS reductions")

        for n in keep:
            bds = dt[n].ds
            if "PSF" not in bds or "PSFPARSN" not in bds:
                log.error_and_raise(
                    f"{dt_name}/{n} has no PSF (imager run with --no-psf?) -- re-run pfb imager with --psf to restore",
                    ValueError,
                )
            for name in (model_name, residual_name):
                if name not in bds:
                    log.error_and_raise(
                        f"{dt_name}/{n} has no {name}; run pfb deconv first, or name an "
                        "existing variable with --model-name/--residual-name",
                        ValueError,
                    )

        # A fully flagged band carries WSUM == 0 (core/imager.py emits these --
        # freq_eff falls back to freq_nominal when wsum_tot == 0). Dividing the
        # residual by it yields inf/NaN, which would land in the stored products
        # AND in the MFS accumulators, poisoning every other band's MFS image.
        # Skip them exactly as --drop-bands does.
        dead = [n for n in keep if not float(dt[n].ds.WSUM.values.sum()) > 0]
        if dead:
            log.info(
                f"time {timeid}: skipping fully flagged bands "
                f"{[int(dt[n].ds.attrs['bandid']) for n in dead]} (WSUM == 0)"
            )
            keep = [n for n in keep if n not in set(dead)]
        if not keep:
            log.error_and_raise(
                f"No bands left at timeid {timeid} after --drop-bands and fully flagged bands were excluded",
                ValueError,
            )

        ref = dt[keep[0]].ds
        nband = len(keep)
        ncorr, ny, nx = ref[model_name].shape
        cell_deg = np.rad2deg(ref.attrs["cell_rad"])
        otype = ref[residual_name].dtype
        wsums = np.stack([dt[n].ds.WSUM.values for n in keep])  # (nband, ncorr)
        wsum_tot = wsums.sum(axis=0)
        psfparsn = np.stack([dt[n].ds.PSFPARSN.values for n in keep])  # (nband, ncorr, 3)

        # accumulate the MFS PSF one band at a time: stacking every band's PSF
        # would hold 4x the image cube (PSFs are 2nx by 2ny)
        psf_sum = np.zeros(dt[keep[0]].ds.PSF.shape, dtype=np.float64)
        for n in keep:
            psf_sum += dt[n].ds.PSF.values
        gausspar_mfs_native = clean_beam(psf_sum, wsum_tot)
        del psf_sum

        # resolve the restoring resolutions. gausspari is None wherever the
        # target already is the residual's own resolution, which skips the
        # residual reconvolution entirely.
        if gausspar is not None and np.any(np.asarray(gausspar, dtype=float)):
            target = np.tile([gausspar[0] / cell_deg, gausspar[1] / cell_deg, np.deg2rad(gausspar[2])], (ncorr, 1))
            log.info(
                f"Using specified resolution of ({gausspar[0]:.3e} deg, {gausspar[1]:.3e} deg, {gausspar[2]:.3e} deg)"
            )
            gaussparf = np.tile(target, (nband, 1, 1))
            gaussparf_mfs = target
            gausspari_band, gausspari_mfs = psfparsn, gausspar_mfs_native
        elif gausspar is not None:
            target = lowest_resolution(psfparsn)
            log.info(
                f"Using lowest resolution of ({target[0, 0] * cell_deg:.3e} deg, "
                f"{target[0, 1] * cell_deg:.3e} deg, {np.rad2deg(target[0, 2]):.3e} deg)"
            )
            gaussparf = np.tile(target, (nband, 1, 1))
            gaussparf_mfs = target
            gausspari_band, gausspari_mfs = psfparsn, gausspar_mfs_native
        else:
            gaussparf = psfparsn
            gaussparf_mfs = gausspar_mfs_native
            gausspari_band, gausspari_mfs = None, None
        log.info(
            f"MFS restoring beam: ({gaussparf_mfs[0, 0] * cell_deg:.3e} deg, "
            f"{gaussparf_mfs[0, 1] * cell_deg:.3e} deg, {np.rad2deg(gaussparf_mfs[0, 2]):.3e} deg)"
        )

        # per-band restore, accumulating the wsum-weighted MFS reductions as we
        # go so only one band's arrays are resident at a time
        m_mfs = np.zeros((ncorr, ny, nx), dtype=np.float64)
        r_mfs = np.zeros((ncorr, ny, nx), dtype=np.float64)
        b_mfs = np.zeros((ncorr, ny, nx), dtype=np.float64)
        for b, n in enumerate(keep):
            bds = dt[n].ds
            model = bds[model_name].values
            residual = bds[residual_name].values / wsums[b][:, None, None]
            beam = bds.BEAM.values
            w = wsums[b][:, None, None]
            m_mfs += w * model
            r_mfs += w * residual
            b_mfs += w * beam

            data_vars = {"PSFPARSF": (("corr", "bpar"), gaussparf[b].astype(otype))}
            if products:
                out = restore_products(
                    model,
                    residual,
                    beam,
                    gaussparf[b],
                    gausspari=None if gausspari_band is None else gausspari_band[b],
                    products=products,
                    pb_min=pb_min,
                    nthreads=nthreads,
                )
                for key, arr in out.items():
                    # convolve2gaussres returns a non-contiguous view under
                    # yx_order=True; zarr wants a contiguous buffer
                    data_vars[PRODUCT_VARS[key]] = (("corr", "y", "x"), np.ascontiguousarray(arr, dtype=otype))
            # to_zarr(mode="a") replaces a group's attrs wholesale, so start
            # from the band's own attrs (core/deconv.py:409-424)
            xr.Dataset(
                data_vars,
                attrs={
                    **dict(bds.attrs),
                    "psfparsf_mfs": [float(v) for v in gaussparf_mfs[0]],
                    "pb_min": float(pb_min),
                },
            ).to_zarr(dt_name, group=n, mode="a")

        m_mfs /= wsum_tot[:, None, None]
        r_mfs /= wsum_tot[:, None, None]
        b_mfs /= wsum_tot[:, None, None]

        psfpars_by_time[timeid] = gaussparf_mfs
        log.info(f"time {timeid}: restored {nband} bands")

    log.info(f"All done after {time.time() - time_start:.1f}s")
