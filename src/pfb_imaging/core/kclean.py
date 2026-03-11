import time

import numpy as np
import psutil
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool
from numba import set_num_threads
from scipy import ndimage

from pfb_imaging import pfb_version, set_envs
from pfb_imaging.deconv.clark import clark
from pfb_imaging.operators.gridder import compute_residual
from pfb_imaging.operators.hessian import HessPSF
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import dds2fits, load_fits, save_fits, set_wcs
from pfb_imaging.utils.modelspec import fit_image_cube
from pfb_imaging.utils.naming import get_opts, set_output_names, xds_from_url

log = pfb_logging.get_logger("KCLEAN")


def kclean(
    output_filename: str,
    suffix: str = "main",
    mask: str | None = None,
    dirosion: int = 1,
    mop_flux: bool = True,
    mop_gamma: float = 0.65,
    niter: int = 5,
    nthreads: int | None = None,
    threshold: float | None = None,
    rmsfactor: float = 3.0,
    eta: float = 0.001,
    gamma: float = 0.1,
    peak_factor: float = 0.15,
    sub_peak_factor: float = 0.75,
    minor_maxit: int = 50,
    subminor_maxit: int = 1000,
    verbose: int = 1,
    report_freq: int = 10,
    cg_tol: float = 0.01,
    cg_maxit: int = 100,
    cg_verbose: int = 1,
    cg_report_freq: int = 100,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
):
    """
    Modified single-scale clean.
    """
    # for logging options
    opts_dict = locals().copy()

    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )
    opts_dict["output_filename"] = output_filename
    opts_dict["fits_output_folder"] = fits_output_folder
    opts_dict["log_directory"] = log_directory

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2

    resize_thread_pool(nthreads)
    set_num_threads(nthreads)
    set_envs(nthreads, ncpu, log=log)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/kclean_{timestamp}.log"
    pfb_logging.log_to_file(logname)

    log.log_options_dict(opts_dict, title="KCLEAN options")

    basename = f"{output_filename}"
    fits_oname = f"{fits_output_folder}/{oname}"
    dds_name = f"{basename}_{suffix}.dds"

    time_start = time.time()

    # Only support product='I' at the moment
    if product != "I":
        log.error_and_raise(
            "Only product='I' is currently supported. Full Stokes deconvolution is not yet available.",
            NotImplementedError,
        )

    if fits_output_folder is not None:
        fits_oname = fits_output_folder + "/" + basename.split("/")[-1]
    else:
        fits_oname = basename

    dds, dds_list = xds_from_url(dds_name)

    if dds[0].corr.size > 1:
        log.error_and_raise(
            "Joint polarisation deconvolution not yet supported for kclean algorithm", NotImplementedError
        )

    nx, ny = dds[0].x.size, dds[0].y.size
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))

    nband = freq_out.size

    # stitch dirty/psf in apparent scale
    # drop_vars to avoid duplicates in memory
    if "RESIDUAL" in dds[0]:
        residual = np.stack([ds.RESIDUAL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("RESIDUAL") for ds in dds]
    else:
        residual = np.stack([ds.DIRTY.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("DIRTY") for ds in dds]
    if "MODEL" in dds[0]:
        model = np.stack([ds.MODEL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("MODEL") for ds in dds]
    else:
        model = np.zeros((nband, nx, ny))
    psf = np.stack([ds.PSF.values[0] for ds in dds], axis=0)
    psfhat = np.stack([ds.PSFHAT.values[0] for ds in dds], axis=0)
    abspsf = np.stack([np.abs(ds.PSFHAT.values[0]) for ds in dds], axis=0)
    beam = np.stack([ds.BEAM.values[0] for ds in dds], axis=0)
    dds = [ds.drop_vars(("PSF", "PSFHAT")) for ds in dds]
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands

    wsum = np.sum(wsums)
    # TODO - do we really need to keep all three versions of the PSF in memory here?
    psf /= wsum
    psfhat /= wsum
    abspsf /= wsum
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)

    # for intermediary results
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq, casambm=False)
    if "niters" in dds[0].attrs:
        iter0 = dds[0].niters
    else:
        iter0 = 0

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if mask is not None:
        mask = load_fits(mask, dtype=residual.dtype).squeeze()
        assert mask.shape == (nx, ny)
        log.info("Using provided fits mask")
    else:
        mask = np.ones((nx, ny), dtype=residual.dtype)

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count = 0
    if threshold is None:
        threshold = rmsfactor * rms
    else:
        threshold = threshold

    precond = HessPSF(
        nx,
        ny,
        abspsf,
        beam=mask[None, :, :] * beam,
        eta=eta * wsums,
        nthreads=nthreads,
        cgtol=cg_tol,
        cgmaxit=cg_maxit,
        cgverbose=cg_verbose,
        cgrf=cg_report_freq,
        taper_width=np.minimum(int(0.1 * nx), 32),
    )

    log.info(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}")
    for k in range(iter0, iter0 + niter):
        log.info("Cleaning")
        x, status = clark(
            residual,
            psf,
            psfhat,
            wsums / wsum,
            mask,
            threshold=threshold,
            gamma=gamma,
            pf=peak_factor,
            maxit=minor_maxit,
            subpf=sub_peak_factor,
            submaxit=subminor_maxit,
            verbosity=verbose,
            report_freq=report_freq,
            nthreads=nthreads,
        )

        model += x

        # write component model
        log.info(f"Writing model at iter {k + 1} to {basename}_{suffix}_model.mds")
        try:
            coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
                time_out,
                freq_out[fsel],
                model[None, fsel, :, :],
                wgt=wsums[None, fsel],
                nbasisf=int(np.sum(fsel)),
                method="Legendre",
            )
            # save interpolated dataset
            data_vars = {
                "coefficients": (("par", "comps"), coeffs),
            }
            coords = {
                "location_x": (("x",), x_index),
                "location_y": (("y",), y_index),
                "params": (("par",), params),  # already converted to list
                "times": (("t",), time_out),  # to allow rendering to original grid
                "freqs": (("f",), freq_out),
            }
            attrs = {
                "pfb-imaging-version": pfb_version,
                "spec": "genesis",
                "cell_rad_x": cell_rad,
                "cell_rad_y": cell_rad,
                "npix_x": nx,
                "npix_y": ny,
                "texpr": texpr,
                "fexpr": fexpr,
                "center_x": dds[0].x0,
                "center_y": dds[0].y0,
                "flip_u": dds[0].flip_u,
                "flip_v": dds[0].flip_v,
                "flip_w": dds[0].flip_w,
                "ra": dds[0].ra,
                "dec": dds[0].dec,
                "stokes": product,  # I,Q,U,V, IQ/IV, IQUV
                "parametrisation": expr,  # already converted to str
            }

            coeff_dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            coeff_dataset.to_zarr(f"{basename}_{suffix}_model.mds", mode="w")
        except Exception as e:
            log.info(f"Exception {e} raised during model fit .")

        if fits_mfs:
            save_fits(np.mean(model[fsel], axis=0), fits_oname + f"_{suffix}_model_{k + 1}.fits", hdr_mfs)

        log.info("Computing residual")
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            resid, fut = compute_residual(
                ds_name,
                nx,
                ny,
                cell_rad,
                cell_rad,
                ds_name,
                model[b][None, :, :],  # add back corr axis
                nthreads=nthreads,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                async_write=False,
            )
            residual[b] = resid[0]  # remove corr axis
        residual /= wsum
        residual_mfs = np.sum(residual, axis=0)
        if fits_mfs:
            save_fits(residual_mfs, fits_oname + f"_{suffix}_residual_{k + 1}.fits", hdr_mfs)

        # report rms and rmax inside mask
        rmsp = rms
        # tmp_mask = ~np.any(model, axis=0)
        rms = np.std(residual_mfs[mask > 0])
        rmax = np.abs(residual_mfs[mask > 0]).max()

        # base this on rmax?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        # these are not updated in compute_residual
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            ds = ds.assign(
                **{
                    "MODEL": (("corr", "x", "y"), model[b][None, :, :]),
                    "MODEL_BEST": (("corr", "x", "y"), best_model[b][None, :, :]),
                }
            )
            attrs = {}
            attrs["rms"] = best_rms
            attrs["rmax"] = best_rmax
            attrs["niters"] = k + 1
            ds = ds.assign_attrs(**attrs)
            ds.to_zarr(ds_name, mode="a")

        if threshold is None:
            threshold = rmsfactor * rms
        else:
            threshold = threshold

        # trigger flux mop if clean has stalled, not converged or
        # we have reached the final iteration/threshold
        status |= k == iter0 + niter - 1
        status |= rmax <= threshold
        if mop_flux and status:
            log.info(f"Extracting flux at iter {k + 1}")
            mopmask = np.any(model, axis=0)
            if dirosion:
                struct = ndimage.generate_binary_structure(2, dirosion)
                mopmask = ndimage.binary_dilation(mopmask, structure=struct)
                mopmask = ndimage.binary_erosion(mopmask, structure=struct)

            # TODO - applying mask as beam is wasteful
            mopmask = (mopmask.astype(residual.dtype) * mask)[None, :, :]
            precond.set_beam(mopmask * beam)
            x = precond.idot(residual * beam, mode="psf", init_x0=False)
            model += mop_gamma * x

            log.info("Computing residual")
            for ds_name, ds in zip(dds_list, dds):
                b = int(ds.bandid)
                resid, _ = compute_residual(
                    ds_name,
                    nx,
                    ny,
                    cell_rad,
                    cell_rad,
                    ds_name,
                    model[b][None, :, :],  # add back corr axis
                    nthreads=nthreads,
                    epsilon=epsilon,
                    do_wgridding=do_wgridding,
                    double_accum=double_accum,
                    async_write=False,
                )
                residual[b] = resid[0]  # remove corr axis
            residual /= wsum
            residual_mfs = np.sum(residual, axis=0)

            if fits_mfs:
                save_fits(residual_mfs, f"{fits_oname}_{suffix}_postmop{k + 1}_residual_mfs.fits", hdr_mfs)
                save_fits(np.mean(model[fsel], axis=0), f"{fits_oname}_{suffix}_postmop{k + 1}_model_mfs.fits", hdr_mfs)

            rmsp = rms
            # tmp_mask = ~np.any(model, axis=0)
            rms = np.std(residual_mfs[mask > 0])
            rmax = np.abs(residual_mfs[mask > 0]).max()

            # base this on rmax?
            if rms < best_rms:
                best_rms = rms
                best_rmax = rmax
                best_model = model.copy()
                for ds_name, ds in zip(dds_list, dds):
                    b = int(ds.bandid)
                    ds = ds.assign(**{"MODEL_BEST": (("corr", "x", "y"), best_model[b][None, :, :])})

                    attrs = {}
                    attrs["rms"] = best_rms
                    attrs["rmax"] = best_rmax
                    attrs["niters"] = k + 1
                    ds = ds.assign_attrs(**attrs)
                    ds.to_zarr(ds_name, mode="a")
                    ds.to_zarr(ds_name, mode="a")

            if threshold is None:
                threshold = rmsfactor * rms
            else:
                threshold = threshold

        log.info(f"Iter {k + 1}: peak residual = {rmax:.3e}, rms = {rms:.3e}")

        if rmax <= threshold:
            log.info("Terminating because final threshold has been reached")
            break

        if rms > rmsp:
            diverge_count += 1
            if diverge_count > 3:
                log.info("Algorithm is diverging. Terminating.")
                break

    # Write FITS outputs
    dds, dds_list = xds_from_url(dds_name)

    if fits_mfs or fits_cubes:
        # get the psfpars for the mfs cube
        dds_store = DaskMSStore(dds_name)
        if "://" in dds_store.url:
            protocol = dds_store.url.split("://")[0]
        else:
            protocol = "file"
        psfpars_mfs = get_opts(dds_store.url, protocol, name="psfparsn_mfs.pkl")
        log.info(f"Writing fits files to {fits_oname}_{suffix}")
        dds2fits(
            dds_list,
            "RESIDUAL",
            f"{fits_oname}_{suffix}",
            norm_wsum=True,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )
        dds2fits(
            dds_list,
            "MODEL",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )

    log.info(f"All done after {time.time() - time_start}s")
