import concurrent.futures as cf
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import torch
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool, thread_pool_size
from numba import threading_layer

from pfb_imaging import set_envs
from pfb_imaging.operators.gridder import compute_residual
from pfb_imaging.operators.hessian import HessPSF
from pfb_imaging.deconv.ista_fb import ISTA
from pfb_imaging.opt.power_method import power_method
from pfb_imaging.prox.prox_airi import ProxOpAIRI
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import dds2fits, save_fits, set_wcs
from pfb_imaging.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
from pfb_imaging.utils.naming import get_opts, set_output_names, xds_from_url

log = pfb_logging.get_logger("AIRI")


@pfb_logging.log_inputs(log)
def airi(
    output_filename: str,
    suffix: str = "main",
    hess_norm: float | None = None,
    hess_approx: str = "psf",
    rmsfactor: float = 1.0,
    eta: float = 1.0,
    gamma: float = 1.0,
    nbasisf: int | None = None,
    positivity: int = 1,
    niter: int = 10,
    nthreads: int | None = None,
    tol: float = 0.0005,
    diverge_count: int = 5,
    verbosity: int = 1,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    fb_tol: float = 3e-4,
    fb_maxit: int = 450,
    fb_verbose: int = 1,
    fb_report_freq: int = 50,
    pm_tol: float = 0.001,
    pm_maxit: int = 100,
    pm_verbose: int = 1,
    pm_report_freq: int = 100,
    cg_tol: float = 0.001,
    cg_maxit: int = 150,
    cg_verbose: int = 1,
    cg_report_freq: int = 10,
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
    shelf_path: Path | None = None,
    airi_heuristic_scale: float = 1.0,
    airi_adapt_update: bool = True,
    airi_peak_tol: float = 0.05,
    airi_device: str = "cuda",
    airi_tile_size: int | None = None,
    airi_tile_margin: int = 16,
):
    """
    Deconvolution using AIRI regularisation
    """
    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2
    else:
        ncpu = np.minimum(nthreads, psutil.cpu_count(logical=False))

    resize_thread_pool(nthreads)
    set_envs(nthreads, ncpu)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/airi_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    basename = output_filename
    fits_oname = f"{fits_output_folder}/{oname}"
    dds_name = f"{basename}_{suffix}.dds"

    time_start = time.time()

    dds, dds_list = xds_from_url(dds_name)

    if dds[0].corr.size > 1:
        log.error_and_raise(
            "Joint polarisation deconvolution not yet supported for sara algorithm", NotImplementedError
        )

    nx, ny = dds[0].x.size, dds[0].y.size
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))
    if time_out.size > 1:
        log.error_and_raise("Only static models currently supported", NotImplementedError)

    nband = freq_out.size

    # drop_vars after access to avoid duplicates in memory
    # and avoid unintentional side effects?
    if "RESIDUAL" in dds[0]:
        residual = np.stack([ds.RESIDUAL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("DIRTY") for ds in dds]
        dds = [ds.drop_vars("RESIDUAL") for ds in dds]
    else:
        residual = np.stack([ds.DIRTY.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("DIRTY") for ds in dds]
    if "MODEL" in dds[0]:
        model = np.stack([ds.MODEL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("MODEL") for ds in dds]
    else:
        model = np.zeros((nband, nx, ny))
    if "UPDATE" in dds[0]:
        update = np.stack([ds.UPDATE.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("UPDATE") for ds in dds]
    else:
        update = np.zeros((nband, nx, ny))
    abspsf = np.stack([np.abs(ds.PSFHAT.values[0]) for ds in dds], axis=0)
    dds = [ds.drop_vars("PSFHAT") for ds in dds]
    beam = np.stack([ds.BEAM.values[0] for ds in dds], axis=0)
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands

    wsum = np.sum(wsums)
    wsums /= wsum
    abspsf /= wsum
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)
    model_mfs = np.mean(model[fsel], axis=0)

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

    # image space hessian
    precond = HessPSF(
        nx,
        ny,
        abspsf,
        beam=beam,
        eta=eta * wsums,
        nthreads=nthreads,
        cgtol=cg_tol,
        cgmaxit=cg_maxit,
        cgverbose=cg_verbose,
        cgrf=cg_report_freq,
        taper_width=np.minimum(int(0.1 * nx), 32),
    )

    if hess_norm is None:
        # if the grid worker had been rerun hess_norm won't be in attrs
        if "hess_norm" in dds[0].attrs:
            hess_norm = dds[0].hess_norm
            log.info(f"Using previously estimated hess_norm of {hess_norm:.3e}")
        else:
            log.info("Finding spectral norm of Hessian approximation")
            hess_norm, hessbeta = power_method(
                precond.dot,
                (nband, nx, ny),
                tol=pm_tol,
                maxit=pm_maxit,
                verbosity=pm_verbose,
                report_freq=pm_report_freq,
            )
            # inflate slightly for stability
            hess_norm *= 1.05
    else:
        hess_norm = hess_norm
        log.info(f"Using provided hess-norm of = {hess_norm:.3e}")

    log.info(f"Using {thread_pool_size()} threads for gridding")
    # number of frequency basis functions
    if nbasisf is None:
        nbasisf = int(np.sum(fsel))
    else:
        nbasisf = nbasisf
    log.info(f"Using {nbasisf} frequency basis functions")

    # Initialize AIRI prox operator
    if shelf_path is not None:
        log.info("Initializing AIRI prox operator")

        # Heuristic noise level calculation
        # Based on BASP small-scale-radio-imaging implementation (fb_airi.py)
        heuristic_noise_scale = 1.0 / np.sqrt(2 * hess_norm)
        heuristic_noise_scale *= airi_heuristic_scale
        noise_std = np.std(residual_mfs)
        heuristic_noise_scale *= noise_std

        log.info(f"AIRI noise heuristic: {heuristic_noise_scale:.6e}")

        # Initialize ProxOpAIRI
        device = torch.device(airi_device if (torch.cuda.is_available() and airi_device == "cuda") else "cpu")
        log.info(f"AIRI prox operator will run on: {device}")

        airi_prox = ProxOpAIRI(shelf_path=shelf_path, device=device, dtype=torch.float32, verbose=True)

        # Initial peak estimate from residual
        peak_est = np.abs(residual_mfs).max()
        log.info(f"Using residual peak as estimate: {peak_est:.6e}")

        # Initialize prox with heuristic and peak estimate
        peak_range = airi_prox.update(heuristic_noise_scale, peak_est)
        prev_peak_val = peak_est

        use_airi_prox = True
        log.info("AIRI prox operator initialized successfully")
        log.info(f"Expected peak range: [{peak_range[0]:.6e}, {peak_range[1]:.6e}]")
        if airi_tile_size is not None:
            log.info(f"Using tiled processing: tile_size={airi_tile_size}, margin={airi_tile_margin}")
    else:
        use_airi_prox = False
        log.warning("No shelf-path provided, AIRI prox operator will not be used")
        log.warning("This implementation is incomplete - using placeholder")

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count = 0
    eps = 1.0
    write_futures = None
    log.info(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}")
    mrange = range(iter0, iter0 + niter)
    for k in mrange:
        log.info("Solving for update")
        residual *= beam  # avoid copy
        update = precond.idot(residual, mode=hess_approx, x0=update if update.any() else None)
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs, fits_oname + f"_{suffix}_update_{k + 1}.fits", hdr_mfs)

        modelp = deepcopy(model)
        xtilde = model + gamma * update
        lam = rmsfactor * rms
        log.info(f"Solving for model with lambda = {lam}")

        if use_airi_prox:
            # Apply AIRI prox operator (backward step)
            # Convert to torch tensor with shape (1, 1, nx, ny) for MFS
            # NOTE: Multi-band handling may need adjustment by team
            xtilde_mfs = np.mean(xtilde, axis=0, keepdims=True)
            xtilde_torch = torch.from_numpy(xtilde_mfs[None, :, :, :]).float().to(airi_prox.get_device())

            if airi_tile_size is not None:
                model_mfs_torch = airi_prox.call_tiled(xtilde_torch, tile_size=airi_tile_size, margin=airi_tile_margin)
            else:
                model_mfs_torch = airi_prox(xtilde_torch)
            model_mfs_denoised = model_mfs_torch.cpu().numpy().squeeze()

            # Broadcast MFS result back to all bands
            # TODO: Team to implement proper multi-band AIRI handling
            model = np.broadcast_to(model_mfs_denoised, xtilde.shape).copy()

            # Apply positivity constraint
            if positivity > 0:
                model = np.maximum(model, 0)

            # Adaptive denoiser update (from BASP fb_airi.py)
            if airi_adapt_update:
                curr_peak_val = np.abs(model).max()
                peak_var = abs(curr_peak_val - prev_peak_val) / (prev_peak_val + 1e-10)
                log.info(f"Peak: {curr_peak_val:.6e}, variation: {peak_var:.6e}")

                if peak_var < airi_peak_tol and (curr_peak_val < peak_range[0] or curr_peak_val > peak_range[1]):
                    log.info("Updating AIRI denoiser based on new peak estimate")
                    peak_range = airi_prox.update(heuristic_noise_scale, curr_peak_val)
                prev_peak_val = curr_peak_val
        else:
            # Fallback to ISTA
            prox_solver = ISTA(
                lmbda=5e-7,
                gamma=hess_norm,
                max_iter=20,
                step_size=1,
                precond=precond,
            )
            model = prox_solver(xtilde)

        # write component model
        log.info(f"Writing model to {basename}_{suffix}_model.mds")
        try:
            coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
                time_out,
                freq_out[fsel],
                model[None, fsel, :, :],
                wgt=wsums[None, fsel],
                nbasisf=nbasisf,
                method="Legendre",
                sigmasq=1e-6,
            )
            # save interpolated dataset
            data_vars = {
                "coefficients": (("par", "comps"), coeffs),
            }
            coords = {
                "location_x": (("x",), x_index),
                "location_y": (("y",), y_index),
                # 'shape_x':,
                "params": (("par",), params),  # already converted to list
                "times": (("t",), time_out),  # to allow rendering to original grid
                "freqs": (("f",), freq_out),
            }
            mattrs = {
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

            coeff_dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=mattrs)
            coeff_dataset.to_zarr(f"{basename}_{suffix}_model.mds", mode="w")

            for b in range(nband):
                model[b] = eval_coeffs_to_slice(
                    time_out[0],
                    freq_out[b],
                    coeffs,
                    x_index,
                    y_index,
                    expr,
                    params,
                    texpr,
                    fexpr,
                    nx,
                    ny,
                    cell_rad,
                    cell_rad,
                    dds[0].x0,
                    dds[0].y0,
                    nx,
                    ny,
                    cell_rad,
                    cell_rad,
                    dds[0].x0,
                    dds[0].y0,
                )
        except Exception as e:
            log.info(f"Exception {e} raised during model fit .")

        model = model[np.newaxis, ...] if len(model.shape) == 2 else model
        model_mfs = np.mean(model[fsel], axis=0)
        save_fits(model_mfs, fits_oname + f"_{suffix}_model_{k + 1}.fits", hdr_mfs)

        # make sure write futures have finished
        if write_futures is not None:
            cf.wait(write_futures)

        log.info("Computing residual")
        write_futures = []
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            resid, fut = compute_residual(
                ds_name,
                nx,
                ny,
                cell_rad,
                cell_rad,
                ds_name,
                model[b][None, :, :],  # add corr axis
                nthreads=nthreads,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                verbosity=verbosity,
            )
            write_futures.append(fut)
            residual[b] = resid[0]  # remove corr axis

        residual /= wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs, fits_oname + f"_{suffix}_residual_{k + 1}.fits", hdr_mfs)
        rmsp = rms
        rms = np.std(residual_mfs)
        rmaxp = rmax
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        # what to base this on?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        # these are not updated in compute_residual
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            ds["UPDATE"] = (("corr", "x", "y"), update[b][None, :, :])
            # don't write unecessarily
            for var in ds.data_vars:
                if var != "UPDATE":
                    ds = ds.drop_vars(var)

            if (model == best_model).all():
                ds["MODEL_BEST"] = (("corr", "x", "y"), best_model[b][None, :, :])

            attrs = {}
            attrs["rms"] = best_rms
            attrs["rmax"] = best_rmax
            attrs["niters"] = k + 1
            attrs["hess_norm"] = hess_norm
            ds = ds.assign_attrs(**attrs)

            with cf.ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(ds.to_zarr, ds_name, mode="a")
                write_futures.append(fut)

        log.info(f"Iter {k + 1}: peak residual = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}")

        if (rms > rmsp) and (rmax > rmaxp):
            diverge_count += 1
            if diverge_count > diverge_count:
                log.info("Algorithm is diverging. Terminating.")
                break

    # make sure write futures have finished
    cf.wait(write_futures)

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
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )
        log.info("Done writing RESIDUAL")
        dds2fits(
            dds_list,
            "MODEL",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )
        log.info("Done writing MODEL")
        dds2fits(
            dds_list,
            "UPDATE",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )
        log.info("Done writing UPDATE")

    log.info(f"Numba used the {threading_layer()} threading layer")
    log.info(f"All done after {time.time() - time_start}s")
