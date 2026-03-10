import concurrent.futures as cf
import time
from copy import deepcopy

import numpy as np
import psutil
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool
from numba import threading_layer

from pfb_imaging import pfb_version, set_envs
from pfb_imaging.deconv import DeconvSolver
from pfb_imaging.operators.gridder import compute_residual
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import dds2fits, save_fits, set_wcs
from pfb_imaging.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
from pfb_imaging.utils.naming import get_opts, set_output_names, xds_from_url

log = pfb_logging.get_logger("DECONV")


def deconv(
    output_filename: str,
    suffix: str = "main",
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
    minor_cycle: str = "sara",
    opt_backend: str = "primal-dual",
    bases: list[str] = ["self", "db1", "db2", "db3"],
    nlevels: int = 3,
    l1_reweight_from: int = 5,
    alpha: float = 2.0,
    hess_norm: float | None = None,
    hess_approx: str = "psf",
    rmsfactor: float = 1.0,
    eta: float = 0.001,
    gamma: float = 0.95,
    nbasisf: int | None = None,
    positivity: int = 1,
    niter: int = 10,
    tol: float = 0.0005,
    diverge_count: int = 5,
    rms_outside_model: bool = False,
    init_factor: float = 0.5,
    verbosity: int = 1,
    nthreads: int | None = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    pd_tol: float = 0.0003,
    pd_maxit: int = 1000,
    pd_verbose: int = 1,
    pd_report_freq: int = 100,
    fb_tol: float = 0.0003,
    fb_maxit: int = 1000,
    fb_verbose: int = 1,
    fb_report_freq: int = 100,
    acceleration: bool = True,
    pm_tol: float = 0.001,
    pm_maxit: int = 500,
    pm_verbose: int = 1,
    pm_report_freq: int = 100,
    cg_tol: float = 0.001,
    cg_maxit: int = 150,
    cg_verbose: int = 1,
    cg_report_freq: int = 10,
):
    """
    General preconditioned forward-backward deconvolution loop.

    All regulariser-specific logic lives in the solver that is initialised based on the minor_cycle argument.
    """
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
    else:
        ncpu = np.minimum(nthreads, psutil.cpu_count(logical=False))
    opts_dict["nthreads"] = nthreads
    log.info(f"Using {nthreads} threads total")
    resize_thread_pool(nthreads)
    set_envs(nthreads, ncpu, log)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/deconv_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.log_options_dict(opts_dict, title="DECONV options")

    basename = output_filename
    fits_oname = f"{fits_output_folder}/{oname}"
    dds_name = f"{basename}_{suffix}.dds"

    time_start = time.time()

    dds, dds_list = xds_from_url(dds_name)

    if dds[0].corr.size > 1:
        log.error_and_raise("Joint polarisation deconvolution not yet supported", NotImplementedError)

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
    fsel = wsums > 0

    wsum = np.sum(wsums)
    wsums /= wsum
    abspsf /= wsum
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)
    model_mfs = np.mean(model[fsel], axis=0)

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

    if nbasisf is None:
        nbasisf = int(np.sum(fsel))

    # hess_norm from DDS cache if available; solver estimates it otherwise
    hess_norm = dds[0].hess_norm if "hess_norm" in dds[0].attrs else None
    if hess_norm is not None:
        log.info(f"Using previously estimated hess_norm of {hess_norm:.3e}")

    # construct solver based on minor_cycle + opt_backend
    sara_common = dict(
        nband=nband,
        nx=nx,
        ny=ny,
        abspsf=abspsf,
        beam=beam,
        wsums=wsums,
        model=model,
        update=update,
        nthreads=nthreads,
        eta=eta,
        hess_approx=hess_approx,
        hess_norm=hess_norm,
        cg_tol=cg_tol,
        cg_maxit=cg_maxit,
        cg_verbose=cg_verbose,
        cg_report_freq=cg_report_freq,
        pm_tol=pm_tol,
        pm_maxit=pm_maxit,
        pm_verbose=pm_verbose,
        pm_report_freq=pm_report_freq,
        bases=bases,
        nlevels=nlevels,
        gamma=gamma,
        positivity=positivity,
        l1_reweight_from=l1_reweight_from,
        rmsfactor=rmsfactor,
        alpha=alpha,
    )
    if minor_cycle == "sara":
        if opt_backend == "primal-dual":
            from pfb_imaging.deconv.sara_pd import SARAPrimalDual

            solver = SARAPrimalDual(
                **sara_common,
                pd_tol=pd_tol,
                pd_maxit=pd_maxit,
                pd_verbose=pd_verbose,
                pd_report_freq=pd_report_freq,
            )
        elif opt_backend == "forward-backward":
            from pfb_imaging.deconv.sara_fb import SARAForwardBackward

            solver = SARAForwardBackward(
                **sara_common,
                fb_tol=fb_tol,
                fb_maxit=fb_maxit,
                fb_verbose=fb_verbose,
                fb_report_freq=fb_report_freq,
                acceleration=acceleration,
            )
        else:
            raise ValueError(f"Unknown opt_backend '{opt_backend}' for minor cycle 'sara'")
    else:
        raise NotImplementedError(f"Minor cycle '{minor_cycle}' not implemented")

    if not isinstance(solver, DeconvSolver):
        raise TypeError(f"Solver must be a DeconvSolver, got {type(solver)}")

    if rms_outside_model and model.any():
        rms_mask = model_mfs == 0
        rms = np.std(residual_mfs[rms_mask])
    else:
        rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count_curr = 0
    eps = 1.0
    write_futures = None
    log.info(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}")

    for k in range(niter):
        log.info("Solving for update")
        solver.first(residual)
        update = solver.forward(residual)
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs, fits_oname + f"_{suffix}_update_{iter0 + k + 1}.fits", hdr_mfs)

        modelp = deepcopy(model)
        lam = (init_factor if iter0 == 0 and k == 0 else 1.0) * rmsfactor * rms
        log.info(f"Solving for model with lambda = {lam:.3e}")
        model = solver.backward(lam)

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
            data_vars = {
                "coefficients": (("par", "comps"), coeffs),
            }
            coords = {
                "location_x": (("x",), x_index),
                "location_y": (("y",), y_index),
                "params": (("par",), params),
                "times": (("t",), time_out),
                "freqs": (("f",), freq_out),
            }
            mattrs = {
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
                "stokes": product,
                "parametrisation": expr,
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
            log.info(f"Exception {e} raised during model fit.")

        model_mfs = np.mean(model[fsel], axis=0)
        save_fits(model_mfs, fits_oname + f"_{suffix}_model_{iter0 + k + 1}.fits", hdr_mfs)

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
                model[b][None, :, :],
                nthreads=nthreads,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                verbosity=verbosity,
            )
            write_futures.append(fut)
            residual[b] = resid[0]

        residual /= wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs, fits_oname + f"_{suffix}_residual_{iter0 + k + 1}.fits", hdr_mfs)

        # post-iteration hook (e.g. l1 reweighting)
        solver.last()

        rmsp = rms
        if rms_outside_model:
            rms_mask = model_mfs == 0
            rms = np.std(residual_mfs[rms_mask])
        else:
            rms = np.std(residual_mfs)
        rmaxp = rmax
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        # cache hess_norm from solver if available
        hess_norm = getattr(solver, "hess_norm", hess_norm)

        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            ds["UPDATE"] = (("corr", "x", "y"), update[b][None, :, :])
            for var in ds.data_vars:
                if var != "UPDATE":
                    ds = ds.drop_vars(var)
            if (model == best_model).all():
                ds["MODEL_BEST"] = (("corr", "x", "y"), best_model[b][None, :, :])
            attrs = {
                "rms": best_rms,
                "rmax": best_rmax,
                "niters": iter0 + k + 1,
                "hess_norm": hess_norm,
            }
            ds = ds.assign_attrs(**attrs)
            with cf.ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(ds.to_zarr, ds_name, mode="a")
                write_futures.append(fut)

        log.info(f"Iter {iter0 + k + 1}: peak residual = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}")

        if eps < tol:
            reweight_active = getattr(solver, "reweight_active", True)
            if not reweight_active:
                # trigger reweighting instead of stopping
                solver.trigger_reweight()
            else:
                log.info(f"Converged after {iter0 + k + 1} iterations.")
                break

        if (rms > rmsp) and (rmax > rmaxp):
            diverge_count_curr += 1
            if diverge_count_curr > diverge_count:
                log.info("Algorithm is diverging. Terminating.")
                break

    cf.wait(write_futures)

    dds, dds_list = xds_from_url(dds_name)

    if fits_mfs or fits_cubes:
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

    log.info(f"Numba uses the {threading_layer()} threading layer")
    log.info(f"All done after {time.time() - time_start:.1f}s")
