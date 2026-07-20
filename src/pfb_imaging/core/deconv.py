import json
import os
import time
from copy import deepcopy

import numpy as np
import psutil
import xarray as xr
from ducc0.misc import resize_thread_pool

from pfb_imaging import init_ray, pfb_version, set_envs, setup_ray_worker
from pfb_imaging.deconv import DeconvSolver
from pfb_imaging.deconv.presets import PRESETS
from pfb_imaging.operators.band_worker import BandWorkerPool
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import dt2fits, save_fits, set_wcs
from pfb_imaging.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
from pfb_imaging.utils.naming import set_output_names

log = pfb_logging.get_logger("DECONV")


def deconv(
    output_filename: str,
    suffix: str = "main",
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
    fits_per_partition: bool = False,
    debug: bool = False,
    minor_cycle: str = "sara",
    opt_backend: str = "primal-dual",
    bases: list[str] = ["self", "db1", "db2", "db3"],
    nlevels: int = 3,
    l1_reweight_from: int = 5,
    alpha: float = 2.0,
    hess_norm: float | None = None,
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
    nthreads: int = 2,
    nworkers: int = 1,
    ray_address: str = "local",
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
    General preconditioned forward-backward deconvolution of the imager DataTree.

    The minor_cycle preset assembles (hess, forward_alg, backward_alg, prox) into
    a PFBSolver; any object satisfying the DeconvSolver Protocol can drive the loop.
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

    # Lazy load at the outset because we need nband to set default worker count
    # unlike the legacy .dds (which the `suffix` convention comes from),
    # imager() never appends a suffix to the .dt tree name (see
    # architecture.md §8: "<output>_<PRODUCT>.dt") -- `suffix` still
    # namespaces the FITS/.mds outputs below, just not the .dt path itself.
    basename = output_filename
    fits_oname = f"{fits_output_folder}/{oname}"
    dt_name = f"{basename}.dt"

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    image_names = sorted(n for n in dt.children if n.startswith("band"))
    if not image_names:
        log.error_and_raise(f"No band nodes found in {dt_name}", ValueError)

    timeids = {int(dt[n].ds.attrs["timeid"]) for n in image_names}
    if len(timeids) > 1:
        log.error_and_raise("Only static models currently supported", NotImplementedError)

    nodes = sorted(image_names, key=lambda n: int(dt[n].ds.attrs["bandid"]))
    first = dt[nodes[0]].ds
    if first.DIRTY.dims != ("corr", "y", "x"):
        log.error_and_raise(
            f"{dt_name} has image dims {first.DIRTY.dims}; this version reads the "
            "(corr, y, x) layout introduced in 0.1.0 -- re-run pfb imager to regenerate the .dt",
            ValueError,
        )
    if first.corr.size > 1:
        log.error_and_raise("Joint polarisation deconvolution not yet supported", NotImplementedError)

    # a psf=False imager tree is quicklook-only: the PSF products the minor
    # cycle needs were never gridded
    first_part = next(iter(dt[nodes[0]].children.values()), None)
    if first_part is None or "PSFHAT" not in first_part.ds:
        log.error_and_raise(
            f"{dt_name} has no per-partition PSFHAT (imager run with --no-psf?) -- "
            "re-run pfb imager with --psf to deconvolve",
            ValueError,
        )
    if "BDIRTY" not in first:
        log.error_and_raise(
            f"{dt_name} has no BDIRTY (beam-attenuated dirty) -- this version feeds "
            "the forward solver the exact beam-attenuated gradient (D23); re-run "
            "pfb imager to regenerate the .dt",
            ValueError,
        )

    nband = len(nodes)
    if nworkers is None:
        nworkers = nband
    ncpu = np.minimum(nthreads, psutil.cpu_count(logical=False))
    opts_dict["nthreads"] = nthreads
    opts_dict["nworkers"] = nworkers
    log.info(f"Using {nworkers} workers with {nthreads} threads per worker")
    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu, log=log)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/deconv_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.log_options_dict(opts_dict, title="DECONV options")

    # every deconv worker claims a nominal CPU (they are thread-pool-bound,
    # see BandWorkerPool), so num_cpus only sets the raylet's worker-process
    # startup throttle (max(1, num_cpus)); size it for the nband band
    # workers plus the driver so startup is not serialised and the raylet
    # does not warn about exceeding its expected startup concurrency.
    init_ray(
        nworkers,
        ray_address=ray_address,
        runtime_env={
            "env_vars": env_vars,
            "worker_process_setup_hook": setup_ray_worker,
        },
        log=log,
    )
    ny, nx = first.y.size, first.x.size
    ny_psf, nx_psf = first.y_psf.size, first.x_psf.size
    cell_rad = first.attrs["cell_rad"]
    cell_deg = np.rad2deg(cell_rad)
    radec = [first.attrs["ra"], first.attrs["dec"]]
    freq_out = np.array([dt[n].ds.attrs["freq_out"] for n in nodes])
    time_out = np.array([first.attrs["time_out"]])
    iter0 = int(first.attrs.get("niters", 0))
    geometry = {"nx": nx, "ny": ny, "nx_psf": nx_psf, "ny_psf": ny_psf}

    # load band cubes and attrs required on driver (MODEL/RESIDUAL/UPDATE/WSUM)
    # partition data (UVW/WEIGHT/MASK/FREQ/BEAM/PSFHAT/DIRTY) is read worker-side by BandWorkerPool.load_bands
    # They never enter the driver or the Ray object store
    # image rasters are (Y, X)-ordered end to end (wiki D19)
    residual_raw = np.zeros((nband, ny, nx))
    bresidual_raw = np.zeros((nband, ny, nx))  # beam-attenuated gradient (D23)
    model = np.zeros((nband, ny, nx))
    update = np.zeros((nband, ny, nx))
    wsums = np.zeros(nband)
    band_attrs = []  # original band attrs (bandid, freq_out, cell_rad, ...) to merge back on write

    for b, n in enumerate(nodes):
        bds = dt[n].ds
        band_attrs.append(dict(bds.attrs))
        residual_raw[b] = bds.RESIDUAL.values[0] if "RESIDUAL" in bds else bds.DIRTY.values[0]
        if "MODEL" in bds:
            model[b] = bds.MODEL.values[0]
        if "UPDATE" in bds:
            update[b] = bds.UPDATE.values[0]
        if "BRESIDUAL" in bds:
            bresidual_raw[b] = bds.BRESIDUAL.values[0]
        elif not model[b].any():
            # fresh tree: with a zero model the gradient is its model-free term
            bresidual_raw[b] = bds.BDIRTY.values[0]
        else:
            log.error_and_raise(
                f"{n} has a MODEL but no BRESIDUAL; resuming needs a .dt written by "
                "this version -- restart the deconvolution from the imager output",
                ValueError,
            )
        wsums[b] = bds.WSUM.values[0]

    wsum = wsums.sum()
    residual = residual_raw / wsum
    bresidual = bresidual_raw / wsum
    residual_mfs = np.sum(residual, axis=0)
    fsel = wsums > 0
    model_mfs = np.mean(model[fsel], axis=0)
    if nbasisf is None:
        nbasisf = int(np.sum(fsel))

    l0 = float(first.attrs.get("l0", 0.0))
    m0 = float(first.attrs.get("m0", 0.0))
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, np.mean(freq_out), casambm=False, l0=l0, m0=m0)

    # hess_norm from the tree cache when available; solver estimates it otherwise
    if hess_norm is None and "hess_norm" in first.attrs:
        hess_norm = first.attrs["hess_norm"]
        opts_dict["hess_norm"] = hess_norm
        log.info(f"Using previously estimated hess_norm of {hess_norm:.3e}")

    if minor_cycle not in PRESETS:
        log.error_and_raise(f"Unknown minor_cycle '{minor_cycle}'", ValueError)

    # one worker process per band co-locating all per-band state:
    # the Hessian and Psi (initialised by the preset's facades) and the exact residual.
    # Each worker reads its own band's vis-scale inputs straight from the store and pins them for the life of the run.
    # TODO - this only works for a local cluster, nworkers not configurable on remote cluster.
    # Need to round robin bands across workers.
    workers = BandWorkerPool(nband, nthreads)
    workers.load_bands(dt_name, nodes)

    solver = PRESETS[minor_cycle](None, geometry, model, update, opts_dict, workers=workers, wsums=wsums)
    if not isinstance(solver, DeconvSolver):
        raise TypeError(f"Solver must be a DeconvSolver, got {type(solver)}")

    if rms_outside_model and model.any():
        rms = np.std(residual_mfs[model_mfs == 0])
    else:
        rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    if "best_rms" in first.attrs:
        best_rms = first.attrs["best_rms"]
        best_rmax = first.attrs["best_rmax"]
        best_model = np.zeros((nband, ny, nx))
        for b, n in enumerate(nodes):
            bds = dt[n].ds
            if "MODEL_BEST" in bds:
                best_model[b] = bds.MODEL_BEST.values[0]
    else:
        best_rms, best_rmax = rms, rmax
        best_model = model.copy()
    diverge_count_curr = 0
    log.info(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}")

    # --debug: per-iteration per-partition chi2 trajectories, collected into a
    # machine-readable record next to the FITS outputs (D23 debugging aid)
    debug_record = {"iterations": [], "uv_profiles": None} if debug else None

    def _chi2_snapshot(iter_id):
        stats = workers.partition_chi2(model[:, None], cell_rad, epsilon=epsilon, do_wgridding=do_wgridding)
        entry = {"iter": int(iter_id), "bands": []}
        for n, plist in zip(nodes, stats):
            entry["bands"].append({"band": n, "partitions": plist})
            for pid, p in enumerate(plist):
                rchi2 = p["chi2"][0] / max(p["ndata"], 1.0)
                log.info(f"debug chi2 iter {iter_id} {n}/part{pid:04d} field={p['field']} rchi2={rchi2:.6e}")
        debug_record["iterations"].append(entry)

    if debug:
        _chi2_snapshot(iter0)  # baseline against the starting model

    mrange = range(iter0, iter0 + niter)
    for k in mrange:
        log.info("Solving for update")
        # the forward solve consumes the beam-attenuated gradient (D23):
        # H = B GtWG B, so its rhs is B*r, exactly legacy sara's
        # `residual *= beam`. Stats/FITS/lambda stay on the apparent residual.
        solver.first(bresidual)
        update = solver.forward(bresidual)
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs, fits_oname + f"_{suffix}_update_{k + 1}.fits", hdr_mfs, yx_order=True)

        modelp = deepcopy(model)
        lam = (init_factor if iter0 == 0 and k == 0 else 1.0) * rmsfactor * rms
        log.info(f"Solving for model with lambda = {lam:.3e}")
        model = solver.backward(lam)

        # write component model (carried over from the legacy driver; .dt-native attrs)
        log.info(f"Writing model to {basename}_{suffix}.mds")
        # TODO - this should be a function call to pfb-model-spec
        try:
            # the .mds stays x-major (degrid/model2comps convention; see the
            # pfb-model-spec migration in #277) -- adapt with zero-copy views
            coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
                time_out,
                freq_out[fsel],
                model[None, fsel, :, :].transpose(0, 1, 3, 2),
                wgt=(wsums / wsum)[None, fsel],
                nbasisf=nbasisf,
                method="Legendre",
                sigmasq=1e-6,
            )
            flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
            coeff_dataset = xr.Dataset(
                data_vars={"coefficients": (("par", "comps"), coeffs)},
                coords={
                    "location_x": (("x",), x_index),
                    "location_y": (("y",), y_index),
                    "params": (("par",), params),
                    "times": (("t",), time_out),
                    "freqs": (("f",), freq_out),
                },
                attrs={
                    "pfb-imaging-version": pfb_version,
                    "spec": "genesis",
                    "cell_rad_x": cell_rad,
                    "cell_rad_y": cell_rad,
                    "npix_x": nx,
                    "npix_y": ny,
                    "texpr": texpr,
                    "fexpr": fexpr,
                    "center_x": x0,
                    "center_y": y0,
                    "flip_u": flip_u,
                    "flip_v": flip_v,
                    "flip_w": flip_w,
                    "ra": radec[0],
                    "dec": radec[1],
                    "stokes": product,
                    "parametrisation": expr,
                },
            )
            coeff_dataset.to_zarr(f"{basename}_{suffix}.mds", mode="w")

            # need to re-evaluate the model after the fit to keep it consistent
            # this can be used to enforce smoothness in the model at the expense of increased residuals
            # does not respect the positivity constraint
            for b in range(nband):
                # eval returns an x-major (nx, ny) raster (mds convention);
                # transpose back into the (Y, X) model cube
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
                    x0,
                    y0,
                    nx,
                    ny,
                    cell_rad,
                    cell_rad,
                    x0,
                    y0,
                ).T
        except Exception as e:
            log.info(f"Exception {e} raised during model fit.")

        model_mfs = np.mean(model[fsel], axis=0)
        save_fits(model_mfs, fits_oname + f"_{suffix}_model_{k + 1}.fits", hdr_mfs, yx_order=True)

        log.info("Computing residual")
        residual_raw, bresidual_raw = workers.residual(
            model[:, None],
            cell_rad,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
        )
        residual_raw = residual_raw[:, 0]
        bresidual_raw = bresidual_raw[:, 0]
        residual = residual_raw / wsum
        bresidual = bresidual_raw / wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs, fits_oname + f"_{suffix}_residual_{k + 1}.fits", hdr_mfs, yx_order=True)

        if debug:
            _chi2_snapshot(k + 1)

        # post-iteration hook (e.g. arming l1 reweighting)
        solver.last()

        # per-actor post-gc memory telemetry (docs/wiki/memory-and-ray.md)
        if verbosity > 1 and hasattr(getattr(solver, "hess", None), "get_mem"):
            for m in solver.hess.get_mem():
                log.info(f"hess actor pid {m['pid']} rss {m['rss_gb']:.2f} GB peak {m['peak_gb']:.2f} GB")

        rmsp, rmaxp = rms, rmax
        if rms_outside_model:
            rms = np.std(residual_mfs[model_mfs == 0])
        else:
            rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        if rms < best_rms:
            best_rms, best_rmax = rms, rmax
            best_model = model.copy()

        hess_norm = getattr(solver, "hess_norm", hess_norm)

        # write back into the band nodes (native DataTree API)
        is_best = (model == best_model).all()
        for b, n in enumerate(nodes):
            data_vars = {
                "MODEL": (("corr", "y", "x"), model[b][None]),
                "UPDATE": (("corr", "y", "x"), update[b][None]),
                "RESIDUAL": (("corr", "y", "x"), residual_raw[b][None]),
                "BRESIDUAL": (("corr", "y", "x"), bresidual_raw[b][None]),
            }
            if is_best:
                data_vars["MODEL_BEST"] = (("corr", "y", "x"), best_model[b][None])
            # to_zarr(mode="a") replaces a group's attrs wholesale (unlike
            # variables, which merge), so start from the band's original
            # attrs (bandid, freq_out, cell_rad, ra, dec, ...) -- mirrors the
            # legacy sara() convention of ds.assign_attrs(**attrs) onto a
            # dataset read from the existing dds, which merges rather than
            # replaces.
            ds_out = xr.Dataset(
                data_vars,
                attrs={
                    **band_attrs[b],
                    "rms": best_rms,
                    "rmax": best_rmax,
                    "niters": k + 1,
                    "hess_norm": hess_norm,
                },
            )
            ds_out.to_zarr(dt_name, group=n, mode="a")

        log.info(f"Iter {k + 1}: peak residual = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}")

        if eps < tol:
            if not getattr(solver, "reweight_active", True):
                solver.trigger_reweight()  # reweight instead of stopping
            else:
                log.info(f"Converged after {k + 1} iterations.")
                break

        if (rms > rmsp) and (rmax > rmaxp):
            diverge_count_curr += 1
            if diverge_count_curr > diverge_count:
                log.info("Algorithm is diverging. Terminating.")
                break

    if fits_per_partition:
        # per-partition misfit localisation (D23 debugging aid): re-gridded
        # dirty, residual against the final model, apparent model B_p*m and
        # vis-space chi2 for every data partition, computed worker-side
        pdir = f"{fits_oname}_{suffix}_partitions"
        os.makedirs(pdir, exist_ok=True)
        log.info(f"Writing per-partition debug FITS to {pdir}")
        parts_per_band = workers.partition_debug(
            model[:, None],
            cell_rad,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
        )
        for b, (n, plist) in enumerate(zip(nodes, parts_per_band)):
            for pid, part in enumerate(plist):
                field = str(part["field"]).replace(" ", "_").replace("/", "-")
                rchi2 = part["chi2"][0] / max(part["ndata"], 1.0)
                hdr = set_wcs(
                    cell_deg,
                    cell_deg,
                    nx,
                    ny,
                    radec,
                    freq_out[b],
                    ms_time=time_out[0],
                    time_is_unix=True,
                    l0=float(band_attrs[b].get("l0", 0.0)),
                    m0=float(band_attrs[b].get("m0", 0.0)),
                )
                hdr["FIELDNAM"] = part["field"]
                hdr["WSUMP"] = (float(part["wsum"][0]), "partition weight sum")
                hdr["CHI2"] = (float(part["chi2"][0]), "sum w|V - G(B m)|^2 over unflagged")
                hdr["NDATA"] = (float(part["ndata"]), "unflagged vis samples")
                hdr["RCHI2"] = (float(rchi2), "CHI2 / NDATA")
                wsum_p = part["wsum"][:, None, None]
                with np.errstate(invalid="ignore", divide="ignore"):
                    dirty_n = np.where(wsum_p > 0, part["dirty"] / wsum_p, 0.0)
                    resid_n = np.where(wsum_p > 0, part["residual"] / wsum_p, 0.0)
                stem = f"{pdir}/{{var}}_{n}_part{pid:04d}_{field}.fits"
                save_fits(dirty_n, stem.format(var="dirty"), hdr, yx_order=True)
                save_fits(resid_n, stem.format(var="residual"), hdr, yx_order=True)
                save_fits(part["amodel"], stem.format(var="model_apparent"), hdr, yx_order=True)
                log.info(
                    f"{n}/part{pid:04d} field={part['field']}: "
                    f"wsum_frac={float(part['wsum'][0]) / wsum:.3f} "
                    f"resid_rms={float(np.std(resid_n[0])):.3e} Jy/beam "
                    f"resid_peak={float(np.abs(resid_n[0]).max()):.3e} rchi2={rchi2:.6e}"
                )

    if debug:
        # baseline-length-binned residual power per partition: flat for
        # noise-like residuals, structured when the misfit lives at specific
        # baseline lengths (calibration/beam/astrometry signatures differ)
        log.info("Computing baseline-binned residual profiles")
        profs = workers.partition_uvprofile(model[:, None], cell_rad, epsilon=epsilon, do_wgridding=do_wgridding)
        debug_record["uv_profiles"] = dict(zip(nodes, profs))
        debug_name = f"{fits_oname}_{suffix}_debug.json"
        with open(debug_name, "w") as f:
            json.dump(debug_record, f, indent=2)
        log.info(f"Debug record written to {debug_name}")

    if fits_mfs or fits_cubes:
        log.info(f"Writing fits files to {fits_oname}_{suffix}")
        for column, norm in (("RESIDUAL", True), ("MODEL", False), ("UPDATE", False)):
            dt2fits(
                dt_name,
                column,
                f"{fits_oname}_{suffix}",
                norm_wsum=norm,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
            )
            log.info(f"Done writing {column}")

    log.info(f"All done after {time.time() - time_start:.1f}s")
