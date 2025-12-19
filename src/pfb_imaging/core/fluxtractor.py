import time

import numpy as np
import psutil
import ray
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool

from pfb_imaging import set_envs
from pfb_imaging.opt.pcg import pcg_dds
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import load_fits
from pfb_imaging.utils.modelspec import fit_image_cube
from pfb_imaging.utils.naming import set_output_names, xds_from_url

log = pfb_logging.get_logger("FLUXTRACTOR")


def fluxtractor(
    output_filename: str,
    suffix: str = "main",
    mask: str | None = None,
    zero_model_outside_mask: bool = False,
    or_mask_with_model: bool = False,
    min_model: float = 1e-5,
    eta: float = 1e-5,
    model_name: str = "MODEL",
    residual_name: str = "RESIDUAL",
    gamma: float = 1.0,
    use_psf: bool = True,
    memory_greedy: bool = False,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    cg_tol: float = 1e-3,
    cg_maxit: int = 150,
    cg_minit: int = 10,
    cg_verbose: int = 1,
    cg_report_freq: int = 10,
    backtrack: bool = False,
    host_address: str | None = None,
    nworkers: int = 1,
    nthreads: int | None = None,
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
):
    """
    Forward step aka flux mop.
    """

    output_filename, fits_output_folder, log_directory, basedir, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )

    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True)
        ncpu = psutil.cpu_count(logical=False)
        if nworkers > 1:
            ntpw = nthreads // nworkers
            nthreads = ntpw // 2
            ncpu = ntpw // 2
        else:
            nthreads = nthreads // 2
            ncpu = ncpu // 2

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/fluxtractor_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    # pfb_logging.log_options_dict(log, opts)

    fits_oname = f"{fits_output_folder}/{oname}"
    dds_name = f"{output_filename}_{suffix}.dds"

    # these are passed through to child Ray processes
    if nworkers == 1:
        env_vars["RAY_DEBUG_POST_MORTEM"] = "1"

    ray.init(num_cpus=nworkers, logging_level="INFO", ignore_reinit_error=True, runtime_env={"env_vars": env_vars})

    time_start = time.time()

    dds_name = f"{output_filename}_{suffix}.dds"
    dds_store = DaskMSStore(dds_name)
    dds, dds_list = xds_from_url(dds_store.url)

    nx, ny = dds[0].x.size, dds[0].y.size
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))

    nband = freq_out.size
    nx = dds[0].x.size
    ny = dds[0].y.size
    cell_rad = dds[0].cell_rad

    if residual_name in dds[0]:
        residual = np.stack([getattr(ds, residual_name).values for ds in dds], axis=0)
    else:
        log.info("Using dirty image as residual")
        residual = np.stack([ds.DIRTY.values for ds in dds], axis=0)
    if model_name in dds[0]:
        model = np.stack([getattr(ds, model_name).values for ds in dds], axis=0)
    else:
        model = np.zeros((nband, nx, ny))
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands
    wsum = np.sum(wsums)
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)
    model_mfs = np.mean(model, axis=0)

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if mask is not None:
        if mask == "model":
            mask = np.any(model > min_model, axis=0)
            assert mask.shape == (nx, ny)
            mask = mask.astype(residual.dtype)
            log.info("Using model > 0 to create mask")
        else:
            mask = load_fits(mask, dtype=residual.dtype).squeeze()
            assert mask.shape == (nx, ny)
            if or_mask_with_model:
                log.info("Combining model with input mask")
                mask = np.logical_or(mask > 0, model_mfs > 0).astype(residual.dtype)

            mask = mask.astype(residual.dtype)
            log.info("Using provided fits mask")
    else:
        mask = np.ones((nx, ny), dtype=residual.dtype)
        log.info("Caution - No mask is being applied")

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    log.info(f"Initial peak residual = {rmax:.3e}, rms = {rms:.3e}")

    log.info("Solving for update")
    tasks = []
    for ds_name in dds_list:
        task = pcg_dds.remote(
            ds_name,
            eta,
            use_psf=use_psf,
            residual_name=residual_name,
            model_name=model_name,
            mask=mask,
            do_wgridding=do_wgridding,
            epsilon=epsilon,
            double_accum=double_accum,
            nthreads=nthreads,
            zero_model_outside_mask=zero_model_outside_mask,
            tol=cg_tol,
            maxit=cg_maxit,
            verbosity=cg_verbose,
            report_freq=cg_report_freq,
        )
        tasks.append(task)

    nds = len(tasks)
    n_launched = 1
    remaining_tasks = tasks.copy()
    while remaining_tasks:
        # Wait for at least 1 task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        r, b = ray.get(ready)
        residual[b] = r
        n_launched += 1

        print(f"\rProcessed: {n_launched}/{nds}", end="\n", flush=True)

    residual_mfs = np.sum(residual / wsum, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    log.info(f"Final peak residual = {rmax:.3e}, rms = {rms:.3e}")

    log.info(f"Writing model to {output_filename}_{suffix}_model.mds")

    try:
        coeffs, ix, iy, expr, params, texpr, fexpr = fit_image_cube(
            time_out,
            freq_out[fsel],
            model[None, fsel, :, :],
            wgt=wsums[None, fsel],
            nbasisf=fsel.size,
            method="Legendre",
            sigmasq=1e-10,
        )
        # save interpolated dataset
        data_vars = {
            "coefficients": (("par", "comps"), coeffs),
        }
        coords = {
            "location_x": (("x",), ix),
            "location_y": (("y",), iy),
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
            "stokes": product,
            "parametrisation": expr,  # already converted to str
        }
        # for key, val in opts.items():
        #     mattrs[key] = val

        coeff_dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=mattrs)
        coeff_dataset.to_zarr(f"{output_filename}_{suffix}_model_mopped.mds", mode="w")
    except Exception as e:
        log.info(f"Exception {e} raised during model fit .")

    _, dds_list = xds_from_url(dds_name)

    # convert to fits files
    if fits_mfs or fits_cubes:
        from pfb_imaging.utils.fits import rdds2fits

        log.info(f"Writing fits files to {fits_oname}_{suffix}")
        tasks = []
        task = rdds2fits.remote(
            dds_list,
            "RESIDUAL_MOPPED",
            f"{fits_oname}_{suffix}",
            norm_wsum=True,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
        )
        tasks.append(task)

        task = rdds2fits.remote(
            dds_list,
            "MODEL_MOPPED",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
        )
        tasks.append(task)

        task = rdds2fits(
            dds_list,
            "UPDATE",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
        )
        tasks.append(task)

        task = rdds2fits(
            dds_list,
            "X0",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
        )
        tasks.append(task)

        remaining_tasks = tasks.copy()
        while remaining_tasks:
            # Wait for at least 1 task to complete
            ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

            # Process the completed task
            for task in ready:
                column = ray.get(task)
                log.info(f"Done writing {column}")

        log.info(f"All done after {time.time() - time_start}s")
        ray.shutdown()
