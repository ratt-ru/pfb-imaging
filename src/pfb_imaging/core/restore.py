import time

import numpy as np
import psutil
import ray
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool

from pfb_imaging import set_envs
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import rdds2fits
from pfb_imaging.utils.naming import get_opts, set_output_names, xds_from_list, xds_from_url
from pfb_imaging.utils.restoration import rrestore_image

log = pfb_logging.get_logger("RESTORE")

@pfb_logging.log_inputs(log)
def restore(
    output_filename: str,
    model_name: str = "MODEL",
    residual_name: str = "RESIDUAL",
    suffix: str = "main",
    outputs: str = "mMrRiI",
    overwrite: bool = True,
    gausspar: list[float] | None = None,
    inflate_factor: float = 1.5,
    drop_bands: list[int] | None = None,
    host_address: str | None = None,
    nworkers: int = 1,
    nthreads: int | None = None,
    direct_to_workers: bool = True,
    log_level: str = "error",
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
):
    """
    Create fits image cubes from data products (eg. restored images).
    """
    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )

    nthreads_total = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = nthreads_total // 2
        ncpu = ncpu // 2

    resize_thread_pool(nthreads)
    set_envs(nthreads, ncpu)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{log_directory}/restore_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    # these are passed through to child Ray processes
    renv = {"env_vars": {}}
    if nworkers == 1:
        renv["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"

    ray.init(num_cpus=nworkers, logging_level="INFO", ignore_reinit_error=True, runtime_env=renv)

    ti = time.time()

    # Inlined _restore() function logic
    basename = output_filename
    if fits_output_folder is not None:
        fits_oname = fits_output_folder + "/" + basename.split("/")[-1]
    else:
        fits_oname = basename

    dds_name = f"{basename}_{suffix}.dds"
    dds, dds_list = xds_from_url(dds_name)
    dds_store = DaskMSStore(dds_name)
    if "://" in dds_store.url:
        protocol = dds_store.url.split("://")[0]
    else:
        protocol = "file"

    # get MFS PSF PARS
    try:
        psfpars_mfs = get_opts(dds_store.url, protocol, name="psfparsn_mfs.pkl")
    except Exception:
        log.error_and_raise("Could not load MFS PSF pamaters. Run grid worker with psf=true to remake.", RuntimeError)

    if drop_bands is not None:
        ddso = []
        ddso_list = []
        for ds, dsl in zip(dds, dds_list):
            b = int(ds.bandid)
            if b not in drop_bands:
                ddso.append(ds)
                ddso_list.append(dsl)
        dds = ddso
        dds_list = ddso_list

    timeids = np.unique(np.array([int(ds.timeid) for ds in dds]))
    freqs = [ds.freq_out for ds in dds]
    freqs = np.unique(freqs)
    nband = freqs.size
    ntime = timeids.size
    ncorr = dds[0].corr.size
    cell_asec = dds[0].cell_rad * 180 / np.pi * 3600
    log.info(f"Number of output times = {ntime}")
    log.info(f"Number of output bands = {nband}")

    if gausspar is None:
        gaussparf = None
        gaussparf_mfs = None
        log.info("Using native resolution")
    elif gausspar == [0, 0, 0]:
        # This is just to figure out what resolution to convolve to
        dds = xds_from_list(dds_list, nthreads=nthreads, drop_all_but=["PSFPARSN"])
        emaj = 0.0
        emin = 0.0
        pas = []
        for ds in dds:
            gausspars = ds.PSFPARSN.values
            emaj = np.maximum(emaj, gausspars[:, 0].max())
            emin = np.maximum(emin, gausspars[:, 1].max())
            pas.append(np.mean(gausspars[:, 2]))
        pa = np.mean(np.array(pas))
        log.info(
            f"Using lowest resolution of ({emaj * cell_asec:.3e} asec, "
            f"{emin * cell_asec:.3e} asec, {pa:.3e} degrees) for restored images"
        )
        # these are n pixel units
        gaussparf_mfs = [emaj, emin, pa]
        gaussparf = gaussparf_mfs * ncorr
    else:
        gaussparf_mfs = list(gausspar)
        emaj = gaussparf_mfs[0]
        emin = gaussparf_mfs[1]
        pa = gaussparf_mfs[2]
        if "i" in outputs.lower():
            log.info(
                f"Using specified resolution of ({emaj:.3e} asec, {emin:.3e} asec, {pa:.3e} degrees) for restored image"
            )
        # convert to pixel units
        for i in range(2):
            gaussparf_mfs[i] /= cell_asec
        gaussparf = [gaussparf_mfs] * ncorr

    # create restored images
    tasksi = []
    for ds_name in dds_list:
        task = rrestore_image.remote(ds_name, model_name, residual_name, gaussparf=gaussparf, nthreads=nthreads)
        tasksi.append(task)

    tasks = []
    if "d" in outputs.lower():
        task = rdds2fits.remote(
            dds_list,
            "DIRTY",
            f"{fits_oname}_{suffix}",
            norm_wsum=True,
            nthreads=nthreads,
            do_mfs="d" in outputs,
            do_cube="D" in outputs,
            psfpars_mfs=psfpars_mfs,
        )
        tasks.append(task)

    if "m" in outputs.lower():
        task = rdds2fits.remote(
            dds_list,
            model_name,
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs="m" in outputs,
            do_cube="M" in outputs,
            psfpars_mfs=psfpars_mfs,
        )
        tasks.append(task)

    if "r" in outputs.lower():
        task = rdds2fits.remote(
            dds_list,
            residual_name,
            f"{fits_oname}_{suffix}",
            norm_wsum=True,
            nthreads=nthreads,
            do_mfs="r" in outputs,
            do_cube="R" in outputs,
            psfpars_mfs=psfpars_mfs,
        )
        tasks.append(task)

    # we need to wait for tasksiI before rendering restored to fits
    ray.get(tasksi)

    if "i" in outputs.lower():
        task = rdds2fits.remote(
            dds_list,
            "IMAGE",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs="i" in outputs,
            do_cube="I" in outputs,
            psfpars_mfs=psfpars_mfs,
        )
        tasks.append(task)

    # TODO(LB) - we may want to add these outputs back in, at least the useful ones
    # if 'f' in outputs:
    #     rhat_mfs = c2c(residual_mfs, forward=True,
    #                    nthreads=nthreads, inorm=0)
    #     rhat_mfs = np.fft.fftshift(rhat_mfs)
    #     save_fits(np.abs(rhat_mfs),
    #               f'{fits_oname}_{suffix}.abs_fft_residual_mfs.fits',
    #               hdr_mfs,
    #               overwrite=overwrite)
    #     save_fits(np.angle(rhat_mfs),
    #               f'{fits_oname}_{suffix}.phase_fft_residual_mfs.fits',
    #               hdr_mfs,
    #               overwrite=overwrite)

    # if 'F' in outputs:
    #     rhat = c2c(residual, axes=(1,2), forward=True,
    #                nthreads=nthreads, inorm=0)
    #     rhat = np.fft.fftshift(rhat, axes=(1,2))
    #     save_fits(np.abs(rhat),
    #               f'{fits_oname}_{suffix}.abs_fft_residual.fits',
    #               hdr,
    #               overwrite=overwrite)
    #     save_fits(np.angle(rhat),
    #               f'{fits_oname}_{suffix}.phase_fft_residual.fits',
    #               hdr,
    #               overwrite=overwrite)

    # if 'c' in outputs:
    #     if GaussPar is None:
    #         raise ValueError("Clean beam in output but no PSF in dds")
    #     cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)
    #     save_fits(cpsf_mfs,
    #               f'{fits_oname}_{suffix}.cpsf_mfs.fits',
    #               hdr_mfs,
    #               overwrite=overwrite)

    # if 'C' in outputs:
    #     if GaussPars is None:
    #         raise ValueError("Clean beam in output but no PSF in dds")
    #     cpsf = np.zeros(residual.shape, dtype=output_type)
    #     for v in range(nband):
    #         gpar = GaussPars[v]
    #         if not np.isnan(gpar).any():
    #             cpsf[v] = Gaussian2D(xx, yy, gpar, normalise=False)
    #     save_fits(cpsf,
    #               f'{fits_oname}_{suffix}.cpsf.fits',
    #               hdr,
    #               overwrite=overwrite)

    # wait for all tasks to finish before returning
    ray.get(tasks)

    log.info(f"All done after {time.time() - ti}s")

    ray.shutdown()
