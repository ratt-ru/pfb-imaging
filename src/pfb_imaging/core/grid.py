import time

import fsspec
import numpy as np
import psutil
import ray
import xarray as xr
from africanus.coordinates import radec_to_lm
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool

from pfb_imaging import set_envs
from pfb_imaging.operators.gridder import rimage_data_products, wgridder_conventions
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.astrometry import get_coordinates
from pfb_imaging.utils.fits import rdds2fits
from pfb_imaging.utils.misc import fitcleanbeam, set_image_size
from pfb_imaging.utils.naming import cache_opts, get_opts, set_output_names, xds_from_url

log = pfb_logging.get_logger("GRID")


def grid(
    output_filename: str,
    xds: str | None = None,
    suffix: str = "main",
    concat_row: bool = True,
    overwrite: bool = False,
    transfer_model_from: str | None = None,
    use_best_model: bool = False,
    robustness: float | None = None,
    dirty: bool = True,
    psf: bool = True,
    residual: bool = True,
    noise: bool = True,
    beam: bool = True,
    weight: bool = True,
    psf_oversize: float = 1.4,
    field_of_view: float | None = None,
    super_resolution_factor: float = 2,
    cell_size: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    filter_counts_level: float = 5.0,
    target: str | None = None,
    l2_reweight_dof: int = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    host_address: str | None = None,
    nworkers: int = 1,
    nthreads: int | None = None,
    log_level: str = "error",
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
):
    """
    Compute imaging weights and create a dirty image, psf etc.

    TODO - enable single precision gridding
    """

    output_filename, fits_output_folder, log_directory, oname = set_output_names(
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

    output_filename = output_filename
    fits_oname = f"{fits_output_folder}/{oname}"
    dds_store = DaskMSStore(f"{output_filename}_{suffix}.dds")

    if xds is not None:
        xds_store = DaskMSStore(xds.rstrip("/"))
        xds_name = xds
    else:
        xds_store = DaskMSStore(f"{output_filename}.xds")
        xds_name = f"{output_filename}.xds"
    try:
        assert xds_store.exists()
    except Exception:
        log.error_and_raise(f"There must be an xds at {xds_name}. ", RuntimeError)
    xds = xds_store.url

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/grid_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    opts = {
        "output_filename": output_filename,
        "xds": xds,
        "suffix": suffix,
        "concat_row": concat_row,
        "overwrite": overwrite,
        "transfer_model_from": transfer_model_from,
        "use_best_model": use_best_model,
        "robustness": robustness,
        "dirty": dirty,
        "psf": psf,
        "residual": residual,
        "noise": noise,
        "beam": beam,
        "weight": weight,
        "psf_oversize": psf_oversize,
        "field_of_view": field_of_view,
        "super_resolution_factor": super_resolution_factor,
        "cell_size": cell_size,
        "nx": nx,
        "ny": ny,
        "filter_counts_level": filter_counts_level,
        "target": target,
        "l2_reweight_dof": l2_reweight_dof,
        "epsilon": epsilon,
        "do_wgridding": do_wgridding,
        "double_accum": double_accum,
        "host_address": host_address,
        "nworkers": nworkers,
        "nthreads": nthreads,
        "log_level": log_level,
        "log_directory": log_directory,
        "product": product,
        "fits_output_folder": fits_output_folder,
        "fits_mfs": fits_mfs,
        "fits_cubes": fits_cubes,
    }

    pfb_logging.log_options_dict(log, opts)

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu)

    # these are passed through to child Ray processes
    if nworkers == 1:
        env_vars["RAY_DEBUG_POST_MORTEM"] = "1"

    ray.init(num_cpus=nworkers, logging_level="INFO", ignore_reinit_error=True, runtime_env=env_vars)

    time_start = time.time()

    # xds contains vis products, no imaging weights applied
    xds_name = f"{output_filename}.xds" if xds is None else xds
    xds_store = DaskMSStore(xds_name)
    try:
        assert xds_store.exists()
    except Exception:
        log.error_and_raise(f"There must be a dataset at {xds_store.url}", RuntimeError)

    log.info(f"Lazy loading xds from {xds_store.url}")
    xds, xds_list = xds_from_url(xds_store.url)
    valid_bands = np.unique([ds.bandid for ds in xds])

    times_in = []
    freqs_in = []
    for ds in xds:
        times_in.append(ds.time_out)
        freqs_in.append(ds.freq_out)

    times_in = np.unique(times_in)
    freqs_out = np.unique(freqs_in)

    ntime_in = times_in.size
    nband = freqs_out.size

    # max uv coords over all datasets
    uv_max = 0
    max_freq = 0
    for ds in xds:
        uv_max = np.maximum(uv_max, ds.uv_max)
        max_freq = np.maximum(max_freq, ds.max_freq)

    nx, ny, nx_psf, ny_psf, cell_n, cell_rad = set_image_size(
        uv_max, max_freq, field_of_view, super_resolution_factor, cell_size, nx, ny, psf_oversize
    )
    cell_deg = np.rad2deg(cell_rad)
    cell_size = cell_deg * 3600
    log.info(f"Super resolution factor = {cell_n / cell_rad}")
    log.info(f"Cell size set to {cell_size:.5e} arcseconds")
    log.info(f"Field of view is ({nx * cell_deg:.3e},{ny * cell_deg:.3e}) degrees")

    # create dds and cache
    dds_name = output_filename + f"_{suffix}" + ".dds"
    dds_store = DaskMSStore(dds_name)
    if "://" in dds_store.url:
        protocol = xds_store.url.split("://")[0]
    else:
        protocol = "file"

    if dds_store.exists() and overwrite:
        log.info(f"Removing {dds_store.url}")
        dds_store.rm(recursive=True)

    fs = fsspec.filesystem(protocol)
    if dds_store.exists() and not overwrite:
        # get opts from previous run
        optsp = get_opts(dds_store.url, protocol, name="opts.pkl")

        # check if we need to remake the data products
        verify_attrs = ["epsilon", "do_wgridding", "double_accum", "field_of_view", "super_resolution_factor"]
        try:
            for attr in verify_attrs:
                assert optsp[attr] == opts[attr]

            from_cache = True
            log.info("Initialising from cached data products")
        except Exception:
            log.info(f"Cache verification failed on {attr}. Will remake image data products")
            dds_store.rm(recursive=True)
            fs.makedirs(dds_store.url, exist_ok=True)
            # dump opts to validate cache on rerun
            cache_opts(opts, dds_store.url, protocol, name="opts.pkl")
            from_cache = False
    else:
        fs.makedirs(dds_store.url, exist_ok=True)
        cache_opts(opts, dds_store.url, protocol, name="opts.pkl")
        from_cache = False
        log.info("Initialising from scratch.")

    log.info(f"Data products will be cached in {dds_store.url}")

    # filter datasets by time and band
    xds_dct = {}
    if concat_row:
        ntime = 1
        times_out = np.mean(times_in, keepdims=True)
        for b, bid in enumerate(valid_bands):
            for ds, ds_name in zip(xds, xds_list):
                if ds.bandid == bid:
                    tbid = f"time0000_band{b:04d}"
                    xds_dct.setdefault(tbid, {})
                    xds_dct[tbid].setdefault("dsl", [])
                    xds_dct[tbid]["dsl"].append(ds_name)
                    xds_dct[tbid]["radec"] = (ds.ra, ds.dec)
                    xds_dct[tbid]["time_out"] = times_out[0]
                    xds_dct[tbid]["freq_out"] = freqs_out[b]
                    xds_dct[tbid]["chan_low"] = ds.chan_low
                    xds_dct[tbid]["chan_high"] = ds.chan_high
    else:
        ntime = ntime_in
        times_out = times_in
        for t in range(times_in.size):
            for b, bid in enumerate(valid_bands):
                for ds, ds_name in zip(xds, xds_list):
                    if ds.time_out == times_in[t] and ds.bandid == bid:
                        tbid = f"time{t:04d}_band{b:04d}"
                        xds_dct.setdefault(tbid, {})
                        xds_dct[tbid].setdefault("dsl", [])
                        xds_dct[tbid]["dsl"].append(ds_name)
                        xds_dct[tbid]["radec"] = (ds.ra, ds.dec)
                        xds_dct[tbid]["time_out"] = times_out[t]
                        xds_dct[tbid]["freq_out"] = freqs_out[b]
                        xds_dct[tbid]["chan_low"] = ds.chan_low
                        xds_dct[tbid]["chan_high"] = ds.chan_high

    ncorr = ds.corr.size
    corrs = ds.corr.values
    if dirty:
        log.info(f"Image size = (ntime={ntime}, nband={nband}, ncorr={ncorr}, nx={nx}, ny={ny})")

    if psf:
        log.info(f"PSF size = (ntime={ntime}, nband={nband}, ncorr={ncorr}, nx={nx_psf}, ny={ny_psf})")

    # check if model exists
    if transfer_model_from:
        try:
            mds = xr.open_zarr(transfer_model_from, chunks=None)
        except Exception:
            log.error_and_raise(f"No dataset found at {transfer_model_from}", RuntimeError)

        # should we load these inside the worker calls?
        model_coeffs = mds.coefficients.values
        locx = mds.location_x.values
        locy = mds.location_y.values
        params = mds.params.values

        log.info(f"Loading model from {transfer_model_from}. ")

    tasks = []
    for tbid, ds_dct in xds_dct.items():
        bandid = tbid[-4:]
        timeid = tbid[4:8]
        ra = ds_dct["radec"][0]
        dec = ds_dct["radec"][1]
        dsl = ds_dct["dsl"]
        time_out = ds_dct["time_out"]
        freq_out = ds_dct["freq_out"]
        chan_low = ds_dct["chan_low"]
        chan_high = ds_dct["chan_high"]
        iter0 = 0
        if from_cache:
            out_ds_name = f"{dds_store.url}/time{timeid}_band{bandid}.zarr"
            out_ds = xr.open_zarr(out_ds_name, chunks=None)
            if "niters" in out_ds:
                iter0 = out_ds.niters
        else:
            out_ds_name = None

        # compute lm coordinates of target
        if target is not None:
            tmp = target.split(",")
            if len(tmp) == 1 and tmp[0] == target:
                tra, tdec = get_coordinates(time_out, target=target)
            else:  # we assume a HH:MM:SS,DD:MM:SS format has been passed in
                from astropy import units as u
                from astropy.coordinates import SkyCoord

                c = SkyCoord(tmp[0], tmp[1], frame="fk5", unit=(u.hourangle, u.deg))
                tra = np.deg2rad(c.ra.value)
                tdec = np.deg2rad(c.dec.value)

            tcoords = np.zeros((1, 2))
            tcoords[0, 0] = tra
            tcoords[0, 1] = tdec
            coords0 = np.array((ra, dec))
            lm0 = radec_to_lm(tcoords, coords0).squeeze()
            l0 = lm0[0]
            m0 = lm0[1]
        else:
            l0 = 0.0
            m0 = 0.0
            tra = ds.ra
            tdec = ds.dec

        attrs = {
            "ra": tra,
            "dec": tdec,
            "l0": l0,
            "m0": m0,
            "cell_rad": cell_rad,
            "bandid": bandid,
            "timeid": timeid,
            "freq_out": freq_out,
            "time_out": time_out,
            "robustness": robustness,
            "super_resolution_factor": super_resolution_factor,
            "field_of_view": field_of_view,
            "product": product,
            "niters": iter0,
            "chan_low": chan_low,
            "chan_high": chan_high,
        }

        # get the model
        if transfer_model_from:
            from pfb_imaging.utils.modelspec import eval_coeffs_to_slice

            _, _, _, x0, y0 = wgridder_conventions(l0, m0)
            model = eval_coeffs_to_slice(
                time_out,
                freq_out,
                model_coeffs,
                locx,
                locy,
                mds.parametrisation,
                params,
                mds.texpr,
                mds.fexpr,
                mds.npix_x,
                mds.npix_y,
                mds.cell_rad_x,
                mds.cell_rad_y,
                mds.center_x,
                mds.center_y,
                nx,
                ny,
                cell_rad,
                cell_rad,
                x0,
                y0,
            )
            model = model[None, :, :]  # hack to get the corr axis

        elif from_cache:
            if use_best_model and "MODEL_BEST" in out_ds:
                model = out_ds.MODEL_BEST.values
            elif "MODEL" in out_ds:
                model = out_ds.MODEL.values
            else:
                model = None
        else:
            model = None

        task = rimage_data_products.remote(
            dsl,
            out_ds_name,
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            cell_rad,
            dds_store.url + "/" + tbid + ".zarr",
            attrs,
            model=model,
            robustness=robustness,
            l0=l0,
            m0=m0,
            nthreads=nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            l2_reweight_dof=l2_reweight_dof,
            do_dirty=dirty,
            do_psf=psf,
            do_weight=weight,
            do_residual=residual,
            do_noise=noise,
            do_beam=beam,
        )
        tasks.append(task)

    residual_mfs = {}
    wsum = {}
    if psf:
        psf_mfs = {}
    nds = len(tasks)
    n_launched = 1
    remaining_tasks = tasks.copy()
    while remaining_tasks:
        # Wait for at least 1 task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

        # Process the completed task
        for task in ready:
            print(f"\rProcessing: {n_launched}/{nds}", end="\n", flush=True)
            outputs = ray.get(task)
            timeid = outputs["timeid"]
            residual_mfs.setdefault(timeid, np.zeros((ncorr, nx, ny), dtype=float))
            residual_mfs[timeid] += outputs["residual"]
            wsum.setdefault(timeid, np.zeros(ncorr, dtype=float))
            wsum[timeid] += outputs["wsum"]
            if psf:
                psf_mfs.setdefault(timeid, np.zeros((ncorr, nx_psf, ny_psf), dtype=float))
                psf_mfs[timeid] += outputs["psf"]
            n_launched += 1

    for timeid in residual_mfs.keys():
        # get MFS PSFPARSN
        # resolution in pixels
        if psf:
            psfparsn = {}
            psf_mfs[timeid] /= wsum[timeid][:, None, None]
            psfparsn[timeid] = fitcleanbeam(psf_mfs[timeid])
        else:
            psfparsn = None

        residual_mfs[timeid] /= wsum[timeid][:, None, None]
        for c in range(ncorr):
            rms = np.std(residual_mfs[timeid][c])
            if np.isnan(rms):
                log.error_and_raise("RMS of residual in nan, something went wrong", RuntimeError)
            rmax = np.abs(residual_mfs[timeid][c]).max()
            log.info(f"Time ID {timeid}: {corrs[c]} - resid max = {rmax:.3e}, rms = {rms:.3e}")

    # put these in the dds for future reference
    if psfparsn is not None:
        cache_opts(psfparsn, dds_store.url, protocol, name="psfparsn_mfs.pkl")

    dds, dds_list = xds_from_url(dds_store.url)

    # convert to fits files
    tasks = []
    if fits_mfs or fits_cubes:
        log.info(f"Writing fits files to {fits_oname}_{suffix}")
        if dirty:
            fut = rdds2fits.remote(
                dds_list,
                "DIRTY",
                f"{fits_oname}_{suffix}",
                norm_wsum=True,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
                psfpars_mfs=psfparsn,
            )
            tasks.append(fut)
        if psf:
            fut = rdds2fits.remote(
                dds_list,
                "PSF",
                f"{fits_oname}_{suffix}",
                norm_wsum=True,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
                psfpars_mfs=psfparsn,
            )
            tasks.append(fut)
        if residual and "RESIDUAL" in dds[0]:
            fut = rdds2fits.remote(
                dds_list,
                "MODEL",
                f"{fits_oname}_{suffix}",
                norm_wsum=False,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
                psfpars_mfs=psfparsn,
            )
            tasks.append(fut)
        if "MODEL" in dds[0]:
            fut = rdds2fits.remote(
                dds_list,
                "RESIDUAL",
                f"{fits_oname}_{suffix}",
                norm_wsum=True,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
                psfpars_mfs=psfparsn,
            )
            tasks.append(fut)
        if noise:
            fut = rdds2fits.remote(
                dds_list,
                "NOISE",
                f"{fits_oname}_{suffix}",
                norm_wsum=True,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
                psfpars_mfs=psfparsn,
            )
            tasks.append(fut)

        if "BEAM" in dds[0]:
            fut = rdds2fits.remote(
                dds_list,
                "BEAM",
                f"{fits_oname}_{suffix}",
                norm_wsum=False,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
                psfpars_mfs=psfparsn,
            )
            tasks.append(fut)

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
