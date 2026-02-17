import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import dask
import dask.array as da
import fsspec
import numpy as np
import psutil
import ray
import xarray as xr
import yaml
from africanus.constants import c as lightspeed
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from daskms import xds_from_storage_ms as xds_from_ms
from daskms.fsspec_store import DaskMSStore
from ducc0.fft import good_size
from ducc0.misc import resize_thread_pool
from zarr import ProcessSynchronizer

from pfb_imaging import set_envs
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.misc import construct_mappings
from pfb_imaging.utils.stokes2im import batch_stokes_image
from pfb_imaging.utils.transients import generate_transient_spectra

log = pfb_logging.get_logger("HCI")


@pfb_logging.log_inputs(log)
def hci(
    ms: list[Path],
    output_dataset: str,
    log_directory: str | None = None,
    product: str = "I",
    scans: str | None = None,
    ddids: str | None = None,
    fields: str | None = None,
    freq_range: str | None = None,
    overwrite: bool = False,
    transfer_model_from: str | None = None,
    data_column: str = "DATA",
    model_column: str | None = None,
    weight_column: str | None = None,
    sigma_column: str | None = None,
    flag_column: str = "FLAG",
    gain_table: list[Path] | None = None,
    max_simul_chunks: int = 4,
    images_per_chunk: int = 16,
    integrations_per_image: int = 1,
    channels_per_image: int = -1,
    precision: str = "double",
    beam_model: str = None,
    field_of_view: float | None = 1.0,
    super_resolution_factor: float = 1.0,
    cell_size: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    psf_relative_size: float | None = None,
    robustness: float = None,
    target: str | None = None,
    l2_reweight_dof: int | None = None,
    eta: float = 1e-05,
    psf_out: bool = False,
    weight_grid_out: bool = False,
    natural_grad: bool = False,
    check_ants: bool = False,
    inject_transients: str | None = None,
    filter_counts_level: float = 10,
    min_padding: float = 2.0,
    phase_dir: str | None = None,
    epsilon: float = 1e-07,
    do_wgridding: bool = True,
    double_accum: bool = True,
    nworkers: int = 1,
    nthreads: int | None = None,
    cg_tol: float = 1e-3,
    cg_maxit: int = 150,
    object_store_memory: float | None = None,
    temp_dir: str | None = None,
    cube_to_fits: bool = False,
    wgt_mode: str = "l2",
    obs_label: str | None = None,
    keep_ray_alive: bool = False,  # not used by CLI
):
    """
    Produce high cadence residual images.
    """
    # start timer
    time_start = time.time()

    if "://" in str(output_dataset):
        protocol = str(output_dataset).split("://")[0]
        prefix = f"{protocol}://"
    else:
        protocol = "file"
        prefix = ""

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path("/".join(str(output_dataset).split("/")[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = str(output_dataset).split("/")[-1]

    output_dataset = f"{prefix}{basedir}/{oname}"

    # this should be a file system
    if log_directory is None:
        log_directory = Path(basedir) / "pfb_logs"
    else:
        log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/hci_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True)
        ncpu = psutil.cpu_count(logical=False)
        nthreads = nthreads // 2
        ncpu = ncpu // 2
    else:
        ncpu = np.minimum(psutil.cpu_count(logical=False), nthreads)

    remprod = product.upper().strip("IQUV")
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported", NotImplementedError)

    msnames = []
    for ms_name in map(str, ms):
        msstore = DaskMSStore(ms_name.rstrip("/"))
        mslist = msstore.fs.glob(ms_name.rstrip("/"))
        try:
            assert len(mslist) > 0
            msnames += list(map(msstore.fs.unstrip_protocol, mslist))
        except Exception:
            log.error_and_raise(f"No MS at {ms_name}", ValueError)
    ms = msnames
    if gain_table is not None:
        gainnames = []
        for gt in map(str, gain_table):
            gainstore = DaskMSStore(gt.rstrip("/"))
            gtlist = gainstore.fs.glob(gt.rstrip("/"))
            try:
                assert len(gtlist) > 0
                gainnames += list(map(gainstore.fs.unstrip_protocol, gtlist))
            except Exception:
                log.error_and_raise(f"No gain table  at {gt}", ValueError)
        gain_table = gainnames

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu)
    runtime_env = {"env_vars": env_vars}

    if object_store_memory is not None:
        mem_limit = object_store_memory * 1e9  # convert GB -> B
    else:
        mem_limit = None

    ray.init(
        num_cpus=nworkers,
        logging_level="INFO",
        ignore_reinit_error=True,
        object_store_memory=mem_limit,
        runtime_env=runtime_env,
    )

    fds_store = DaskMSStore(f"{output_dataset}")
    if fds_store.exists():
        if overwrite:
            log.info(f"Overwriting {output_dataset}")
            fds_store.rm(recursive=True)
        else:
            log.error_and_raise(f"{output_dataset} exists. Set overwrite to overwrite it. ", RuntimeError)

    fs = fsspec.filesystem(fds_store.url.split(":", 1)[0])
    fs.makedirs(fds_store.url, exist_ok=True)

    if gain_table is not None:

        def tmpf(x):
            return "::".join(x.rsplit("/", 1))

        gain_names = list(map(tmpf, gain_table))
    else:
        gain_names = None

    if freq_range is not None:
        fmin, fmax = freq_range.strip(" ").split(":")
        if len(fmin) > 0:
            freq_min = float(fmin)
        else:
            freq_min = -np.inf
        if len(fmax) > 0:
            freq_max = float(fmax)
        else:
            freq_max = np.inf
    else:
        freq_min = -np.inf
        freq_max = np.inf

    group_by = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

    # write model to tmp ds
    if transfer_model_from is not None:
        log.error_and_raise("Use the degrid app to populate a model column instead", NotImplementedError)

    log.info("Constructing mapping")
    (
        row_mapping,
        freq_mapping,
        time_mapping,
        freqs,
        utimes,
        ms_chunks,
        gains,
        radecs,
        chan_widths,
        uv_max,
        antpos,
        poltype,
    ) = construct_mappings(
        ms,
        gain_names,
        ipi=images_per_chunk * integrations_per_image,
        cpi=channels_per_image,
        freq_min=freq_min,
        freq_max=freq_max,
        field_ids=fields,
        ddids=ddids,
        scans=scans,
    )

    max_freq = 0

    # get the full (time, freq) domain
    all_times = []
    all_freqs = []
    for ms_name in ms:
        for idt in freqs[ms_name].keys():
            freq = freqs[ms_name][idt]
            all_freqs.extend(freq)
            all_times.extend(utimes[ms_name][idt])
            mask = (freq <= freq_max) & (freq >= freq_min)
            max_freq = np.maximum(max_freq, freq[mask].max())

    all_freqs = np.unique(all_freqs)
    all_times = np.unique(all_times)

    # cell size
    cell_n = 1.0 / (2 * uv_max * max_freq / lightspeed)

    if cell_size is not None:
        cell_size = cell_size
        cell_rad = cell_size * np.pi / 60 / 60 / 180
        if cell_n / cell_rad < 1:
            log.info(f"Requested cell size of {cell_size} arcseconds could be sub-Nyquist.")
        log.info(f"Super resolution factor = {cell_n / cell_rad}")
    else:
        cell_rad = cell_n / super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi
        log.info(f"Cell size set to {cell_size} arcseconds")

    if nx is None:
        fov = field_of_view * 3600
        npix = int(fov / cell_size)
        npix = good_size(npix)
        while npix % 2:
            npix += 1
            npix = good_size(npix)
        nx = npix
        ny = npix
    else:
        nx = nx
        ny = ny if ny is not None else nx
        cell_deg = np.rad2deg(cell_rad)
        if nx % 2 or ny % 2:
            raise NotImplementedError("Only even number of pixels currently supported")
        fovx = nx * cell_deg
        fovy = ny * cell_deg
        log.info(f"Field of view is ({fovx:.3e},{fovy:.3e}) degrees")

    cell_deg = np.rad2deg(cell_rad)

    # crude column arithmetic
    dc = data_column.replace(" ", "")
    if "+" in dc:
        dc1, dc2 = dc.split("+")
        operator = "+"
    elif "-" in dc:
        dc1, dc2 = dc.split("-")
        operator = "-"
    else:
        dc1 = dc
        dc2 = None
        operator = None

    columns = (dc1, flag_column, "UVW", "ANTENNA1", "ANTENNA2", "TIME", "INTERVAL", "FLAG_ROW")
    schema = {}
    schema[flag_column] = {"dims": ("chan", "corr")}
    schema[dc1] = {"dims": ("chan", "corr")}
    if dc2 is not None:
        columns += (dc2,)
        schema[dc2] = {"dims": ("chan", "corr")}

    # only WEIGHT column gets special treatment
    # any other column must have channel axis
    if sigma_column is not None:
        log.info(f"Initialising weights from {sigma_column} column")
        columns += (sigma_column,)
        schema[sigma_column] = {"dims": ("chan", "corr")}
    elif weight_column is not None:
        log.info(f"Using weights from {weight_column} column")
        columns += (weight_column,)
        # hack for https://github.com/ratt-ru/dask-ms/issues/268
        if weight_column != "WEIGHT":
            schema[weight_column] = {"dims": ("chan", "corr")}
    else:
        log.info("No weights provided, using unity weights")

    if model_column is not None:
        columns += (model_column,)
        schema[model_column] = {"dims": ("chan", "corr")}

    # distinct freq groups
    sgroup = 0
    freq_groups = []
    freq_sgroups = []
    for ms_name in ms:
        for idt, freq in freqs[ms_name].items():
            ilo = idt.find("DDID") + 4
            ihi = idt.rfind("_")
            ddid = int(idt[ilo:ihi])
            if (ddids is not None) and (ddid not in ddids):
                continue
            if not len(freq_groups):
                freq_groups.append(freq)
                freq_sgroups.append(sgroup)
                sgroup += freq_mapping[ms_name][idt]["counts"].size
            else:
                in_group = False
                for fs in freq_groups:
                    if freq.size == fs.size and np.all(freq == fs):
                        in_group = True
                        break
                if not in_group:
                    freq_groups.append(freq)
                    freq_sgroups.append(sgroup)
                    sgroup += freq_mapping[ms_name][idt]["counts"].size

    # band mapping
    msddid2bid = {}
    for ms_name in ms:
        msddid2bid[ms_name] = {}
        for idt, freq in freqs[ms_name].items():
            # find group where it matches
            for sgroup, fs in zip(freq_sgroups, freq_groups):
                if freq.size == fs.size and np.all(freq == fs):
                    msddid2bid[ms_name][idt] = sgroup

    log.info("Creating scaffold for stacked cube")
    attrs, ntasks, n_timeo, n_freqo = make_dummy_dataset(
        ms,
        output_dataset,
        fields,
        ddids,
        scans,
        phase_dir,
        product,
        beam_model,
        psf_out,
        psf_relative_size,
        utimes,
        freqs,
        radecs,
        time_mapping,
        freq_mapping,
        freq_min,
        freq_max,
        nx,
        ny,
        cell_deg,
        integrations_per_image=integrations_per_image,
        obs_label=obs_label,
    )

    log.info("Scaffolding complete")

    n_stokes = len(set(product))
    log.info(f"Cube size = (n_stokes={n_stokes}, n_freq={n_freqo}, n_time={n_timeo}, n_y={ny}, n_x={nx})")

    if inject_transients is not None:
        # we need to do this here because we don't have access
        # to the full domain inside the call to stokes2im
        log.info("Computing transient spectra")

        with open(inject_transients, "r") as f:
            config = yaml.safe_load(f)
        transients = config["transients"]
        coords = {
            "TIME": (("TIME",), all_times),
            "FREQ": (("FREQ",), all_freqs),
        }
        names = []
        ras = []
        decs = []
        transient_ds = xr.Dataset(coords=coords)
        for transient in transients:
            tprofile, fprofile = generate_transient_spectra(all_times, all_freqs, transient)
            name = transient["name"]
            transient_ds[f"{name}_time_profile"] = (("TIME",), tprofile)
            transient_ds[f"{name}_freq_profile"] = (("FREQ",), fprofile)
            names.append(name)
            ras.append(transient["position"]["ra"])
            decs.append(transient["position"]["dec"])
        transient_ds = transient_ds.assign_attrs({"names": names, "ras": ras, "decs": decs})
        transient_baseoname = inject_transients.removesuffix("yaml")
        transient_ds.to_zarr(f"{transient_baseoname}zarr", mode="a")
        log.info("Spectra computed")

    tasks = []
    channel_width = {}
    ncompleted = 0
    # if opts.temp-dir is None dir will be /tmp
    with TemporaryDirectory(prefix="hci-", dir=temp_dir) as tempdir:
        synchronizer = ProcessSynchronizer(tempdir)
        for ims, ms_name in enumerate(ms):
            xds = xds_from_ms(ms_name, columns=columns, table_schema=schema, group_cols=group_by)

            for ds in xds:
                fid = ds.FIELD_ID
                ddid = ds.DATA_DESC_ID
                scanid = ds.SCAN_NUMBER
                if (fields is not None) and (fid not in fields):
                    continue
                if (ddids is not None) and (ddid not in ddids):
                    continue
                if (scans is not None) and (scanid not in scans):
                    continue

                idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

                idx = (freqs[ms_name][idt] >= freq_min) & (freqs[ms_name][idt] <= freq_max)
                if not idx.any():
                    continue

                titr = enumerate(zip(time_mapping[ms_name][idt]["start_indices"], time_mapping[ms_name][idt]["counts"]))
                for ti, (tlow, tcounts) in titr:
                    time_slice = slice(tlow, tlow + tcounts)
                    row_index = row_mapping[ms_name][idt]["start_indices"][time_slice]
                    row_counts = row_mapping[ms_name][idt]["counts"][time_slice]
                    # select all rows for output dataset
                    row_slice = slice(row_index[0], row_index[-1] + row_counts[-1])

                    fitr = enumerate(
                        zip(freq_mapping[ms_name][idt]["start_indices"], freq_mapping[ms_name][idt]["counts"])
                    )
                    b0 = msddid2bid[ms_name][idt]
                    for fi, (flow, fcounts) in fitr:
                        freq_slice = slice(flow, flow + fcounts)

                        subds = ds[{"row": row_slice, "chan": freq_slice}]
                        subds = subds.chunk({"row": -1, "chan": -1})
                        if gains[ms_name][idt] is not None:
                            subgds = gains[ms_name][idt][{"gain_time": time_slice, "gain_freq": freq_slice}]
                            jones = subgds.gains.data
                        else:
                            jones = None

                        fut = batch_stokes_image.remote(
                            dc1=dc1,
                            dc2=dc2,
                            operator=operator,
                            ds=subds,
                            jones=jones,
                            nx=nx,
                            ny=ny,
                            freq=freqs[ms_name][idt][freq_slice],
                            utime=utimes[ms_name][idt][time_slice],
                            rbin_idx=row_index,
                            rbin_counts=row_counts,
                            cell_rad=cell_rad,
                            radec=radecs[ms_name][idt],
                            antpos=antpos[ms_name],
                            poltype=poltype[ms_name],
                            fds_store=fds_store,
                            bandid=b0 + fi,
                            timeid=ti,
                            msid=ims,
                            attrs=attrs,
                            integrations_per_image=integrations_per_image,
                            nthreads=nthreads,
                            precision=precision,
                            sigma_column=sigma_column,
                            weight_column=weight_column,
                            model_column=model_column,
                            product=product,
                            check_ants=check_ants,
                            phase_dir=phase_dir,
                            beam_model=beam_model,
                            target=target,
                            robustness=robustness,
                            min_padding=min_padding,
                            filter_counts_level=filter_counts_level,
                            psf_relative_size=psf_relative_size,
                            inject_transients=inject_transients,
                            epsilon=epsilon,
                            do_wgridding=do_wgridding,
                            double_accum=double_accum,
                            natural_grad=natural_grad,
                            eta=eta,
                            cg_tol=cg_tol,
                            cg_maxit=cg_maxit,
                            psf_out=psf_out,
                            weight_grid_out=weight_grid_out,
                            l2_reweight_dof=l2_reweight_dof,
                            synchronizer=synchronizer,
                            wgt_mode=wgt_mode,
                        )
                        tasks.append(fut)
                        channel_width[b0 + fi] = (
                            freqs[ms_name][idt][freq_slice].max() - freqs[ms_name][idt][freq_slice].min()
                        )

                        # limit the number of chunks that are processed simultaneously
                        if len(tasks) > max_simul_chunks:
                            ncompleted += 1
                            ready, tasks = ray.wait(tasks, num_returns=1)
                            timeid, bandid = ray.get(ready)[0]
                            print(f"Processed {ncompleted}/{ntasks}", end="\n", flush=True)

        remaining_tasks = tasks.copy()
        while remaining_tasks:
            ncompleted += 1
            ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
            timeid, bandid = ray.get(ready)[0]
            print(f"Processed {ncompleted}/{ntasks}", end="\n", flush=True)

    log.info("Computing means")
    cwidths = []
    for _, val in channel_width.items():
        cwidths.append(val)
    # reduction over FREQ and TIME so use max chunk sizes
    ds = xr.open_zarr(fds_store.url, chunks={"FREQ": -1, "TIME": -1})
    cube = ds.cube.data
    wsums = ds.weight.data
    # all variables have been normalised by wsum so we first
    # undo the normalisation (wsum=0 where there is no data)
    weighted_cube = cube * wsums[:, :, :, None, None]
    taxis = ds.cube.get_axis_num("TIME")
    faxis = ds.cube.get_axis_num("FREQ")
    wsum = da.sum(wsums, axis=taxis)
    # we need this for the where clause in da.divide, should be cheap
    wsumc = wsum.compute()[:, :, None, None]
    weighted_sum = da.sum(weighted_cube, axis=taxis)
    weighted_mean = da.divide(weighted_sum, wsum[:, :, None, None], where=wsumc > 0)
    if psf_out:
        psfsq = (ds.psf.data * wsums[:, :, :, None, None]) ** 2
        weighted_psfsq_sum = da.sum(psfsq, axis=(faxis, taxis))
        wsumsq = da.sum(wsums**2, axis=(faxis, taxis))
        wsumsqc = wsumsq.compute()[:, None, None]
        weighted_psfsq_mean = da.divide(weighted_psfsq_sum, wsumsq[:, None, None], where=wsumsqc > 0)
        if psf_relative_size == 1:
            xpsf = "X"
            ypsf = "Y"
        else:
            xpsf = "X_PSF"
            ypsf = "Y_PSF"
        ds["psf2"] = (("STOKES", ypsf, xpsf), weighted_psfsq_mean)
    # only write new variables
    drop_vars = [key for key in ds.data_vars.keys() if key != "psf2"]
    ds = ds.drop_vars(drop_vars)
    ds["mean"] = (("STOKES", "FREQ", "Y", "X"), weighted_mean)
    ds["channel_width"] = (("FREQ",), da.from_array(cwidths, chunks=1))
    with dask.config.set(pool=ThreadPoolExecutor(8)):
        ds.to_zarr(fds_store.url, mode="r+")
    log.info("Reduction complete")

    log.info(f"All done after {time.time() - time_start}s")

    if not keep_ray_alive:
        ray.shutdown()


def make_dummy_dataset(
    ms,
    output_dataset,
    fields,
    ddids,
    scans,
    phase_dir,
    product,
    beam_model,
    psf_out,
    psf_relative_size,
    utimes,
    freqs,
    radecs,
    time_mapping,
    freq_mapping,
    freq_min,
    freq_max,
    nx,
    ny,
    cell_deg,
    spatial_chunk=128,
    integrations_per_image=1,
    obs_label=None,
):
    out_ra = []
    out_dec = []
    out_times = []
    out_freqs = []
    ntasks = 0
    for ms_name in ms:
        xds = xds_from_ms(ms_name, columns="TIME", group_cols=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"])
        for ds in xds:
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            if (fields is not None) and (fid not in fields):
                continue
            if (ddids is not None) and (ddid not in ddids):
                continue
            if (scans is not None) and (scanid not in scans):
                continue

            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

            idx = (freqs[ms_name][idt] >= freq_min) & (freqs[ms_name][idt] <= freq_max)
            if not idx.any():
                continue

            out_ra.append(radecs[ms_name][idt][0])
            out_dec.append(radecs[ms_name][idt][1])

            titr = enumerate(zip(time_mapping[ms_name][idt]["start_indices"], time_mapping[ms_name][idt]["counts"]))
            for ti, (tlow, tcounts) in titr:
                fitr = enumerate(zip(freq_mapping[ms_name][idt]["start_indices"], freq_mapping[ms_name][idt]["counts"]))
                for fi, (flow, fcounts) in fitr:
                    freq_slice = slice(flow, flow + fcounts)
                    out_freqs.append(np.mean(freqs[ms_name][idt][freq_slice]))
                    ntasks += 1

                    for t0 in range(tlow, tlow + tcounts, integrations_per_image):
                        tmax = np.minimum(tlow + tcounts, t0 + integrations_per_image)
                        time_slice = slice(t0, tmax)
                        out_times.append(np.mean(utimes[ms_name][idt][time_slice]))

    # remove duplicates
    out_times = np.unique(out_times)
    out_freqs = np.unique(out_freqs)
    n_times = out_times.size
    n_freqs = out_freqs.size

    # spatial coordinates
    if phase_dir is None:
        out_ra_deg = np.rad2deg(np.unique(out_ra))
        out_dec_deg = np.rad2deg(np.unique(out_dec))
        if out_ra_deg.size > 1 or out_dec_deg.size > 1:
            raise ValueError("phase-dir must be specified when stacking multiple fields")
    else:
        ra_str, dec_str = phase_dir.split(",")
        coord = SkyCoord(ra_str, dec_str, frame="fk5", unit=(units.hourangle, units.deg))
        out_ra_deg = np.array([coord.ra.value])
        out_dec_deg = np.array([coord.dec.value])

    # remove duplicates
    out_times = np.unique(out_times)
    out_freqs = np.unique(out_freqs)

    n_stokes = len(product)
    n_times = out_times.size
    n_freqs = out_freqs.size

    cube_dims = (n_stokes, n_freqs, n_times, ny, nx)
    cube_chunks = (1, 1, 1, spatial_chunk, spatial_chunk)

    mean_dims = (n_stokes, n_freqs, ny, nx)
    mean_chunks = (1, 1, spatial_chunk, spatial_chunk)

    rms_dims = (n_stokes, n_freqs, n_times)
    rms_chunks = (1, 1, 1)

    ra_dim = "RA--SIN"
    dec_dim = "DEC--SIN"
    if out_freqs.size > 1:
        delta_freq = np.diff(out_freqs).min()
    else:
        delta_freq = 1
    if out_times.size > 1:
        delta_time = np.diff(out_times).min()
    else:
        delta_time = 1

    # construct reference header
    w = WCS(naxis=5)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "INTEGRATION", "FREQ", "STOKES"]
    w.wcs.cdelt = [-cell_deg, cell_deg, delta_time, delta_freq, 1]
    w.wcs.cunit = ["deg", "deg", "s", "Hz", ""]
    w.wcs.crval = [out_ra_deg[0], out_dec_deg[0], out_times[0], out_freqs[0], 1]
    w.wcs.crpix = [1 + nx // 2, 1 + ny // 2, 1, 1, 1]
    w.wcs.equinox = 2000.0
    wcs_hdr = w.to_header()

    # the order does seem to matter here,
    # especially when using with StreamingHDU
    hdr = fits.Header()
    hdr["SIMPLE"] = (True, "conforms to FITS standard")
    hdr["BITPIX"] = (-32, "array data type")
    hdr["NAXIS"] = (5, "number of array dimensions")
    data_shape = (nx, ny, n_times, n_freqs, n_stokes)
    for i, size in enumerate(data_shape, 1):
        hdr[f"NAXIS{i}"] = (size, f"length of data axis {i}")

    hdr["EXTEND"] = True
    hdr["BSCALE"] = 1.0
    hdr["BZERO"] = 0.0
    hdr["BUNIT"] = "Jy/beam"
    hdr["EQUINOX"] = 2000.0
    hdr["BTYPE"] = "Intensity"
    hdr.update(wcs_hdr)
    hdr["TIMESCAL"] = delta_time
    if obs_label is not None:
        hdr["OBSLABEL"] = obs_label

    # if we don't pass these into stokes2im they get overwritten
    attrs = {
        "fits_header": list(dict(hdr).items()),
        "radec_dims": (ra_dim, dec_dim),
        "fits_dims": (("X", ra_dim), ("Y", dec_dim), ("TIME", "TIME"), ("FREQ", "FREQ"), ("STOKES", "STOKES")),
    }

    # TODO - why do we sometimes need to round here?
    out_ras = out_ra_deg + np.arange(nx // 2, -(nx // 2), -1) * cell_deg
    out_decs = out_dec_deg + np.arange(-(ny // 2), ny // 2) * cell_deg
    out_ras = np.round(out_ras, decimals=12)
    out_decs = np.round(out_decs, decimals=12)

    dummy_ds = xr.Dataset(
        data_vars={
            "cube": (("STOKES", "FREQ", "TIME", "Y", "X"), da.empty(cube_dims, chunks=cube_chunks, dtype=np.float32)),
            "mean": (("STOKES", "FREQ", "Y", "X"), da.empty(mean_dims, chunks=mean_chunks, dtype=np.float32)),
            "rms": (("STOKES", "FREQ", "TIME"), da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32)),
            "weight": (("STOKES", "FREQ", "TIME"), da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32)),
            "nonzero": (
                (
                    "STOKES",
                    "FREQ",
                    "TIME",
                ),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.bool_),
            ),
            "psf_maj": (
                (
                    "STOKES",
                    "FREQ",
                    "TIME",
                ),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "psf_min": (
                (
                    "STOKES",
                    "FREQ",
                    "TIME",
                ),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "psf_pa": (
                (
                    "STOKES",
                    "FREQ",
                    "TIME",
                ),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32),
            ),
            "channel_width": (("FREQ",), da.empty((n_freqs), chunks=(1), dtype=np.float32)),
        },
        coords={
            "TIME": (("TIME",), out_times),
            "STOKES": (("STOKES",), list(sorted(product))),
            "FREQ": (("FREQ",), out_freqs),
            "X": (("X",), out_ras),
            "Y": (("Y",), out_decs),
        },
        # Store dicts as tuples as zarr doesn't seem to maintain dict order.
        attrs=attrs,
    )

    if beam_model is not None:
        dummy_ds["beam_weight"] = (
            ("STOKES", "FREQ", "TIME", "Y", "X"),
            da.empty(cube_dims, chunks=cube_chunks, dtype=np.float32),
        )

    if psf_out:
        nx_psf = good_size(int(psf_relative_size * nx))
        while nx_psf % 2:
            nx_psf = good_size(nx_psf + 1)
        ny_psf = good_size(int(psf_relative_size * ny))
        while ny_psf % 2:
            ny_psf = good_size(ny_psf + 1)
        psf_dims = (n_stokes, n_freqs, n_times, ny_psf, nx_psf)
        if psf_relative_size == 1:
            xpsf = "X"
            ypsf = "Y"
        else:
            xpsf = "X_PSF"
            ypsf = "Y_PSF"
            dummy_ds = dummy_ds.assign_coords(
                {
                    "Y_PSF": ((ypsf,), out_dec_deg + np.arange(-(ny_psf // 2), ny_psf // 2) * cell_deg),
                    "X_PSF": ((xpsf,), out_ra_deg + np.arange(nx_psf // 2, -(nx_psf // 2), -1) * cell_deg),
                }
            )
        dummy_ds["psf"] = (
            ("STOKES", "FREQ", "TIME", ypsf, xpsf),
            da.empty(psf_dims, chunks=cube_chunks, dtype=np.float32),
        )
        dummy_ds["psf2"] = (
            ("STOKES", ypsf, xpsf),
            da.empty((n_stokes, ny_psf, nx_psf), chunks=(1, spatial_chunk, spatial_chunk), dtype=np.float32),
        )

    # Write scaffold and metadata to disk.
    cds = f"{output_dataset}"
    dummy_ds.to_zarr(cds, mode="w", compute=False)
    return attrs, ntasks, n_times, n_freqs
