import gc
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

from pfb_imaging.utils.optional import optional_dependency

# dask-ms (and therefore python-casacore, which has no linux-aarch64 wheel)
# lives behind the optional [casacore] extra. Guard the import here rather than
# in cli/hci.py: hip-cargo round-trips the CLI module against the cab
# definition and requires it to match the generated form byte for byte, so the
# command body cannot carry extra statements.
with optional_dependency("The hci command"):
    from daskms import xds_from_storage_ms as xds_from_ms
from ducc0.fft import good_size
from ducc0.misc import resize_thread_pool
from meerkat_beams.utils import BeamWizard
from zarr import ProcessSynchronizer

from pfb_imaging import init_ray, set_envs, setup_ray_worker
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.hci_common import (
    FITS_AXIS_KEYWORDS as _FITS_AXIS_KEYWORDS,  # noqa: F401
)
from pfb_imaging.utils.hci_common import (
    STOKES_FITS_INDEX as _STOKES_FITS_INDEX,  # noqa: F401
)
from pfb_imaging.utils.hci_common import (
    build_scaffold,
    image_centre_deg,
    write_fits_vars,
    write_transient_spectra,
)
from pfb_imaging.utils.hci_common import (
    default_phase_dir as _default_phase_dir,
)
from pfb_imaging.utils.hci_common import (
    make_fits_header as _make_fits_header,  # noqa: F401
)
from pfb_imaging.utils.misc import check_gridder_epsilon, construct_mappings, set_image_size
from pfb_imaging.utils.naming import glob_uris, uri_and_fs
from pfb_imaging.utils.stokes2im import batch_stokes_image

log = pfb_logging.get_logger("HCI")


@pfb_logging.log_exceptions
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
    channels_per_bin: int = -1,
    precision: str = "double",
    beam_model: str | None = None,
    primary_beam_band: str | None = None,
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
    npix_super: int = 0,
    min_padding: float = 2.0,
    phase_dir: str | None = None,
    epsilon: float = 1e-05,
    do_wgridding: bool = True,
    double_accum: bool = True,
    nworkers: int = 1,
    nthreads: int | None = None,
    cg_tol: float = 1e-3,
    cg_maxit: int = 150,
    object_store_memory: float | None = None,
    fits_output_folder: str | None = None,
    fits_vars: list[str] = None,
    wgt_mode: str = "l2",
    obs_label: str | None = None,
    flag_excess_rms: float = 1.5,
    temp_dir: str | None = None,
    ray_address: str = "local",
    keep_ray_alive: bool = False,  # not used by CLI
):
    """
    Produce high cadence residual images.
    """
    # for logging options
    opts_dict = locals().copy()

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
    opts_dict["output_dataset"] = output_dataset

    # logs and fits need a file system
    if fits_output_folder is None:
        fits_output_folder = Path(basedir) / f"{oname}_fits"
    else:
        fits_output_folder = Path(fits_output_folder)
    opts_dict["fits_output_folder"] = fits_output_folder
    # only materialise it if something will be written there
    if fits_vars is not None:
        fits_output_folder.mkdir(parents=True, exist_ok=True)

    if log_directory is None:
        log_directory = Path(basedir) / "pfb_logs"
    else:
        log_directory = Path(log_directory)
    opts_dict["log_directory"] = log_directory
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/hci_{timestamp}.log"
    pfb_logging.log_to_file(logname)

    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True)
        ncpu = psutil.cpu_count(logical=False)
        nthreads = nthreads // 2
        ncpu = ncpu // 2
    else:
        ncpu = np.minimum(psutil.cpu_count(logical=False), nthreads)
    opts_dict["nthreads"] = nthreads
    log.info(f"Using {nworkers} workers with {nthreads} threads per worker")

    remprod = product.upper().strip("IQUV")
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported", NotImplementedError)

    # ducc only refuses an unreachable epsilon when it is handed data; say so up
    # front instead (#340).
    try:
        check_gridder_epsilon(precision, epsilon, do_wgridding=do_wgridding)
    except ValueError as e:
        log.error_and_raise(str(e), ValueError)

    msnames = []
    for ms_name in map(str, ms):
        matches = glob_uris(ms_name)
        if not matches:
            log.error_and_raise(f"No MS at {ms_name}", ValueError)
        msnames += matches
    ms = msnames
    opts_dict["ms"] = ms
    if gain_table is not None:
        gainnames = []
        for gt in map(str, gain_table):
            matches = glob_uris(gt)
            if not matches:
                log.error_and_raise(f"No gain table at {gt}", ValueError)
            gainnames += matches
        gain_table = gainnames
        opts_dict["gain_table"] = gain_table

    log.log_options_dict(opts_dict, title="HCI options")

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu, log=log)
    runtime_env = {
        "env_vars": env_vars,
        "worker_process_setup_hook": setup_ray_worker,
    }

    if object_store_memory is not None:
        mem_limit = object_store_memory * 1e9  # convert GB -> B
    else:
        mem_limit = None

    init_ray(
        nworkers,
        ray_address=ray_address,
        runtime_env=runtime_env,
        object_store_memory=mem_limit,
        log=log,
    )

    fds_fs, fds_url = uri_and_fs(output_dataset)
    if fds_fs.exists(fds_url):
        if overwrite:
            log.info(f"Overwriting {output_dataset}")
            fds_fs.rm(fds_url, recursive=True)
        else:
            log.error_and_raise(f"{output_dataset} exists. Set overwrite to overwrite it. ", RuntimeError)

    fds_fs.makedirs(fds_url, exist_ok=True)

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
        max_blength,
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

    # image size
    nx, ny, nx_psf, ny_psf, cell_n, cell_rad, cell_deg = set_image_size(
        max_blength,
        max_freq,
        field_of_view,
        super_resolution_factor,
        cell_size=cell_size,
        nx=nx,
        ny=ny,
        psf_oversize=psf_relative_size,  # 128 pix if not specified
        log=log,
    )

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
    attrs, ntasks, n_timeo, n_freqo, phase_dir = make_dummy_dataset(
        ms,
        output_dataset,
        fields,
        ddids,
        scans,
        phase_dir,
        product,
        beam_model,
        psf_out,
        utimes,
        freqs,
        radecs,
        time_mapping,
        freq_mapping,
        freq_min,
        freq_max,
        nx,
        ny,
        nx_psf,
        ny_psf,
        cell_deg,
        target=target,
        spatial_chunk=128,  # hardcode for now
        images_per_chunk=images_per_chunk,
        integrations_per_image=integrations_per_image,
        obs_label=obs_label,
    )

    log.info("Scaffolding complete")

    # this is a bit hacky but I doubt we use anything but MeerKAT beams for now
    if beam_model is None:
        log.info("No beam model specified, not applying primary beam correction")
    elif beam_model.lower() != "meerkat-beams":
        raise NotImplementedError("Only meerkat-beams is currently supported for beam_model")
    else:
        if primary_beam_band not in ("U", "L", "S0", "S4"):
            raise ValueError("primary_beam_band must be one of U, L, S0 or S4")
        log.info("Using meerkat-beams to initialise BeamWizard")
        beam_model = BeamWizard(image_name=output_dataset, band=primary_beam_band)

    # get the size of the padded grid
    nx_pad = good_size(int(min_padding * nx))
    while nx_pad % 2:
        nx_pad = good_size(nx_pad + 1)
    ny_pad = good_size(int(min_padding * ny))
    while ny_pad % 2:
        ny_pad = good_size(ny_pad + 1)
    n_stokes = len(set(product))
    log.info(f"Cube size = (n_stokes={n_stokes}, n_freq={n_freqo}, n_time={n_timeo}, n_y={ny}, n_x={nx})")
    log.info(f"Padded grid size = (n_y={ny_pad}, n_x={nx_pad})")
    log.info(f"Cube chunks = (n_stokes={n_stokes}, n_freq={1}, n_time={images_per_chunk}, n_y={128}, n_x={128})")

    if inject_transients is not None:
        # we need to do this here because we don't have access
        # to the full domain inside the call to stokes2im
        log.info("Computing transient spectra")

        write_transient_spectra(inject_transients, all_times, all_freqs)
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
                            fds_store=fds_url,
                            bandid=b0 + fi,
                            timeid=ti,
                            msid=ims,
                            attrs=attrs,
                            integrations_per_image=integrations_per_image,
                            channels_per_bin=channels_per_bin,
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
                            npix_super=npix_super,
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
                            log.info(f"Processed {ncompleted}/{ntasks}")

            # release the main MS TableProxy before opening the next MS — see
            # https://github.com/ratt-ru/pfb-imaging/issues/241
            del xds
            gc.collect()

        remaining_tasks = tasks.copy()
        while remaining_tasks:
            ncompleted += 1
            ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
            timeid, bandid = ray.get(ready)[0]
            log.info(f"Processed {ncompleted}/{ntasks}")

    log.info("Computing means")
    cwidths = []
    for _, val in channel_width.items():
        cwidths.append(val)
    # reduction over FREQ and TIME so use max chunk sizes
    ds = xr.open_zarr(fds_url, chunks={"FREQ": -1, "TIME": -1})
    chunk_sizes = ds.chunks
    x_chunk = chunk_sizes["X"][0]
    y_chunk = chunk_sizes["Y"][0]
    cube = ds.cube.data
    wsums = ds.weight.data
    # needed for flagging
    nonzero = ds.nonzero.values
    rms = ds.rms.values
    ncorr, nband, ntime = nonzero.shape
    flag = np.zeros((ncorr, nband, ntime), dtype=bool)
    for corr in range(ncorr):
        for band in range(nband):
            nzero = nonzero[corr, band, :]
            medianrms = np.median(rms[corr, band, nzero])
            flag[corr, band, :] = (rms[corr, band, :] > flag_excess_rms * medianrms) | ~nzero
    # need flag (as mask) in dask graph
    mask = da.from_array((~flag).astype(np.uint8), chunks=(1, -1, -1))
    wsums = wsums * mask
    # all variables have been normalised by wsum so we first
    # undo the normalisation (wsum=0 where there is no data)
    taxis = ds.cube.get_axis_num("TIME")
    faxis = ds.cube.get_axis_num("FREQ")
    wsum = da.sum(wsums, axis=taxis)
    # we need this for the where clause in da.divide, should be cheap
    wsumc = wsum.compute()[:, :, None, None]
    weighted_sum = da.sum(cube * wsums[:, :, :, None, None], axis=taxis)
    weighted_mean = da.divide(weighted_sum, wsum[:, :, None, None], where=wsumc > 0)
    if psf_out:
        psfsq = (ds.psf.data * wsums[:, :, :, None, None]) ** 2
        weighted_psfsq_sum = da.sum(psfsq, axis=(faxis, taxis))
        wsumsq = da.sum(wsums**2, axis=(faxis, taxis))
        wsumsqc = wsumsq.compute()[:, None, None]
        weighted_psfsq_mean = da.divide(weighted_psfsq_sum, wsumsq[:, None, None], where=wsumsqc > 0)
        ds["psf2"] = (("STOKES", "Y_PSF", "X_PSF"), weighted_psfsq_mean.rechunk(chunks=(1, y_chunk, x_chunk)))
    # only write new variables
    drop_vars = [key for key in ds.data_vars.keys() if key != "psf2"]
    ds = ds.drop_vars(drop_vars)
    ds["cube_mean"] = (("STOKES", "FREQ", "Y", "X"), weighted_mean.rechunk(chunks=(1, 1, y_chunk, x_chunk)))
    ds["channel_width"] = (("FREQ",), da.from_array(cwidths, chunks=1))
    ds["flag"] = (("STOKES", "FREQ", "TIME"), da.from_array(flag, chunks=(1, 1, images_per_chunk)))
    with dask.config.set(pool=ThreadPoolExecutor(8)):
        ds.to_zarr(fds_url, mode="r+")
    log.info("Reduction complete")

    if fits_vars is not None:
        write_fits_vars(fds_url, fits_vars, fits_output_folder, images_per_chunk, log)

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
    utimes,
    freqs,
    radecs,
    time_mapping,
    freq_mapping,
    freq_min,
    freq_max,
    nx,
    ny,
    nx_psf,
    ny_psf,
    cell_deg,
    target=None,
    spatial_chunk=128,
    images_per_chunk=16,
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

            titr = zip(time_mapping[ms_name][idt]["start_indices"], time_mapping[ms_name][idt]["counts"])
            for tlow, tcounts in titr:
                fitr = zip(freq_mapping[ms_name][idt]["start_indices"], freq_mapping[ms_name][idt]["counts"])
                for flow, fcounts in fitr:
                    freq_slice = slice(flow, flow + fcounts)
                    out_freqs.append(np.mean(freqs[ms_name][idt][freq_slice]))
                    ntasks += 1

                    for t0 in range(tlow, tlow + tcounts, integrations_per_image):
                        tmax = np.minimum(tlow + tcounts, t0 + integrations_per_image)
                        time_slice = slice(t0, tmax)
                        out_times.append(np.mean(utimes[ms_name][idt][time_slice]))

        # release the main MS TableProxy before opening the next MS — see
        # https://github.com/ratt-ru/pfb-imaging/issues/241
        del xds
        gc.collect()

    # remove duplicates
    out_times = np.unique(out_times)
    out_freqs = np.unique(out_freqs)
    n_times = out_times.size
    n_freqs = out_freqs.size

    if n_times == 0 or n_freqs == 0:
        # the time loop steps by integrations_per_image, so a value < 1 gives an
        # empty range and hence no output images at all
        raise ValueError(
            f"selection produced {n_times} time bins and {n_freqs} frequency bins, so there is "
            f"nothing to image. integrations_per_image must be >= 1 (got {integrations_per_image}); "
            "pass the scan length for a single time bin."
        )

    # spatial coordinates
    if phase_dir is None:
        phase_dir = _default_phase_dir(out_ra, out_dec)
    out_ra_deg, out_dec_deg = image_centre_deg(np.asarray(out_ra), np.asarray(out_dec), phase_dir, target, out_times)

    dummy_ds, attrs = build_scaffold(
        out_times,
        out_freqs,
        out_ra_deg,
        out_dec_deg,
        product,
        beam_model is not None,
        psf_out,
        nx,
        ny,
        nx_psf,
        ny_psf,
        cell_deg,
        spatial_chunk=spatial_chunk,
        images_per_chunk=images_per_chunk,
        obs_label=obs_label,
    )
    dummy_ds.to_zarr(f"{output_dataset}", mode="w", compute=False)
    return attrs, ntasks, dummy_ds.sizes["TIME"], dummy_ds.sizes["FREQ"], phase_dir
