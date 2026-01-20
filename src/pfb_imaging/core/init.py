import time
from pathlib import Path

import fsspec
import numpy as np
import psutil
import ray
from daskms import xds_from_storage_ms as xds_from_ms
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool

from pfb_imaging import set_envs
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.misc import construct_mappings
from pfb_imaging.utils.naming import set_output_names
from pfb_imaging.utils.stokes2vis import safe_stokes_vis

log = pfb_logging.get_logger("INIT")

@pfb_logging.log_inputs(log)
def init(
    ms: list[Path],
    output_filename: str,
    scans: list[int] | None = None,
    ddids: list[int] | None = None,
    fields: list[int] | None = None,
    freq_range: str | None = None,
    overwrite: bool = False,
    radec: str | None = None,
    data_column: str = "DATA",
    weight_column: str | None = None,
    sigma_column: str | None = None,
    flag_column: str = "FLAG",
    gain_table: list[Path] | None = None,
    integrations_per_image: int = -1,
    channels_per_image: int = -1,
    precision: str = "double",
    bda_decorr: float = 1.0,
    max_field_of_view: float = 3.0,
    beam_model: str | None = None,
    chan_average: int = 1,
    target: str | None = None,
    progressbar: bool = True,
    check_ants: bool = False,
    log_directory: str | None = None,
    product: str = "I",
    host_address: str | None = None,
    nworkers: int = 1,
    nthreads: int | None = None,
):
    """
    Initialise Stokes data products for imaging
    """

    output_filename, _, log_directory, oname = set_output_names(
        output_filename,
        product,
        None,
        log_directory,
    )

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2

    remprod = product.upper().strip("IQUV")
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported", NotImplementedError)

    msnames = []
    for ms_path in ms:
        msstore = DaskMSStore(str(ms_path).rstrip("/"))
        mslist = msstore.fs.glob(str(ms_path).rstrip("/"))
        try:
            assert len(mslist) > 0
            msnames.append(*list(map(msstore.fs.unstrip_protocol, mslist)))
        except Exception:
            log.error_and_raise(f"No MS at {ms_path}", ValueError)
    ms = msnames

    if gain_table is not None:
        gainnames = []
        for gt in gain_table:
            gainstore = DaskMSStore(str(gt).rstrip("/"))
            gtlist = gainstore.fs.glob(str(gt).rstrip("/"))
            try:
                assert len(gtlist) > 0
                gainnames.append(*list(map(gainstore.fs.unstrip_protocol, gtlist)))
            except Exception:
                log.error_and_raise(f"No gain table at {gt}", ValueError)
        gain_table = gainnames

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/init_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu)

    # these are passed through to child Ray processes
    if nworkers == 1:
        env_vars["RAY_DEBUG_POST_MORTEM"] = "1"

    ray.init(num_cpus=nworkers, logging_level="INFO", ignore_reinit_error=True, runtime_env={"env_vars": env_vars})

    time_start = time.time()

    # Main implementation logic (previously in _init)
    basename = f"{output_filename}"

    xds_store = DaskMSStore(f"{basename}.xds")
    if xds_store.exists():
        if overwrite:
            log.info(f"Overwriting {basename}.xds")
            xds_store.rm(recursive=True)
        else:
            log.error_and_raise(f"{basename}.xds exists. Set overwrite to overwrite it. ", RuntimeError)

    fs = fsspec.filesystem(xds_store.protocol)
    fs.makedirs(xds_store.url, exist_ok=True)

    log.info(f"Data products will be stored in {xds_store.url}")

    if gain_table is not None:

        def tmpf(x):
            return "::".join(x.rsplit("/", 1))

        gain_names = list(map(tmpf, gain_table))
    else:
        gain_names = None

    if freq_range is not None and len(freq_range):
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
        ipi=integrations_per_image,
        cpi=channels_per_image,
        freq_min=freq_min,
        freq_max=freq_max,
        FIELD_IDs=fields,
        DDIDs=ddids,
        SCANs=scans,
    )

    group_by = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

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

    columns = (dc1, "UVW", "ANTENNA1", "ANTENNA2", "TIME", "INTERVAL", "FLAG_ROW")
    schema = {}
    if flag_column != "None":
        columns += (flag_column,)
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

    tasks = []
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
                t_index = slice(tlow, tlow + tcounts)
                ridx = row_mapping[ms_name][idt]["start_indices"][t_index]
                rcnts = row_mapping[ms_name][idt]["counts"][t_index]
                # select all rows for output dataset
                row_index = slice(ridx[0], ridx[-1] + rcnts[-1])

                fitr = enumerate(zip(freq_mapping[ms_name][idt]["start_indices"], freq_mapping[ms_name][idt]["counts"]))
                b0 = msddid2bid[ms_name][idt]
                for fi, (flow, fcounts) in fitr:
                    nu_index = slice(flow, flow + fcounts)
                    subds = ds[{"row": row_index, "chan": nu_index}]
                    if gains[ms_name][idt] is not None:
                        subgds = gains[ms_name][idt][{"gain_time": t_index, "gain_freq": nu_index}]
                        jones = subgds.gains
                    else:
                        jones = None

                    fut = safe_stokes_vis.remote(
                        dc1=dc1,
                        dc2=dc2,
                        operator=operator,
                        ds=subds,
                        jones=jones,
                        freq=freqs[ms_name][idt][nu_index],
                        chan_width=chan_widths[ms_name][idt][nu_index],
                        utime=utimes[ms_name][idt][t_index],
                        tbin_idx=ridx,
                        tbin_counts=rcnts,
                        chan_low=flow,
                        chan_high=flow + fcounts,
                        radec=radecs[ms_name][idt],
                        antpos=antpos[ms_name],
                        poltype=poltype[ms_name],
                        xds_store=xds_store.url,
                        bandid=b0 + fi,
                        timeid=ti,
                        msid=ims,
                        # Parameters previously from opts:
                        precision=precision,
                        sigma_column=sigma_column,
                        weight_column=weight_column,
                        product=product,
                        check_ants=check_ants,
                        chan_average=chan_average,
                        bda_decorr=bda_decorr,
                        max_field_of_view=max_field_of_view,
                        beam_model=beam_model,
                    )
                    tasks.append(fut)

    nds = len(tasks)
    ncomplete = 0
    remaining_tasks = tasks.copy()
    times_out = []
    freqs_out = []
    while remaining_tasks:
        # Wait for at least 1 task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

        # Process the completed task
        for task in ready:
            result = ray.get(task)
            if result is not None:
                times_out.append(result[0])
                freqs_out.append(result[1])
            ncomplete += 1
            if progressbar:
                print(f"Completed: {ncomplete} / {nds}", end="\n", flush=True)

    times_out = np.unique(times_out)
    freqs_out = np.unique(freqs_out)

    nband = freqs_out.size
    ntime = times_out.size

    log.info(f"Freq and time selection resulted in {nband} output bands and {ntime} output times")

    log.info(f"All done after {time.time() - time_start}s")

    ray.shutdown()

    return
