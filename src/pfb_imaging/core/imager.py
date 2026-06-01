import time
import warnings
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import psutil
import ray
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool
from msv4_utils import MSv4Backend, infer_backend
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
from xarray_ms.errors import FrameConversionWarning, IrregularGridWarning, MissingMetadataWarning

from pfb_imaging import set_envs, setup_ray_worker
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.naming import set_output_names
from pfb_imaging.utils.stokes2vis_msv4 import safe_stokes_vis

warnings.filterwarnings("ignore", category=IrregularGridWarning)
warnings.filterwarnings("ignore", category=MissingMetadataWarning)
warnings.filterwarnings("ignore", category=FrameConversionWarning)


log = pfb_logging.get_logger("IMAGER")


def imager(
    ms: list[Path],
    output_filename: str,
    scan_names: list[str] | None = None,
    spw_names: list[str] | None = None,
    field_names: list[str] | None = None,
    freq_range: str | None = None,
    overwrite: bool = False,
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
    progressbar: bool = True,
    log_directory: str | None = None,
    product: str = "I",
    nworkers: int = 1,
    nthreads: int | None = None,
    wgt_mode: str = "l2",
    keep_ray_alive: bool = False,  # not used by CLI
):
    """
    Initialise Stokes data products for imaging
    """
    # for logging options
    opts_dict = locals().copy()

    output_filename, _, log_directory, oname = set_output_names(
        output_filename,
        product,
        None,
        log_directory,
    )
    opts_dict["output_filename"] = output_filename
    opts_dict["log_directory"] = log_directory

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2
    log.info(f"Using {nworkers} workers with {nthreads} thread per worker")

    remprod = product.upper().strip("IQUV")
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported", NotImplementedError)

    msnames = []
    for ms_path in ms:
        msstore = DaskMSStore(str(ms_path).rstrip("/"))
        mslist = msstore.fs.glob(str(ms_path).rstrip("/"))
        try:
            assert len(mslist) > 0
            msnames += list(map(msstore.fs.unstrip_protocol, mslist))
        except Exception:
            log.error_and_raise(f"No MS at {ms_path}", ValueError)
    ms = msnames
    opts_dict["ms"] = ms

    if gain_table is not None:
        gainnames = []
        for gt in gain_table:
            gainstore = DaskMSStore(str(gt).rstrip("/"))
            gtlist = gainstore.fs.glob(str(gt).rstrip("/"))
            try:
                assert len(gtlist) > 0
                gainnames += list(map(gainstore.fs.unstrip_protocol, gtlist))
            except Exception:
                log.error_and_raise(f"No gain table at {gt}", ValueError)
        gain_table = gainnames
        opts_dict["gain_table"] = gain_table

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/imager_{timestamp}.log"
    pfb_logging.log_to_file(logname)

    log.log_options_dict(opts_dict, title="IMAGER options")

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu, log=log)

    ray.init(
        num_cpus=nworkers,
        logging_level="INFO",
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": env_vars,
            "worker_process_setup_hook": setup_ray_worker,
        },
    )

    time_start = time.time()

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

    # DATA is always mapped to VISIBILITY in MSv4, so we need to rename it here
    if dc1 == "DATA":
        dc1 = "VISIBILITY"
    if dc2 == "DATA":
        dc2 = "VISIBILITY"

    # figure out where band edges are
    # note mapping currently maps partitions to the band it has most overlap with
    # partitions are not sub-divided
    all_freqs = []
    all_chan_widths = []
    max_blength = 0
    for ims, ms_name in enumerate(ms):
        dt_kwargs = get_engine(ms_name)
        if "file://" in ms_name:
            ms_name = ms_name.replace("file://", "")
        dt = xr.open_datatree(
            ms_name,
            **dt_kwargs,
        )
        for node in dt.children.values():
            if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
                continue
            ds = node.ds.sel(frequency=slice(freq_min, freq_max))
            if ds.frequency.size == 0:
                continue
            field_name = np.unique(ds.field_name.load().values).item()  # partitioned by FIELD_ID → single value
            scan_name = np.unique(ds.scan_name.load().values).item()  # partitioned by SCAN_NUMBER → single value
            spw_name = ds.frequency.attrs["spectral_window_name"]  # always in partition schema → single value
            # skip if data not in selection
            if (field_names is not None) and (field_name not in field_names):
                continue
            if (spw_names is not None) and (spw_name not in spw_names):
                continue
            if (scan_names is not None) and (scan_name not in scan_names):
                continue
            all_freqs.append(ds.frequency.values)
            all_chan_widths.append(ds.frequency.attrs["channel_width"]["data"])
            # xarray-ms establishes a regular grid over irregular or missing data.
            # Nans are inserted in these cases, mostly because xarray interprets nans as missing data.
            # This is different from xarray-kat which inherits katdal's missing data behaviour
            # (zeroed visibilities and weights).
            uvw = ds.UVW.load().values
            uvw_mask = np.isnan(uvw).all(axis=-1)
            # this forces a reshape (t, bl, 3) -> (row, 3)
            uvw = uvw[~uvw_mask]
            if uvw.size:
                max_blength = max(max_blength, np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max())

    # guard against irregular channel widths
    cw = np.asarray(all_chan_widths)
    cw = cw[np.isfinite(cw)]
    if cw.size == 0:
        log.error_and_raise("No SPW has a usable channel_width", ValueError)
    min_chan_width = np.min(cw)
    if channels_per_image in (0, None, -1):
        nband = len(np.unique([f.tobytes() for f in all_freqs]))  # one per spw
    else:
        flat_freqs = np.concatenate(all_freqs)
        nband = int(np.ceil((flat_freqs.max() - flat_freqs.min()) / (min_chan_width * channels_per_image)))
        nband = max(nband, 1)
    all_freqs = np.unique(np.concatenate(all_freqs))
    log.info(f"Number of output bands determined to be {nband} based on channel width and freq range")
    band_edges = np.linspace(all_freqs.min() - min_chan_width / 2, all_freqs.max() + min_chan_width / 2, nband + 1)
    half_band_width = (band_edges[1] - band_edges[0]) / 2
    freq_out = band_edges[0:-1] + half_band_width

    tasks = []
    scan_block_to_tid = {}  # (scan_name, block_idx) -> tid
    next_tid = 0
    for ims, ms_name in enumerate(ms):
        dt_kwargs = get_engine(ms_name)
        if "file://" in ms_name:
            ms_name = ms_name.replace("file://", "")
        dt = xr.open_datatree(
            ms_name,
            **dt_kwargs,
        )
        for node in dt.children.values():
            if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
                continue

            ds = node.ds.sel(frequency=slice(freq_min, freq_max))
            if ds.frequency.size == 0:
                continue
            field_name = np.unique(ds.field_name.load().values).item()  # partitioned by FIELD_ID → single value
            scan_name = np.unique(ds.scan_name.load().values).item()  # partitioned by SCAN_NUMBER → single value
            spw_name = ds.frequency.attrs["spectral_window_name"]  # always in partition schema → single value
            # skip if data not in selection
            if (field_names is not None) and (field_name not in field_names):
                continue
            if (spw_names is not None) and (spw_name not in spw_names):
                continue
            if (scan_names is not None) and (scan_name not in scan_names):
                continue

            freqs_node = ds.frequency.load().values
            times_node = ds.time.load().values
            nchan_node = freqs_node.size
            ntimes_node = times_node.size
            if integrations_per_image in (0, None, -1):
                ipi_node = ntimes_node
            else:
                ipi_node = integrations_per_image
            if channels_per_image in (0, None, -1):
                cpi_node = nchan_node
            else:
                cpi_node = channels_per_image
            for tlow in range(0, ntimes_node, ipi_node):
                thigh = min(tlow + ipi_node, ntimes_node)
                t_index = slice(tlow, thigh)
                key = (scan_name, tlow // ipi_node)
                if key not in scan_block_to_tid:
                    scan_block_to_tid[key] = next_tid
                    next_tid += 1
                timeid = scan_block_to_tid[key]
                for flow in range(0, nchan_node, cpi_node):
                    fhigh = min(flow + cpi_node, nchan_node)
                    nu_index = slice(flow, fhigh)
                    bandid = int(np.argmin(np.abs(freq_out - freqs_node[nu_index].mean())))

                    # slice out subset of node
                    subdt = node.isel(time=t_index, frequency=nu_index)

                    fut = safe_stokes_vis.remote(
                        dc1=dc1,
                        dc2=dc2,
                        operator=operator,
                        node_dt=subdt,
                        xds_store=xds_store.url,
                        bandid=bandid,
                        timeid=timeid,
                        msid=ims,
                        freq_out=freq_out[bandid],
                        precision=precision,
                        sigma_column=sigma_column,
                        weight_column=weight_column,
                        product=product,
                        chan_average=chan_average,
                        bda_decorr=bda_decorr,
                        max_field_of_view=max_field_of_view,
                        beam_model=beam_model,
                        wgt_mode=wgt_mode,
                        max_blength=max_blength,
                        max_freq=all_freqs.max(),
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

    if not keep_ray_alive:
        ray.shutdown()

    return


def get_engine(ms_path: str) -> dict[str, Any]:
    if "file://" in ms_path:
        ms_path = ms_path.replace("file://", "")
    backend = infer_backend(ms_path)
    if backend == MSv4Backend.CASA_TABLE:
        import xarray_ms  # noqa: F401

        return {
            "engine": "xarray-ms:msv2",
            "partition_schema": ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
        }
    elif backend == MSv4Backend.ZARR:
        return {
            "engine": "zarr",
            "chunks": None,
        }
    elif backend == MSv4Backend.MEERKAT:
        import xarray_kat  # noqa: F401

        return {
            "engine": "xarray-kat",
            "applycal": "all",
            "chunked_array_type": "xarray-kat",
            "chunks": {},
            "uvw_sign_convention": "casa",
        }
    else:
        raise ValueError(f"Unhandled MSv4 backend {backend!r} for {ms_path}")
