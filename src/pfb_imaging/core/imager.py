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

from pfb_imaging import init_ray, pfb_version, set_envs, setup_ray_worker
from pfb_imaging.operators.gridder import grid_partition
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import rdt2fits
from pfb_imaging.utils.misc import fitcleanbeam, set_image_size
from pfb_imaging.utils.naming import set_output_names
from pfb_imaging.utils.stokes2vis_msv4 import safe_stokes_vis
from pfb_imaging.utils.weighting import box_sum_counts, filter_extreme_counts, reduce_counts

warnings.filterwarnings("ignore", category=IrregularGridWarning)
warnings.filterwarnings("ignore", category=MissingMetadataWarning)
warnings.filterwarnings("ignore", category=FrameConversionWarning)


log = pfb_logging.get_logger("IMAGER")


@ray.remote
def _grid_image(
    scratch_store,
    dt_store,
    image_name,
    counts,
    nx,
    ny,
    nx_psf,
    ny_psf,
    cell_rad,
    freq_out,
    meta,
    robustness=None,
    nx_pad=None,
    ny_pad=None,
    filter_counts_level=5.0,
    npix_super=0,
    nthreads=1,
    epsilon=1e-7,
    do_wgridding=True,
    double_accum=True,
):
    """Pass-2 worker for one output image.

    Groups the image's fine scratch pieces into partitions ``(msid, field, spw,
    baseline_group)``, concatenates scans along ``row``, grids each partition with
    :func:`grid_partition`, sums the image-space products into the band node, and
    writes the band node + ``part####`` children into the ``.dt`` store. Returns a
    light summary (``timeid``/``psf``/``wsum``) for MFS beam-parameter fitting.
    """
    resize_thread_pool(nthreads)
    image = xr.open_datatree(scratch_store, engine="zarr", chunks=None)[image_name]
    pieces = [child.ds.load() for _, child in image.children.items()]

    groups = {}
    for ds in pieces:
        key = (ds.attrs["msid"], ds.attrs["field_name"], ds.attrs["spw_name"], ds.attrs["baseline_group"])
        groups.setdefault(key, []).append(ds)

    # filter/box the reduced counts once per image (grid_partition copies before use)
    if robustness is not None:
        counts = filter_extreme_counts(counts.copy(), level=filter_counts_level)
        counts = box_sum_counts(counts, npix_super)

    corr = pieces[0].corr.values
    ncorr = corr.size
    bpar = ["BMAJ", "BMIN", "BPA"]
    dirty_sum = np.zeros((ncorr, nx, ny))
    psf_sum = np.zeros((ncorr, nx_psf, ny_psf))
    wsum_sum = np.zeros(ncorr)

    for pid, key in enumerate(sorted(groups)):
        plist = groups[key]
        # concat scans of the same partition along row (beam/freq identical across scans)
        if len(plist) == 1:
            part = plist[0]
        else:
            rowvars = ["VIS", "WEIGHT", "MASK", "UVW"]
            cat = xr.concat([p[rowvars] for p in plist], dim="row", coords="minimal", compat="override")
            part = plist[0].drop_vars(rowvars)
            for v in rowvars:
                part[v] = cat[v]

        prod = grid_partition(
            part,
            counts,
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            robustness=robustness,
            nx_pad=nx_pad,
            ny_pad=ny_pad,
            nthreads=nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
        )

        part_out = xr.Dataset(
            {
                "VIS": (("corr", "row", "chan"), part.VIS.values),
                "WEIGHT": (("corr", "row", "chan"), prod["WEIGHT"]),
                "MASK": (("row", "chan"), part.MASK.values),
                "UVW": (("row", "three"), part.UVW.values),
                "FREQ": (("chan",), part.FREQ.values),
                "PSF": (("corr", "x_psf", "y_psf"), prod["PSF"]),
                "PSFHAT": (("corr", "x_psf", "yo2"), prod["PSFHAT"]),
                "BEAM": (("corr", "x", "y"), prod["BEAM"]),
                "PSFPARSN": (("corr", "bpar"), prod["PSFPARSN"]),
            },
            coords={"corr": corr, "bpar": bpar},
            attrs={
                "msid": int(key[0]),
                "field_name": key[1],
                "spw_name": key[2],
                "baseline_group": key[3],
                "ra": meta["ra"],
                "dec": meta["dec"],
                "l0": 0.0,
                "m0": 0.0,
                "wsum": prod["WSUM"].tolist(),
            },
        )
        part_out.to_zarr(dt_store, group=f"{image_name}/part{pid:04d}", mode="a")

        dirty_sum += prod["DIRTY"]
        psf_sum += prod["PSF"]
        wsum_sum += prod["WSUM"]

    band_attrs = {
        "bandid": meta["bandid"],
        "timeid": meta["timeid"],
        "freq_out": float(freq_out),
        "time_out": meta["time_out"],
        "ra": meta["ra"],
        "dec": meta["dec"],
        "cell_rad": cell_rad,
        "robustness": robustness,
        "niters": 0,
    }
    band_attrs = {k: v for k, v in band_attrs.items() if v is not None}
    band_ds = xr.Dataset(
        {
            "DIRTY": (("corr", "x", "y"), dirty_sum),
            "RESIDUAL": (("corr", "x", "y"), dirty_sum.copy()),  # no model yet
            "PSF": (("corr", "x_psf", "y_psf"), psf_sum),
            "PSFPARSN": (("corr", "bpar"), np.array(fitcleanbeam(psf_sum / wsum_sum[:, None, None]))),
            "WSUM": (("corr",), wsum_sum),
        },
        coords={"corr": corr, "bpar": bpar},
        attrs=band_attrs,
    )
    band_ds.to_zarr(dt_store, group=image_name, mode="a")

    return {"timeid": meta["timeid"], "psf": psf_sum, "wsum": wsum_sum}


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
    weight_grouping: str = "per-band-time",
    robustness: float | None = None,
    field_of_view: float | None = None,
    super_resolution_factor: float = 2.0,
    cell_size: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    psf_oversize: float = 1.4,
    filter_counts_level: float = 5.0,
    npix_super: int = 0,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    keep_scratch: bool = True,
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
    ray_address: str = "local",
    keep_ray_alive: bool = False,  # not used by CLI
):
    """
    Initialise Stokes data products for imaging
    """
    # for logging options
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

    init_ray(
        nworkers,
        ray_address=ray_address,
        runtime_env={
            "env_vars": env_vars,
            "worker_process_setup_hook": setup_ray_worker,
        },
        log=log,
    )

    time_start = time.time()

    basename = f"{output_filename}"

    # pass-1 fine averaged Stokes pieces are written into a .scratch DataTree
    scratch_store = DaskMSStore(f"{basename}.scratch")
    if scratch_store.exists():
        if overwrite:
            log.info(f"Overwriting {basename}.scratch")
            scratch_store.rm(recursive=True)
        else:
            log.error_and_raise(f"{basename}.scratch exists. Set overwrite to overwrite it. ", RuntimeError)

    fs = fsspec.filesystem(scratch_store.protocol)
    fs.makedirs(scratch_store.url, exist_ok=True)

    log.info(f"Pass-1 scratch products will be stored in {scratch_store.url}")

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

    # shared imaging geometry (also fixes the padded uv-grid used for COUNTS)
    max_freq = float(all_freqs.max())
    nx, ny, nx_psf, ny_psf, cell_n, cell_rad, cell_deg = set_image_size(
        max_blength, max_freq, field_of_view, super_resolution_factor, cell_size, nx, ny, psf_oversize
    )
    min_padding = 1.7
    nx_pad = int(np.ceil(min_padding * nx))
    nx_pad += nx_pad % 2
    ny_pad = int(np.ceil(min_padding * ny))
    ny_pad += ny_pad % 2
    log.info(f"Image size (nx={nx}, ny={ny}), cell={np.rad2deg(cell_rad) * 3600:.4e} arcsec")

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
                        scratch_store=scratch_store.url,
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
                        max_freq=max_freq,
                        nx_pad=nx_pad,
                        ny_pad=ny_pad,
                        cell_rad=cell_rad,
                        baseline_group="all",
                    )
                    tasks.append(fut)

    nds = len(tasks)
    ncomplete = 0
    remaining_tasks = tasks.copy()
    bandids_out = []
    timeids_out = []
    while remaining_tasks:
        # Wait for at least 1 task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

        # Process the completed task
        for task in ready:
            result = ray.get(task)
            if result is not None:
                bandids_out.append(result[0])
                timeids_out.append(result[1])
            ncomplete += 1
            if progressbar:
                print(f"Completed: {ncomplete} / {nds}", end="\n", flush=True)

    ntime = len(set(timeids_out))
    nband_out = len(set(bandids_out))

    log.info(f"Pass 1 wrote fine pieces for {nband_out} bands and {ntime} time chunks to {scratch_store.url}")
    log.info(f"Pass 1 done after {time.time() - time_start}s")

    # ---- between passes: accumulate per-(band,time) counts and reduce ----
    log.info(f"Reducing uv counts with grouping '{weight_grouping}'")
    scratch_dt = xr.open_datatree(scratch_store.url, engine="zarr", chunks=None)
    counts_map = {}
    image_meta = {}
    ncorr = None
    for name in scratch_dt.children:
        if not name.startswith("band"):
            continue
        pieces = [child.ds for _, child in scratch_dt[name].children.items()]
        if not pieces:
            continue
        bandid = int(pieces[0].attrs["bandid"])
        timeid = int(pieces[0].attrs["timeid"])
        ncorr = pieces[0].corr.size
        acc = None
        for ds in pieces:
            c = ds.COUNTS.values
            acc = c.copy() if acc is None else acc + c
        counts_map[(bandid, timeid)] = acc
        image_meta[name] = {
            "bandid": bandid,
            "timeid": timeid,
            "ra": float(pieces[0].attrs["ra"]),
            "dec": float(pieces[0].attrs["dec"]),
            "time_out": float(np.mean([ds.attrs["time_out"] for ds in pieces])),
        }
    if not image_meta:
        log.error_and_raise("Pass 1 produced no output images (all data flagged?)", RuntimeError)
    reduced = reduce_counts(counts_map, weight_grouping)

    # ---- initialise the .dt store root ----
    dt_store = DaskMSStore(f"{basename}.dt")
    if dt_store.exists():
        if overwrite:
            dt_store.rm(recursive=True)
        else:
            log.error_and_raise(f"{basename}.dt exists. Set overwrite to overwrite it.", RuntimeError)
    fs.makedirs(dt_store.url, exist_ok=True)
    root_attrs = {
        "pfb-imaging-version": pfb_version,
        "product": product,
        "nband": int(nband),
        "ntime": int(ntime),
        "nx": int(nx),
        "ny": int(ny),
        "nx_psf": int(nx_psf),
        "ny_psf": int(ny_psf),
        "cell_rad": float(cell_rad),
        "max_blength": float(max_blength),
        "max_freq": float(max_freq),
    }
    xr.Dataset(attrs=root_attrs).to_zarr(dt_store.url, mode="w")
    log.info(f"Imaging products will be written to {dt_store.url}")

    # ---- pass 2: grid each output image into the .dt tree (parallel over images) ----
    tasks = []
    for name, meta in image_meta.items():
        fut = _grid_image.remote(
            scratch_store.url,
            dt_store.url,
            name,
            reduced[(meta["bandid"], meta["timeid"])],
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            freq_out[meta["bandid"]],
            meta,
            robustness=robustness,
            nx_pad=nx_pad,
            ny_pad=ny_pad,
            filter_counts_level=filter_counts_level,
            npix_super=npix_super,
            nthreads=nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
        )
        tasks.append(fut)

    psf_mfs = {}
    wsum_mfs = {}
    nds = len(tasks)
    ncomplete = 0
    remaining_tasks = tasks.copy()
    while remaining_tasks:
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        for task in ready:
            res = ray.get(task)
            tid = res["timeid"]
            psf_mfs.setdefault(tid, np.zeros((ncorr, nx_psf, ny_psf)))
            psf_mfs[tid] += res["psf"]
            wsum_mfs.setdefault(tid, np.zeros(ncorr))
            wsum_mfs[tid] += res["wsum"]
            ncomplete += 1
            if progressbar:
                print(f"Gridded: {ncomplete} / {nds}", end="\n", flush=True)

    # MFS beam parameters per time chunk (from the wsum-normalised MFS PSF)
    psfparsn = {}
    for tid in psf_mfs:
        psfparsn[tid] = np.array(fitcleanbeam(psf_mfs[tid] / wsum_mfs[tid][:, None, None]))

    # ---- FITS ----
    if fits_mfs or fits_cubes:
        fits_oname = f"{fits_output_folder}/{oname}"
        log.info(f"Writing fits files to {fits_oname}")
        fits_tasks = [
            rdt2fits.remote(
                dt_store.url,
                column,
                fits_oname,
                norm_wsum=True,
                nthreads=nthreads,
                do_mfs=fits_mfs,
                do_cube=fits_cubes,
                psfpars_mfs=psfparsn,
            )
            for column in ("DIRTY", "PSF", "RESIDUAL")
        ]
        for task in fits_tasks:
            ray.get(task)

    if not keep_scratch:
        log.info(f"Removing scratch store {scratch_store.url}")
        scratch_store.rm(recursive=True)

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
