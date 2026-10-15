"""pfb hci-msv4: high cadence imaging of MSv4 data with worker actors and an actor writer."""

import gc
import time
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import psutil
import ray
import xarray as xr
from ducc0.misc import resize_thread_pool
from meerkat_beams.utils import BeamWizard

from pfb_imaging import init_ray, set_envs, setup_ray_worker
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.gains import open_term, term_path
from pfb_imaging.utils.hci_actors import CubeWriterImpl, HciWorkerImpl
from pfb_imaging.utils.hci_common import (
    build_scaffold,
    default_phase_dir,
    image_centre_deg,
    write_fits_vars,
    write_transient_spectra,
)
from pfb_imaging.utils.memprof import format_memory
from pfb_imaging.utils.misc import check_gridder_epsilon, set_image_size, to_mjd_time
from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes
from pfb_imaging.utils.naming import glob_uris, uri_and_fs
from pfb_imaging.utils.stokes2im_msv4 import HciConfig

log = pfb_logging.get_logger("HCI-MSV4")

_WHOLE = (-1, 0, None)


@dataclass(frozen=True)
class WorkItem:
    """One output image: a (time, frequency) region of one MSv4 node."""

    ms_index: int
    node_path: str
    time: slice  # full-node indices
    frequency: slice  # full-node indices (already offset by SelectedNode.chan0)
    time_index: int  # into the output TIME axis
    band_index: int  # into the output FREQ axis
    block: int  # time_index // images_per_chunk

    @property
    def region(self) -> dict[str, slice]:
        return {"time": self.time, "frequency": self.frequency}


@dataclass
class HciPlan:
    """The full set of work items plus the output axes and global scalars."""

    items: list[WorkItem]  # sorted by (block, band_index, time_index)
    out_times: np.ndarray  # (ntime,) MJD seconds, sorted unique image mean times
    out_freqs: np.ndarray  # (nband,) Hz, sorted unique band mean frequencies
    expected: dict[tuple[int, int], int]  # (band_index, block) -> images planned in that slab
    channel_width: np.ndarray  # (nband,) max - min channel frequency of the band
    field_radecs: np.ndarray  # (nnode, 2) radians, one row per selected node
    max_blength: float
    max_freq: float
    all_times: np.ndarray  # unique MJD integration times over the selection (transients)
    all_freqs: np.ndarray  # unique channel frequencies of the selected nodes' full spws (transients)


def plan_images(
    dts: Sequence[xr.DataTree],
    *,
    integrations_per_image: int,
    channels_per_image: int,
    images_per_chunk: int,
    data_group: str = "base",
    field_names: list[str] | None = None,
    spw_names: list[str] | None = None,
    scan_names: list[str] | None = None,
    freq_min: float = -np.inf,
    freq_max: float = np.inf,
) -> HciPlan:
    """One WorkItem per output image, with hci's time and band semantics.

    Args:
        dts: Opened MSv4 DataTrees, one per measurement set.
        integrations_per_image: Integrations per image; -1/0/None means one image per scan.
        channels_per_image: Channels per image; -1/0/None means all selected channels.
        images_per_chunk: Output images per time block.
        data_group: Data group used to resolve the field_and_source subtable.
        field_names: Field names to keep; None keeps all.
        spw_names: Spectral window names to keep; None keeps all.
        scan_names: Scan names to keep; None keeps all.
        freq_min: Lower frequency bound in Hz, inclusive.
        freq_max: Upper frequency bound in Hz, inclusive.

    Returns:
        The HciPlan.

    Raises:
        ValueError: on an empty selection, a non-positive integrations_per_image or
            channels_per_image other than the whole-scan/spw sentinels, a non-positive
            images_per_chunk, or two partitions mapping to the same output image.
    """
    if integrations_per_image not in _WHOLE and integrations_per_image < 1:
        raise ValueError(
            f"integrations_per_image must be >= 1, or -1 for one image per scan (got {integrations_per_image})"
        )
    if channels_per_image not in _WHOLE and channels_per_image < 1:
        raise ValueError(
            f"channels_per_image must be >= 1, or -1 for one image per spectral window (got {channels_per_image})"
        )
    if images_per_chunk < 1:
        raise ValueError(f"images_per_chunk must be >= 1 (got {images_per_chunk})")
    raw = []  # (ms_index, path, tslice, fslice, tmean_mjd, fmean, fwidth)
    radecs, all_times, all_freqs = [], [], []
    max_blength, max_freq = 0.0, 0.0
    for ims, dt in enumerate(dts):
        for sel in select_vis_nodes(
            dt,
            data_group=data_group,
            field_names=field_names,
            spw_names=spw_names,
            scan_names=scan_names,
            freq_min=freq_min,
            freq_max=freq_max,
        ):
            node = dt[sel.path]
            tmjd = to_mjd_time(node.ds.time.values)
            freqs = node.ds.frequency.values[sel.chan0 : sel.chan0 + sel.nchan]
            uvw = node.ds.UVW.values  # coordinates only; never a blanket node load
            uvw = uvw[~np.isnan(uvw).any(axis=-1)]
            if uvw.size:
                max_blength = max(max_blength, float(np.hypot(uvw[:, 0], uvw[:, 1]).max()))
            max_freq = max(max_freq, float(freqs.max()))
            radec = np.array(sel.field_radec, dtype=np.float64)
            radec[0] = radec[0] % (2 * np.pi)  # as hci's construct_mappings and read_item
            radecs.append(radec)
            all_times.append(tmjd)
            all_freqs.append(node.ds.frequency.values)  # full spw, unfiltered by freq range, as hci
            ipi = tmjd.size if integrations_per_image in _WHOLE else integrations_per_image
            cpi = freqs.size if channels_per_image in _WHOLE else min(channels_per_image, freqs.size)
            for t0 in range(0, tmjd.size, ipi):
                t1 = min(t0 + ipi, tmjd.size)
                for f0 in range(0, freqs.size, cpi):
                    f1 = min(f0 + cpi, freqs.size)
                    raw.append(
                        (
                            ims,
                            sel.path,
                            slice(t0, t1),
                            slice(sel.chan0 + f0, sel.chan0 + f1),
                            float(np.mean(tmjd[t0:t1])),
                            float(np.mean(freqs[f0:f1])),
                            float(freqs[f0:f1].max() - freqs[f0:f1].min()),
                        )
                    )
    if not raw:
        raise ValueError("Selection matched no data")

    out_times = np.unique([r[4] for r in raw])
    out_freqs = np.unique([r[5] for r in raw])
    channel_width = np.zeros(out_freqs.size)
    items, seen = [], set()
    for ims, path, ts, fs, tm, fm, fw in raw:
        ti = int(np.searchsorted(out_times, tm))
        bi = int(np.searchsorted(out_freqs, fm))
        if (ti, bi) in seen:
            raise ValueError(
                f"Two partitions map to the same output image (TIME {tm}, FREQ {fm}); "
                "select fields/spws/scans so each output image has one source"
            )
        seen.add((ti, bi))
        channel_width[bi] = fw
        items.append(WorkItem(ims, path, ts, fs, ti, bi, ti // images_per_chunk))
    items.sort(key=lambda it: (it.block, it.band_index, it.time_index))
    expected = Counter((it.band_index, it.block) for it in items)
    return HciPlan(
        items=items,
        out_times=out_times,
        out_freqs=out_freqs,
        expected=dict(expected),
        channel_width=channel_width,
        field_radecs=np.asarray(radecs),
        max_blength=max_blength,
        max_freq=max_freq,
        all_times=np.unique(np.concatenate(all_times)),
        all_freqs=np.unique(np.concatenate(all_freqs)),
    )


@pfb_logging.log_exceptions
def hci_msv4(
    ms: list[Path],
    output_dataset: str,
    log_directory: str | None = None,
    product: str = "I",
    scan_names: list[str] | None = None,
    spw_names: list[str] | None = None,
    field_names: list[str] | None = None,
    freq_range: str | None = None,
    overwrite: bool = False,
    transfer_model_from: str | None = None,
    data_column: str = "DATA",
    data_group: str = "base",
    model_column: str | None = None,
    weight_column: str | None = None,
    sigma_column: str | None = None,
    flag_column: str = "FLAG",
    gain_table: list[Path] | None = None,
    max_simul_chunks: int = 2,
    images_per_chunk: int = 128,
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
    robustness: float | None = None,
    target: str | None = None,
    l2_reweight_dof: int | None = None,
    eta: float = 1e-05,
    psf_out: bool = False,
    weight_grid_out: bool = False,
    natural_grad: bool = False,
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
    fits_vars: list[str] | None = None,
    wgt_mode: str = "l2",
    obs_label: str | None = None,
    flag_excess_rms: float = 1.5,
    ray_address: str = "local",
    keep_ray_alive: bool = False,  # not used by CLI
):
    """Produce high cadence residual images from MSv4 data.

    The output store is hci's, up to chunking: a long TIME chunk
    (``images_per_chunk``) is assembled by a single writer actor while
    ``nworkers`` image actors read and grid one image each.
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
    logname = f"{str(log_directory)}/hci_msv4_{timestamp}.log"
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

    # ducc only refuses an unreachable epsilon when it is handed data; say so up front (#340)
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
        # gains.qc/NET or gains.qc::NET; an unmatched pattern is kept so the count
        # check below reports first and open_term then names the missing term
        gainnames = []
        for gt in map(str, gain_table):
            gainnames += glob_uris(term_path(gt)) or [term_path(gt)]
        gain_table = [g.removeprefix("file://") for g in gainnames]
        opts_dict["gain_table"] = gain_table

    log.log_options_dict(opts_dict, title="HCI-MSV4 options")

    # refusals before any I/O-heavy work
    if weight_grid_out:
        log.error_and_raise(
            "--weight-grid-out is not supported by hci-msv4 (hci's scaffold has no weight_grid either)",
            NotImplementedError,
        )
    if transfer_model_from is not None:
        log.error_and_raise("Use the degrid app to populate a model column instead", NotImplementedError)
    if gain_table is not None:
        if len(gain_table) != len(ms):
            log.error_and_raise(f"Need one gain table per MS: got {len(gain_table)} for {len(ms)} MSs", ValueError)
        for g in gain_table:
            open_term(g)  # fail now, not in a worker
    if beam_model is not None and beam_model.lower() != "meerkat-beams":
        log.error_and_raise("Only meerkat-beams is currently supported for beam_model", NotImplementedError)
    if beam_model is not None and primary_beam_band not in ("U", "L", "S0", "S4"):
        log.error_and_raise("primary_beam_band must be one of U, L, S0 or S4", ValueError)

    fds_fs, fds_url = uri_and_fs(output_dataset)
    if fds_fs.exists(fds_url):
        if overwrite:
            log.info(f"Overwriting {output_dataset}")
            fds_fs.rm(fds_url, recursive=True)
        else:
            log.error_and_raise(f"{output_dataset} exists. Set overwrite to overwrite it. ", RuntimeError)
    fds_fs.makedirs(fds_url, exist_ok=True)

    if freq_range is not None:
        fmin, fmax = freq_range.strip(" ").split(":")
        freq_min = float(fmin) if len(fmin) > 0 else -np.inf
        freq_max = float(fmax) if len(fmax) > 0 else np.inf
    else:
        freq_min = -np.inf
        freq_max = np.inf

    log.info("Planning images")
    dts = [xr.open_datatree(m, **get_engine(m)) for m in ms]
    plan = plan_images(
        dts,
        integrations_per_image=integrations_per_image,
        channels_per_image=channels_per_image,
        images_per_chunk=images_per_chunk,
        data_group=data_group,
        field_names=field_names,
        spw_names=spw_names,
        scan_names=scan_names,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    # crude column arithmetic, as hci
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

    # MSv2 column names -> the data group's MSv4 variables
    first = dts[plan.items[0].ms_index][plan.items[0].node_path]
    grp = first.ds.attrs["data_groups"][data_group]

    col_map = {"DATA": grp["correlated_data"], "WEIGHT_SPECTRUM": grp["weight"], "FLAG": grp["flag"]}

    def resolve(name: str | None) -> str | None:
        return None if name is None else col_map.get(name, name)

    wanted = [dc1, dc2, model_column, sigma_column if sigma_column else weight_column, flag_column]
    missing = [n for n in wanted if n is not None and resolve(n) not in first.ds.data_vars]
    if missing:  # fail now, not in a worker
        log.error_and_raise(
            f"{missing} not found in data group {data_group!r} of {first.path}; have {sorted(first.ds.data_vars)}",
            ValueError,
        )

    if sigma_column is not None:
        log.info(f"Initialising weights from {sigma_column} column")
    elif weight_column is not None:
        log.info(f"Using weights from {weight_column} column")
    else:
        log.info("No weights provided, using unity weights")

    # image size and centre
    if phase_dir is None:
        phase_dir = default_phase_dir(plan.field_radecs[:, 0], plan.field_radecs[:, 1])
    nx, ny, nx_psf, ny_psf, cell_n, cell_rad, cell_deg = set_image_size(
        plan.max_blength,
        plan.max_freq,
        field_of_view,
        super_resolution_factor,
        cell_size=cell_size,
        nx=nx,
        ny=ny,
        psf_oversize=psf_relative_size,  # 128 pix if not specified
        log=log,
    )
    out_ra_deg, out_dec_deg = image_centre_deg(
        plan.field_radecs[:, 0], plan.field_radecs[:, 1], phase_dir, target, plan.out_times
    )

    log.info("Creating scaffold for stacked cube")
    scaffold, _ = build_scaffold(
        plan.out_times,
        plan.out_freqs,
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
        spatial_chunk=128,  # hardcode for now, as hci
        images_per_chunk=images_per_chunk,
        obs_label=obs_label,
    )
    scaffold.to_zarr(output_dataset, mode="w", compute=False)
    nstokes = len(set(product))
    nband = plan.out_freqs.size
    log.info(
        f"Cube size = (n_stokes={nstokes}, n_freq={nband}, n_time={plan.out_times.size}, n_y={ny}, n_x={nx}); "
        f"{len(plan.items)} images"
    )

    per_image = nx * ny * (1 + (beam_model is not None)) + (nx_psf * ny_psf if psf_out else 0)
    est = max_simul_chunks * nband * images_per_chunk * per_image * nstokes * 4  # float32 slabs
    log.info(
        f"Writer holds up to {max_simul_chunks} time blocks of {images_per_chunk} images per band: "
        f"~{est / 2**30:.1f} GiB"
    )
    if est > psutil.virtual_memory().available:
        log.warning("Writer estimate exceeds available memory; reduce --images-per-chunk or --max-simul-chunks")

    if inject_transients is not None:
        # needs the full (time, freq) domain, which no single worker sees
        log.info("Computing transient spectra")
        write_transient_spectra(inject_transients, plan.all_times, plan.all_freqs)

    cfg = HciConfig(
        dc1=resolve(dc1),
        dc2=resolve(dc2),
        operator=operator,
        model_column=resolve(model_column),
        weight_column=resolve(weight_column),
        sigma_column=sigma_column,
        flag_column=resolve(flag_column),
        product=product,
        precision=precision,
        nx=nx,
        ny=ny,
        cell_rad=cell_rad,
        psf_relative_size=psf_relative_size,
        phase_dir=phase_dir,
        target=target,
        robustness=robustness,
        min_padding=min_padding,
        filter_counts_level=filter_counts_level,
        npix_super=npix_super,
        channels_per_bin=channels_per_bin,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        double_accum=double_accum,
        natural_grad=natural_grad,
        eta=eta,
        cg_tol=cg_tol,
        cg_maxit=cg_maxit,
        psf_out=psf_out,
        l2_reweight_dof=l2_reweight_dof,
        wgt_mode=wgt_mode,
        inject_transients=inject_transients,
        nthreads=nthreads,
    )
    del dts, first, grp  # the driver needs no open MS handles from here on
    gc.collect()

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu, log=log)
    init_ray(
        nworkers + 2,
        ray_address=ray_address,
        runtime_env={"env_vars": env_vars, "worker_process_setup_hook": setup_ray_worker},
        object_store_memory=None if object_store_memory is None else object_store_memory * 1e9,
        log=log,
    )
    actors = []
    try:
        beam_ref = None
        if beam_model is not None:
            log.info("Using meerkat-beams to initialise BeamWizard")
            beam_ref = ray.put(BeamWizard(image_name=output_dataset, band=primary_beam_band))
        # nominal CPU claims: a real claim can deadlock scheduling (D8)
        writer = ray.remote(num_cpus=1e-2)(CubeWriterImpl).remote(
            output_dataset, plan.expected, images_per_chunk, plan.channel_width, flag_excess_rms
        )
        actors.append(writer)
        workers = [
            ray.remote(num_cpus=1e-2)(HciWorkerImpl).remote(ms, cfg, nx_psf, ny_psf, data_group, gain_table, beam_ref)
            for _ in range(nworkers)
        ]
        actors += workers
        _dispatch(plan, workers, writer, max_simul_chunks)
        ray.get([w.close.remote() for w in workers])
        out = ray.get(writer.finalise.remote())
        log.info(f"Flagged {out['nflagged']} images with excess rms")
    finally:
        # actors would outlive this call under keep_ray_alive; killing one also
        # fails its pending tasks, which is the error path's cancellation
        for actor in actors:
            ray.kill(actor)
        if not keep_ray_alive:
            ray.shutdown()

    if fits_vars is not None:
        write_fits_vars(fds_url, fits_vars, fits_output_folder, images_per_chunk, log)

    log.info(f"All done after {time.time() - time_start}s")


def _dispatch(plan: HciPlan, workers: list, writer, max_open: int) -> None:
    """Deal items to workers with a one-item lookahead; at most ``max_open`` time blocks in the writer.

    Each worker's next item is promised at call time so its prefetch is always for an item it
    will receive. Image refs go straight from worker to writer; the driver only sees telemetry.

    Args:
        plan: The image plan; items are sorted by (block, band, time).
        workers: HciWorkerImpl actor handles.
        writer: The CubeWriterImpl actor handle.
        max_open: Maximum number of time blocks open in the writer at once.

    Raises:
        Exception: The first worker or writer error, re-raised as Ray delivers it.
        RuntimeError: If dispatch stalls with work remaining (a bug).
    """
    queue = deque(plan.items)
    slabs_left = Counter(block for _, block in plan.expected)  # block -> slabs not yet written
    promised: dict[int, WorkItem] = {}
    idle = list(range(len(workers)))
    inflight: dict = {}  # writer ack -> (worker index, item, telemetry ref)
    nitems = len(plan.items)
    ndone = 0

    def allowed(item: WorkItem) -> bool:
        lowest_open = min(slabs_left) if slabs_left else np.iinfo(np.int64).max
        return item.block < lowest_open + max_open

    while queue or promised or inflight:
        for w in list(idle):
            if w in promised:
                item = promised.pop(w)
            elif queue and allowed(queue[0]):
                item = queue.popleft()
            else:
                continue
            nxt = queue.popleft() if queue and allowed(queue[0]) else None
            if nxt is not None:
                promised[w] = nxt
            img_ref, mem_ref = workers[w].image.options(num_returns=2).remote(item, nxt)
            inflight[writer.add.remote(img_ref)] = (w, item, mem_ref)
            idle.remove(w)
        if not inflight:
            raise RuntimeError("hci-msv4 dispatch stalled with work remaining; this is a bug")
        (ack,), _ = ray.wait(list(inflight), num_returns=1)
        w, item, mem_ref = inflight.pop(ack)
        try:
            slab = ray.get(ack)
        except Exception:
            log.error(f"Failed on {item}")
            ray.get(mem_ref)  # re-raises the worker's own error if it failed there
            raise
        idle.append(w)
        ndone += 1
        log.info(f"Imaged: {ndone} / {nitems} [{format_memory(ray.get(mem_ref))}]", extra=pfb_logging.NO_MARKUP)
        if slab is not None:
            slabs_left[slab[1]] -= 1
            if slabs_left[slab[1]] == 0:
                del slabs_left[slab[1]]
