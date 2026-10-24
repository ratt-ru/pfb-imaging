"""`pfb degrid`: degrid a `.mds` component model into MSv4 data (#278).

The MSv4 <-> pfb-model-spec seam lives in `utils/degrid.py` and is pure;
this module owns the guards, the Ray Serve deployment and the driver.

It replaced the MSv2/dask-ms `degrid` in #330, taking over its name. That
retired the codebase's last consumer of `distributed`, but it is not a
drop-in: the selection options are MSv4 names rather than MSv2 integer ids
(`--scan-names`/`--spw-names`/`--field-names` for `--scans`/`--ddids`/
`--fields`), chunking is `--integrations-per-chunk`/`--channels-per-chunk`
rather than `..._per_image`, and the cluster address is `--ray-address`
rather than `--host-address`. Recipes written against the old command fail
with an unrecognised-option error rather than silently doing the wrong thing.
"""

import asyncio
import gc
import logging
import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import psutil
import xarray as xr
from ducc0.misc import resize_thread_pool
from meerkat_beams.utils import BeamWizard, set_console_logging_level
from pfb_model_spec.utils.degrid import RegionMask, model_geometry, model_sampling
from rarg_python_patterns.multiton import Multiton
from ray import serve

from pfb_imaging import init_ray, set_envs, setup_ray_worker
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.baselines import (
    GROUP_LABELS,
    node_baseline_groups,
    parse_antenna_groups,
    preflight_baseline_groups,
    resolve_group_beam_cache,
)
from pfb_imaging.utils.beam import BEAM_FREQ_EXTRAPOLATION
from pfb_imaging.utils.degrid import (
    SPECTRAL,
    assert_writable,
    build_region_masks,
    check_stokes_subset,
    check_writable_backend,
    chunk_eval_freqs,
    default_channels_per_chunk,
    degrid_region,
    ensure_model_columns,
    flux_scale_warning,
)
from pfb_imaging.utils.memprof import PhaseClock, dump_timeline, format_memory, profile_env, profile_task, task_memory
from pfb_imaging.utils.msv4 import (
    SelectedNode,
    get_engine,
    ignore_xarray_ms_warnings,
    select_vis_nodes,
    wrapped_angle_diff,
)
from pfb_imaging.utils.naming import glob_uris, set_output_names

ignore_xarray_ms_warnings()

log = pfb_logging.get_logger("DEGRID")

# Ray Serve application name. Used for both serve.run and the targeted
# serve.delete that replaces a cluster-wide serve.shutdown.
_SERVE_APP_NAME = "degrid"

# Requests a replica will accept at once: one running, one ready. The driver's
# in-flight bound is derived from it (see `max_inflight`), and the two must
# stay tied -- a request the replicas cannot accept sits in Serve's router
# being retried on a 0.5 s backoff, and at attempt 1024 (~510 s of waiting)
# the router's own backoff arithmetic overflows and kills the routing task
# that owns it (docs/msv4_issues.md, "Adjacent, not MSv4").
_MAX_ONGOING_REQUESTS = 2


def _default_mds(output_filename: str, suffix: str) -> str:
    """Locate the component model when `--mds` was not given.

    Two producers write a `.mds` with different names, and both are legitimate
    inputs here:

    * `pfb deconv` writes `{basename}_{suffix}.mds` (`core/deconv.py`).
    * `pfbspec model2comps` writes `{basename}_{suffix}_{model_name}.mds`,
      defaulting to `..._model.mds`.

    The legacy `degrid` only ever looked for the second, so the common
    `imager -> deconv -> degrid` path never worked without an explicit
    `--mds`. Prefer `deconv`'s name, fall back to `model2comps`'.

    Args:
        output_filename: Basename, already carrying the `_PRODUCT` suffix.
        suffix: The `--suffix` value.

    Returns:
        The path that exists.

    Raises:
        ValueError: If neither candidate exists, naming both.
    """
    fs = fsspec.filesystem("file")
    candidates = [f"{output_filename}_{suffix}.mds", f"{output_filename}_{suffix}_model.mds"]
    for candidate in candidates:
        if fs.exists(candidate):
            return candidate
    raise ValueError(f"No mds found. Looked for {' and '.join(candidates)}. Pass --mds to name one explicitly.")


def check_model(model_ds: xr.Dataset, product: str, *, beam: bool = False) -> dict:
    """Validate the `.mds` and the requested product before any work starts.

    `model_geometry` already raises on an unknown `spec` and on non-square
    pixels; calling it here means those fire before a Ray cluster is stood up
    and before a column is added to the user's MS. Without a beam the product
    must be a subset of the model's Stokes planes (`check_stokes_subset`).

    Args:
        model_ds: An opened `.mds` dataset.
        product: The `--product` string.
        beam: Whether `--beam-model` is set, which lets the product name planes
            the model lacks (leakage, D55).

    Returns:
        `model_geometry(model_ds)`: nx, ny, cell_rad, x0, y0, flip_u/v/w, stokes.

    Raises:
        ValueError: From `check_stokes_subset`, or if `model_geometry` rejects
            the spec or the pixel shape.
    """
    geom = model_geometry(model_ds)  # raises on unknown spec / non-square pixels
    check_stokes_subset(product, geom["stokes"], beam=beam)
    return geom


def check_tangent_point(
    model_ds: xr.Dataset,
    nodes: list[SelectedNode],
    tol_rad: float = 1e-9,
) -> None:
    """Refuse to degrid a model against a field it was not made for.

    A model on a different tangent point needs a per-row inverse w-phase to be
    predicted correctly (wiki D21, mosaics). v1 does not do that, so this is a
    refusal rather than a warning.

    Compared as a **wrapped magnitude**: a naive signed difference silently
    accepts half of all real mismatches, and a bare `abs(a - b)` reads two RAs
    either side of zero as ~2*pi apart.

    Args:
        model_ds: An opened `.mds` dataset, carrying `ra`/`dec` attrs.
        nodes: Selected visibility nodes, each with a `field_radec`.
        tol_rad: Tolerance in radians.

    Raises:
        ValueError: If any node's field centre differs from the model's.
    """
    model_radec = np.array([float(model_ds.ra), float(model_ds.dec)])
    for node in nodes:
        sep = wrapped_angle_diff(node.field_radec, model_radec)
        if np.any(sep > tol_rad):
            raise ValueError(
                f"Tangent point mismatch for field {node.field_name!r} in "
                f"{node.path}: field is (ra, dec) = "
                f"{np.rad2deg(node.field_radec)} deg, model is "
                f"{np.rad2deg(model_radec)} deg (separation "
                f"{np.rad2deg(sep)} deg). Degridding a rephased or mosaic "
                "model is not supported in v1 (see wiki D21)."
            )


def _load_model(mds: str) -> xr.Dataset:
    """Load a `.mds` into memory. Module level so a `Multiton` can pickle it."""
    return xr.open_zarr(mds).load()


def _load_masks(mds: str, region_file: str | None) -> list[RegionMask]:
    """Rebuild the region masks in a replica from the paths alone.

    Goes through `Multiton(_load_model, mds)` rather than loading the `.mds`
    again: the key is the one the `model` Multiton uses, so in a replica this
    is a cache hit on the dataset that replica has already loaded, and on the
    driver (which never touches either) it costs nothing.
    """
    return build_region_masks(Multiton(_load_model, mds).instance, region_file)


@dataclass(frozen=True)
class WorkItem:
    """One `(time, frequency)` region of one node of one MS.

    `region` is exactly what `isel` and `to_msv2(region=...)` both consume.
    **Slices only** -- an integer index makes the write raise
    `MismatchedWriteRegion`.
    """

    ms_index: int
    node_path: str
    region: Mapping[str, slice]

    def __hash__(self):
        # Slices only became hashable in Python 3.12, and core code runs on
        # 3.11 (the container is python:3.11-slim; CI covers 3.11-3.13), so a
        # slice cannot go into the hash as itself -- hash its components.
        # Sorted rather than a frozenset so the hash does not depend on the
        # mapping's iteration order.
        return hash(
            (
                self.ms_index,
                self.node_path,
                tuple(sorted((k, v.start, v.stop, v.step) for k, v in self.region.items())),
            )
        )


@serve.deployment
class Degridder:
    """Degrids one region and writes it straight back to its own MS region.

    The write is fused into the compute deliberately: the visibilities are the
    large object here (~413 MB of complex64 for a 100x2016x64x4 chunk), so
    shipping them to a separate writer would cost an object-store round trip
    per item for nothing. We keep tricolour#106's vocabulary -- WorkItem,
    region driving both isel and the write, Multiton, a bounded in-flight
    queue -- and drop its three-way load/compute/write split, which suits a
    read-heavy pipeline rather than this write-heavy one.
    """

    def __init__(
        self,
        datatrees: Sequence[Multiton],
        model: Multiton,
        masks: Multiton,
        columns: Sequence[str],
        stokes: Sequence[str],
        accumulate: bool = False,
        epsilon: float = 1e-5,
        do_wgridding: bool = True,
        nthreads: int = 1,
        beams: Mapping[str, Multiton] | None = None,
        antenna_groups: tuple[str, str] | None = None,
        baseline_groups: bool = False,
        spectral: str = "flat",
    ):
        # Multitons, not DataTrees: each replica reconstructs its own tree
        # rather than receiving a pickled one. The driver must not touch
        # `.instance` before the model columns exist, or replicas inherit a
        # tree that predates them.
        self._datatrees = list(datatrees)
        # The .mds and the masks are Multitons for the same reason, and here it
        # is load bearing rather than tidy: Serve cloudpickles these init args
        # and checkpoints them into the GCS internal KV store, whose gRPC
        # message cap is 512 MiB (RAY_max_grpc_message_size). A .mds is only
        # "small" relative to visibilities -- a real deconvolution of a crowded
        # field runs to millions of components (633 MB for 8.8M of them), and
        # an all-ones 6720^2 mask adds another 361 MB, so binding the loaded
        # objects blows the cap before a single region is degridded (#278).
        self._model = model
        self._masks = masks
        # Dereferenced once per replica on the first work item and held. The
        # Multiton cache has a 300 s *inactivity* TTL, so leaving a 633 MB .mds
        # behind it would let a quiet replica drop and reload it mid-run.
        self._model_ds: xr.Dataset | None = None
        self._mask_list: list[np.ndarray] | None = None
        # Beam wizards are Multitons too (D38: bind recipes, never loaded objects)
        self._beams = beams
        # The driver already built every wizard once, so its meerkat-beams INFO
        # (download, staging, "opening BDS ...") is in the log; a replica's
        # rebuild from the warm cache would only repeat it per replica
        set_console_logging_level(logging.WARNING)
        self._override = antenna_groups
        self._use_groups = baseline_groups
        self._wizards: dict | None = None
        self._node_groups: dict = {}
        self._columns = list(columns)
        self._stokes = list(stokes)
        self._accumulate = accumulate
        self._epsilon = epsilon
        self._do_wgridding = do_wgridding
        self._nthreads = nthreads
        self._spectral = spectral
        # Process-global, so it covers `_degrid`'s worker thread too (measured:
        # a ducc FFT at nthreads=8 runs 0.71 s with the pool at 1 and 0.16 s
        # with it at 8, from the resizing thread and from another thread alike).
        resize_thread_pool(nthreads)
        # One worker thread, and the reason is Serve's user-loop watchdog.
        # `degrid` used to be a sync def, which Serve runs *on* the replica's
        # asyncio loop (RAY_SERVE_RUN_SYNC_IN_THREADPOOL defaults to 0), so a
        # multi-minute item wedged the loop; the watchdog probes every 60 s
        # with a 300 s timeout and kills the replica after 3 misses, which
        # surfaces to the driver as a bare ActorDiedError. max_workers=1 keeps
        # one item in flight per replica -- and keeps arcae's handles on a
        # single thread -- no matter what Serve's concurrency ends up being.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="degrid")

    async def degrid(self, item: WorkItem) -> dict:
        """Degrid and write one region. Returns post-gc memory telemetry.

        Async purely so the replica's event loop stays free to answer health
        probes; all the work happens on `_executor`'s single thread.
        """
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._degrid, item)

    def _degrid(self, item: WorkItem) -> dict:
        with profile_task("degrid") as clock:
            try:
                if self._model_ds is None:
                    self._model_ds = self._model.instance
                    self._mask_list = self._masks.instance
                # dereferenced once and held; a Multiton's TTL is inactivity-based
                if self._wizards is None and self._beams is not None:
                    self._wizards = {k: v.instance for k, v in self._beams.items()}
                dt = self._datatrees[item.ms_index].instance
                groups = None
                if self._use_groups:
                    key = (item.ms_index, item.node_path)
                    if key not in self._node_groups:
                        self._node_groups[key] = node_baseline_groups(dt[item.node_path], self._override)
                    groups = self._node_groups[key]
                node_ds = dt[item.node_path].ds
                write_ds = degrid_region(
                    node_ds,
                    region=item.region,
                    model_ds=self._model_ds,
                    masks=self._mask_list,
                    columns=self._columns,
                    corr_types=tuple(str(p) for p in node_ds.polarization.values),
                    stokes=self._stokes,
                    accumulate=self._accumulate,
                    epsilon=self._epsilon,
                    do_wgridding=self._do_wgridding,
                    nthreads=self._nthreads,
                    wizards=self._wizards,
                    groups=groups,
                    spectral=self._spectral,
                )
                assert_writable(write_ds)
                # region MUST be explicit: the default "auto" expands each dim to
                # slice(0, size) and would write every chunk to the start of the
                # array. write_map is identity so MSV4_WRITE_MAP cannot redirect a
                # column that happens to share an MSv4 variable name.
                write_ds.to_msv2(
                    compute=True,
                    region=dict(item.region),
                    write_map={c: c for c in self._columns},
                )
            finally:
                # Never wipe xarray-ms's class-level Multiton cache here: this
                # deployment keys its model and masks on Multitons, and a wholesale
                # clear reloaded a 633 MB .mds on every work item (wiki
                # memory-and-ray). Reference cycles in deserialised xarray objects
                # still need the collect.
                gc.collect()
        # Measured after the collect, as pass 1 does: post-gc anon that ratchets
        # for a pid across items is retention below Python; shm is Ray
        # object-store pages (wiki memory-and-ray).
        return {**task_memory(), **clock}


def _work_items(ms_index: int, node, integrations_per_chunk: int, channels_per_chunk: int):
    """Yield the `(time, frequency)` regions of one selected node.

    Frequency slices are shifted by `node.chan0` because `--freq-range` trims
    the axis for compute while write regions index the unsliced node.

    Args:
        ms_index: Index into the driver's MS list.
        node: A `SelectedNode`.
        integrations_per_chunk: Times per chunk; `-1`/`0` means the whole node.
        channels_per_chunk: Channels per chunk.

    Yields:
        `WorkItem`s covering the node exactly once.
    """
    ntime = node.ntime
    tstep = ntime if integrations_per_chunk in (-1, 0, None) else int(integrations_per_chunk)
    fstep = int(channels_per_chunk)
    for t in range(0, ntime, tstep):
        for f in range(0, node.nchan, fstep):
            yield WorkItem(
                ms_index=ms_index,
                node_path=node.path,
                region={
                    "time": slice(t, min(t + tstep, ntime)),
                    "frequency": slice(node.chan0 + f, node.chan0 + min(f + fstep, node.nchan)),
                },
            )


@pfb_logging.log_exceptions
def degrid(
    ms: list[Path],
    output_filename: str,
    channels_per_chunk: int | None = None,
    suffix: str = "main",
    mds: str | None = None,
    model_column: str = "MODEL_DATA",
    product: str = "I",
    scan_names: list[str] | None = None,
    spw_names: list[str] | None = None,
    field_names: list[str] | None = None,
    freq_range: str | None = None,
    data_group: str = "base",
    partition_columns: list[str] | None = None,
    auto_corrs: bool = False,
    integrations_per_chunk: int = -1,
    accumulate: bool = False,
    region_file: str | None = None,
    epsilon: float = 1e-5,
    do_wgridding: bool = True,
    beam_model: str | None = None,
    baseline_groups: bool = False,
    antenna_groups: str | None = None,
    spectral: str = "flat",
    ray_address: str = "local",
    nworkers: int = 1,
    nthreads: int | None = None,
    progressbar: bool = True,
    log_directory: str | None = None,
) -> None:
    """Degrid a `.mds` component model into MSv4 measurement sets.

    Args:
        ms: Measurement sets to write to.
        output_filename: Output basename; only used for naming and the default
            `.mds` location -- nothing image-space is written.
        channels_per_chunk: Channels per degridding chunk. Defaults to the
            channels spanning one imaging band, from the model's `freq_bounds`.
        mds: Path to the component model. Defaults to whichever of
            `{output_filename}_{suffix}.mds` (what `deconv` writes) or
            `{output_filename}_{suffix}_model.mds` (what `pfbspec model2comps`
            writes) exists -- see `_default_mds`.
        suffix: Product suffix used to build the default `.mds` path.
        model_column: Column to write. `--region-file` adds
            `{model_column}1`, `{model_column}2`, ... one per region.
        product: Stokes product; must be a subset of the model's `stokes` coord.
        field_names: Field names to degrid into. Defaults to all.
        spw_names: Spectral window names. Defaults to all.
        scan_names: Scan names. Defaults to all.
        freq_range: `'fmin:fmax'` in Hz; either side may be empty.
        data_group: MSv4 data group used to resolve the field_and_source subtable.
        partition_columns: xarray-ms partition schema override.
        integrations_per_chunk: Times per chunk; `-1` means the whole node.
        accumulate: Add to the existing column rather than replacing it.
        region_file: Region file splitting the model across columns (#115).
        epsilon: Gridder accuracy.
        do_wgridding: Perform w-correction via improved w-stacking.
        beam_model: meerkat-beams band name of the primary beam to apply. The
            model must be in intrinsic flux; katbeam is refused.
        baseline_groups: Apply a separate beam per MeerKAT baseline group.
            Requires `beam_model`.
        antenna_groups: Two comma separated glob patterns naming the MeerKAT
            and MeerKAT+ antennas. Requires `baseline_groups`.
        spectral: `"flat"` holds the model and beam at the weighted frequency
            of the band holding each chunk (`flat_freq`), deconv's own forward
            model; `"linear"` interpolates between the apparent model at each
            chunk's end channels (D59).
        ray_address: Ray cluster address, or `"local"` for a private one.
        nworkers: Maximum `Degridder` replicas.
        nthreads: ducc threads per replica. Defaults to half the logical CPUs.
        progressbar: Log per-item progress with memory telemetry.
        log_directory: Directory for the run log.

    Raises:
        ValueError: If `channels_per_chunk` is not positive, `spectral` is
            unknown, no MS matches, the selection matches no data, a beam
            frequency lies outside the beam's range, or any guard in
            `check_model` / `check_tangent_point` fires.
    """
    opts_dict = locals().copy()
    time_start = time.time()
    phases = PhaseClock(log)

    # a negative step other than -1 would reach range(0, ntime, step) as a
    # negative stride, yielding zero work items -- the command would start
    # Serve, write nothing and exit successfully
    if integrations_per_chunk is not None and int(integrations_per_chunk) < -1:
        raise ValueError(
            f"integrations-per-chunk must be -1 (whole partition), 0 (same) or positive, got {integrations_per_chunk}"
        )
    if channels_per_chunk is not None and int(channels_per_chunk) <= 0:
        raise ValueError(f"channels-per-chunk must be positive, got {channels_per_chunk}")
    if spectral not in SPECTRAL:
        raise ValueError(f"spectral must be one of {SPECTRAL}, got {spectral!r}")

    output_filename, _, log_directory, _ = set_output_names(output_filename, product, log_directory=log_directory)
    opts_dict["output_filename"] = output_filename
    opts_dict["log_directory"] = log_directory

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2
    opts_dict["nthreads"] = nthreads

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pfb_logging.log_to_file(f"{log_directory}/degrid_{timestamp}.log")
    log.log_options_dict(opts_dict, title="DEGRID options")
    log.info(f"Using {nworkers} workers with {nthreads} threads per worker")

    # --- resolve inputs -------------------------------------------------
    msnames = []
    for ms_path in ms:
        matches = glob_uris(ms_path)
        if not matches:
            raise ValueError(f"No MS at {ms_path}")
        # strip file:// -- these go on to arcae/python-casacore, which want a
        # plain filesystem path. A remote protocol survives and fails later at
        # the backend check, which is where it should fail.
        msnames += [m.replace("file://", "") for m in matches]

    if mds is None:
        mds = _default_mds(output_filename, suffix)
    elif not fsspec.filesystem("file").exists(mds):
        raise ValueError(f"No mds at {mds}")
    # absolute, because these two paths are now reopened by the replicas rather
    # than shipped to them, and a Ray worker's cwd is its own session directory
    mds = str(Path(mds).resolve())
    if region_file is not None:
        region_file = str(Path(region_file).resolve())
    # The options block above was logged before this defaulting and
    # resolution, so it carries `mds: None` for a defaulted path and the
    # unresolved string for a relative one. These are the paths every replica
    # reopens by name, so name them once they are final.
    log.info(f"Model: {mds}" + (f", regions: {region_file}" if region_file else ""))
    # the driver loads it for the guards below; replicas reopen it themselves
    model_ds = _load_model(mds)

    geom = check_model(model_ds, product, beam=beam_model is not None)
    log.info(
        f"Model grid {geom['nx']}x{geom['ny']} at {np.rad2deg(geom['cell_rad']) * 3600:.4e} arcsec, "
        f"stokes {geom['stokes']}"
    )
    sampling = model_sampling(model_ds)
    if channels_per_chunk is None and sampling["freq_bounds"] is None:
        raise ValueError(
            "channels-per-chunk was not given and the model records no freq_bounds to derive it "
            "from. Pass --channels-per-chunk, or regenerate the .mds with a current pfb deconv."
        )
    if beam_model is None:
        if (msg := flux_scale_warning(sampling["flux_scale"])) is not None:
            log.warning(msg)
        if baseline_groups:
            # without a beam, groups change nothing in a prediction (D47's reasoning)
            log.warning("--baseline-groups does nothing without --beam-model; predicting one group")
    else:
        if str(beam_model).lower() == "katbeam":
            log.error_and_raise(
                "degrid --beam-model needs a meerkat-beams band (for example L): katbeam has no "
                "Stokes Mueller block, and partition_mueller refuses it (meerkat-beams D17).",
                ValueError,
            )
        if sampling["flux_scale"] == "apparent":
            log.error_and_raise(
                "The model is already in apparent flux; applying --beam-model would attenuate it twice.",
                ValueError,
            )
        if sampling["flux_scale"] is None:
            log.warning("The model records no flux_scale; assuming intrinsic flux and applying the beam.")
    if antenna_groups is not None:
        if not baseline_groups:
            # Refuse rather than ignore, as the imager does
            log.error_and_raise(
                "--antenna-groups only has an effect with --baseline-groups, which is not set. "
                "Add --baseline-groups, or drop --antenna-groups.",
                ValueError,
            )
        # parsed even when --baseline-groups is dropped for lack of a beam
        try:
            parse_antenna_groups(antenna_groups)
        except ValueError as e:
            log.error_and_raise(str(e), ValueError)
    use_groups = beam_model is not None and baseline_groups
    group_override = (
        preflight_baseline_groups(msnames, partition_columns, beam_model, antenna_groups) if use_groups else None
    )

    freq_min, freq_max = -np.inf, np.inf
    if freq_range:
        fmin, fmax = freq_range.strip().split(":")
        freq_min = float(fmin) if fmin else -np.inf
        freq_max = float(fmax) if fmax else np.inf

    # built here only to count the columns and to fire the overlapping-region
    # guard before a Ray cluster exists; the replicas build their own
    masks = build_region_masks(model_ds, region_file)
    columns = [model_column] + [f"{model_column}{i}" for i in range(1, len(masks))]
    log.info(f"Writing {len(columns)} column(s): {', '.join(columns)}")

    # --- guards, then the beam preflight, then column creation ----------
    selected: list[tuple[int, SelectedNode]] = []
    present_groups: set[str] = set()
    for ims, ms_name in enumerate(msnames):
        # before opening anything: degrid writes, and only the CASA backend can
        # be written back to
        check_writable_backend(ms_name)
        # metadata only, closed before any replica starts: one MAIN instance
        # loses no parallelism and matches the column pass below
        dt_kwargs = get_engine(ms_name, partition_columns, auto_corrs=auto_corrs, main_ninstances=1)
        dt = xr.open_datatree(ms_name, **dt_kwargs)
        try:
            nodes = select_vis_nodes(
                dt,
                data_group=data_group,
                field_names=field_names,
                spw_names=spw_names,
                scan_names=scan_names,
                freq_min=freq_min,
                freq_max=freq_max,
            )
            if not nodes:
                continue
            check_tangent_point(model_ds, nodes)
            if use_groups:
                for node in nodes:
                    try:
                        present_groups |= set(node_baseline_groups(dt[node.path], group_override)[0])
                    except ValueError as e:
                        log.error_and_raise(str(e), ValueError)
            selected += [(ims, node) for node in nodes]
        finally:
            dt.close()

    if not selected:
        raise ValueError("Selection matched no data")

    beams = None
    if beam_model is not None:
        if use_groups:
            labels = sorted(present_groups, key=GROUP_LABELS.index)
            if (missing := resolve_group_beam_cache(str(beam_model).upper(), labels)) is not None:
                log.error_and_raise(missing, RuntimeError)
            log.info(f"Beam {beam_model} for baseline groups {', '.join(labels)}")
            kwargs = {g: {"band": beam_model, "group": g} for g in labels}
        else:
            log.info(f"Beam {beam_model}")
            kwargs = {"all": {"band": beam_model}}
        # band-edge chunks lie up to one BDS channel outside the beam cube (D55)
        kwargs = {label: {**kw, "freq_extrapolation": BEAM_FREQ_EXTRAPOLATION} for label, kw in kwargs.items()}
        # Build each wizard once here, single process: meerkat-beams does not
        # guard concurrent first downloads, and a staging error should stop the
        # run before Serve starts. Replicas rebuild their own from the warmed
        # cache through a Multiton (D38: bind recipes, never loaded objects).
        driver_wizards = {label: BeamWizard(**kw) for label, kw in kwargs.items()}
        beams = {label: Multiton(BeamWizard, **kw) for label, kw in kwargs.items()}

    # Build the work list here, before any column exists and before Ray: a bad
    # chunk spec, or a chunk whose beam the wizard cannot evaluate, must stop
    # the run with every MS untouched.
    def node_fstep(node) -> int:
        if channels_per_chunk is not None:
            return int(channels_per_chunk)
        return default_channels_per_chunk(sampling["freq_bounds"], node.chan_width)

    if channels_per_chunk is None:
        derived = sorted({node_fstep(node) for _, node in selected})
        log.info(f"Derived channels-per-chunk {derived} from the model's band width")
    items = [
        item for ims, node in selected for item in _work_items(ims, node, integrations_per_chunk, node_fstep(node))
    ]
    log.info(f"Degridding {len(items)} chunks over {len(selected)} partition(s)")

    if beams is not None:
        # Exactly the frequencies the replicas will evaluate the beam at,
        # through the same helper they use (D59)
        beam_freqs = []
        for _, node in selected:
            fstep = node_fstep(node)
            for f in range(0, node.nchan, fstep):
                beam_freqs += chunk_eval_freqs(model_ds, node.freq[f : f + fstep], spectral)
        for label, wizard in driver_wizards.items():
            try:
                wizard.check_freqs(np.asarray(beam_freqs, dtype=float))
            except ValueError as e:
                log.error_and_raise(f"beam {label}: {e}", ValueError)

    # Columns last, so a failed guard or beam leaves every MS untouched. One
    # MAIN instance: with several, arcae can answer the follow-up reads from one
    # that has not seen the new column (see get_engine).
    for ims in sorted({ims for ims, _ in selected}):
        ms_name = msnames[ims]
        dt = xr.open_datatree(
            ms_name, **get_engine(ms_name, partition_columns, auto_corrs=auto_corrs, main_ninstances=1)
        )
        try:
            ensure_model_columns(ms_name, dt, columns)
        finally:
            # LOAD BEARING, do not tidy away: newly added columns are invisible
            # to other processes until the table is closed (CTDS property), and
            # every Ray replica is another process.
            dt.close()

    # No cache eviction after creating columns: arcae >= 0.4.0a13 resyncs table
    # handles that were already open when a column was added (ska-sa/arcae#241,
    # pinned by test_handles_open_before_a_column_is_added_still_read).

    # --- distribute -----------------------------------------------------
    # model_ds and masks are dead once the guards above have run, and since the
    # bind site now passes Multitons rather than the objects themselves they
    # are no longer needed to build the app. Drop them before Serve starts: at
    # 6720^2 that is 633 MB plus 361 MB per mask the driver would otherwise
    # hold for the whole run while replicas compete for the same node's memory.
    del model_ds, masks

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu, log=log)
    # +1 CPU for the Serve controller, which would otherwise contend with the
    # replicas for the cluster's only slots. `init_ray` is a no-op (with a
    # warning) if Ray is already up, which is the case under pytest -- hence
    # the nominal replica CPU claim below.
    init_ray(
        nworkers + 1,
        ray_address=ray_address,
        runtime_env={"env_vars": {**env_vars, **profile_env()}, "worker_process_setup_hook": setup_ray_worker},
        log=log,
    )

    datatrees = [
        Multiton(xr.open_datatree, name, **get_engine(name, partition_columns, auto_corrs=auto_corrs))
        for name in msnames
    ]

    app = Degridder.options(
        num_replicas="auto",
        # nominal CPU claim, as BandWorkerPool does: replicas are ducc
        # thread-pool bound, and a real per-replica claim deadlocks scheduling
        # on a small cluster (the test session's is num_cpus=2). max_replicas
        # is what actually caps concurrency.
        ray_actor_options={"num_cpus": 1e-2},
        # One running plus one ready per replica. This MUST live here and not
        # in `autoscaling_config`: that is a pydantic model which silently
        # drops the key, leaving the default of 5 in force -- five items then
        # queue on the replica, and back when `degrid` was sync that wedged
        # the event loop until Serve's watchdog killed the replica.
        #
        # 2 rather than 1 so a replica has its next item in hand the moment it
        # finishes one, without the driver holding requests the router cannot
        # place. Concurrency is NOT what this bounds: `_executor` has a single
        # worker, so a replica still degrids strictly one item at a time. The
        # second request simply awaits that executor, which is harmless now
        # that `degrid` is async and the event loop stays free.
        max_ongoing_requests=_MAX_ONGOING_REQUESTS,
        # Serve's per-request "CALL degrid OK" access lines duplicate the
        # driver's "Completed:" progress line and are streamed to the terminal
        # interleaved with it
        logging_config={"enable_access_log": False},
        autoscaling_config={
            "upscale_delay_s": 1.0,
            "min_replicas": 1,
            "initial_replicas": 1,
            # must not exceed max_ongoing_requests, or a saturated replica
            # never reaches the target and the deployment never scales out
            "target_ongoing_requests": 1,
            "max_replicas": nworkers,
        },
    ).bind(
        datatrees=datatrees,
        model=Multiton(_load_model, mds),
        masks=Multiton(_load_masks, mds, region_file),
        columns=columns,
        stokes=check_stokes_subset(product, geom["stokes"], beam=beam_model is not None),
        accumulate=accumulate,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
        beams=beams,
        antenna_groups=group_override,
        baseline_groups=use_groups,
        spectral=spectral,
    )

    # No HTTP surface at all. route_prefix=None keeps this application off the
    # proxy's routing table, but Serve still starts a ProxyActor per node and
    # that actor binds 127.0.0.1:8000 -- so an unrelated listener on 8000 (a
    # port-forward, another Serve cluster) fails the whole run at startup with
    # `RuntimeError: Failed to bind to address '127.0.0.1:8000'`. We drive the
    # deployment entirely through the handle returned below, so the proxy is a
    # port dependency we never use. location="NoServer" removes it.
    #
    # Serve's own chatter is silenced. The driver-side "ray.serve" logger is
    # held at WARNING for the app's lifetime (start-up, router and delete INFO
    # lines), and at ERROR across start/run, where serve.run re-calls
    # serve_start with its EveryNode default and WARNs that our NoServer is
    # already in force -- benign, and unavoidable through the public API. The
    # controller's lifecycle INFO (replicas added, upscaling) goes through its
    # log_level; its warnings, e.g. a replica slow to start, still show. The
    # "Completed:" telemetry names each replica pid anyway.
    phases.lap("Setup")
    log.info(f"Starting Ray Serve with up to {nworkers} Degridder replica(s)")
    serve_log = logging.getLogger("ray.serve")
    serve_level = serve_log.level
    serve_log.setLevel(logging.ERROR)
    try:
        serve.start(http_options={"location": "NoServer"}, logging_config={"log_level": "WARNING"})
        handle = serve.run(app, name=_SERVE_APP_NAME, route_prefix=None)
    except BaseException:
        serve_log.setLevel(serve_level)
        raise
    serve_log.setLevel(logging.WARNING)
    phases.lap("Serve start")

    try:
        # Bounded in-flight queue: drain the oldest response once more than
        # `max_inflight` are outstanding, so submission cannot outrun the
        # replicas and pile up unwritten regions.
        #
        # Sized to exactly what the replicas can accept. Anything beyond that
        # is not extra throughput -- Serve's router cannot place it, so it
        # spins on a 0.5 s backoff for as long as the queue ahead of it takes,
        # and past ~510 s of that the router's backoff overflows (see
        # `_MAX_ONGOING_REQUESTS`). Items here run for minutes, so a queue
        # deeper than capacity reaches that reliably rather than rarely.
        max_inflight = _MAX_ONGOING_REQUESTS * max(nworkers, 1)
        inflight: deque = deque()
        ncomplete = 0

        def drain(target: int) -> None:
            nonlocal ncomplete
            while len(inflight) > target:
                mem = inflight.popleft().result()
                ncomplete += 1
                if progressbar:
                    # a post-gc rss that ratchets for a pid across items means
                    # retention below Python; peak is the lifetime high-water
                    log.info(
                        f"Completed: {ncomplete} / {len(items)} [{format_memory(mem)}]", extra=pfb_logging.NO_MARKUP
                    )

        for item in items:
            # `- 1` because we are about to add one: draining *to* max_inflight
            # and then appending peaks at max_inflight + 1, which is one more
            # than the replicas can accept. That one surplus request is never
            # routable, so it sits in Serve's router being retried until a slot
            # frees -- and if it stays there past ~510 s the router's backoff
            # overflows and kills the routing task holding it. Measured on a
            # real run: exactly one homeless request at a time, rotating, and
            # two OverflowErrors by item 36.
            drain(max_inflight - 1)
            inflight.append(handle.degrid.remote(item))
        drain(0)
        phases.lap("Degridding")
    finally:
        # delete only this application. serve.shutdown() is cluster-wide --
        # it deletes every app and tears down the Serve system actors -- so on
        # a shared --ray-address cluster it would kill unrelated workloads.
        serve.delete(_SERVE_APP_NAME)
        serve_log.setLevel(serve_level)
    phases.lap("Serve delete")

    phases.summary()
    log.info(f"All done after {time.time() - time_start}s.")
    dump_timeline(log)
