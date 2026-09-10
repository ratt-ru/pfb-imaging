"""`pfb degrid-msv4`: degrid a `.mds` component model into MSv4 data (#278).

The MSv4 <-> pfb-model-spec seam lives in `utils/degrid_msv4.py` and is pure;
this module owns the guards, the Ray Serve deployment and the driver. It is
the MSv4 replacement for `core/degrid.py`, which is the codebase's last
consumer of `distributed`.

Naming: the user-facing sub-command is `degrid-msv4` (registered in
`cli/__init__.py`), but the stimela cab is `degrid_msv4`. hip-cargo defines a
cab's name to be the CLI function's `__name__` (`hip_cargo/utils/spec.py`) and
`tests/test_roundtrip.py` regenerates each CLI module from its cab to check it,
so a hyphen there cannot round-trip. Container execution is unaffected:
`run_in_container` replays `sys.argv`, so `--backend docker` invokes
`pfb degrid-msv4` exactly as typed.
"""

import gc
import os
import resource
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import psutil
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool
from pfb_model_spec.utils.degrid import model_geometry
from rarg_python_patterns.multiton import Multiton
from ray import serve

from pfb_imaging import init_ray, set_envs, setup_ray_worker
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.degrid_msv4 import (
    assert_writable,
    build_region_masks,
    degrid_region,
    ensure_model_columns,
)
from pfb_imaging.utils.msv4 import SelectedNode, get_engine, select_vis_nodes, wrapped_angle_diff
from pfb_imaging.utils.naming import set_output_names
from pfb_imaging.utils.stokes2vis_msv4 import _release_ms_caches

log = pfb_logging.get_logger("DEGRID_MSV4")


def check_model(model_ds: xr.Dataset, product: str) -> dict:
    """Validate the `.mds` and the requested product before any work starts.

    `model_geometry` already raises on an unknown `spec` and on non-square
    pixels; calling it here means those fire before a Ray cluster is stood up
    and before a column is added to the user's MS.

    The product checks are stricter than the old `degrid`'s and deliberately
    so. The `genesis` spec stores a single Stokes plane, but the old command
    took `len(product)` planes and degridded the same image into each, so
    `--product IQ` emitted `XX = 2I, YY = 0`. Refusing is the fix.

    Args:
        model_ds: An opened `.mds` dataset.
        product: The `--product` string.

    Returns:
        `model_geometry(model_ds)`: nx, ny, cell_rad, x0, y0, flip_u/v/w, stokes.

    Raises:
        ValueError: If the product is not a subset of IQUV, names more than one
            Stokes product, or disagrees with the model's own `stokes` attr; or
            if `model_geometry` rejects the spec or the pixel shape.
    """
    product = product.upper().strip()
    remainder = product.strip("IQUV")
    if remainder:
        raise ValueError(f"Product {remainder} not yet supported")
    if len(product) != 1:
        raise ValueError(
            f"Product {product!r} names {len(product)} Stokes products, but the "
            "'genesis' .mds spec carries a single Stokes plane. Degrid one "
            "product at a time until pfb-model-spec#19 lands."
        )

    geom = model_geometry(model_ds)  # raises on unknown spec / non-square pixels
    if geom["stokes"].upper() != product:
        raise ValueError(
            f"Requested product {product!r} but the model's stokes attr is "
            f"{geom['stokes']!r}. Degridding a model as a different Stokes "
            "product produces confidently wrong correlations."
        )
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
        return hash((self.ms_index, self.node_path, frozenset(self.region.items())))


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
        model_ds: xr.Dataset,
        masks: Sequence[np.ndarray],
        columns: Sequence[str],
        accumulate: bool = False,
        epsilon: float = 1e-7,
        do_wgridding: bool = True,
        nthreads: int = 1,
    ):
        # Multitons, not DataTrees: each replica reconstructs its own tree
        # rather than receiving a pickled one. The driver must not touch
        # `.instance` before the model columns exist, or replicas inherit a
        # tree that predates them.
        self._datatrees = list(datatrees)
        # the .mds is small (coefficients at component locations) and every
        # region needs all of it, so it travels as a plain loaded Dataset
        self._model_ds = model_ds
        self._masks = list(masks)
        self._columns = list(columns)
        self._accumulate = accumulate
        self._epsilon = epsilon
        self._do_wgridding = do_wgridding
        self._nthreads = nthreads
        resize_thread_pool(nthreads)

    def degrid(self, item: WorkItem) -> dict:
        """Degrid and write one region. Returns post-gc memory telemetry."""
        try:
            dt = self._datatrees[item.ms_index].instance
            node_ds = dt[item.node_path].ds
            write_ds = degrid_region(
                node_ds,
                region=item.region,
                model_ds=self._model_ds,
                masks=self._masks,
                columns=self._columns,
                corr_types=tuple(str(p) for p in node_ds.polarization.values),
                accumulate=self._accumulate,
                epsilon=self._epsilon,
                do_wgridding=self._do_wgridding,
                nthreads=self._nthreads,
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
            # xarray-ms's process-level table cache has a 300 s *inactivity*
            # TTL and per-partition keys, so a busy replica never lets it
            # expire; deserialised xarray objects sit in reference cycles.
            # Both must go at every task boundary (wiki memory-and-ray).
            _release_ms_caches()
            gc.collect()

        return {
            "pid": os.getpid(),
            "rss_gb": psutil.Process().memory_info().rss / 2**30,
            "peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30,
        }


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


def degrid_msv4(
    ms: list[Path],
    output_filename: str,
    channels_per_chunk: int,
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
    integrations_per_chunk: int = -1,
    accumulate: bool = False,
    region_file: str | None = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
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
        channels_per_chunk: Channels per degridding chunk. Required, and must
            be positive: it also sets how finely the model's spectrum is
            sampled, so there is no defensible default while the `.mds` does
            not record the imaging run's channelisation (#327).
        mds: Path to the component model. Defaults to
            `{output_filename}_{suffix}_model.mds`.
        suffix: Product suffix used to build the default `.mds` path.
        model_column: Column to write. `--region-file` adds
            `{model_column}1`, `{model_column}2`, ... one per region.
        product: Stokes product; must match the model's own `stokes` attr.
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
        ray_address: Ray cluster address, or `"local"` for a private one.
        nworkers: Maximum `Degridder` replicas.
        nthreads: ducc threads per replica. Defaults to half the logical CPUs.
        progressbar: Print per-item progress with memory telemetry.
        log_directory: Directory for the run log.

    Raises:
        ValueError: If `channels_per_chunk` is not positive, no MS matches, the
            selection matches no data, or any guard in `check_model` /
            `check_tangent_point` fires.
    """
    opts_dict = locals().copy()
    time_start = time.time()

    if int(channels_per_chunk) <= 0:
        raise ValueError(
            "channels-per-chunk must be positive. There is no "
            "defensible default while the .mds does not record the imaging "
            "run's channelisation (see issue #327); it also sets how finely "
            "the model's spectrum is sampled, so it is a science choice."
        )

    output_filename, _, log_directory, _ = set_output_names(output_filename, product, log_directory=log_directory)
    opts_dict["output_filename"] = output_filename
    opts_dict["log_directory"] = log_directory

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2
    opts_dict["nthreads"] = nthreads

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pfb_logging.log_to_file(f"{log_directory}/degrid_msv4_{timestamp}.log")
    log.log_options_dict(opts_dict, title="DEGRID-MSV4 options")
    log.info(f"Using {nworkers} workers with {nthreads} threads per worker")

    # --- resolve inputs -------------------------------------------------
    msnames = []
    for ms_path in ms:
        store = DaskMSStore(str(ms_path).rstrip("/"))
        matches = store.fs.glob(str(ms_path).rstrip("/"))
        if not matches:
            raise ValueError(f"No MS at {ms_path}")
        msnames += [m.replace("file://", "") for m in map(store.fs.unstrip_protocol, matches)]

    if mds is None:
        mds = f"{output_filename}_{suffix}_model.mds"
    if not fsspec.filesystem("file").exists(mds):
        raise ValueError(f"No mds at {mds}")
    # small enough to hold in memory, and every region needs all of it
    model_ds = xr.open_zarr(mds).load()

    geom = check_model(model_ds, product)
    log.info(
        f"Model grid {geom['nx']}x{geom['ny']} at {np.rad2deg(geom['cell_rad']) * 3600:.4e} arcsec, "
        f"stokes {geom['stokes']}"
    )

    freq_min, freq_max = -np.inf, np.inf
    if freq_range:
        fmin, fmax = freq_range.strip().split(":")
        freq_min = float(fmin) if fmin else -np.inf
        freq_max = float(fmax) if fmax else np.inf

    masks = build_region_masks(model_ds, region_file)
    columns = [model_column] + [f"{model_column}{i}" for i in range(1, len(masks))]
    log.info(f"Writing {len(columns)} column(s): {', '.join(columns)}")

    # --- guards, then column creation, then close -----------------------
    selected: list[tuple[int, SelectedNode]] = []
    for ims, ms_name in enumerate(msnames):
        dt_kwargs = get_engine(ms_name, partition_columns)
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
            ensure_model_columns(ms_name, dt, columns)
            selected += [(ims, node) for node in nodes]
        finally:
            # LOAD BEARING, do not tidy away: newly added columns are invisible
            # to other processes until the table is closed (CTDS property), and
            # every Ray replica is another process.
            dt.close()

    if not selected:
        raise ValueError("Selection matched no data")

    # --- distribute -----------------------------------------------------
    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu, log=log)
    # +1 CPU for the Serve controller, which would otherwise contend with the
    # replicas for the cluster's only slots. `init_ray` is a no-op (with a
    # warning) if Ray is already up, which is the case under pytest -- hence
    # the nominal replica CPU claim below.
    init_ray(
        nworkers + 1,
        ray_address=ray_address,
        runtime_env={"env_vars": env_vars, "worker_process_setup_hook": setup_ray_worker},
        log=log,
    )

    datatrees = [Multiton(xr.open_datatree, name, **get_engine(name, partition_columns)) for name in msnames]

    app = Degridder.options(
        num_replicas="auto",
        # nominal CPU claim, as BandWorkerPool does: replicas are ducc
        # thread-pool bound, and a real per-replica claim deadlocks scheduling
        # on a small cluster (the test session's is num_cpus=2). max_replicas
        # is what actually caps concurrency.
        ray_actor_options={"num_cpus": 1e-2},
        autoscaling_config={
            "upscale_delay_s": 1.0,
            "min_replicas": 1,
            "initial_replicas": 1,
            "max_ongoing_requests": 1,
            "max_replicas": nworkers,
        },
    ).bind(
        datatrees=datatrees,
        model_ds=model_ds,
        masks=masks,
        columns=columns,
        accumulate=accumulate,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
    )
    # build the work list before standing the app up, so a bad chunk spec
    # cannot leave a Serve deployment running with nothing to shut it down
    items = [
        item for ims, node in selected for item in _work_items(ims, node, integrations_per_chunk, channels_per_chunk)
    ]
    log.info(f"Degridding {len(items)} chunks over {len(selected)} partition(s)")

    # route_prefix=None: a batch job must not bind an HTTP route
    handle = serve.run(app, name="degrid-msv4", route_prefix=None)

    try:
        # bounded in-flight queue: drain the oldest response once more than
        # `max_inflight` are outstanding, so submission cannot outrun the
        # replicas and pile up unwritten regions
        max_inflight = 4 * max(nworkers, 1)
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
                    print(
                        f"Completed: {ncomplete} / {len(items)} "
                        f"[pid {mem['pid']} rss {mem['rss_gb']:.2f} GB peak {mem['peak_gb']:.2f} GB]",
                        end="\n",
                        flush=True,
                    )

        for item in items:
            drain(max_inflight)
            inflight.append(handle.degrid.remote(item))
        drain(0)
    finally:
        serve.shutdown()

    log.info(f"All done after {time.time() - time_start}s.")
