"""Shared MSv4 access helpers.

`get_engine` is lifted verbatim out of `core/imager.py` so both MSv4 front
ends (`imager`, `degrid`) resolve backend kwargs the same way. The
selection and angle helpers below are new and used only by `degrid`:
the imager's own node loop also computes imaging geometry and the rephasing
barycentre, so sharing it would be a contortion, not a simplification.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from msv4_utils import MSv4Backend, infer_backend
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES


def get_engine(
    ms_path: str,
    partition_columns: list[str] | None = None,
    auto_corrs: bool = False,
    main_ninstances: int | None = None,
) -> dict[str, Any]:
    """Resolve `xr.open_datatree` kwargs for the backend `ms_path` lives on.

    Args:
        ms_path: Path to the MS, zarr store or MeerKAT dataset.
        partition_columns: MSv2 partition schema override.
        auto_corrs: Include autocorrelation baselines (MSv2 only).
        main_ninstances: MSv2 only. Number of casacore table instances on the
            MAIN table; `None` keeps xarray-ms's default (8). Pass `1` for a tree
            that will add columns: arcae adds a column on instance 0 but serves
            reads from whichever instance is least busy, and any other instance
            then either fails to resync ("another process changed the number of
            columns") or returns a stale column list (ska-sa/arcae#241).

    Returns:
        Keyword arguments for `xr.open_datatree`.
    """
    if "file://" in ms_path:
        ms_path = ms_path.replace("file://", "")
    backend = infer_backend(ms_path)
    if backend == MSv4Backend.CASA_TABLE:
        # deferred: registers the xarray-ms engine; only needed for this backend
        import xarray_ms  # noqa: F401

        # default schema suits mv4toms.py-style MSs; other instruments may need
        # extra columns (e.g. SOURCE_ID) -- override via partition_columns.
        # (sjperkins, PR #252 review; see xarray-ms partitioning docs.)
        kwargs = {
            "engine": "xarray-ms:msv2",
            "partition_schema": partition_columns or ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
            "auto_corrs": auto_corrs,
        }
        if main_ninstances is not None:
            # deferred: xarray-ms internals, only needed for this backend
            from xarray_ms.backend.msv2.entrypoint_utils import (
                DEFAULT_DRIVER_KWARGS,
                MAIN_TABLE,
                TABLE_OVERRIDES,
            )

            # start from xarray-ms's defaults: an explicit driver_kwargs replaces
            # them wholesale, which would silently drop the cache_size bound
            kwargs["driver_kwargs"] = {
                **DEFAULT_DRIVER_KWARGS,
                TABLE_OVERRIDES: {MAIN_TABLE: {"ninstances": main_ninstances}},
            }
        return kwargs
    elif backend == MSv4Backend.ZARR:
        return {
            "engine": "zarr",
            "chunks": None,
        }
    elif backend == MSv4Backend.MEERKAT:
        # deferred: optional dependency; registers the xarray-kat engine
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


def wrapped_angle_diff(a, b):
    """Smallest absolute angular separation between two angles, in radians.

    Compare sky coordinates with this, never with a bare ``abs(a - b)``: two
    RAs either side of zero differ by ~2*pi under subtraction while being the
    same direction, and a signed difference accepts half of all real
    mismatches. Correct for declination too, since |dec| <= pi/2 never wraps.

    Args:
        a: Angle(s) in radians.
        b: Angle(s) in radians, broadcastable against ``a``.

    Returns:
        Elementwise separation in radians, in [0, pi].
    """
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class SelectedNode:
    """A visibility node that passed selection, plus what the driver needs.

    Attributes:
        path: DataTree node path, e.g. `/name_partition_000`. Replicas index
            their own reconstructed tree with this.
        chan0: Offset of the frequency selection in **full-node** channel
            indices. Write regions are expressed against the unsliced node, so
            every frequency slice this node produces is shifted by `chan0`.
            Derived from matching channel indices, not a label slice, so it is
            correct for a descending spectral window.
        nchan: Number of selected channels.
        ntime: Number of times on the node (never trimmed).
        field_name: The node's single field name.
        spw_name: The node's spectral window name.
        scan_name: The node's single scan name.
        field_radec: `(ra, dec)` of `FIELD_PHASE_CENTER_DIRECTION`, radians.
        corr_types: The `polarization` coord values, e.g. `("XX", "XY", "YX", "YY")`.
    """

    path: str
    chan0: int
    nchan: int
    ntime: int
    field_name: str
    spw_name: str
    scan_name: str
    field_radec: np.ndarray
    corr_types: tuple[str, ...]


def select_vis_nodes(
    dt: xr.DataTree,
    *,
    data_group: str = "base",
    field_names: list[str] | None = None,
    spw_names: list[str] | None = None,
    scan_names: list[str] | None = None,
    freq_min: float = -np.inf,
    freq_max: float = np.inf,
) -> list[SelectedNode]:
    """Enumerate the visibility nodes of an MSv4 DataTree that pass selection.

    Iterates `dt.subtree`, not `dt.children`: `sync_msv2` counts correlated
    nodes over the whole subtree, and the two must agree about which nodes
    exist or column creation silently drops variables.

    Args:
        dt: An opened MSv4 DataTree.
        data_group: Data group used to resolve the field_and_source subtable.
        field_names: Field names to keep; `None` keeps all.
        spw_names: Spectral window names to keep; `None` keeps all.
        scan_names: Scan names to keep; `None` keeps all.
        freq_min: Lower frequency bound in Hz, inclusive.
        freq_max: Upper frequency bound in Hz, inclusive.

    Returns:
        One `SelectedNode` per surviving node, in tree order.
    """
    out: list[SelectedNode] = []
    for node in dt.subtree:
        if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
            continue
        # Select by index, not by label slice. `sel(frequency=slice(lo, hi))`
        # silently returns nothing for a descending spectral window, and
        # `searchsorted` below would carry the same assumption. MS spectral
        # windows can be descending, so find the matching channels explicitly.
        full_freqs = node.ds.frequency.load().values
        keep = np.nonzero((full_freqs >= freq_min) & (full_freqs <= freq_max))[0]
        if keep.size == 0:
            continue
        if keep.size != keep[-1] - keep[0] + 1:
            raise ValueError(
                f"Frequency selection on {node.path} is not contiguous in channel "
                f"index (channels {keep.tolist()}); a write region must be a slice. "
                "This means the spectral window is not monotonic in frequency."
            )
        chan0 = int(keep[0])
        ds = node.ds.isel(frequency=slice(chan0, int(keep[-1]) + 1))
        # partitioned by FIELD_ID / SCAN_NUMBER, so each is a single value
        field_name = np.unique(ds.field_name.load().values).item()
        scan_name = np.unique(ds.scan_name.load().values).item()
        spw_name = ds.frequency.attrs["spectral_window_name"]
        if (field_names is not None) and (field_name not in field_names):
            continue
        if (spw_names is not None) and (spw_name not in spw_names):
            continue
        if (scan_names is not None) and (scan_name not in scan_names):
            continue

        grp = node.ds.attrs["data_groups"][data_group]
        fns = node[grp["field_and_source"].rsplit("/", 1)[-1]].ds
        field_radec = np.asarray(fns.FIELD_PHASE_CENTER_DIRECTION.sel(field_name=field_name).values).squeeze()

        out.append(
            SelectedNode(
                path=node.path,
                chan0=chan0,
                nchan=int(ds.frequency.size),
                ntime=int(node.ds.sizes["time"]),
                field_name=str(field_name),
                spw_name=str(spw_name),
                scan_name=str(scan_name),
                field_radec=field_radec,
                corr_types=tuple(str(p) for p in node.ds.polarization.values),
            )
        )
    return out
