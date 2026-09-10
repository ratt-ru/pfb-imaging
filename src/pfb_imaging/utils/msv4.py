"""Shared MSv4 access helpers.

`get_engine` is lifted verbatim out of `core/imager.py` so both MSv4 front
ends (`imager`, `degrid-msv4`) resolve backend kwargs the same way. The
selection and angle helpers below are new and used only by `degrid-msv4`:
the imager's own node loop also computes imaging geometry and the rephasing
barycentre, so sharing it would be a contortion, not a simplification.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from msv4_utils import MSv4Backend, infer_backend
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES


def get_engine(ms_path: str, partition_columns: list[str] | None = None) -> dict[str, Any]:
    if "file://" in ms_path:
        ms_path = ms_path.replace("file://", "")
    backend = infer_backend(ms_path)
    if backend == MSv4Backend.CASA_TABLE:
        # deferred: registers the xarray-ms engine; only needed for this backend
        import xarray_ms  # noqa: F401

        # default schema suits mv4toms.py-style MSs; other instruments may need
        # extra columns (e.g. SOURCE_ID) -- override via partition_columns.
        # (sjperkins, PR #252 review; see xarray-ms partitioning docs.)
        return {
            "engine": "xarray-ms:msv2",
            "partition_schema": partition_columns or ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
        }
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
        ds = node.ds.sel(frequency=slice(freq_min, freq_max))
        if ds.frequency.size == 0:
            continue
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

        # write regions index the *unsliced* node, so record where the
        # frequency selection starts in full-node channel indices
        full_freqs = node.ds.frequency.load().values
        chan0 = int(np.searchsorted(full_freqs, ds.frequency.values[0]))

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
