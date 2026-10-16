"""Read QuartiCal gain terms onto an MSv4 node's (time, frequency) grid.

A QuartiCal gain term is a directory of zarr groups, one per (field, spw[, scan])
partition -- ``gains.qc/NET/NET_0``, ``NET_1``... -- each holding ``gains``
``(gain_time, gain_freq, antenna, direction, correlation)`` and ``gain_flags``
without the last axis, written by dask-ms with MSv2 ids and names (``FIELD_ID``,
``FIELD_NAME``, ``DATA_DESC_ID``, optionally ``SCAN_NUMBER``) as attributes. MSv4
nodes carry names, not ids, so partitions are matched on every wanted channel being
a gain channel (so a channel subset of the term's grid matches), on
``FIELD_NAME == field_name`` and ``SCAN_NUMBER == int(scan_name)`` when the term
records them, and on time overlap.

Gains apply on-grid only: each wanted time and channel is a gain solution point, in
either axis order, or the gain axis has length one. Only the wanted region is read
from the store, so nothing of a term outlives the call that reads it.
Ported from surfvis (serve-ui); temporary until QuartiCal writes plain xarray.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from rarg_python_patterns.multiton import Multiton

from pfb_imaging.utils.misc import to_mjd_time

# seconds; QuartiCal times are MJD seconds read from TIME
TIME_TOLERANCE = 1e-3
FREQ_RTOL = 1e-6


@dataclass(frozen=True)
class NodeGains:
    """Gains for one node, on its (time, frequency) grid.

    Attributes:
        jones: ``(gtime, gchan, nant, 2)`` diagonal or ``(gtime, gchan, nant, 2, 2)``
            full Jones; length-1 ``gtime``/``gchan`` broadcast.
        flags: ``(gtime, gchan, nant)`` bool.
    """

    jones: np.ndarray
    flags: np.ndarray


def term_path(spec: str) -> str:
    """Accept ``gains.qc/NET`` or pfb's ``gains.qc::NET`` spelling."""
    spec = str(spec).rstrip("/")
    if "::" in spec:
        root, term = spec.rsplit("::", 1)
        return f"{root.rstrip('/')}/{term}"
    return spec


def open_term(path: str) -> tuple[xr.Dataset, ...]:
    """Open every ``<term>_<n>`` group of a gain term, lazily. Module level so a Multiton can pickle it."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No gain term at {path}")
    term = os.path.basename(path)
    names = sorted(
        (n for n in os.listdir(path) if n.startswith(f"{term}_") and n.removeprefix(f"{term}_").isdigit()),
        key=lambda n: int(n.removeprefix(f"{term}_")),
    )
    if not names:
        raise ValueError(f"{path} holds no {term}_<n> groups -- is this a QuartiCal gain term?")
    try:
        return tuple(xr.open_zarr(os.path.join(path, n), chunks=None, consolidated=False) for n in names)
    except KeyError as e:
        # older dask-ms wrote no _ARRAY_DIMENSIONS, which xarray needs to name the axes
        raise ValueError(
            f"{path} is not readable as plain zarr ({e}); it was probably written by an older "
            "dask-ms. Rewrite it with a current QuartiCal, or use pfb hci."
        ) from e


def load_node_gains(
    spec: str,
    scan_name: str,
    times_unix: np.ndarray,
    freqs: np.ndarray,
    antenna_names: Sequence[str],
    field_name: str | None = None,
) -> NodeGains:
    """Gains for one MSv4 node on its grid, antennas in ``antenna_names`` order.

    Args:
        spec: Gain term, ``gains.qc/NET`` or ``gains.qc::NET``.
        scan_name: The node's scan name (``str(SCAN_NUMBER)`` for MSv2-backed trees).
        times_unix: The times to load (unix seconds), all or part of the node's ``time`` coordinate.
        freqs: The channel frequencies to load in Hz, all or a subset of the node's channels.
        antenna_names: Antenna names, indexed the way the caller's ant1/ant2 are.
        field_name: The node's field name; selects on the term's ``FIELD_NAME`` when it has one.

    Returns:
        :class:`NodeGains`. Antennas absent from the term get identity gains and are flagged.

    Raises:
        ValueError: if zero or several partitions match, the grids disagree, no antenna
            name is in the term, or the term has several directions or an unsupported
            correlation count.
    """
    ds, tsel, fsel = _resolve(spec, scan_name, field_name, times_unix, freqs)
    position = _antenna_positions(spec, ds, antenna_names)

    jones = _read_region(ds.gains, tsel, fsel, ("gain_time", "gain_freq", "antenna", "correlation"))
    jones = jones.astype(np.complex128)
    gflags = _read_region(ds.gain_flags, tsel, fsel, ("gain_time", "gain_freq", "antenna"))
    # int8 from QuartiCal, float from some writers; NaN is not a solution, nor is a non-finite gain
    flags = (gflags != 0) | ~np.isfinite(gflags) | ~np.isfinite(jones).all(axis=-1)

    ncorr = jones.shape[-1]
    if ncorr == 1:
        jones = np.repeat(jones, 2, axis=-1)
    elif ncorr not in (2, 4):
        raise ValueError(f"{spec} has {ncorr} gain correlations; expected 1, 2 or 4")
    full = jones.shape[-1] == 4

    identity = np.array([1, 0, 0, 1] if full else [1, 1], dtype=np.complex128)
    nt, nf = jones.shape[:2]
    out_j = np.broadcast_to(identity, (nt, nf, len(antenna_names), identity.size)).copy()
    out_f = np.ones((nt, nf, len(antenna_names)), dtype=bool)
    for a, name in enumerate(antenna_names):
        k = position.get(str(name))
        if k is not None:
            out_j[:, :, a] = jones[:, :, k]
            out_f[:, :, a] = flags[:, :, k]
    if full:
        out_j = out_j.reshape(nt, nf, len(antenna_names), 2, 2)
    return NodeGains(jones=out_j, flags=out_f)


def check_node_grid(
    spec: str,
    scan_name: str,
    times_unix: np.ndarray,
    freqs: np.ndarray,
    field_name: str | None = None,
    antenna_names: Sequence[str] | None = None,
) -> list[str]:
    """Check, on coordinates only, that a gain term applies to a node's (time, frequency) grid.

    Runs exactly the partition selection and alignment :func:`load_node_gains` runs, so a
    driver can refuse a mismatched term before any worker starts.

    Args:
        spec: Gain term, ``gains.qc/NET`` or ``gains.qc::NET``.
        scan_name: The node's scan name.
        times_unix: The node's ``time`` coordinate (unix seconds).
        freqs: Every channel frequency the node will be imaged at, in Hz.
        field_name: The node's field name.
        antenna_names: The node's antenna names; None skips the antenna check.

    Returns:
        The antenna names absent from the term (their data will be flagged).

    Raises:
        ValueError: as :func:`load_node_gains`.
    """
    ds, _, _ = _resolve(spec, scan_name, field_name, times_unix, freqs)
    if antenna_names is None:
        return []
    position = _antenna_positions(spec, ds, antenna_names)
    return [str(n) for n in antenna_names if str(n) not in position]


def _antenna_positions(spec: str, ds: xr.Dataset, antenna_names: Sequence[str]) -> dict[str, int]:
    """Term antenna name -> index, refusing a term that shares no name with the data."""
    position = {str(n): k for k, n in enumerate(ds.antenna.values)}
    if not any(str(n) in position for n in antenna_names):
        raise ValueError(
            f"No antenna of the data is in {spec} (data: {list(antenna_names)[:3]}..., "
            f"term: {list(position)[:3]}...); gains are matched to antennas by name"
        )
    return position


def _read_region(da: xr.DataArray, tsel, fsel, dims: tuple[str, ...]) -> np.ndarray:
    """Read only the wanted (time, frequency) points of ``da``, axes by name, direction 0."""
    da = da.isel(direction=0)
    picks = []
    for dim, sel in (("gain_time", tsel), ("gain_freq", fsel)):
        if isinstance(sel, slice):
            picks.append(slice(None))
            continue
        # sorted unique indices make an orthogonal store read; the inverse restores order
        uniq, inv = np.unique(sel, return_inverse=True)
        da = da.isel({dim: uniq})
        picks.append(inv)
    out = np.asarray(da.transpose(*dims).values)
    return out[picks[0]][:, picks[1]]


def _resolve(spec: str, scan_name: str, field_name: str | None, times_unix: np.ndarray, freqs: np.ndarray):
    """The matching partition and the time/frequency indices of the wanted grid in it."""
    path = os.path.abspath(term_path(spec))  # one cache entry per term, however it is spelt
    dss = Multiton(open_term, path).instance
    times_mjd = to_mjd_time(np.asarray(times_unix, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    ds = _select(path, dss, scan_name, field_name, times_mjd, freqs)
    if ds.sizes.get("direction", 1) > 1:
        raise ValueError(f"{spec} has {ds.sizes['direction']} directions; only direction-independent gains apply")
    tsel = _align(ds.gain_time.values, times_mjd, "time", atol=TIME_TOLERANCE)
    fsel = _align(ds.gain_freq.values, freqs, "frequency", rtol=FREQ_RTOL)
    return ds, tsel, fsel


def _select(
    path: str, dss, scan_name: str, field_name: str | None, times_mjd: np.ndarray, freqs: np.ndarray
) -> xr.Dataset:
    lo, hi = float(times_mjd.min()), float(times_mjd.max())

    def overlaps(d):
        gt = d.gain_time.values
        # a single solution interval broadcasts over the node
        return gt.size == 1 or (gt.min() - TIME_TOLERANCE <= hi and gt.max() + TIME_TOLERANCE >= lo)

    def freq_matches(d):
        # every wanted channel is a gain channel: a channel subset of the term's grid matches
        gf = d.gain_freq.values
        return gf.size == 1 or _on_grid(gf, freqs, rtol=FREQ_RTOL)

    where = f"scan {scan_name}" if field_name is None else f"field {field_name} scan {scan_name}"
    cands = [d for d in dss if freq_matches(d)]
    if not cands:
        raise ValueError(f"No gain partition in {path} has the channels of {where}")
    if field_name is not None and any("FIELD_NAME" in d.attrs for d in cands):
        cands = [d for d in cands if str(d.attrs.get("FIELD_NAME")) == str(field_name)]
    if any("SCAN_NUMBER" in d.attrs for d in cands):
        cands = [d for d in cands if str(int(d.attrs.get("SCAN_NUMBER", -1))) == str(scan_name)]
    # solved across scans, or several partitions share a field and scan: time overlap decides
    if len(cands) != 1:
        cands = [d for d in cands if overlaps(d)]
    if len(cands) != 1:
        raise ValueError(f"{'No' if not cands else len(cands)} gain partitions in {path} for {where}")
    return cands[0]


def _nearest(have: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Index into ``have`` (any order) of the element nearest each of ``want``."""
    order = np.argsort(have, kind="stable")
    srt = have[order]
    idx = np.clip(np.searchsorted(srt, want), 0, srt.size - 1)
    lower = np.clip(idx - 1, 0, srt.size - 1)
    idx = np.where(np.abs(srt[lower] - want) < np.abs(srt[idx] - want), lower, idx)
    return order[idx]


def _on_grid(have: np.ndarray, want: np.ndarray, atol: float = 0.0, rtol: float = 0.0) -> bool:
    return bool(np.allclose(have[_nearest(have, want)], want, atol=atol, rtol=rtol))


def _align(have: np.ndarray, want: np.ndarray, name: str, atol: float = 0.0, rtol: float = 0.0):
    """Indices into ``have`` matching each of ``want``, or a broadcast slice."""
    if have.size == 1:
        return slice(None)
    idx = _nearest(have, want)
    if not np.allclose(have[idx], want, atol=atol, rtol=rtol):
        raise ValueError(
            f"Gain {name} axis does not match the data ({have.size} solutions for {want.size} data points). "
            f"Gains apply on-grid only; write net gains at the data's {name} resolution."
        )
    return idx
