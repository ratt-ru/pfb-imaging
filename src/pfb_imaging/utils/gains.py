"""Read QuartiCal gain terms onto an MSv4 node's (time, frequency) grid.

A QuartiCal gain term is a directory of zarr groups, one per (field, spw[, scan])
partition -- ``gains.qc/NET/NET_0``, ``NET_1``... -- each holding ``gains``
``(gain_time, gain_freq, antenna, direction, correlation)`` and ``gain_flags``
without the last axis, written by dask-ms with MSv2 ids (``FIELD_ID``,
``DATA_DESC_ID``, optionally ``SCAN_NUMBER``) as attributes. MSv4 nodes carry
names, not ids, so partitions are matched on ``SCAN_NUMBER == int(scan_name)``
when present, the frequency grid, and time overlap.

Gains apply on-grid only: each gain axis matches the data or has length one.
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

    def subset(self, tsel: slice, fsel: slice) -> "NodeGains":
        """Slice to a region, leaving broadcast (length-1) axes alone."""
        tsel = tsel if self.jones.shape[0] > 1 else slice(None)
        fsel = fsel if self.jones.shape[1] > 1 else slice(None)
        return NodeGains(self.jones[tsel, fsel], self.flags[tsel, fsel])


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
    return tuple(xr.open_zarr(os.path.join(path, n), chunks=None, consolidated=False) for n in names)


def load_node_gains(
    spec: str,
    scan_name: str,
    times_unix: np.ndarray,
    freqs: np.ndarray,
    antenna_names: Sequence[str],
) -> NodeGains:
    """Gains for one MSv4 node on its grid, antennas in ``antenna_names`` order.

    Args:
        spec: Gain term, ``gains.qc/NET`` or ``gains.qc::NET``.
        scan_name: The node's scan name (``str(SCAN_NUMBER)`` for MSv2-backed trees).
        times_unix: The node's ``time`` coordinate (unix seconds), ascending.
        freqs: The node's channel frequencies in Hz.
        antenna_names: Antenna names, indexed the way the caller's ant1/ant2 are.

    Returns:
        :class:`NodeGains`. Antennas absent from the term get identity gains and are flagged.

    Raises:
        ValueError: if zero or several partitions match, the grids disagree, or the term
            has several directions or an unsupported correlation count.
    """
    path = term_path(spec)
    dss = Multiton(open_term, path).instance
    times_mjd = to_mjd_time(np.asarray(times_unix, dtype=float))
    ds = _select(path, dss, scan_name, times_mjd, np.asarray(freqs, dtype=float))

    if ds.sizes.get("direction", 1) > 1:
        raise ValueError(f"{spec} has {ds.sizes['direction']} directions; only direction-independent gains apply")

    tsel = _align(ds.gain_time.values, times_mjd, "time", atol=TIME_TOLERANCE)
    fsel = _align(ds.gain_freq.values, np.asarray(freqs, dtype=float), "frequency", rtol=FREQ_RTOL)

    jones = np.asarray(ds.gains.values)[..., 0, :][tsel][:, fsel].astype(np.complex128)
    gflags = np.asarray(ds.gain_flags.values)[..., 0][tsel][:, fsel]
    # int8 from QuartiCal, float from some writers; NaN is not a solution, nor is a non-finite gain
    flags = (gflags != 0) | ~np.isfinite(gflags) | ~np.isfinite(jones).all(axis=-1)

    ncorr = jones.shape[-1]
    if ncorr == 1:
        jones = np.repeat(jones, 2, axis=-1)
    elif ncorr not in (2, 4):
        raise ValueError(f"{spec} has {ncorr} gain correlations; expected 1, 2 or 4")
    full = jones.shape[-1] == 4

    position = {str(n): k for k, n in enumerate(ds.antenna.values)}
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


def _select(path: str, dss, scan_name: str, times_mjd: np.ndarray, freqs: np.ndarray) -> xr.Dataset:
    lo, hi = float(times_mjd.min()), float(times_mjd.max())

    def overlaps(d):
        gt = d.gain_time.values
        return gt.min() - TIME_TOLERANCE <= hi and gt.max() + TIME_TOLERANCE >= lo

    def freq_matches(d):
        gf = d.gain_freq.values
        return gf.size == 1 or (gf.size == freqs.size and np.allclose(gf, freqs, rtol=FREQ_RTOL, atol=0))

    cands = [d for d in dss if freq_matches(d)]
    if any("SCAN_NUMBER" in d.attrs for d in cands):
        cands = [d for d in cands if str(int(d.attrs.get("SCAN_NUMBER", -1))) == str(scan_name)]
    # solved across scans, or several fields share a scan number: time overlap decides
    if len(cands) != 1:
        cands = [d for d in cands if overlaps(d)]
    if len(cands) != 1:
        raise ValueError(f"{'No' if not cands else len(cands)} gain partitions in {path} for scan {scan_name}")
    return cands[0]


def _align(have: np.ndarray, want: np.ndarray, name: str, atol: float = 0.0, rtol: float = 0.0):
    """Indices into ``have`` matching each of ``want``, or a broadcast slice."""
    if have.size == 1:
        return slice(None)
    idx = np.clip(np.searchsorted(have, want), 0, have.size - 1)
    lower = np.clip(idx - 1, 0, have.size - 1)
    idx = np.where(np.abs(have[lower] - want) < np.abs(have[idx] - want), lower, idx)
    if not np.allclose(have[idx], want, atol=atol, rtol=rtol):
        raise ValueError(
            f"Gain {name} axis does not match the data ({have.size} solutions for {want.size} data points). "
            f"Gains apply on-grid only; write net gains at the data's {name} resolution."
        )
    return idx
