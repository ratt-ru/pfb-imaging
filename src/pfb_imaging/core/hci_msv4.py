"""pfb hci-msv4: high cadence imaging of MSv4 data with worker actors and an actor writer."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from pfb_imaging.utils.misc import to_mjd_time
from pfb_imaging.utils.msv4 import select_vis_nodes

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
    all_freqs: np.ndarray  # unique channel frequencies over the selection (transients)


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
        ValueError: on an empty selection, a non-positive integrations_per_image
            other than the whole-scan sentinels, or two partitions mapping to the
            same output image.
    """
    if integrations_per_image not in _WHOLE and integrations_per_image < 1:
        raise ValueError(
            f"integrations_per_image must be >= 1, or -1 for one image per scan (got {integrations_per_image})"
        )
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
            radecs.append(sel.field_radec)
            all_times.append(tmjd)
            all_freqs.append(freqs)
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
