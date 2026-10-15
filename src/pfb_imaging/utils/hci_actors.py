"""Ray actor bodies for pfb hci-msv4: the image workers and the single cube writer.

Plain classes so they can be exercised in-process; the driver wraps them with
ray.remote. The writer is the only process that writes to the cube store while
it lives, which is what makes the slab writes safe without a lock.
"""

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import zarr
from rarg_python_patterns.multiton import Multiton

from pfb_imaging.utils.memprof import task_memory
from pfb_imaging.utils.msv4 import get_engine
from pfb_imaging.utils.stokes2im_msv4 import image_item, read_item

_IMAGE_VARS = ("cube", "beam_weight", "psf")
_SCALAR_VARS = ("weight", "rms", "nonzero", "psf_maj", "psf_min", "psf_pa")
_TIME_VARS = ("orig_phase_dir_ra", "orig_phase_dir_dec")


def _fill(arr) -> object:
    return arr.fill_value if arr.fill_value is not None else 0


class CubeWriterImpl:
    """Assemble (band, time-block) slabs in RAM and write each one once.

    Args:
        store: The scaffolded output zarr store.
        expected: (band_index, block) -> number of images the plan sends to that slab.
        images_per_chunk: TIME chunk of the store.
        channel_width: (nband,) channel width per band, written at finalise.
        flag_excess_rms: Images whose rms exceeds this multiple of the band median are flagged.
    """

    def __init__(
        self,
        store: str,
        expected: dict[tuple[int, int], int],
        images_per_chunk: int,
        channel_width: np.ndarray,
        flag_excess_rms: float,
    ) -> None:
        """Open the scaffolded store for update.

        Args:
            store: Path of the scaffolded zarr store.
            expected: (band_index, block) -> number of images the plan sends to that slab.
            images_per_chunk: TIME chunk of the store.
            channel_width: (nband,) channel width per band.
            flag_excess_rms: Rms flagging threshold as a multiple of the band median.
        """
        self._root = zarr.open_group(store, mode="r+")
        self._expected = dict(expected)
        self._ipc = images_per_chunk
        self._channel_width = np.asarray(channel_width)
        self._flag_excess_rms = flag_excess_rms
        self._img_vars = [v for v in _IMAGE_VARS if v in self._root]
        nstokes, nband, ntime = self._root["rms"].shape
        self._ntime = ntime
        self._scalars = {
            v: np.full((nstokes, nband, ntime), _fill(self._root[v]), dtype=self._root[v].dtype) for v in _SCALAR_VARS
        }
        self._times = {v: np.full(ntime, _fill(self._root[v]), dtype=self._root[v].dtype) for v in _TIME_VARS}
        self._received = np.zeros((nband, ntime), dtype=bool)
        ny, nx = self._root["cube"].shape[-2:]
        self._cube_sum = np.zeros((nstokes, nband, ny, nx))
        self._wsum = np.zeros((nstokes, nband))
        self._psf = "psf" in self._root
        if self._psf:
            self._psf_sum = np.zeros((nstokes,) + self._root["psf"].shape[-2:])
            self._w2sum = np.zeros(nstokes)
        self._slabs: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._counts: dict[tuple[int, int], int] = {}

    def open_blocks(self) -> int:
        """Number of slabs currently held in memory."""
        return len(self._slabs)

    def add(self, result: dict) -> tuple[tuple[int, int], dict] | None:
        """Store one image; write its slab once every expected image has arrived.

        Args:
            result: Task 4 image dict plus ``time_index`` and ``band_index``.

        Returns:
            ``((band, block), mem)`` when that slab was written, with ``mem`` this process's
            post-gc ``task_memory()``; otherwise None.

        Raises:
            KeyError: If the image's slab is not in the plan.
        """
        try:
            return self._add(result)
        finally:
            gc.collect()  # actor-method boundary: deserialised results sit in cycles

    def _add(self, result: dict) -> tuple[tuple[int, int], dict] | None:
        t = int(result["time_index"])
        b = int(result["band_index"])
        key = (b, t // self._ipc)
        if key not in self._expected:
            raise KeyError(f"slab {key} (band, block) is not in the plan")
        t0 = key[1] * self._ipc
        if key not in self._slabs:
            tlen = min(self._ipc, self._ntime - t0)
            self._slabs[key] = {
                v: np.full(
                    (self._root[v].shape[0], tlen) + self._root[v].shape[-2:],
                    _fill(self._root[v]),
                    dtype=self._root[v].dtype,
                )
                for v in self._img_vars
            }
            self._counts[key] = 0
        slab = self._slabs[key]
        for v in self._img_vars:
            slab[v][:, t - t0] = result[v]
        for v in _SCALAR_VARS:
            self._scalars[v][:, b, t] = result[v]
        for v in _TIME_VARS:
            self._times[v][t] = result[v]
        self._received[b, t] = True

        w = result["weight"].astype(np.float64)
        self._cube_sum[:, b] += w[:, None, None] * result["cube"]
        self._wsum[:, b] += w
        if self._psf:
            self._psf_sum += (w[:, None, None] * result["psf"]) ** 2
            self._w2sum += w**2

        self._counts[key] += 1
        if self._counts[key] < self._expected[key]:
            return None
        tlen = slab[self._img_vars[0]].shape[1]
        for v in self._img_vars:
            self._root[v][:, b, t0 : t0 + tlen] = slab[v]
        del self._slabs[key], self._counts[key], slab
        gc.collect()  # so the telemetry shows the slab released
        return key, task_memory()

    def finalise(self) -> dict:
        """Write per-image scalars, flag, cube_mean, psf2 and channel_width.

        Returns:
            ``{"nflagged": int}``, the number of received, nonzero images flagged for excess rms.

        Raises:
            RuntimeError: If any slab is still incomplete.
        """
        try:
            return self._finalise()
        finally:
            gc.collect()

    def _finalise(self) -> dict:
        if self._slabs:
            raise RuntimeError(f"finalise called with {len(self._slabs)} incomplete slabs: {sorted(self._slabs)}")
        rms = self._scalars["rms"]
        nonzero = self._scalars["nonzero"].astype(bool)
        nstokes, nband, _ = rms.shape
        flag = np.zeros(rms.shape, dtype=bool)
        for c in range(nstokes):
            for b in range(nband):
                nz = nonzero[c, b]
                # median over an empty selection is NaN, so nothing exceeds it: as hci
                med = np.median(rms[c, b, nz]) if nz.any() else np.nan
                flag[c, b] = (rms[c, b] > self._flag_excess_rms * med) | ~nz

        # remove flagged-but-received images from the running sums; unreceived and
        # zero-weight images never contributed
        weight = self._scalars["weight"].astype(np.float64)
        drop = flag & nonzero & self._received[None]
        # read each (stokes, band, time block) once: a chunk spans images_per_chunk times
        for c, b, blk in sorted({(c, b, t // self._ipc) for c, b, t in zip(*np.nonzero(drop))}):
            t0 = blk * self._ipc
            t1 = min(t0 + self._ipc, self._ntime)
            ts = np.nonzero(drop[c, b, t0:t1])[0]
            w = weight[c, b, t0 + ts]
            cube = self._root["cube"][c, b, t0:t1][ts]
            self._cube_sum[c, b] -= np.tensordot(w, cube, axes=1)
            self._wsum[c, b] -= w.sum()
            if self._psf:
                psf = self._root["psf"][c, b, t0:t1][ts]
                self._psf_sum[c] -= ((w[:, None, None] * psf) ** 2).sum(axis=0)
                self._w2sum[c] -= (w**2).sum()

        wsum = self._wsum[:, :, None, None]
        mean = np.divide(self._cube_sum, wsum, out=np.zeros_like(self._cube_sum), where=wsum > 0)
        self._root["cube_mean"][:] = mean.astype(self._root["cube_mean"].dtype)
        if self._psf:
            w2 = self._w2sum[:, None, None]
            psf2 = np.divide(self._psf_sum, w2, out=np.zeros_like(self._psf_sum), where=w2 > 0)
            self._root["psf2"][:] = psf2.astype(self._root["psf2"].dtype)
        for v in _SCALAR_VARS:
            self._root[v][:] = self._scalars[v]
        for v in _TIME_VARS:
            self._root[v][:] = self._times[v]
        self._root["flag"][:] = flag
        self._root["channel_width"][:] = self._channel_width.astype(self._root["channel_width"].dtype)
        return {"nflagged": int(drop.sum())}


def _key(item) -> tuple:
    return (item.ms_index, item.node_path, item.time_index, item.band_index)


class HciWorkerImpl:
    """Reads and images work items; reads the next item on a thread while imaging the current one.

    Visibilities stay in this process. Actor execution is serial, so the only
    concurrency is the single prefetch thread (arcae reads and ducc gridding both
    release the GIL).

    Args:
        msnames: Measurement sets, indexed by ``WorkItem.ms_index``.
        cfg: Imaging configuration.
        nx_psf: PSF size in x.
        ny_psf: PSF size in y.
        data_group: MSv4 data group.
        gain_specs: One gain term (or None) per MS.
        beam_model: A BeamWizard or None.
    """

    def __init__(
        self,
        msnames: list[str],
        cfg,
        nx_psf: int,
        ny_psf: int,
        data_group: str = "base",
        gain_specs: list[str | None] | None = None,
        beam_model=None,
    ) -> None:
        self._dts = [Multiton(xr.open_datatree, name, **get_engine(name)) for name in msnames]
        self._cfg = cfg
        self._nx_psf = nx_psf
        self._ny_psf = ny_psf
        self._data_group = data_group
        self._gain_specs = gain_specs or [None] * len(msnames)
        self._beam_model = beam_model
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hci-read")
        self._pending = None  # (key, Future[ItemData])

    def _read(self, item):
        node = self._dts[item.ms_index].instance[item.node_path]
        return read_item(node, item.region, self._cfg, self._data_group, self._gain_specs[item.ms_index])

    def image(self, item, next_item=None) -> tuple[dict, dict]:
        """Image ``item`` (prefetched if it was the previous call's ``next_item``) and start reading ``next_item``.

        Args:
            item: The work item to image.
            next_item: The item the driver will send next, or None.

        Returns:
            (result, mem): the image result dict plus ``time_index``/``band_index``, and the
            post-gc ``task_memory()`` telemetry. Call with ``num_returns=2`` so the image ref
            goes to the writer and only the small telemetry returns to the driver.

        Raises:
            RuntimeError: A prefetch is still pending for a different item (driver bug).
        """
        try:
            if self._pending is not None and self._pending[0] == _key(item):
                _, fut = self._pending
                self._pending = None  # cleared first so a failed read surfaces here, once
                inp = fut.result()
            else:
                inp = self._read(item)
            if next_item is not None:
                if self._pending is not None:
                    raise RuntimeError(
                        f"prefetch pending for {self._pending[0]} while asked to prefetch {_key(next_item)}"
                    )
                self._pending = (_key(next_item), self._executor.submit(self._read, next_item))
            result = image_item(inp, self._cfg, self._beam_model, self._nx_psf, self._ny_psf)
            result["time_index"] = item.time_index
            result["band_index"] = item.band_index
            del inp
            gc.collect()
            return result, task_memory()
        except Exception as e:
            e.add_note(f"while imaging {item}")
            raise
        finally:
            gc.collect()

    def close(self) -> None:
        """Drop any pending prefetch and stop the read thread."""
        try:
            self._pending = None
            self._executor.shutdown(wait=True, cancel_futures=True)
        finally:
            gc.collect()
