"""Ray actor bodies for pfb hci-msv4: the image workers and the single cube writer.

Plain classes so they can be exercised in-process; the driver wraps them with
ray.remote. The writer is the only process that writes to the cube store while
it lives, which is what makes the slab writes safe without a lock.
"""

import gc

import numpy as np
import zarr

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

    def __init__(self, store, expected, images_per_chunk, channel_width, flag_excess_rms):
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

    def add(self, result):
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
        del self._slabs[key], self._counts[key]
        gc.collect()
        return key

    def finalise(self):
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
        for c, b, t in zip(*np.nonzero(drop)):
            w = weight[c, b, t]
            self._cube_sum[c, b] -= w * self._root["cube"][c, b, t]
            self._wsum[c, b] -= w
            if self._psf:
                self._psf_sum[c] -= (w * self._root["psf"][c, b, t]) ** 2
                self._w2sum[c] -= w**2

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
