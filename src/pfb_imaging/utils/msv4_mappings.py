import warnings

import numpy as np
import xarray as xr
import xarray_ms  # noqa: F401
from daskms.experimental.zarr import xds_from_zarr
from msv4_utils.msv4_types import (
    VISIBILITY_XDS_TYPES,
)
from omegaconf import ListConfig
from xarray_ms.errors import FrameConversionWarning, IrregularGridWarning, MissingMetadataWarning

from pfb_imaging.utils.misc import chunkify_rows

warnings.filterwarnings("ignore", category=IrregularGridWarning)
warnings.filterwarnings("ignore", category=MissingMetadataWarning)
warnings.filterwarnings("ignore", category=FrameConversionWarning)


def construct_mappings_xms(
    ms_name,
    gain_name=None,
    ipi=None,
    cpi=None,
    freq_min=-np.inf,
    freq_max=np.inf,
    field_names=None,
    spw_names=None,
    scan_names=None,
    applycal="all",
):
    """
    Construct dictionaries containing per MS, FIELD, DDID and SCAN
    time and frequency mappings.

    Input:
    ms_name     - list of ms names
    gain_name   - list of paths to gains, must be in same order as ms_names
    ipi         - integrations (i.e. unique times) per output image.
                  Defaults to one image per scan.
    cpi         - Channels per image. Defaults to one per spw.

    The chan <-> band mapping is determined by:

    freqs           - dict[MS][IDT] frequencies chunked by band
    fbin_idx        - dict[MS][IDT] freq bin starting indices
    fbin_counts     - dict[MS][IDT] freq bin counts
    band_mapping    - dict[MS][IDT] output bands in dataset
    freq_out        - array of linearly spaced output frequencies
                      over entire frequency range.
                      freq_out[band_mapping[MS][IDT]] gives the
                      model frequencies in a dataset

    where IDT is constructed as FIELD#_DDID#_SCAN#.

    Similarly, the row <-> time mapping is determined by:

    utimes          - dict[MS][IDT] unique times per output image
    tbin_idx        - dict[MS][IDT] time bin starting indices
    tbin_counts     - dict[MS][IDT] time bin counts
    time_mapping    - dict[MS][IDT] utimes per dataset

    """

    if not isinstance(ms_name, list) and not isinstance(ms_name, ListConfig):
        ms_name = [ms_name]

    if gain_name is not None:
        if not isinstance(gain_name, list) and not isinstance(gain_name, ListConfig):
            gain_name = [gain_name]
        assert len(ms_name) == len(gain_name)

    # collect times and frequencies per ms and ds
    freqs = {}
    chan_widths = {}
    times = {}
    gains = {}
    radecs = {}
    antpos = {}
    poltype = {}
    max_blengths = []
    idts = {}
    for ims, ms in enumerate(ms_name):
        dt = xr.open_datatree(ms, partition_schema=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"])

        idts[ms] = []
        freqs[ms] = {}
        times[ms] = {}
        radecs[ms] = {}
        chan_widths[ms] = {}

        # subtables
        for node in dt.children.values():
            if node.get("type") not in VISIBILITY_XDS_TYPES:
                continue
            base = node.attrs["data_groups"]["base"]
            fs = dt[base["field_and_source"]]  # absolute path lookup off the root
            radecs[ms] = fs.FIELD_PHASE_CENTER_DIRECTION.values
            ant = node["antenna_xds"]
            antpos[ms] = ant.ANTENNA_POSITION.values
            # get the actual xarray dataset
            ds = node.ds
            field_name = np.unique(ds.field_name.values).item()  # partitioned by FIELD_ID → single value
            scan_name = np.unique(ds.scan_name.values).item()  # partitioned by SCAN_NUMBER → single value
            spw_name = ds.frequency.attrs["spectral_window_name"]  # always in partition schema → single value
            idt = f"FIELD-{field_name}_SPW-{spw_name}_SCAN-{scan_name}"
            if (field_name is not None) and (field_name not in field_names):
                continue
            if (spw_name is not None) and (spw_name not in spw_names):
                continue
            if (scan_name is not None) and (scan_name not in scan_names):
                continue

            idts[ms].append(idt)
            freqs[ms][idt] = ds.frequency.values
            chan_widths[ms][idt] = ds.frequency.attrs["channel_width"]["data"]
            times[ms][idt] = ds.time.values
            uvw = ds.UVW.values
            max_blengths.append(np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max())

    max_blength = max(max_blengths)
    all_times = []
    freq_mapping = {}
    row_mapping = {}
    time_mapping = {}
    utimes = {}
    ms_chunks = {}
    for ims, ms in enumerate(ms_name):
        freq_mapping[ms] = {}
        row_mapping[ms] = {}
        time_mapping[ms] = {}
        utimes[ms] = {}
        ms_chunks[ms] = []
        gains[ms] = {}

        if gain_name is not None:
            gds = xds_from_zarr(gain_name[ims])

        for idt in idts[ms]:
            ilo = idt.find("FIELD") + 5
            ihi = idt.find("_")
            fid = int(idt[ilo:ihi])
            ilo = idt.find("DDID") + 4
            ihi = idt.rfind("_")
            ddid = int(idt[ilo:ihi])
            ilo = idt.find("SCAN") + 4
            scanid = int(idt[ilo:])
            freq = freqs[ms][idt]
            nchan_in = freq.size
            idx = (freq >= freq_min) & (freq <= freq_max)
            if not idx.any():
                continue
            idx0 = np.argmax(idx)  # returns index of first True element
            try:
                # returns zero if not idx.any()
                assert idx[idx0]
            except Exception:
                continue
            freq = freq[idx]
            nchan = freq.size
            if cpi in [-1, 0, None]:
                cpit = nchan
                cpi = nchan_in
            else:
                cpit = np.minimum(cpi, nchan)
                cpi = np.minimum(cpi, nchan_in)
            freq_mapping[ms][idt] = {}
            tmp = np.arange(idx0, idx0 + nchan, cpit)
            freq_mapping[ms][idt]["start_indices"] = tmp
            if cpit != nchan:
                tmp2 = np.append(tmp, [idx0 + nchan])
                freq_mapping[ms][idt]["counts"] = tmp2[1:] - tmp2[0:-1]
            else:
                freq_mapping[ms][idt]["counts"] = np.array((nchan,), dtype=int)

            time = times[ms][idt]
            if not np.all(time[1:] >= time[:-1]):
                raise NotImplementedError(
                    f"Time column in {ms} for {idt} is not monotonically "
                    f"non-decreasing. pfb-imaging currently requires "
                    f"time-ordered measurement sets. See "
                    f"https://github.com/ratt-ru/pfb-imaging/issues/153"
                )
            utime = np.unique(time)
            utimes[ms][idt] = utime
            all_times.append(utime)

            ntime = utimes[ms][idt].size
            if ipi in [0, -1, None]:
                ipit = ntime
            else:
                ipit = np.minimum(ipi, ntime)
            row_chunks, ridx, rcounts = chunkify_rows(time, ipit, daskify_idx=False)
            # these are for applying gains
            # essentially the number of rows per unique time
            row_mapping[ms][idt] = {}
            row_mapping[ms][idt]["start_indices"] = ridx
            row_mapping[ms][idt]["counts"] = rcounts

            freq_idx0 = freq_mapping[ms][idt]["start_indices"][0]
            if freq_idx0 != 0:
                freq_chunks = (freq_idx0,) + tuple(freq_mapping[ms][idt]["counts"])
            else:
                freq_chunks = tuple(freq_mapping[ms][idt]["counts"])
            freq_idxf = np.sum(freq_chunks)
            if freq_idxf != nchan_in:
                freq_chunkf = nchan_in - freq_idxf
                freq_chunks += (freq_chunkf,)

            try:
                assert np.sum(freq_chunks) == nchan_in
            except Exception:
                raise RuntimeError("Something went wrong constructing the frequency mapping. sum(fchunks != nchan)")

            ms_chunks[ms].append({"row": row_chunks, "chan": freq_chunks})

            time_mapping[ms][idt] = {}
            tmp = np.arange(0, ntime, ipit)
            time_mapping[ms][idt]["start_indices"] = tmp
            if ipit != ntime:
                tmp2 = np.append(tmp, [ntime])
                time_mapping[ms][idt]["counts"] = tmp2[1:] - tmp2[0:-1]
            else:
                time_mapping[ms][idt]["counts"] = np.array((ntime,))

            if gain_name is not None:
                gdsf = [dsg for dsg in gds if dsg.FIELD_ID == fid]
                gdsfd = [dsg for dsg in gdsf if dsg.DATA_DESC_ID == ddid]
                # gains may have been solved over scans
                if "SCAN_NUMBER" in gdsfd[0].attrs:
                    gdsfds = [dsg for dsg in gdsfd if dsg.SCAN_NUMBER == scanid]
                    try:
                        assert len(gdsfds) == 1
                    except Exception:
                        raise RuntimeError(
                            f"Gain datasets don't align for "
                            f"ms {ms} at FIELD_ID = {fid}, "
                            f"DATA_DESC_ID = {ddid}, "
                            f"SCAN_NUMBER = {scanid}. "
                            f"len(gds) = {len(gdsfds)}"
                        )
                    try:
                        assert (gdsfds[0].gain_time == utime).all()
                    except Exception:
                        raise ValueError(f"Mismatch between gain and MS utimes for {ms} at {idt}")

                    gains[ms][idt] = gdsfds[0]
                else:
                    try:
                        assert len(gdsfd) == 1
                    except Exception:
                        raise RuntimeError(
                            "Multiple gain datasets per FIELD and DDID but SCAN_NUMBER not in attributes"
                        )
                    t0 = utime[0]
                    tf = utime[-1]
                    gains[ms][idt] = gdsfd[0].sel(gain_time=slice(t0, tf))
            else:
                gains[ms][idt] = None

    return (
        row_mapping,
        freq_mapping,
        time_mapping,
        freqs,
        utimes,
        ms_chunks,
        gains,
        radecs,
        chan_widths,
        max_blength,
        antpos,
        poltype,
    )
