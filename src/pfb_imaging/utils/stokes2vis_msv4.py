import gc
from datetime import datetime, timezone

import numexpr as ne
import numpy as np
import ray
import xarray as xr

# from astropy.time import Time
from katbeam import JimBeam
from scipy import ndimage
from scipy.constants import c as lightspeed

from pfb_imaging import pfb_version
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.misc import to_unix_time
from pfb_imaging.utils.weighting import _compute_counts, weight_data


# wrapper to facilitate calling stokes_vis without ray
@ray.remote
def safe_stokes_vis(*args, **kwargs):
    # Loaded xarray objects (deserialised DataTree nodes and the datasets built
    # from them) end up in reference cycles that refcounting cannot free, so
    # each completed task would otherwise leave its fully loaded node behind
    # until a rare gen-2 GC. Ray workers run many tasks sequentially, ramping
    # RSS by ~a node per task (observed >100 GB/worker OOM); collect on exit.
    try:
        return stokes_vis(*args, **kwargs)
    finally:
        gc.collect()


def stokes_vis(
    dc1=None,
    dc2=None,
    operator=None,
    node_dt=None,
    scratch_store=None,
    bandid=None,
    timeid=None,
    msid=None,
    freq_out=None,
    precision="double",
    sigma_column=None,
    weight_column=None,
    product="I",
    chan_average=1,
    bda_decorr=1.0,
    max_field_of_view=3.0,
    beam_model=None,
    wgt_mode="minvar",
    max_blength=None,
    max_freq=None,
    nx_pad=None,
    ny_pad=None,
    cell_rad=None,
    baseline_group="all",
    data_group="base",
):
    """
    MSv4 version of stokes2vis.

    inputs:
        dc1, dc2, operator: data column names and operator to combine them with.
        node_dt: sub-node of datatree with the relevant time and frequency selection already applied.
        scratch_store: parent .scratch DataTree store; the piece is written to the group
            ``band{bandid:04d}_time{timeid:04d}/<piece_name>`` within it.
        bandid, timeid, msid: identifiers for the output dataset.
        nx_pad, ny_pad, cell_rad: padded uv-grid size and image cell (rad) for the per-piece
            COUNTS grid used to build imaging weights in pass 2.
        baseline_group: partition baseline-group label (single ``"all"`` group for now).
        freq_out: output frequency for selected data.
        precision: "single" or "double" for output data and weights.
        sigma_column: name of column containing sigma values to convert to weights.
        weight_column: name of column containing weights to use instead of sigma.
        product: Stokes product. Can be any subset of IQUV.
        chan_average: number of channels to average together.
        bda_decorr: decorrelation threshold for baseline dependent averaging.
        max_field_of_view: maximum field of view in degrees for baseline dependent averaging.
        beam_model: model to use for beam correction. Can be "katbeam" or None.
        wgt_mode: weighting mode to use in weight_data. Can be "l2" or "minvar".
    """
    # Load only the variables this task consumes. The MSv4 node exposes every
    # correlated-data column (VISIBILITY, CORRECTED_DATA, MODEL_DATA, ...) plus
    # auxiliaries like TIME_CENTROID; a blanket node_dt.load() reads them all
    # from disk and holds them in memory even though only the selected
    # data/weight/flag columns are used. The small antenna_xds/field_and_source
    # subtables are loaded lazily on access below.
    needed = [dc1]
    if dc2 is not None:
        needed.append(dc2)
    if sigma_column is not None:
        needed.append(sigma_column)
    elif weight_column is not None:
        needed.append(weight_column)
    needed += ["FLAG", "UVW"]
    for name in ("EFFECTIVE_INTEGRATION_TIME", "INTEGRATION_TIME", "CHANNEL_WIDTH"):
        if name in node_dt.ds.data_vars:
            needed.append(name)
    needed = list(dict.fromkeys(needed))
    ds = node_dt.ds[needed].load()
    field_name = np.unique(ds.field_name.values).item()
    spw_name = ds.frequency.attrs["spectral_window_name"]
    scan_name = np.unique(ds.scan_name.values).item()
    # partition identity: (msid, field, spw, baseline_group); scan distinguishes fine pieces
    piece_name = f"ms{msid:04d}_fid{field_name}_spw{spw_name}_bg{baseline_group}_scan{scan_name}"

    if precision.lower() == "single":
        real_type = np.float32
        complex_type = np.complex64
    elif precision.lower() == "double":
        real_type = np.float64
        complex_type = np.complex128
    else:
        raise ValueError(f"Unsupported precision {precision!r}; expected 'single' or 'double'")

    data = getattr(ds, dc1).values
    if dc2 is not None:
        if operator not in ("+", "-"):
            raise ValueError(f"Unsupported operator {operator!r}; expected '+' or '-'")
        # can't use out here, data are immutable as passed through Ray's object store
        data = ne.evaluate(
            f"data {operator} data2",
            local_dict={"data": data, "data2": getattr(ds, dc2).values},
            casting="same_kind",
        )

    # we now get utimes and freqs from the coords
    freq = ds.frequency.values
    freq_min = freq.min()
    freq_max = freq.max()
    ntime, nbl, nchan, ncorr = data.shape
    nrow = ntime * nbl
    utime = ds.time.values
    time_out = np.mean(utime)

    # per-row integration time. MSv4 carries the actual per-sample value in
    # EFFECTIVE_INTEGRATION_TIME (time, baseline) which ravels to row order; prefer
    # it over the regular-grid integration_time attr (xarray-ms signals an irregular
    # grid with NaN in that attr). Fall back to the legacy INTEGRATION_TIME var, then
    # the scalar attr. (sjperkins, PR #252 review.)
    if "EFFECTIVE_INTEGRATION_TIME" in ds.data_vars:
        interval = ds.EFFECTIVE_INTEGRATION_TIME.values.ravel()
    elif "INTEGRATION_TIME" in ds.data_vars:
        interval = ds.INTEGRATION_TIME.values.ravel()
    else:
        it_attr = ds.time.attrs.get("integration_time", {}).get("data")
        interval = np.full(nrow, it_attr, dtype=np.float64)

    # get MSv2 style ANTENNA1 and ANTENNA2 as indices into the antenna_xds
    # subtable. A plain np.unique on the names only establishes baseline relations
    # and need not map to antenna_xds order, which is wrong as soon as antenna_xds
    # variables (e.g. ANTENNA_POSITION) are indexed by these. searchsorted against
    # the antenna_xds antenna_name ordering gives the canonical mapping.
    # (sjperkins, PR #252 review.)
    ant1_names = ds.baseline_antenna1_name.values
    ant2_names = ds.baseline_antenna2_name.values
    ant_names = node_dt["antenna_xds"].antenna_name.values
    order = np.argsort(ant_names)
    sorted_n = ant_names[order]
    ant1_bl = order[np.searchsorted(sorted_n, ant1_names)]
    ant2_bl = order[np.searchsorted(sorted_n, ant2_names)]
    time, ant1, ant2 = np.broadcast_arrays(utime[:, None], ant1_bl[None, :], ant2_bl[None, :])
    time = time.ravel()
    # averaging routines expect int32 antenna indices
    ant1 = ant1.ravel().astype(np.int32)
    ant2 = ant2.ravel().astype(np.int32)
    uvw = ds.UVW.values.reshape(nrow, 3)
    flag = ds.FLAG.values.reshape(nrow, nchan, ncorr)

    # MS may contain auto-correlations
    frow = ant1 == ant2

    # combine flag and frow
    flag = np.logical_or(flag, frow[:, None, None])

    # we rely on this to check the number of output bands and
    # to ensure we don't end up with fully flagged chunks
    if flag.all():
        return None

    if sigma_column is not None:
        weight = ne.evaluate("1.0/sigma**2", local_dict={"sigma": getattr(ds, sigma_column).values})
        weight = weight.reshape(nrow, nchan, ncorr)
    elif weight_column is not None:
        weight = getattr(ds, weight_column).values.reshape(nrow, nchan, ncorr)
    else:
        weight = np.ones((nrow, nchan, ncorr), dtype=real_type)

    # antenna positions
    antpos = node_dt["antenna_xds"].ANTENNA_POSITION.values
    nant = antpos.shape[0]

    # polarization types
    if set(ds.polarization.values).issubset({"XX", "XY", "YX", "YY"}):
        poltype = "linear"
    elif set(ds.polarization.values).issubset({"RR", "RL", "LR", "LL"}):
        poltype = "circular"
    else:
        raise ValueError("Unknown polarization types")

    # get phase dir from the field_and_source subtable of the chosen data group,
    # selecting the row that matches this partition's field_name rather than
    # assuming index 0. (sjperkins, PR #252 review.)
    grp = node_dt.ds.attrs["data_groups"][data_group]
    fns = node_dt[grp["field_and_source"].rsplit("/", 1)[-1]].ds
    radec = fns.FIELD_PHASE_CENTER_DIRECTION.sel(field_name=field_name).values

    if data.dtype != complex_type:
        data = data.astype(complex_type)

    if weight.dtype != real_type:
        weight = weight.astype(real_type)

    # Fake jones for now
    jones = np.ones((ntime, nant, nchan, 1, 2), dtype=complex_type)

    # need to rewrite weight_data for MSv4, this is temporary
    tbin_idx = np.arange(ntime) * nbl
    tbin_counts = np.full(ntime, nbl)

    # apply gains and convert to Stokes
    data = data.reshape(nrow, nchan, ncorr)
    data, weight = weight_data(
        data,
        weight,
        flag,
        jones,
        tbin_idx,
        tbin_counts,
        ant1,
        ant2,
        poltype,
        product,
        str(ncorr),
        wgt_mode,
    )

    # TODO - check if wsum for any of the correlations is zero
    # This happens e.g. if selecting out diagonal correlations
    # with QC and making CORRECTED_WEIGHTS

    # do after weight_data otherwise mappings need to be recomputed
    # drop fully flagged rows
    mrow = ~frow
    data = data[mrow]
    time = time[mrow]
    interval = interval[mrow]
    ant1 = ant1[mrow]
    ant2 = ant2[mrow]
    uvw = uvw[mrow]
    flag = flag[mrow]
    weight = weight[mrow]

    # number of output correlations will be set by required Stokes products
    ncorr = data.shape[-1]
    # we need this for averaging
    flag = np.tile(flag.any(axis=-1, keepdims=True), (1, 1, ncorr))

    # set corr coords (removing duplicates and sorting)
    corr = list("".join(dict.fromkeys(sorted(product))))
    ncorr = len(corr)

    # CHANNEL_WIDTH will only be in data_vars if it's not regular, otherwise it's an attr of frequency coord
    if "CHANNEL_WIDTH" in ds.data_vars:
        chan_width = ds.CHANNEL_WIDTH.values
    else:
        cw = ds.frequency.attrs["channel_width"]["data"]
        chan_width = np.full(ds.frequency.size, cw, dtype=np.float64)

    # simple average over channels
    if chan_average > 1:
        from africanus.averaging import time_and_channel

        res = time_and_channel(
            time,
            interval,
            ant1,
            ant2,
            uvw=uvw,
            flag=flag,
            weight_spectrum=weight,
            visibilities=data,
            chan_freq=freq,
            chan_width=chan_width,
            time_bin_secs=1e-15,
            chan_bin_size=chan_average,
        )

        data = res.visibilities
        weight = res.weight_spectrum
        flag = res.flag
        freq = res.chan_freq
        chan_width = res.chan_width
        uvw = res.uvw
        nchan = freq.size

    if bda_decorr < 1:
        from africanus.averaging import bda

        res = bda(
            time,
            interval,
            ant1,
            ant2,
            uvw=uvw,
            flag=flag,
            weight_spectrum=weight,
            visibilities=data,
            chan_freq=freq,
            chan_width=chan_width,
            decorrelation=bda_decorr,
            min_nchan=freq.size,
            max_fov=max_field_of_view,
        )

        offsets = res.offsets
        uvw = res.uvw[offsets[:-1], :]
        weight = res.weight_spectrum.reshape(-1, nchan, ncorr)
        data = res.visibilities.reshape(-1, nchan, ncorr)
        flag = res.flag.reshape(-1, nchan, ncorr)

    flag = flag.any(axis=-1)
    mask = (~flag).astype(np.uint8)

    # TODO - better beam interpolation
    if max_blength is None or max_freq is None:
        raise ValueError("max_blength and max_freq must be provided to size the beam grid")
    # NB: the beam evaluation grid is sized at the critically-sampled (Nyquist)
    # cell, which is a *separate* quantity from the imaging cell_rad passed in for
    # the COUNTS uv-grid below. Keep them in distinct variables so the imaging
    # cell (used to bin COUNTS) is not clobbered -- counts must be gridded on the
    # same cell that grid_partition's counts_to_weights later looks them up on,
    # i.e. the imaging cell (matching the legacy image_data_products path).
    fov = max_field_of_view
    beam_cell_rad = 1.0 / (max_blength * max_freq / lightspeed)
    beam_cell_deg = np.rad2deg(beam_cell_rad)
    npix = int(fov / beam_cell_deg)
    l_beam = (-(npix // 2) + np.arange(npix)) * beam_cell_deg
    m_beam = (-(npix // 2) + np.arange(npix)) * beam_cell_deg
    if beam_model is None:
        beam = np.ones((ncorr, npix, npix), dtype=real_type)
    elif beam_model.lower() == "katbeam":
        if freq_min >= 8.5e8 and freq_max <= 1.8e9:
            beamo = JimBeam("MKAT-AA-L-JIM-2020")
        elif freq_min >= 5.4e8 and freq_max <= 1.1e9:
            beamo = JimBeam("MKAT-AA-UHF-JIM-2020")
        # elif freq_min >= 8.56e8 and freq_max <= 1.71179102e+09:
        #     beamo = JimBeam('MKAT-AA-S-JIM-2020')
        else:
            raise ValueError("Freq range not covered by katbeam")
        xx, yy = np.meshgrid(l_beam, m_beam, indexing="ij")
        # katbeam expects freq in MHz
        fmhz = freq_out / 1e6
        beam = np.zeros((ncorr, npix, npix), dtype=np.float64)
        for i, product in enumerate(corr):
            beam0 = getattr(beamo, product)(xx, yy, fmhz)
            step = 25
            angles = np.linspace(0, 359, step)
            for angle in angles:
                beam[i] += ndimage.rotate(beam0, angle, reshape=False, mode="nearest")
            beam[i] /= angles.size
            # how to normalise the center for other Stokes products?
            # beam[i] /= beam[i].max()
    else:
        raise ValueError(f"Unknown beam model {beam_model}")

    # for operations that follow it will be preferable to have the corr axis
    # first for contiguity
    vis_cf = data.transpose(2, 0, 1)
    wgt_cf = weight.transpose(2, 0, 1)

    # signs occording to wgridder conventions
    flip_u, flip_v, flip_w, _, _ = wgridder_conventions(0, 0)
    counts = _compute_counts(
        uvw,
        freq,
        mask,
        wgt_cf,
        nx_pad,
        ny_pad,
        cell_rad,
        cell_rad,
        wgt_cf.dtype,
        ngrid=1,
        usign=1.0 if flip_u else -1.0,
        vsign=1.0 if flip_v else -1.0,
    )

    data_vars = {}
    data_vars["VIS"] = (("corr", "row", "chan"), vis_cf)
    data_vars["WEIGHT"] = (("corr", "row", "chan"), wgt_cf)
    data_vars["MASK"] = (("row", "chan"), mask)
    data_vars["UVW"] = (("row", "three"), uvw)
    data_vars["FREQ"] = (("chan",), freq)
    data_vars["BEAM"] = (("corr", "l_beam", "m_beam"), beam)
    data_vars["COUNTS"] = (("corr", "u", "v"), counts)

    coords = {
        "chan": (("chan",), freq),
        "l_beam": (("l_beam",), l_beam),
        "m_beam": (("m_beam",), m_beam),
        "corr": (("corr",), corr),
    }

    unix_time = to_unix_time(time_out)
    utc = datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    attrs = {
        "pfb-imaging-version": pfb_version,
        "ra": radec[0],
        "dec": radec[1],
        "msid": msid,
        "field_name": field_name,
        "spw_name": spw_name,
        "baseline_group": baseline_group,
        "scan_name": scan_name,
        "freq_out": freq_out,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "bandid": bandid,
        "time_out": time_out,
        "time_min": utime.min(),
        "time_max": utime.max(),
        "timeid": timeid,
        "product": product,
        "utc": utc,
        "max_freq": max_freq,
        "max_blength": max_blength,
        "beam_model": beam_model,
    }

    out_ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    group = f"band{bandid:04d}_time{timeid:04d}/{piece_name}"
    # consolidated=False: many workers write distinct leaf groups into the shared
    # scratch store concurrently. Per-write consolidation would race on the single
    # root .zmetadata (torn JSON -> JSONDecodeError on networked filesystems); the
    # band/time parent groups are pre-created and the driver consolidates once
    # after all pass-1 tasks finish (see core.imager).
    out_ds.to_zarr(scratch_store, group=group, mode="a", consolidated=False)
    return bandid, timeid, piece_name
