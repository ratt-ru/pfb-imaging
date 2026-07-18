import gc
import os
import resource
from datetime import datetime, timezone

import numexpr as ne
import numpy as np
import psutil
import ray
import xarray as xr
from africanus.averaging import bda, time_and_channel
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from katbeam import JimBeam
from meerkat_beams.utils import BeamWizard
from scipy import ndimage
from scipy.constants import c as lightspeed

from pfb_imaging import pfb_version
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.beam import eval_beam, reproject_and_interp_scat_beam
from pfb_imaging.utils.misc import parse_sky_coords, radec_to_lm, to_mjd_time
from pfb_imaging.utils.weighting import _compute_counts, as_contiguous_readonly_view, weight_data


def _release_ms_caches():
    """Evict xarray-ms's process-level arcae Table cache.

    xarray-ms keeps every opened arcae Table in a class-level Multiton cache
    with a 300 s *inactivity* TTL. Sequential Ray tasks each open their own
    partition selection under a distinct cache key, and a busy worker never
    goes idle long enough for entries to expire, so post-gc worker RSS
    ratchets by ~the task's read footprint per task (observed +3.5 GB/task
    to ~39 GB/worker on an 8-band 10-scan run) and the retained tables then
    sit there for the whole of pass 2. The cache holds strong references --
    gc cannot reclaim it -- so evict it explicitly between tasks; the next
    task reconstructs its own tables anyway. Private API by necessity:
    degrade gracefully if the pattern package changes.
    """
    try:
        # deferred: private xarray-ms internals; degrade gracefully if absent
        from rarg_python_patterns.multiton import Multiton

        with Multiton._INSTANCE_LOCK:
            Multiton._INSTANCE_CACHE.clear()
            Multiton._EXPIRY_HEAP.clear()
    except Exception:
        pass


# wrapper to facilitate calling stokes_vis without ray
@ray.remote
def safe_stokes_vis(*args, **kwargs):
    # Loaded xarray objects (deserialised DataTree nodes and the datasets built
    # from them) end up in reference cycles that refcounting cannot free, so
    # each completed task would otherwise leave its fully loaded node behind
    # until a rare gen-2 GC. Ray workers run many tasks sequentially, ramping
    # RSS by ~a node per task (observed >100 GB/worker OOM); collect on exit.
    try:
        ret = stokes_vis(*args, **kwargs)
    finally:
        _release_ms_caches()
        gc.collect()
    # Per-task memory telemetry, measured *after* the collect: a post-gc RSS
    # that ratchets up across a worker's sequential tasks indicates retention
    # below Python (arcae/casacore caches, allocator arenas), which gc cannot
    # touch. ru_maxrss is the process lifetime high-water mark (kB on Linux).
    mem = {
        "pid": os.getpid(),
        "rss_gb": psutil.Process().memory_info().rss / 2**30,
        "peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30,
    }
    return ret, mem


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
    radec_new=None,
    target=None,
    nx=None,
    ny=None,
    nthreads=1,
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
        radec_new: optional (ra, dec) in radians to rephase to (the common mosaic
            phase centre / tangent point). None means image about the field centre.
        target: optional image-centre target ('HH:MM:SS,DD:MM:SS' or an astropy
            body); resolved to an in-plane (l0, m0) at this piece's own time.
        nx, ny: output image size; the piece's BEAM is placed on this grid here
            in pass 1 (#281) so pass 2 consumes it as is.
        nthreads: threads available to the task (parallelises the COUNTS gridding).
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

    # CHANNEL_WIDTH will only be in data_vars if it's not regular, otherwise it's an attr of frequency coord
    if "CHANNEL_WIDTH" in ds.data_vars:
        chan_width = ds.CHANNEL_WIDTH.values
    else:
        cw = ds.frequency.attrs["channel_width"]["data"]
        chan_width = np.full(freq.size, cw, dtype=np.float64)

    # Everything this task needs is now extracted into plain numpy arrays;
    # release the loaded input Dataset/DataTree so the raw c64/f32 inputs die
    # as soon as their converted copies exist below, instead of staying pinned
    # until task exit (the legacy dask-backed path never holds them at all).
    # The collect is required: deserialised xarray objects sit in reference
    # cycles that plain refcounting cannot free.
    del node_dt, ds, fns, grp
    gc.collect()

    # MS may contain auto-correlations
    frow = ant1 == ant2

    # combine flag and frow
    flag = np.logical_or(flag, frow[:, None, None])

    # we rely on this to check the number of output bands and
    # to ensure we don't end up with fully flagged chunks
    if flag.all():
        return None

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

    # rephase to the common phase centre (mosaic tangent point) BEFORE
    # weighting, averaging and the COUNTS gridding so weights, decorrelation
    # and uv counts are all consistent with the new centre automatically.
    # chgcentre-style w-difference, as in stokes2im.
    if radec_new is not None and not np.allclose(radec_new, np.asarray(radec).squeeze(), rtol=0, atol=1e-9):
        # deferred: synthesize_uvw pulls pyrap (python-casacore); only
        # mosaic / phase_dir runs pay the import
        from pfb_imaging.utils.astrometry import synthesize_uvw

        try:
            # MSv4 time is unix seconds; casacore measures wants MJD seconds
            mjd_time = to_mjd_time(time)
            uvw_new = synthesize_uvw(antpos, mjd_time, ant1, ant2, np.asarray(radec_new))
            # Recompute the *old* UVW via the same measures call (rather than
            # diffing against the MS's own recorded UVW) so any systematic
            # offset between pyrap's earth-orientation handling and whatever
            # produced the MS's UVW (DUT1/precession-nutation model, etc.)
            # cancels in the difference instead of contaminating w_diff.
            # That mismatch is small in absolute terms (~1e-5 relative to the
            # baseline length) but scales with baseline length, so at the
            # longest baselines it is comparable to the true w-difference for
            # small rephasing offsets and visibly decorrelates off-axis
            # sources if left in.
            uvw_old = synthesize_uvw(antpos, mjd_time, ant1, ant2, np.asarray(radec).squeeze())
        except ImportError as e:
            raise ImportError(
                "Rephasing (multi-field mosaics or phase_dir) requires python-casacore. Install pfb-imaging[full]."
            ) from e
        w_diff = uvw_new[:, 2:] - uvw_old[:, 2:]
        freqfactor = -2j * np.pi * freq[None, :] / lightspeed
        # data may be a read-only view out of Ray's object store; don't use *=
        data = data * np.exp(freqfactor * w_diff)[:, :, None]
        # store the differentially rotated MS coordinates rather than the
        # wholesale synthesized ones: the sampling stays anchored to the MS's
        # own UVW (the same measures-vs-MS systematic that w_diff cancels
        # would otherwise re-enter through the coordinates)
        uvw = uvw + (uvw_new - uvw_old)

    data, weight = weight_data(
        as_contiguous_readonly_view(data),
        as_contiguous_readonly_view(weight),
        as_contiguous_readonly_view(flag),
        as_contiguous_readonly_view(jones),
        as_contiguous_readonly_view(tbin_idx),
        as_contiguous_readonly_view(tbin_counts),
        as_contiguous_readonly_view(ant1),
        as_contiguous_readonly_view(ant2),
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

    # simple average over channels
    if chan_average > 1:
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

    # ---- primary beam on the output image grid (pass-1 beam, #281) ----
    # The (rotation-averaged) beam is evaluated about the FIELD's own pointing
    # -- where the antennas point regardless of rephasing -- on a small grid,
    # then placed onto the output image grid HERE in pass 1 (parallel over
    # pieces); pass 2 consumes the stored (corr, ny, nx) BEAM as is.
    if max_blength is None or max_freq is None:
        raise ValueError("max_blength and max_freq must be provided to size the beam grid")
    if nx is None or ny is None:
        raise ValueError("nx and ny must be provided to place the beam on the image grid")
    cell_deg = np.rad2deg(cell_rad)
    radec_grid = np.asarray(radec_new) if radec_new is not None else radec  # image tangent point

    # --target: in-plane image-centre offset, resolved at this piece's own time
    if target is not None:
        if len(target.split(",")) == 2:
            tradec = parse_sky_coords(target)
        else:
            # deferred: get_coordinates pulls pyrap (python-casacore)
            from pfb_imaging.utils.astrometry import get_coordinates

            # named body: D13 -- get_coordinates expects MJD seconds
            tradec = np.array(get_coordinates(to_mjd_time(time_out), target=target))
        l0, m0 = (float(v) for v in radec_to_lm(tradec, radec_grid))
    else:
        l0, m0 = 0.0, 0.0

    # the wgridder's geometric n-term is folded into the stored beam: the
    # effective image-plane response is B(l,m)/n(l,m), computed on exactly
    # the coordinates ducc uses (absolute w.r.t. the phase centre, target
    # offset included). Rationale (wiki design-decisions D22, pinned by
    # tests/test_hessian_nterm.py): diag(B/n) GtWG diag(B/n) is IDENTICAL to
    # gridding with divide_by_n=True, but the diagonal rides in HessianTree's
    # beam slots where the PSF-convolution approximation captures it exactly,
    # whereas divide_by_n=True buries an image-plane envelope inside the
    # gridded PSF that a convolution cannot represent (4-25% worse Hessian
    # accuracy, growing with fov). It also sidesteps ducc's divide_by_n
    # silently no-oping when do_wgridding=False. Every ducc call in the
    # imager+deconv path therefore stays divide_by_n=False, and the
    # deconvolved MODEL comes out in intrinsic flux.
    _, _, _, x0_, y0_ = wgridder_conventions(l0, m0)
    x_abs = (-(nx / 2) + np.arange(nx)) * cell_rad + x0_
    y_abs = (-(ny / 2) + np.arange(ny)) * cell_rad + y0_
    yy_abs, xx_abs = np.meshgrid(y_abs, x_abs, indexing="ij")  # (ny, nx)
    nlm = np.sqrt(1.0 - xx_abs**2 - yy_abs**2)

    if beam_model is None:
        # no aperture beam: the stored response is 1/n (compresses well in zarr)
        beam = (1.0 / nlm)[None, :, :].astype(real_type) * np.ones((ncorr, 1, 1), dtype=real_type)
    else:
        # field-centred evaluation on a small grid first
        if isinstance(beam_model, BeamWizard):
            # the wizard ships detached (BDS-only); the parallactic angles
            # need the FIELD's pointing centre, set per piece here
            rvec = np.asarray(radec).squeeze()
            beam_model.set_field_centre(SkyCoord(ra=rvec[0] * u.rad, dec=rvec[1] * u.rad))
            bds = beam_model.bds
            l_beam = bds.X.values  # deg, East positive
            m_beam = bds.Y.values  # deg, North positive
            # D13: MSv4 time is unix seconds; astropy Time wants MJD days
            t_beam = Time(to_mjd_time(utime) / 86400.0, format="mjd")
            beam_small = np.zeros((ncorr, m_beam.size, l_beam.size), dtype=np.float64)
            for i, p in enumerate(corr):
                # returns (Y, X)-ordered maps (docs/wiki/image-and-beam-orientation.md);
                # signature kept stable so meerkat-beams can grow weighted
                # time/freq averaging underneath (#281)
                beam_small[i], _ = beam_model.get_rotation_averaged_beam(
                    l=l_beam,
                    m=m_beam,
                    times=t_beam,
                    freq=np.atleast_1d(freq_out),
                    time_stepping=1,
                    pixel_stepping=1,
                    var="nstokes",
                    i=p,
                    j=p,
                    verbose=0,
                )
        elif str(beam_model).lower() == "katbeam":
            # NB: the evaluation grid is sized at the critically-sampled
            # (Nyquist) cell, a *separate* quantity from the imaging cell_rad
            # used for the COUNTS uv-grid below.
            fov = max_field_of_view
            beam_cell_deg = np.rad2deg(1.0 / (max_blength * max_freq / lightspeed))
            npix = int(fov / beam_cell_deg)
            l_beam = (-(npix // 2) + np.arange(npix)) * beam_cell_deg
            m_beam = (-(npix // 2) + np.arange(npix)) * beam_cell_deg
            if freq_min >= 8.5e8 and freq_max <= 1.8e9:
                beamo = JimBeam("MKAT-AA-L-JIM-2020")
            elif freq_min >= 5.4e8 and freq_max <= 1.1e9:
                beamo = JimBeam("MKAT-AA-UHF-JIM-2020")
            else:
                raise ValueError("Freq range not covered by katbeam")
            # (Y, X)/(m, l) order end to end (wiki D19): axis 0 is m/Dec-like;
            # katbeam evaluates pointwise, so beam0 takes the (m, l) shape
            mm, ll = np.meshgrid(m_beam, l_beam, indexing="ij")
            # katbeam expects freq in MHz
            fmhz = freq_out / 1e6
            beam_small = np.zeros((ncorr, npix, npix), dtype=np.float64)
            for i, p in enumerate(corr):
                beam0 = getattr(beamo, p)(ll, mm, fmhz)
                step = 25
                angles = np.linspace(0, 359, step)
                for angle in angles:
                    beam_small[i] += ndimage.rotate(beam0, angle, reshape=False, mode="nearest")
                beam_small[i] /= angles.size
        else:
            raise ValueError(f"Unknown beam model {beam_model}")

        offset_centre = (not np.allclose(radec_grid, radec, rtol=0, atol=1e-9)) or l0 != 0.0 or m0 != 0.0
        if offset_centre:
            # mosaic/facet: SIN->SIN reprojection from the field-centred grid
            # onto the image grid (fills 0 outside the beam's coverage, so an
            # off-grid partition contributes nothing where its beam is unknown)
            beam = reproject_and_interp_scat_beam(
                beam_small,
                l_beam,
                m_beam,
                radec,
                radec_grid,
                cell_deg,
                nx,
                ny,
                "".join(corr),
                l0=l0,
                m0=m0,
            ).astype(real_type)
        else:
            # single-field path: same bilinear interpolant grid_partition used
            # to apply (fill 1.0 outside the small grid, as before)
            x = (-(nx / 2) + np.arange(nx)) * cell_rad
            y = (-(ny / 2) + np.arange(ny)) * cell_rad
            yy, xx = np.meshgrid(np.rad2deg(y), np.rad2deg(x), indexing="ij")
            beam = np.zeros((ncorr, ny, nx), dtype=real_type)
            for c in range(ncorr):
                beam[c] = eval_beam(beam_small[c], m_beam, l_beam, yy, xx)
        # fold the n-term (see the D22 comment above)
        beam = (beam / nlm[None, :, :]).astype(real_type)

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
        ngrid=nthreads,
        usign=1.0 if flip_u else -1.0,
        vsign=1.0 if flip_v else -1.0,
    )

    data_vars = {}
    data_vars["VIS"] = (("corr", "row", "chan"), vis_cf)
    data_vars["WEIGHT"] = (("corr", "row", "chan"), wgt_cf)
    data_vars["MASK"] = (("row", "chan"), mask)
    data_vars["UVW"] = (("row", "three"), uvw)
    data_vars["FREQ"] = (("chan",), freq)
    data_vars["BEAM"] = (("corr", "y", "x"), beam)
    data_vars["COUNTS"] = (("corr", "u", "v"), counts)

    coords = {
        "chan": (("chan",), freq),
        "corr": (("corr",), corr),
    }

    # the MSv4 time coordinate is already unix seconds (xarray-ms converts
    # MSv2 MJD seconds); no epoch shift needed here
    utc = datetime.fromtimestamp(time_out, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # ra/dec: the tangent point downstream consumers grid about (the new
    # phase centre when rephasing); ra0/dec0: the field's original pointing
    # (beam centre), needed for the (future) pass-2 beam reprojection (#281)
    radec = np.asarray(radec).squeeze()
    attrs = {
        "pfb-imaging-version": pfb_version,
        "ra": radec_new[0] if radec_new is not None else radec[0],
        "dec": radec_new[1] if radec_new is not None else radec[1],
        "ra0": radec[0],
        "dec0": radec[1],
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
        "beam_model": beam_model
        if beam_model is None or isinstance(beam_model, str)
        else str(getattr(beam_model, "band", type(beam_model).__name__)),
        "l0": l0,
        "m0": m0,
        # the stored BEAM is the effective image-plane response B/n, NOT the
        # bare primary beam (wiki design-decisions D22)
        "beam_includes_n": True,
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
