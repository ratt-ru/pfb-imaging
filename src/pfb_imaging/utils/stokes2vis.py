import gc
from datetime import datetime, timezone

import numexpr as ne
import numpy as np
import ray
import xarray as xr
from africanus.averaging import bda, time_and_channel
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from katbeam import JimBeam
from meerkat_beams.utils import BeamWizard, beam_on_grid
from pfb_model_spec.utils.degrid import pixel_lm
from rarg_python_patterns.multiton import Multiton
from scipy.constants import c as lightspeed

from pfb_imaging import pfb_version
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.beam import BEAM_DISH_DIAMETER, BEAM_PA_TOLERANCE_DEG, beam_grid
from pfb_imaging.utils.memprof import profile_task, task_memory
from pfb_imaging.utils.misc import parse_sky_coords, radec_to_lm, to_mjd_time
from pfb_imaging.utils.msv4 import load_detached
from pfb_imaging.utils.weighting import as_contiguous_readonly_view, compute_sparse_counts, weight_data


# wrapper to facilitate calling stokes_vis without ray
@ray.remote
def safe_stokes_vis(*args, **kwargs):
    # Loaded xarray objects (deserialised DataTree nodes and the datasets built
    # from them) end up in reference cycles that refcounting cannot free, so
    # each completed task would otherwise leave its fully loaded node behind
    # until a rare gen-2 GC. Ray workers run many tasks sequentially, ramping
    # RSS by ~a node per task (observed >100 GB/worker OOM); collect on exit.
    with profile_task("stokes_vis") as clock:
        try:
            ret = stokes_vis(*args, **kwargs)
        finally:
            gc.collect()
    # Per-task memory telemetry, measured *after* the collect: a post-gc anon
    # RSS that ratchets up across a worker's sequential tasks indicates
    # retention below Python (arcae/casacore caches, allocator arenas), which
    # gc cannot touch; shm growth is Ray object-store pages (see memprof).
    return ret, {**task_memory(), **clock}


def real_beam_maps(maps):
    """Stack per-correlation beam maps as real float64, reporting max|Im|/max|Re|.

    meerkat-beams returns complex Stokes beams for the MeerKAT-MeerKAT+ cross
    group: a mixed baseline has no single Jones matrix. We store Re(B) -- the
    imaginary part is the antisymmetric response, which integrates to zero for a
    real sky over conjugate baseline orderings (wiki D46) -- and keep every
    downstream operator, the beam FITS path and the uniform-precision invariant
    (D27) real.

    The conversion is explicit on purpose. Assigning a complex array into a
    float array discards the imaginary part with only a ComplexWarning, so the
    decision would otherwise happen by accident and the ratio below -- the only
    evidence that Re(B) is a safe approximation on real data -- would never be
    computed.

    Args:
        maps: one 2D beam map per correlation, real or complex.

    Returns:
        (stacked float64 array of shape (ncorr, ny, nx), max|Im|/max|Re| over
        correlations; 0.0 when every map is real).
    """
    ratio = 0.0
    real_maps = []
    for m in maps:
        m = np.asarray(m)
        if np.iscomplexobj(m):
            denom = np.abs(m.real).max()
            if denom > 0:
                ratio = max(ratio, float(np.abs(m.imag).max() / denom))
            m = m.real
        real_maps.append(np.asarray(m, dtype=np.float64))
    return np.stack(real_maps, axis=0), ratio


def integration_edges(time: np.ndarray, interval: np.ndarray) -> tuple[float, float]:
    """Outermost integration edges of a chunk's rows (#327).

    Taken per row, so an irregular grid's edge integrations are exact rather
    than padded by a typical interval. NaN rows (xarray-ms's fill for absent
    data) are ignored; with no finite interval the centroids' extent is returned.

    Args:
        time: Per-row centroid times.
        interval: Per-row integration times, same shape as `time`.

    Returns:
        `(lo, hi)`: the earliest row start and the latest row end.
    """
    lo, hi = time - interval / 2, time + interval / 2
    if not np.isfinite(lo).any():
        return float(np.nanmin(time)), float(np.nanmax(time))
    return float(np.nanmin(lo)), float(np.nanmax(hi))


def beam_on_image_grid(
    beam_small: np.ndarray,
    l_beam: np.ndarray,
    m_beam: np.ndarray,
    radec: np.ndarray,
    radec_grid: np.ndarray,
    cell_rad: float,
    nx: int,
    ny: int,
    l0: float = 0.0,
    m0: float = 0.0,
    nthreads: int = 1,
) -> np.ndarray:
    """Place a field-centred beam onto the (Y, X) output image grid.

    One path for every partition: each pixel's sky ``(l, m)`` under the
    wgridder conventions (East at low x, North at high y; the same ``pixel_lm``
    degrid evaluates its beam on, #355) is rotated from the image tangent point
    to the field's pointing and bilinearly interpolated in ``beam_small``
    (meerkat-beams ``beam_on_grid``). A pixel outside the small grid is 0; the
    driver sizes the grid to cover every image within ``--max-field-of-view``,
    less what the beam cube cannot reach (``field_beam_axis``, D66).

    Args:
        beam_small: `(ncorr, m_beam.size, l_beam.size)` (Y, X)-ordered maps
            centred on the field's pointing.
        l_beam: 1D l of the beam columns (deg, East positive), uniform.
        m_beam: 1D m of the beam rows (deg, North positive), uniform.
        radec: The field's pointing centre (rad).
        radec_grid: The image tangent point (rad).
        cell_rad: Image cell size (rad).
        nx: Image pixels along x.
        ny: Image pixels along y.
        l0: Image-centre offset from the tangent point (rad).
        m0: Image-centre offset from the tangent point (rad).
        nthreads: Threads over output rows.

    Returns:
        `(ncorr, ny, nx)` float64 beam on the image grid, without the n-term.
    """
    flip_u, flip_v, _, x0, y0 = wgridder_conventions(l0, m0)
    geom = {"nx": nx, "ny": ny, "cell_rad": cell_rad, "x0": x0, "y0": y0, "flip_u": flip_u, "flip_v": flip_v}
    l_px, m_px = pixel_lm(geom)
    return beam_on_grid(
        np.asarray(beam_small, dtype=np.float64),
        l_beam,
        m_beam,
        SkyCoord(ra=radec[0] * u.rad, dec=radec[1] * u.rad),
        SkyCoord(ra=radec_grid[0] * u.rad, dec=radec_grid[1] * u.rad),
        np.rad2deg(l_px),
        np.rad2deg(m_px),
        nthreads=nthreads,
    )


def partition_beam(
    ds: xr.Dataset,
    radec_grid: np.ndarray,
    cell_rad: float,
    nx: int,
    ny: int,
    l0: float = 0.0,
    m0: float = 0.0,
    dtype=np.float64,
    nthreads: int = 1,
) -> np.ndarray:
    """The stored image-plane response ``B/n`` of a piece or reduced partition.

    Places the field-centred ``BEAM_SMALL`` on the image grid
    (:func:`beam_on_image_grid`) and folds in the wgridder's ``1/n``; without a
    ``BEAM_SMALL`` (no aperture beam) the response is ``1/n`` alone. ``n`` is
    computed on exactly ducc's pixel coordinates, absolute w.r.t. the tangent
    point with the ``--target`` offset included, which is why every ducc call on
    the imager+deconv path keeps ``divide_by_n=False`` (wiki D22). Placement is
    linear in the beam, so placing a partition's mean small beam equals the mean
    of its pieces' placed beams (#365).

    Args:
        ds: Piece or partition with ``corr`` and, given a beam model,
            ``BEAM_SMALL`` ``(corr, m_beam, l_beam)`` plus the ``ra0``/``dec0``
            pointing attrs.
        radec_grid: The image tangent point (rad).
        cell_rad: Image cell size (rad).
        nx: Image pixels along x.
        ny: Image pixels along y.
        l0: Image-centre offset from the tangent point (rad); the value the
            partition is gridded with.
        m0: As ``l0``, for m.
        dtype: Output dtype, the tree's ``--precision`` (D27).
        nthreads: Threads for the placement.

    Returns:
        ``(ncorr, ny, nx)`` ``B/n``, (Y, X)-ordered.
    """
    _, _, _, x0, y0 = wgridder_conventions(l0, m0)
    x_abs = (-(nx / 2) + np.arange(nx)) * cell_rad + x0
    y_abs = (-(ny / 2) + np.arange(ny)) * cell_rad + y0
    yy_abs, xx_abs = np.meshgrid(y_abs, x_abs, indexing="ij")
    nlm = np.sqrt(1.0 - xx_abs**2 - yy_abs**2)
    ncorr = ds.corr.size
    if "BEAM_SMALL" not in ds:
        # no aperture beam: 1/n alone (compresses well in zarr)
        return (1.0 / nlm)[None, :, :].astype(dtype) * np.ones((ncorr, 1, 1), dtype=dtype)
    radec = np.array([ds.attrs["ra0"], ds.attrs["dec0"]])
    beam = beam_on_image_grid(
        ds.BEAM_SMALL.values.astype(np.float64),
        ds.l_beam.values,
        ds.m_beam.values,
        radec,
        np.asarray(radec_grid),
        cell_rad,
        nx,
        ny,
        l0=l0,
        m0=m0,
        nthreads=nthreads,
    ).astype(dtype)
    return (beam / nlm[None, :, :]).astype(dtype)


def stokes_vis(
    dc1=None,
    dc2=None,
    operator=None,
    node_dt=None,
    scratch_store=None,
    bandid=None,
    timeid=None,
    msid=None,
    freq_nominal=None,
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
    counts_nx=None,
    counts_ny=None,
    counts_cell=None,
    baseline_group="all",
    data_group="base",
    radec_new=None,
    target=None,
    beam_lm=None,
    nthreads=1,
    double_accum=True,
):
    """
    MSv4 version of stokes2vis.

    inputs:
        dc1, dc2, operator: data column names and operator to combine them with.
        node_dt: sub-node of datatree with the relevant time and frequency selection already applied.
        scratch_store: parent .scratch DataTree store; the piece is written to the group
            ``band{bandid:04d}_time{timeid:04d}/<piece_name>`` within it.
        bandid, timeid, msid: identifiers for the output dataset.
        counts_nx, counts_ny, counts_cell: the per-piece COUNTS grid used to build
            imaging weights in pass 2, as the padded uv grid of a ``counts_nx`` x
            ``counts_ny`` image of cell ``counts_cell`` (rad): the image's own
            under ``--weight-on-image-grid``, else the driver's ``counts_grid``
            (stored sparse, see ``weighting.compute_sparse_counts``).
        baseline_group: partition baseline-group label (single ``"all"`` group for now).
        freq_nominal: nominal band centre (band-edge midpoint) used by the driver to
            assign channel slices to bands. The piece reports its own effective
            frequency as freq_out; freq_nominal is the fallback when no weight
            survives (issue #296, wiki D28).
        precision: "single" or "double" for output data and weights.
        sigma_column: name of column containing sigma values to convert to weights.
        weight_column: name of column containing weights to use instead of sigma.
        product: Stokes product. Can be any subset of IQUV.
        chan_average: number of channels to average together.
        bda_decorr: decorrelation threshold for baseline dependent averaging.
        max_field_of_view: radius (deg) about the phase centre within which BDA
            bounds the decorrelation, and within which every image must lie.
        beam_model: model to use for beam correction: "katbeam", None, a
            BeamWizard, or a Multiton over one (resolved per worker).
        wgt_mode: weighting mode to use in weight_data. Can be "l2" or "minvar".
        radec_new: optional (ra, dec) in radians to rephase to (the common mosaic
            phase centre / tangent point). None means image about the field centre.
        target: optional image-centre target ('HH:MM:SS,DD:MM:SS' or an astropy
            body); resolved to an in-plane (l0, m0) at this piece's own time.
        beam_lm: 1D field-centred beam axis (deg), used for both l and m; the
            driver sizes it once per run (``utils.beam.beam_grid``). Required for
            a BeamWizard; katbeam defaults to an axis reaching ``max_field_of_view``.
        nthreads: threads available to the task (parallelises the COUNTS gridding
            and meerkat-beams' rotation average).
        double_accum: accumulate the counts in float64 (``--double-accum``);
            they are stored at ``precision`` either way (D27).
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
    ds = load_detached(node_dt.ds[needed])
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
    # before any row masking: the image's time_bounds (#327) are the nominal
    # coverage, like its freq_bounds
    time_lo, time_hi = integration_edges(time, interval)
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

    # identity gains: a length-one (time, chan) jones broadcasts in weight_data
    jones = np.ones((1, 1, nant, 1, 2), dtype=complex_type)

    # rows for the per-row rephasing below
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

    # Convert to Stokes. No gains are applied on this path -- the MSv2 stokes2vis
    # did, and `imager --gain-table` is refused rather than silently ignored (#333).
    # weight_data works on the MSv4 (time, baseline) grid; ant1_bl/ant2_bl are per baseline
    vis4, wgt4 = weight_data(
        as_contiguous_readonly_view(data.reshape(ntime, nbl, nchan, ncorr)),
        as_contiguous_readonly_view(weight.reshape(ntime, nbl, nchan, ncorr)),
        as_contiguous_readonly_view(flag.reshape(ntime, nbl, nchan, ncorr)),
        as_contiguous_readonly_view(jones),
        as_contiguous_readonly_view(ant1_bl.astype(np.int32)),
        as_contiguous_readonly_view(ant2_bl.astype(np.int32)),
        poltype,
        product,
        str(ncorr),
        wgt_mode,
    )
    # (stokes, time, baseline, chan) -> (row, chan, stokes) views for the
    # averaging routines; the row selection below makes the contiguous copy
    ns = vis4.shape[0]
    data = np.moveaxis(vis4.reshape(ns, nrow, nchan), 0, -1)
    weight = np.moveaxis(wgt4.reshape(ns, nrow, nchan), 0, -1)
    del vis4, wgt4

    # TODO - check if wsum for any of the correlations is zero
    # This happens e.g. if selecting out diagonal correlations
    # with QC and making CORRECTED_WEIGHTS

    # Row selection happens after weight_data, which needs the regular
    # (time, baseline) grid.
    #
    # Drop auto-correlations, fully flagged rows, and rows xarray-ms padded to
    # square off the (time, baseline) grid. The padding carries NaN UVW (see
    # the uvw_mask in core/imager.py) and is the reason two things used to blow
    # up: africanus' BDA mapper derives a bin's central UVW from its first and
    # last row irrespective of flags, so a padded row poisons it to NaN and
    # trips `assert fracsizeChanBlockMin >= 1` (max(nan, 1) is nan); and any
    # UNMASKED ducc degrid derives its w range from every row, giving a NaN w
    # extent and "too many w planes". Dropping them here fixes both at source
    # and is pure win otherwise -- they are zero-weight and fully masked, so
    # they contribute nothing while inflating the stored .dt and the gridding.
    mrow = ~frow & ~flag.all(axis=(1, 2)) & np.isfinite(uvw).all(axis=1)
    if not mrow.any():
        return None
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

    # Effective frequency of this piece: the weight-weighted mean over the
    # channels that survive flagging (issue #296, wiki D28). freq_nominal is
    # only the band *assignment* grid and is wrong by up to half a channel
    # whenever nband does not divide nchan. This value is what the beam is
    # evaluated at below and what the piece reports as freq_out.
    # Within a piece these are natural weights, the only ones that exist here.
    # Pieces are then combined on their imaging-weight sums, which the counts
    # reduction predicts before pass 2 (wiki D28, D65).
    w_chan = (weight * mask[:, :, None]).sum(axis=(0, 2), dtype=np.float64)
    wsum_nat = float(w_chan.sum())
    if wsum_nat > 0:
        freq_eff = float((w_chan * freq).sum() / wsum_nat)
    else:
        # rows survived the mrow filter above but carry zero weight
        freq_eff = float(freq_nominal)

    # ---- primary beam, field-centred (pass-1 beam, #281, #365) ----
    # The (rotation-averaged) beam is evaluated about the FIELD's own pointing
    # -- where the antennas point regardless of rephasing -- on a small grid and
    # stored as is. Pass 2 averages a partition's small beams and places the
    # mean on the image grid once (partition_beam): placement is linear in the
    # beam, so that equals placing each piece and averaging, without writing an
    # image-sized beam per piece (#365).
    if max_blength is None or max_freq is None:
        raise ValueError("max_blength and max_freq must be provided to size the beam grid")
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

    beam_imre_ratio = 0.0
    if isinstance(beam_model, Multiton):
        # the driver ships a recipe; this worker's wizard, prefilters and all,
        # persists across its tasks
        beam_model = beam_model.instance
    beam_small = None
    if beam_model is None:
        # no aperture beam: pass 2 stores the bare 1/n (partition_beam)
        pass
    else:
        # field-centred evaluation on a small grid first
        if isinstance(beam_model, BeamWizard):
            # the wizard ships detached (BDS-only); the parallactic angles
            # need the FIELD's pointing centre, set per piece here
            rvec = np.asarray(radec).squeeze()
            beam_model.set_pointing_centre(SkyCoord(ra=rvec[0] * u.rad, dec=rvec[1] * u.rad))
            if beam_lm is None:
                raise ValueError("a BeamWizard beam needs the driver's beam_lm axis (utils.beam.beam_grid)")
            # deg, East/North positive: sized by the driver to the image and to a
            # bilinear-placement tolerance, shared by every piece of the run (#365)
            l_beam = m_beam = np.asarray(beam_lm, dtype=np.float64)
            # D13: MSv4 time is unix seconds; astropy Time wants MJD days
            t_beam = Time(to_mjd_time(utime) / 86400.0, format="mjd")
            raw_maps = []
            for p in corr:
                # returns (Y, X)-ordered maps (docs/wiki/image-and-beam-orientation.md);
                # signature kept stable so meerkat-beams can grow weighted
                # time/freq averaging underneath (#281). Complex for the MPM
                # cross group -- real_beam_maps takes Re and reports the ratio.
                bmap, _ = beam_model.get_rotation_averaged_beam(
                    l=l_beam,
                    m=m_beam,
                    times=t_beam,
                    freq=np.atleast_1d(freq_eff),
                    time_stepping=1,
                    pixel_stepping=1,
                    var="nstokes",
                    i=p,
                    j=p,
                    verbose=0,
                    nthreads=nthreads,
                    pa_tolerance=BEAM_PA_TOLERANCE_DEG,
                )
                raw_maps.append(bmap)
            beam_small, beam_imre_ratio = real_beam_maps(raw_maps)
        elif str(beam_model).lower() == "katbeam":
            # the driver's axis when given (pointing_reach at the beam_grid cell),
            # else one reaching max_field_of_view, for direct callers
            if beam_lm is None:
                beam_lm = beam_grid(max_freq, BEAM_DISH_DIAMETER[None], max_field_of_view)
            l_beam = m_beam = np.asarray(beam_lm, dtype=np.float64)
            if freq_min >= 8.5e8 and freq_max <= 1.8e9:
                beamo = JimBeam("MKAT-AA-L-JIM-2020")
            elif freq_min >= 5.4e8 and freq_max <= 1.1e9:
                beamo = JimBeam("MKAT-AA-UHF-JIM-2020")
            else:
                raise ValueError("Freq range not covered by katbeam")
            # (Y, X)/(m, l) order end to end (wiki D19): axis 0 is m/Dec-like
            mm, ll = np.meshgrid(m_beam, l_beam, indexing="ij")
            # katbeam expects freq in MHz
            fmhz = freq_eff / 1e6
            # Rotation average over 25 angles, rotating the coordinates rather
            # than the image: katbeam is analytic, so this is exact on any grid.
            # ndimage.rotate on a grid erred by 1.3e-3 at this cell (D66).
            angles = np.deg2rad(np.linspace(0, 359, 25))
            beam_small = np.zeros((ncorr, m_beam.size, l_beam.size), dtype=np.float64)
            for angle in angles:
                ca, sa = np.cos(angle), np.sin(angle)
                lr, mr = ca * ll - sa * mm, sa * ll + ca * mm
                for i, p in enumerate(corr):
                    beam_small[i] += getattr(beamo, p)(lr, mr, fmhz)
            beam_small /= angles.size
        else:
            raise ValueError(f"Unknown beam model {beam_model}")

    # for operations that follow it will be preferable to have the corr axis
    # first for contiguity
    vis_cf = data.transpose(2, 0, 1)
    wgt_cf = weight.transpose(2, 0, 1)

    # signs occording to wgridder conventions
    flip_u, flip_v, flip_w, _, _ = wgridder_conventions(0, 0)
    # always gridded, so one scratch serves natural and robust weighting (#354);
    # only the occupied cells are kept: the dense grid is freed inside (#365)
    counts_vars, wsum_offgrid = compute_sparse_counts(
        uvw,
        freq,
        mask,
        wgt_cf,
        counts_nx,
        counts_ny,
        counts_cell,
        counts_cell,
        wgt_cf.dtype,
        usign=1.0 if flip_u else -1.0,
        vsign=1.0 if flip_v else -1.0,
        accum_dtype=np.float64 if double_accum else wgt_cf.dtype,
    )

    data_vars = {}
    data_vars["VIS"] = (("corr", "row", "chan"), vis_cf)
    data_vars["WEIGHT"] = (("corr", "row", "chan"), wgt_cf)
    data_vars["MASK"] = (("row", "chan"), mask)
    data_vars["UVW"] = (("row", "three"), uvw)
    data_vars["FREQ"] = (("chan",), freq)
    if beam_small is not None:
        data_vars["BEAM_SMALL"] = (("corr", "m_beam", "l_beam"), beam_small.astype(real_type))
    data_vars.update(counts_vars)
    del counts_vars

    coords = {
        "chan": (("chan",), freq),
        "corr": (("corr",), corr),
    }
    if beam_small is not None:
        # deg, East/North positive; shared by every piece of a run (pass 2 asserts it)
        coords["l_beam"] = (("l_beam",), np.asarray(l_beam, dtype=np.float64))
        coords["m_beam"] = (("m_beam",), np.asarray(m_beam, dtype=np.float64))

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
        "freq_out": freq_eff,
        "freq_nominal": float(freq_nominal),
        "wsum_nat": wsum_nat,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "bandid": bandid,
        "time_out": time_out,
        "time_min": utime.min(),
        "time_max": utime.max(),
        # outermost integration edges, for the image's time_bounds (#327)
        "time_lo": time_lo,
        "time_hi": time_hi,
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
        # max|Im|/max|Re| of the evaluated beam. Non-zero only for the MPM
        # cross group, where the stored BEAM is Re(B) (wiki D46). Watch it on
        # real data: this is the evidence that Re(B) is a safe approximation.
        "beam_imre_ratio": float(beam_imre_ratio),
    }

    # natural weight beyond the counts grid, for the imaging-wsum prediction (D65)
    attrs["wsum_offgrid"] = wsum_offgrid
    out_ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    group = f"band{bandid:04d}_time{timeid:04d}/{piece_name}"
    # consolidated=False: many workers write distinct leaf groups into the shared
    # scratch store concurrently. Per-write consolidation would race on the single
    # root .zmetadata (torn JSON -> JSONDecodeError on networked filesystems); the
    # band/time parent groups are pre-created and the driver consolidates once
    # after all pass-1 tasks finish (see core.imager).
    out_ds.to_zarr(scratch_store, group=group, mode="a", consolidated=False)
    return bandid, timeid, piece_name
