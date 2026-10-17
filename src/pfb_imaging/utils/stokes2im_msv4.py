"""Per-image read and imaging for ``pfb hci-msv4``.

Pure functions, numpy in and out, no Ray. ``read_item`` does one image's I/O
from an MSv4 node; ``image_item`` is the maths of the legacy
``utils/stokes2im.stokes_image`` on its output, with visibilities kept in
MSv4 ``(time, baseline, chan, corr)`` order up to the Stokes conversion.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import numexpr as ne
import numpy as np
import xarray as xr
from astropy import units
from astropy.coordinates import SkyCoord
from ducc0.fft import good_size
from jax.scipy.sparse.linalg import cg
from numba import set_num_threads
from scipy.constants import c as lightspeed

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.operators.hessian import hessian_slice_jax
from pfb_imaging.utils.gains import load_node_gains
from pfb_imaging.utils.hci_common import beam_for_band, beam_gain_for_source
from pfb_imaging.utils.misc import fitcleanbeam, radec_to_lm, to_mjd_time
from pfb_imaging.utils.msv4 import load_detached
from pfb_imaging.utils.weighting import (
    _compute_counts,
    as_contiguous_readonly_view,
    box_sum_counts,
    counts_to_weights,
    filter_extreme_counts,
    weight_data_msv4,
)


@dataclass(frozen=True)
class HciConfig:
    """Per-image imaging options, resolved against the MSv4 schema by the driver."""

    dc1: str  # resolved MSv4 variable name, e.g. "VISIBILITY"
    dc2: str | None
    operator: str | None  # "+" | "-" | None
    model_column: str | None  # resolved
    weight_column: str | None  # resolved ("WEIGHT" for WEIGHT_SPECTRUM)
    sigma_column: str | None
    flag_column: str  # "FLAG"
    product: str
    precision: str  # "single" | "double"
    nx: int
    ny: int
    cell_rad: float
    psf_relative_size: float | None
    phase_dir: str | None  # "HH:MM:SS,DD:MM:SS"
    target: str | None
    robustness: float | None
    min_padding: float
    filter_counts_level: float
    npix_super: int
    channels_per_bin: int
    epsilon: float
    do_wgridding: bool
    double_accum: bool
    natural_grad: bool
    eta: float
    cg_tol: float
    cg_maxit: int
    psf_out: bool
    l2_reweight_dof: int | None
    wgt_mode: str
    inject_transients: str | None  # path of the transient yaml, spectra zarr next to it
    nthreads: int


@dataclass
class ItemData:
    """One image's inputs as plain numpy, in MSv4 (time, baseline, chan, corr) order."""

    data: np.ndarray  # (time, baseline, chan, corr) complex, column arithmetic applied, NaN -> 0
    model: np.ndarray | None  # same shape, or None
    weight: np.ndarray  # (time, baseline, chan, corr) real
    flag: np.ndarray  # (time, baseline, chan, corr) bool, includes non-finite inputs and gain flags
    uvw: np.ndarray  # (time, baseline, 3) float64, NaN -> 0
    time_mjd: np.ndarray  # (time,) MJD seconds
    freq: np.ndarray  # (chan,) Hz
    ant1: np.ndarray  # (baseline,) int32 into antenna_xds order
    ant2: np.ndarray
    antpos: np.ndarray  # (nant, 3) ITRF
    radec: np.ndarray  # (2,) field phase centre, radians, RA wrapped to [0, 2pi)
    poltype: str  # "linear" | "circular"
    jones: np.ndarray  # (gtime, gchan, nant, 2[, 2]) complex; ones((1, 1, nant, 2)) without gains
    field_name: str
    spw_name: str
    scan_name: str


def read_item(
    node: xr.DataTree,
    region: Mapping[str, slice],
    cfg: HciConfig,
    data_group: str = "base",
    gain_spec: str | None = None,
) -> ItemData:
    """Read one image's visibilities from an MSv4 node into plain numpy.

    Loads only the variables the image needs (never a blanket load) through
    ``load_detached`` so nothing is cached on the open tree, then releases the
    Dataset before returning.

    Args:
        node: The MSv4 visibility node (its ``antenna_xds`` and field_and_source
            children are read for antenna positions and the phase centre).
        region: ``isel`` selection on the node's ``time``/``frequency`` dims.
        cfg: Imaging options; only the column names and precision are used here.
        data_group: Data group whose field_and_source subtable gives the phase centre.
        gain_spec: Optional QuartiCal gain term (``gains.qc/NET``) to apply.

    Returns:
        The image's :class:`ItemData`.

    Raises:
        KeyError: if the node lacks a requested variable.
        ValueError: on an unknown operator or polarisation basis.
    """
    needed = [cfg.dc1] + ([cfg.dc2] if cfg.dc2 else [])
    for name in (cfg.model_column, cfg.sigma_column if cfg.sigma_column else cfg.weight_column):
        if name is not None:
            needed.append(name)
    needed += [cfg.flag_column, "UVW"]
    needed = list(dict.fromkeys(needed))
    missing = [n for n in needed if n not in node.ds.data_vars]
    if missing:
        raise KeyError(f"{node.path} has no {missing}; have {sorted(node.ds.data_vars)}")
    ds = load_detached(node.ds[needed].isel(region))

    data = ds[cfg.dc1].values
    if cfg.dc2 is not None:
        if cfg.operator not in ("+", "-"):
            raise ValueError(f"Unsupported operator {cfg.operator!r}; expected '+' or '-'")
        data = ne.evaluate(f"a {cfg.operator} b", local_dict={"a": data, "b": ds[cfg.dc2].values}, casting="same_kind")
    model = ds[cfg.model_column].values if cfg.model_column else None
    if cfg.sigma_column is not None:
        weight = ne.evaluate("1.0/s**2", local_dict={"s": ds[cfg.sigma_column].values})
    elif cfg.weight_column is not None:
        weight = ds[cfg.weight_column].values
    else:
        weight = np.ones(data.shape, dtype=data.real.dtype)
    flag = ds[cfg.flag_column].values.astype(bool)
    uvw = ds.UVW.values.astype(np.float64)

    # xarray-ms fills the regular grid with NaN where MSv2 had no row: flag and zero
    bad = ~np.isfinite(data) | ~np.isfinite(weight)
    if model is not None:
        bad |= ~np.isfinite(model)
        model = np.where(bad, 0, model)
    bad |= ~np.isfinite(uvw).all(axis=-1)[:, :, None, None]
    flag |= bad
    data = np.where(bad, 0, data)
    weight = np.where(bad, 0, weight)
    uvw = np.nan_to_num(uvw)

    times_unix = ds.time.values
    freq = ds.frequency.values
    field_name = str(np.unique(ds.field_name.values).item())
    scan_name = str(np.unique(ds.scan_name.values).item())
    spw_name = str(ds.frequency.attrs["spectral_window_name"])
    # the Stokes kernels index correlations by position, so the order matters, not just the set
    corr = tuple(str(p) for p in ds.polarization.values)
    if corr in (("XX", "XY", "YX", "YY"), ("XX", "YY")):
        poltype = "linear"
    elif corr in (("RR", "RL", "LR", "LL"), ("RR", "LL")):
        poltype = "circular"
    else:
        raise ValueError(f"Unsupported correlations {corr}; expected XX,XY,YX,YY / XX,YY or RR,RL,LR,LL / RR,LL")

    # MSv2-style antenna indices into antenna_xds order (searchsorted, as stokes_vis does)
    ant_names = node["antenna_xds"].antenna_name.values
    order = np.argsort(ant_names)
    sorted_n = ant_names[order]
    ant1 = order[np.searchsorted(sorted_n, ds.baseline_antenna1_name.values)].astype(np.int32)
    ant2 = order[np.searchsorted(sorted_n, ds.baseline_antenna2_name.values)].astype(np.int32)
    antpos = node["antenna_xds"].ANTENNA_POSITION.values

    # phase centre from the data group's field_and_source subtable, by field name
    grp = node.ds.attrs["data_groups"][data_group]
    fns = node[grp["field_and_source"].rsplit("/", 1)[-1]].ds
    radec = np.array(fns.FIELD_PHASE_CENTER_DIRECTION.sel(field_name=field_name).values, dtype=np.float64).squeeze()
    radec[0] = radec[0] % (2 * np.pi)
    del ds, fns, grp

    nant = antpos.shape[0]
    complex_type = np.complex64 if cfg.precision.lower() == "single" else np.complex128
    if gain_spec is not None:
        # gains on the region's own (time, frequency) grid
        gains = load_node_gains(gain_spec, scan_name, times_unix, freq, list(ant_names), field_name=field_name)
        jones = gains.jones.astype(complex_type)
        gf = gains.flags[:, :, ant1] | gains.flags[:, :, ant2]  # (gtime, gchan, baseline)
        flag |= np.transpose(gf, (0, 2, 1))[:, :, :, None]
    else:
        jones = np.ones((1, 1, nant, 2), dtype=complex_type)

    return ItemData(
        data=data,
        model=model,
        weight=weight,
        flag=flag,
        uvw=uvw,
        time_mjd=to_mjd_time(times_unix),
        freq=freq,
        ant1=ant1,
        ant2=ant2,
        antpos=antpos,
        radec=radec,
        poltype=poltype,
        jones=jones,
        field_name=field_name,
        spw_name=spw_name,
        scan_name=scan_name,
    )


def image_item(inp: ItemData, cfg: HciConfig, beam_model, nx_psf: int, ny_psf: int) -> dict:
    """Image one :class:`ItemData` exactly as the legacy ``stokes_image`` does.

    Args:
        inp: The image's inputs from :func:`read_item`.
        cfg: Imaging options.
        beam_model: Beam model object for primary-beam weighting, or None.
        nx_psf: PSF width in pixels (computed once by the driver).
        ny_psf: PSF height in pixels.

    Returns:
        Dict of numpy arrays in output dtypes: ``cube`` ``(nstokes, ny, nx)``,
        ``psf`` (if ``cfg.psf_out``), ``beam_weight`` (if ``beam_model``), per-Stokes
        ``weight``/``rms``/``nonzero``/``psf_maj``/``psf_min``/``psf_pa`` (degrees),
        and scalars ``orig_phase_dir_ra``/``orig_phase_dir_dec``/``time_mjd``/``freq_out``.
    """
    # deferred: ducc objects break Ray serialisation of the enclosing module's functions (stokes2im precedent)
    from ducc0.misc import resize_thread_pool
    from ducc0.wgridder import vis2dirty

    set_num_threads(cfg.nthreads)
    resize_thread_pool(cfg.nthreads)

    if cfg.precision.lower() == "single":
        real_type = np.float32
        complex_type = np.complex64
    elif cfg.precision.lower() == "double":
        real_type = np.float64
        complex_type = np.complex128
    else:
        raise ValueError(f"Unsupported precision {cfg.precision!r}; expected 'single' or 'double'")

    product = cfg.product
    nx, ny = cfg.nx, cfg.ny
    cell_rad = cfg.cell_rad
    phase_dir = cfg.phase_dir
    beam_model_given = beam_model is not None

    ntime_i, nbl, nchan, ncorr = inp.data.shape
    nrow = ntime_i * nbl
    data = inp.data.reshape(nrow, nchan, ncorr)
    weight = inp.weight.reshape(nrow, nchan, ncorr)
    uvw = inp.uvw.reshape(nrow, 3)
    time = np.repeat(inp.time_mjd, nbl)
    ant1 = np.tile(inp.ant1, ntime_i)
    ant2 = np.tile(inp.ant2, ntime_i)
    # MS may contain auto-correlations
    flag = inp.flag.reshape(nrow, nchan, ncorr) | (ant1 == ant2)[:, None, None]
    freq = inp.freq
    utime = inp.time_mjd
    radec = inp.radec.copy()
    antpos = inp.antpos

    model_vis = None
    if inp.model is not None:
        model_vis = inp.model.reshape(nrow, nchan, ncorr).astype(complex_type)
        if product.lower() == "i":
            model_vis = model_vis[..., [0, -1]].mean(axis=-1, keepdims=True)
        else:
            raise NotImplementedError(
                f"Applying Jones matrcies on the fly is not supported for product {product}. "
                f"Use --data-column 'CORRECTED_DATA - MODEL_DATA' for model subtraction."
            )

    time_out = np.mean(utime)
    freq_out = np.mean(freq)

    if data.dtype != complex_type:
        data = data.astype(complex_type)

    if weight.dtype != real_type:
        weight = weight.astype(real_type)

    # antenna indices are already canonical (antenna_xds order)
    jones = inp.jones
    if jones.dtype != complex_type:
        jones = jones.astype(complex_type)

    cell_deg = np.rad2deg(cell_rad)

    flip_u, flip_v, flip_w, _, _ = wgridder_conventions(0, 0)
    signu = -1.0 if flip_u else 1.0
    signv = -1.0 if flip_v else 1.0
    # we need these because of flipped wgridder convention
    # https://github.com/mreineck/ducc/issues/34
    signx = -1.0 if flip_u else 1.0
    signy = -1.0 if flip_v else 1.0
    freqfactor = -2j * np.pi * freq[None, :] / lightspeed

    # rephase if asked
    if phase_dir is not None:
        # deferred: pyrap (python-casacore); only rephasing runs pay the import
        from pfb_imaging.utils.astrometry import synthesize_uvw

        new_ra, new_dec = phase_dir.split(",")
        c = SkyCoord(new_ra, new_dec, frame="fk5", unit=(units.hourangle, units.deg))
        new_ra_rad = np.deg2rad(c.ra.value)
        new_dec_rad = np.deg2rad(c.dec.value)
        radec_new = np.array((new_ra_rad, new_dec_rad))
        uvw_new = synthesize_uvw(antpos, time, ant1, ant2, radec_new)
        # Recompute the OLD uvw via the same measures call so the systematic
        # between pyrap's earth orientation and the MS UVW cancels in w_diff
        # (pfb-imaging#280; see stokes2im.stokes_image for the full story).
        uvw_ref = synthesize_uvw(antpos, time, ant1, ant2, radec)

        # chgcentre style wdiff rephasing (differential w)
        w_diff = uvw_new[:, 2:] - uvw_ref[:, 2:]
        data = data * np.exp(freqfactor * w_diff)[:, :, None]
        if model_vis is not None:
            model_vis = model_vis * np.exp(freqfactor * w_diff)[:, :, None]

        # sampling stays anchored to the MS's own UVW plus the differential rotation
        uvw_old = uvw
        uvw = uvw + (uvw_new - uvw_ref)
    else:
        uvw_old = uvw
        radec_new = radec

    # compute lm coordinates of target if requested
    if cfg.target is not None:
        # deferred: pyrap (python-casacore); only --target runs pay the import
        from pfb_imaging.utils.astrometry import resolve_target_radec

        # same resolver the driver uses to place the scaffold's coords (D36)
        tra, tdec, _ = resolve_target_radec(cfg.target, time_out)
        lm0 = radec_to_lm((tra, tdec), radec_new)
        # flip for wgridder conventions
        x0 = -lm0[0]
        y0 = -lm0[1]
    else:
        x0 = 0.0
        y0 = 0.0

    vis4, wgt4 = weight_data_msv4(
        as_contiguous_readonly_view(data.reshape(ntime_i, nbl, nchan, ncorr)),
        as_contiguous_readonly_view(weight.reshape(ntime_i, nbl, nchan, ncorr)),
        as_contiguous_readonly_view(flag.reshape(ntime_i, nbl, nchan, ncorr)),
        as_contiguous_readonly_view(jones),
        as_contiguous_readonly_view(inp.ant1),
        as_contiguous_readonly_view(inp.ant2),
        inp.poltype,
        product,
        str(ncorr),
        cfg.wgt_mode,
    )
    ns = vis4.shape[0]
    data = vis4.reshape(ns, nrow, nchan)  # (stokes, row, chan): free view
    weight = wgt4.reshape(ns, nrow, nchan)
    del vis4, wgt4

    # flag if any correlation is flagged
    flag = flag.any(axis=-1)
    mask = (~flag).astype(np.uint8)

    if model_vis is not None:
        data = ne.evaluate("(data - m) * k", local_dict={"data": data, "m": model_vis[None, :, :, 0], "k": mask[None]})

    if cfg.l2_reweight_dof:
        dof = cfg.l2_reweight_dof
        # data should contain residual_vis at this point
        ressq = (data * data.conj()).real * mask[None]
        wcount = mask.sum()
        if wcount:
            # one variance per Stokes: pooling them over a single-Stokes count overestimates it ns-fold
            ovar = ressq.sum(axis=(1, 2), keepdims=True) / wcount  # use 67% quantile?
            if not (ovar > 0).any():
                raise ValueError("Residual visibilities are exactly zero. Cannot compute L2 reweighting.")
            # a Stokes with an exactly zero residual (an unpolarised simulation's V) keeps its weights
            safe = np.where(ovar > 0, ovar, 1.0)
            weight = np.where(ovar > 0, (dof + 1) / (dof + ressq / safe) / safe, weight)
        # with nothing unflagged the (all-masked) weights are left as they are

    n = np.sqrt(1 - x0**2 - y0**2)
    psf_vis = np.exp(
        freqfactor * (signu * uvw[:, 0:1] * x0 * signx + signv * uvw[:, 1:2] * y0 * signy - uvw[:, 2:] * (n - 1))
    ).astype(complex_type)

    # The transient is simulated at the old uvw and carried to the new phase
    # centre with the data's w-difference; the sign conventions are explained
    # in stokes2im.stokes_image (ratt-ru/breifast#263).
    if cfg.inject_transients is not None:
        transient_name = cfg.inject_transients.removesuffix("yaml") + "zarr"
        transient_ds = xr.open_zarr(transient_name, chunks=None)
        all_times = transient_ds.TIME.values
        all_freqs = transient_ds.FREQ.values
        names = transient_ds.names
        ras = transient_ds.ras
        decs = transient_ds.decs
        for name, ra, dec in zip(names, ras, decs):
            ra_rad = np.deg2rad(ra)
            dec_rad = np.deg2rad(dec)
            # use initial radec as reference since rephasing happens after beam application
            lm0t = radec_to_lm((ra_rad, dec_rad), radec)
            x0t = lm0t[0]
            y0t = lm0t[1]
            n0t = np.sqrt(1 - x0t**2 - y0t**2)

            # these are the profiles on the full domain so interpolate
            tprofile = getattr(transient_ds, f"{name}_time_profile").values
            fprofile = getattr(transient_ds, f"{name}_freq_profile").values

            tprofile = np.interp(time, all_times, tprofile)  # interpolate to row times
            fprofile = np.interp(freq, all_freqs, fprofile)

            # outer product gives dynamic spectrum
            dspec = (tprofile[:, None]) * fprofile[None, :]

            # apply beam in original frame
            if beam_model_given:
                dspec *= beam_gain_for_source(beam_model, ra, dec, product, utime, time, freq)

            if phase_dir is not None:
                phase = -w_diff
            else:
                phase = np.zeros((nrow, 1), dtype=real_type)
            phase += signu * uvw_old[:, 0:1] * x0t * signx
            phase += signv * uvw_old[:, 1:2] * y0t * signy
            phase += uvw_old[:, 2:] * (n0t - 1)
            # RIME point-source visibility carries a 1/n term (V = I/n * fringe)
            dspec = dspec / n0t * np.exp(-freqfactor * phase)

            # currently Stokes I only
            data[0] += dspec

    data = data.astype(complex_type)
    weight = weight.astype(real_type)
    if cfg.robustness is not None:
        # we need to compute the weights on the padded grid
        # but we don't have control over the optimal gridding
        # parameters so assume a minimum
        nx_pad = good_size(int(cfg.min_padding * nx))
        while nx_pad % 2:
            nx_pad = good_size(nx_pad + 1)
        ny_pad = good_size(int(cfg.min_padding * ny))
        while ny_pad % 2:
            ny_pad = good_size(ny_pad + 1)
        counts = _compute_counts(
            uvw,
            freq,
            mask,
            weight,
            nx_pad,
            ny_pad,
            cell_rad,
            cell_rad,
            real_type,
            usign=1.0 if flip_u else -1.0,
            vsign=1.0 if flip_v else -1.0,
        )

        counts = filter_extreme_counts(counts, level=cfg.filter_counts_level)
        counts = box_sum_counts(counts, cfg.npix_super)

        weight = counts_to_weights(
            counts,
            uvw,
            freq,
            weight,
            mask,
            nx_pad,
            ny_pad,
            cell_rad,
            cell_rad,
            cfg.robustness,
            usign=1.0 if flip_u else -1.0,
            vsign=1.0 if flip_v else -1.0,
        )

    channels_per_bin = cfg.channels_per_bin
    if channels_per_bin in (0, None, -1):
        channels_per_bin = nchan
    nband = int(np.ceil(nchan / channels_per_bin))
    nstokes = weight.shape[0]
    wsums = np.zeros((nstokes, nband), dtype=real_type)
    # all image-space arrays are cube/FITS (Y, X)-ordered; ducc's x-major world
    # exists only at the vis2dirty seams below via transposed views
    # (see docs/wiki/image-and-beam-orientation.md)
    residual = np.zeros((nstokes, ny, nx), dtype=real_type)
    psf = np.zeros((nstokes, ny_psf, nx_psf), dtype=real_type)
    pbeam = np.zeros((nstokes, ny, nx), dtype=real_type)
    for b, fi in enumerate(range(0, nchan, channels_per_bin)):
        ff = min(fi + channels_per_bin, nchan)
        datab = data[:, :, fi:ff]
        weightb = weight[:, :, fi:ff]
        flagb = flag[:, fi:ff]
        freqb = freq[fi:ff]
        maskb = mask[:, fi:ff]
        psf_visb = psf_vis[:, fi:ff]

        # get beam for subband
        pbeamb = beam_for_band(
            beam_model,
            utime,
            product,
            real_type,
            np.mean(freqb, keepdims=True),
            radec,
            radec_new,
            cell_deg,
            nx,
            ny,
        )

        for c in range(nstokes):
            if weightb[c, ~flagb].sum() == 0:
                continue
            # ducc fills the (Y, X) buffers through zero-copy transposed views
            residualb = np.zeros((ny, nx), dtype=real_type)
            vis2dirty(
                uvw=uvw,
                freq=freqb,
                vis=datab[c],
                wgt=weightb[c],
                mask=maskb,
                npix_x=nx,
                npix_y=ny,
                pixsize_x=cell_rad,
                pixsize_y=cell_rad,
                center_x=x0,
                center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=cfg.epsilon,
                do_wgridding=cfg.do_wgridding,
                divide_by_n=True,
                nthreads=cfg.nthreads,
                sigma_min=cfg.min_padding,
                double_precision_accumulation=cfg.double_accum,
                verbosity=0,
                dirty=residualb.T,
            )

            psfb = np.zeros((ny_psf, nx_psf), dtype=real_type)
            vis2dirty(
                uvw=uvw,
                freq=freqb,
                vis=psf_visb,
                wgt=weightb[c],
                mask=maskb,
                npix_x=nx_psf,
                npix_y=ny_psf,
                pixsize_x=cell_rad,
                pixsize_y=cell_rad,
                center_x=x0,
                center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=cfg.epsilon,
                do_wgridding=cfg.do_wgridding,
                divide_by_n=True,
                nthreads=cfg.nthreads,
                sigma_min=cfg.min_padding,
                double_precision_accumulation=cfg.double_accum,
                verbosity=0,
                dirty=psfb.T,
            )
            # normalize by sum of weights to get Jy/beam units
            # done using psf_max in case some of the data points fell off the grid (sub-Nyquist imaging)
            psf_max = psfb.max()
            if psf_max > 0:
                wsums[c, b] = psf_max
                psfb /= psf_max
                residualb /= psf_max

            if cfg.natural_grad:
                # TODO - add beam application
                abspsf = jnp.abs(jnp.fft.rfft2(jnp.fft.ifftshift(psfb, axes=(0, 1)), axes=(0, 1), norm="backward"))

                # sizes follow the (Y, X) array axes
                hess = partial(hessian_slice_jax, ny, nx, 2 * ny, 2 * nx, cfg.eta, abspsf)

                residualb = cg(hess, residualb, tol=cfg.cg_tol, maxiter=cfg.cg_maxit)[0]

            elif beam_model_given:
                # TODO - spatially varying eta (fit 2D Gaussian to pbeam and use inverse)
                residualb *= pbeamb[c] / (pbeamb[c] ** 2 + cfg.eta)

            residual[c] += residualb * wsums[c, b]
            psf[c] += psfb * wsums[c, b]
            pbeam[c] += pbeamb[c] * wsums[c, b]

    # normalise by wsum
    wsum = wsums.sum(axis=1)
    for c in range(nstokes):
        if wsum[c] > 0:
            residual[c] /= wsum[c]
            psf[c] /= wsum[c]
            pbeam[c] /= wsum[c]

    # should happen after beam correction
    rms = np.std(residual, axis=(1, 2))

    # these will be in degrees; an all-zero (zero-weight) PSF fits to NaN, as in hci
    gausspars = np.asarray(fitcleanbeam(psf, level=0.5, pixsize=cell_deg, yx_order=True), dtype=np.float64)

    out = {"cube": residual.astype(np.float32)}
    if cfg.psf_out:
        out["psf"] = psf.astype(np.float32)
    out["weight"] = wsum.astype(np.float32)
    if beam_model_given:
        out["beam_weight"] = (pbeam**2 + cfg.eta).astype(np.float32)
    out["rms"] = rms.astype(np.float32)
    out["nonzero"] = wsum > 0
    out["psf_maj"] = gausspars[:, 0].astype(np.float32)
    out["psf_min"] = gausspars[:, 1].astype(np.float32)
    # convert bpa to degrees
    out["psf_pa"] = (gausspars[:, 2] * 180 / np.pi).astype(np.float32)
    out["orig_phase_dir_ra"] = np.float64(radec[0])
    out["orig_phase_dir_dec"] = np.float64(radec[1])
    out["time_mjd"] = np.float64(time_out)
    out["freq_out"] = np.float64(freq_out)
    return out
