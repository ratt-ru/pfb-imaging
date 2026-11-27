import gc
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import psutil
import ray
import xarray as xr
import zarr
from ducc0.misc import resize_thread_pool
from meerkat_beams.utils import BeamWizard
from pfb_model_spec.utils.degrid import model_geometry, model_sampling, render_model_region
from pfb_model_spec.utils.spec import open_mds
from rarg_python_patterns.multiton import Multiton
from scipy.constants import c as lightspeed

from pfb_imaging import init_ray, pfb_version, set_envs, setup_ray_worker, thread_env
from pfb_imaging.operators.gridder import grid_partition, wgridder_conventions
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.baselines import (
    GROUP_LABELS,
    check_group_telescope,
    node_baseline_groups,
    preflight_baseline_groups,
    resolve_group_beam_cache,
)
from pfb_imaging.utils.beam import BEAM_DISH_DIAMETER, BEAM_FREQ_EXTRAPOLATION, beam_grid
from pfb_imaging.utils.degrid import check_stokes_subset
from pfb_imaging.utils.fits import rdt2fits, save_fits, set_wcs
from pfb_imaging.utils.memplan import GIB, P1_THREADS, pass1_task_bytes, pass2_image_bytes, plan_pass
from pfb_imaging.utils.memprof import PhaseClock, dump_timeline, format_memory, profile_env, profile_task, task_memory
from pfb_imaging.utils.misc import (
    check_gridder_epsilon,
    fitcleanbeam,
    parse_sky_coords,
    radec_barycentre,
    radec_to_lm,
    set_image_size,
    to_mjd_time,
)
from pfb_imaging.utils.msv4 import (
    get_engine,
    ignore_xarray_ms_warnings,
    load_detached,
    select_vis_nodes,
    wrapped_angle_diff,
)
from pfb_imaging.utils.naming import glob_uris, set_output_names, uri_and_fs
from pfb_imaging.utils.stokes2vis import needed_variables, partition_beam, safe_stokes_vis
from pfb_imaging.utils.weighting import write_group_counts

ignore_xarray_ms_warnings()


log = pfb_logging.get_logger("IMAGER")


def _partition_fits(
    fits_dir, out_name, pid, field_name, baseline_group, prod, meta, freq_out, cell_rad, do_psf, do_beam
):
    """Write one partition's sanity-check FITS (dirty [+residual] [+psf] [+beam]).

    Runs inside the pass-2 worker while the partition products are in memory;
    per-partition DIRTY/RESIDUAL/PSF are wsum-normalised, the beam is written as
    stored (effective response B/n, wiki D22). The residual exists under
    ``--transfer-model-from``: each partition's own misfit to the band model,
    so a partition that calibrated worse than its siblings stands out.
    """
    field = str(field_name).replace(" ", "_").replace("/", "-")
    cell_deg = np.rad2deg(cell_rad)
    ncorr = prod["DIRTY"].shape[0]

    def mk_hdr(nx_, ny_, unit):
        return set_wcs(
            cell_deg,
            cell_deg,
            nx_,
            ny_,
            (meta["ra"], meta["dec"]),
            np.atleast_1d(freq_out),
            unit=unit,
            ms_time=meta["time_out"],
            time_is_unix=True,
            l0=meta.get("l0", 0.0),
            m0=meta.get("m0", 0.0),
            ncorr=ncorr,
        )

    # the group suffix is conditional: with grouping off there is one partition
    # per field and today's filenames must not change
    bg = "" if str(baseline_group) == "all" else f"_{baseline_group}"
    stem = f"{fits_dir}/{{var}}_{out_name}_part{pid:04d}_{field}{bg}.fits"
    wsum = prod["WSUM"][:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        dirty = np.where(wsum > 0, prod["DIRTY"] / wsum, 0.0)
    ny_im, nx_im = prod["DIRTY"].shape[1:]
    save_fits(dirty, stem.format(var="dirty"), mk_hdr(nx_im, ny_im, "Jy/beam"), yx_order=True)
    if "RESIDUAL" in prod:
        with np.errstate(invalid="ignore", divide="ignore"):
            residual = np.where(wsum > 0, prod["RESIDUAL"] / wsum, 0.0)
        save_fits(residual, stem.format(var="residual"), mk_hdr(nx_im, ny_im, "Jy/beam"), yx_order=True)
    if do_psf:
        with np.errstate(invalid="ignore", divide="ignore"):
            psf = np.where(wsum > 0, prod["PSF"] / wsum, 0.0)
        ny_psf_, nx_psf_ = prod["PSF"].shape[1:]
        save_fits(psf, stem.format(var="psf"), mk_hdr(nx_psf_, ny_psf_, "Jy/beam"), yx_order=True)
    if do_beam:
        hdr = mk_hdr(nx_im, ny_im, "")
        hdr["BEAMINCN"] = (True, "beam includes the wgridder n-term (D22)")
        save_fits(prod["BEAM"], stem.format(var="beam"), hdr, yx_order=True)


def _write_partition(ds, store, group, slab_bytes=256 * 2**20):
    """Write one partition node into the ``.dt`` store, a row slab at a time.

    zarr encodes every chunk of a ``to_zarr`` call before an fsspec store
    writes any of them, so one call holds encoded copies of everything it
    writes -- 7.4 GiB at the pass-2 peak of a 5120^2 UHF run, for the
    vis-space variables (#339). The first call writes the non-row variables
    and the first slab of rows; the rest are appended along ``row``, so the
    transient is about one slab. The stored arrays are identical either way.

    Args:
        ds: the partition Dataset (vis-space variables on ``row``).
        store: the ``.dt`` store URL.
        group: the node path, ``band####_time####/part####``.
        slab_bytes: target size of the row variables written per call.
    """
    nrow = ds.sizes.get("row", 0)
    row_vars = [v for v in ds.data_vars if "row" in ds[v].dims]
    row_bytes = sum(ds[v].nbytes for v in row_vars) / max(nrow, 1)
    step = max(1, int(slab_bytes // max(row_bytes, 1)))
    # consolidated=False: each pass-2 worker owns a distinct image_name node,
    # but they share the store root; the driver consolidates once at the end
    ds.isel(row=slice(0, step)).to_zarr(store, group=group, mode="a", consolidated=False)
    for start in range(step, nrow, step):
        slab = ds[row_vars].isel(row=slice(start, start + step))
        slab.to_zarr(store, group=group, mode="a", append_dim="row", consolidated=False)


def effective_robustness(robustness: float | None) -> float | None:
    """Map the CLI's robustness onto the imager's: ``> 2`` is natural (``None``).

    The documented contract, and not a no-op: Briggs at R > 2 is close to
    natural but not equal to it, and only ``None`` skips the uv counts.
    """
    if robustness is not None and robustness > 2:
        return None
    return robustness


def _load_piece(node):
    """Load one scratch piece into memory, minus its sparse COUNTS.

    COUNTS is only consumed by the driver's weight reduction, which has
    already happened. Loaded
    detached from the open scratch tree (see ``load_detached``), or every piece
    would stay reachable from it -- through its concat, past ``del plist`` and
    into the worker's next task.

    Args:
        node: the piece's node in the scratch DataTree.

    Returns:
        The piece as an in-memory Dataset.
    """
    return load_detached(node.ds.drop_vars(["COUNTS_INDEX", "COUNTS_VALUE"], errors="ignore"))


def _load_transfer_model(path: str) -> xr.Dataset:
    """Open and load a `--transfer-model-from` `.mds`. Module level so a `Multiton` can pickle it.

    Args:
        path: Absolute path to the `.mds`.

    Returns:
        The in-memory model Dataset.
    """
    return open_mds(path).load()


def _transfer_planes(model_stokes: list[str], corr: list[str]) -> list[int]:
    """Indices of the model's Stokes planes, in the band's `corr` order (#309).

    Args:
        model_stokes: The `.mds` `stokes` coord labels.
        corr: The band's Stokes labels.

    Returns:
        One index into `model_stokes` per entry of `corr`.

    Raises:
        ValueError: If a band Stokes is missing from the model (the driver
            refuses this upfront).
    """
    return [list(model_stokes).index(str(c)) for c in corr]


def _sampling_bounds(infos: list[dict], band_lo: float, band_hi: float) -> dict:
    """Channel extent and integration-edge time extent of one output image (#327).

    Stored on the band node so `deconv` can hand them to the `.mds`, where
    `degrid` derives its chunking from them. The band edges are the outer
    channel edges of the channels assigned to the band (flagged or not), not
    the equal-width nominal edges used to assign them: those disagree whenever
    the bands do not divide the channels evenly, and degrid must place its
    chunks on the band's real channels to reproduce deconv's forward model (D59).

    Args:
        infos: The image's per-scratch-node summaries (`time_lo`, `time_hi`: the
            outermost integration edges).
        band_lo: Lower channel edge of the band in Hz.
        band_hi: Upper channel edge of the band in Hz.

    Returns:
        `{"freq_bounds": [lo, hi], "time_bounds": [lo, hi]}`.
    """
    return {
        "freq_bounds": [float(band_lo), float(band_hi)],
        "time_bounds": [min(i["time_lo"] for i in infos), max(i["time_hi"] for i in infos)],
    }


def check_transfer_geometry(model_ds, *, grid_radec) -> None:
    """Refuse a transfer model the pass-2 render cannot place correctly (#309).

    Any cell size and image size are fine: the render remaps the model onto this
    run's grid conserving flux, refining or coarsening (pfb-model-spec#32, D52).

    Args:
        model_ds: The opened `--transfer-model-from` `.mds`.
        grid_radec: This run's tangent point `(ra, dec)` in radians.

    Raises:
        ValueError: If the tangent points differ (in-plane offsets are handled
            by x0/y0, a different tangent point needs reprojection).
    """
    model_radec = np.array([float(model_ds.attrs["ra"]), float(model_ds.attrs["dec"])])
    sep = wrapped_angle_diff(np.asarray(grid_radec, dtype=np.float64), model_radec)
    if np.any(sep > 1e-9):
        raise ValueError(
            f"Tangent point mismatch: this run is centred on (ra, dec) = {np.rad2deg(grid_radec)} deg, "
            f"the transfer model on {np.rad2deg(model_radec)} deg. Transferring a model to a different "
            "tangent point needs reprojection, which is not supported."
        )


def transfer_freq_coverage(model_ds) -> tuple[float, float, bool]:
    """The frequency range a transfer model's fit covers (#309, D52).

    Args:
        model_ds: The opened `--transfer-model-from` `.mds`.

    Returns:
        `(lo, hi, from_bounds)` in Hz: the extent of `freq_bounds` when the model
        records them, else of its fitted `freqs` (D50).
    """
    fb = model_sampling(model_ds)["freq_bounds"]
    if fb is not None:
        return float(fb.min()), float(fb.max()), True
    return float(model_ds.freqs.min()), float(model_ds.freqs.max()), False


def transfer_coverage_warning(freqs_out, coverage: tuple[float, float, bool]) -> str | None:
    """Warn when pass 2 rendered the model outside its fitted range (#309, D52).

    Checked against the effective frequencies the model was actually rendered
    at, which are known only after pass 2 (D28): the nominal band centres differ
    from them, and a model from a run with identical settings was fitted at
    exactly these frequencies.

    Args:
        freqs_out: The bands' effective frequencies in Hz.
        coverage: `transfer_freq_coverage` of the model.

    Returns:
        The message, or `None` when every frequency lies inside the coverage.
    """
    lo, hi, from_bounds = coverage
    freqs_out = np.sort(np.asarray(freqs_out, dtype=np.float64))
    # relative slack for float round-off in a recomputed effective frequency
    tol = 1e-9 * hi
    outside = freqs_out[(freqs_out < lo - tol) | (freqs_out > hi + tol)]
    if not outside.size:
        return None
    source = "its freq_bounds" if from_bounds else "its fitted frequencies; it records no freq_bounds"
    return (
        f"The transfer model was rendered at {', '.join(f'{f / 1e6:.3f}' for f in outside)} MHz, outside the "
        f"{lo / 1e6:.3f}-{hi / 1e6:.3f} MHz it covers ({source}); its frequency fit was extrapolated there."
    )


def effective_frequency(pieces, freq_nominal):
    """An output image's effective frequency, from its pieces, before pass 2 (wiki D28).

    The imaging-wsum-weighted mean of the pieces' own effective frequencies:
    the image is a sum of imaging-weighted visibilities, so to first order in
    the spectrum a source appears at this frequency, and it is the one the
    transferred model is rendered at. Partition frequencies and beams are
    reduced on the same weights (``_concat_pieces``), and so is the band level
    in pass 2, which recomputes this from its gridded ``WSUM`` and refuses a
    disagreement.

    Args:
        pieces: ``(wsum_img, freq_out)`` per piece of the image.
        freq_nominal: Fallback when the image carries no weight.

    Returns:
        The effective frequency in Hz.
    """
    w = np.array([p[0] for p in pieces], dtype=np.float64)
    f = np.array([p[1] for p in pieces], dtype=np.float64)
    den = float(w.sum())
    return float((w * f).sum() / den) if den > 0 else float(freq_nominal)


def _concat_pieces(plist):
    """Reduce a partition's scratch pieces to a single Dataset.

    Rows are concatenated; the field-centred BEAM_SMALL and the effective
    frequency are combined as ``wsum_img``-weighted means, the pieces'
    imaging-weight sums the driver predicted before pass 2
    (``weighting.piece_imaging_wsum``). Pieces of a partition can differ in
    both -- BeamWizard evaluates the beam at the piece's own timestamps, and
    post-#296 at its own effective frequency -- so taking piece 0's beam, as
    this used to, silently discarded every other piece (wiki D28). The caller
    places the mean on the image grid once (``partition_beam``); placement is
    linear, so that equals averaging the placed beams (#365). The weights are
    the imaging ones, the same basis as the band-level reduction in pass 2,
    because those are what each piece contributes to the gridded image.

    Args:
        plist: scratch-piece Datasets of one partition, all sharing a FREQ axis.

    Returns:
        The merged partition Dataset, carrying the weighted ``freq_out`` and
        the summed ``wsum_img`` in its attrs. A single-piece list is returned
        unchanged.
    """
    if len(plist) == 1:
        return plist[0]

    # rows are only concatenatable when they share the freq axis (same
    # spw + band channel-chunk); guard the invariant before concat
    f0 = plist[0].FREQ.values
    for p in plist[1:]:
        assert np.array_equal(p.FREQ.values, f0), "concat group has mismatched FREQ; cannot concatenate rows"

    # the small beams share one field-centred grid by construction (the wizard's
    # cube axes, or katbeam's run-wide Nyquist grid); averaging them before the
    # one placement in pass 2 is exact only on a shared grid
    has_beam = "BEAM_SMALL" in plist[0]
    if has_beam:
        lb, mb = plist[0].l_beam.values, plist[0].m_beam.values
        for p in plist[1:]:
            assert np.array_equal(p.l_beam.values, lb) and np.array_equal(p.m_beam.values, mb), (
                "concat group has mismatched beam grids; cannot average BEAM_SMALL"
            )

    w = np.array([float(p.attrs["wsum_img"]) for p in plist], dtype=np.float64)
    wsum_img = float(w.sum())
    beam = None
    if wsum_img > 0:
        freqs = np.array([float(p.attrs["freq_out"]) for p in plist], dtype=np.float64)
        freq_out = float((w * freqs).sum() / wsum_img)
        if has_beam:
            # accumulate rather than stack: one buffer, not npiece of them
            beam = np.zeros(plist[0].BEAM_SMALL.shape, dtype=np.float64)
            for wi, p in zip(w, plist):
                beam += wi * p.BEAM_SMALL.values
            beam /= wsum_img
    else:
        freq_out = float(plist[0].attrs["freq_out"])
        if has_beam:
            beam = plist[0].BEAM_SMALL.values

    rowvars = ["VIS", "WEIGHT", "MASK", "UVW"]
    cat = xr.concat([p[rowvars] for p in plist], dim="row", coords="minimal", compat="override")
    part = plist[0].drop_vars(rowvars)
    for v in rowvars:
        part[v] = cat[v]
    del cat
    if has_beam:
        part["BEAM_SMALL"] = (part.BEAM_SMALL.dims, beam.astype(part.BEAM_SMALL.dtype))
    part.attrs["freq_out"] = freq_out
    part.attrs["wsum_img"] = wsum_img
    # Re(B) diagnostic (wiki D46): report the WORST piece, not piece 0's, which
    # is what `part = plist[0].drop_vars(...)` would otherwise leave here.
    part.attrs["beam_imre_ratio"] = max(float(p.attrs.get("beam_imre_ratio", 0.0)) for p in plist)
    return part


def _check_prediction(what, where, got, predicted, rtol):
    """Refuse a pass-2 product that disagrees with the driver's pre-pass-2 reduction.

    Raises:
        RuntimeError: If ``got`` and ``predicted`` differ by more than ``rtol``.
    """
    if not np.isclose(got, predicted, rtol=rtol, atol=0.0):
        raise RuntimeError(
            f"{what} of {where}: gridded {got!r} but predicted {predicted!r} before pass 2 "
            f"(wiki D28: the counts reduction and pass 2 must agree)"
        )


@ray.remote
def _grid_image(*args, **kwargs):
    """Ray entry point for :func:`_grid_image_body` (opt-in profiling, see memprof)."""
    with profile_task("grid_image") as clock:
        res = _grid_image_body(*args, **kwargs)
    res["mem"].update(clock)
    return res


def _grid_image_body(
    scratch_store,
    dt_store,
    src_names,
    out_name,
    counts_group,
    nx,
    ny,
    nx_psf,
    ny_psf,
    cell_rad,
    freq_nominal,
    meta,
    robustness=None,
    counts_nx=None,
    counts_ny=None,
    counts_cell=None,
    nthreads=1,
    epsilon=1e-5,
    do_wgridding=True,
    double_accum=True,
    do_psf=True,
    part_fits_dir=None,
    part_fits_beam=True,
    transfer_model=None,
    transfer_path=None,
):
    """Pass-2 worker for one output image.

    Groups the image's fine scratch pieces into partitions ``(msid, field, spw,
    baseline_group)`` and processes them one at a time: loads a partition's
    pieces, concatenates scans along ``row``, grids it with
    :func:`grid_partition`, sums the image-space products into the band node, and
    writes the band node + ``part####`` children into the ``.dt`` store. Returns a
    light summary (``timeid``/``wsum``/telemetry); the band PSF goes to the ``.dt``
    only -- returning it would put an image per task in the Ray object store.

    ``transfer_model`` is a ``Multiton`` over ``_load_transfer_model``, or ``None``;
    ``transfer_path`` is its path, recorded on the band node. With a model, the
    band node also gets ``MODEL``/``RESIDUAL``/``BRESIDUAL`` (#309).
    """
    resize_thread_pool(nthreads)
    # cache=False: otherwise the counts read below is memoised on the tree,
    # which outlives the task in a reference cycle until the next gc (#339).
    # Pieces are detached from the tree by _load_piece for the same reason.
    dt = xr.open_datatree(scratch_store, engine="zarr", chunks=None, cache=False)
    # partition key -> its pieces' node paths, from the pieces' attrs alone: a
    # piece is loaded only when its partition is gridded, so the task holds one
    # partition's visibilities at a time, not the whole image's
    groups = {}
    # each partition's imaging wsum as the driver predicted it (write_group_counts)
    wsum_pred = {}
    for sn in src_names:
        for cname, child in dt[sn].children.items():
            a = child.ds.attrs  # lazy: nothing is read
            key = (a["msid"], a["field_name"], a["spw_name"], a["baseline_group"])
            path = f"{sn}/{cname}"
            groups.setdefault(key, []).append(path)
            wsum_pred[key] = wsum_pred.get(key, 0.0) + meta["wsum_img"][path]

    # the applied (filtered, box-summed) group counts, written once by the driver
    # (write_group_counts); grid_partition copies before use
    counts = dt[counts_group].ds.COUNTS.values if robustness is not None else None

    # a lazy piece, for the correlations and the dtype: reading nothing else
    first = dt[next(iter(groups.values()))[0]].ds
    corr = first.corr.values
    ncorr = corr.size
    bpar = ["BMAJ", "BMIN", "BPA"]
    # The whole tree carries the requested --precision: pass 1 stamped it on the
    # scratch pieces, grid_partition's ducc buffers follow the vis dtype (ducc
    # templates every array on one type), and the accumulation below must not
    # upcast it back to f8 (--double-accum is a wgridder-internal control only).
    real_type = np.empty(0, dtype=first.VIS.dtype).real.dtype
    del first
    model = None
    if transfer_model is not None:
        # Render at the image's effective frequency, which the driver predicted
        # from the counts before pass 2 (D28): the residual is then formed per
        # partition while its visibilities are in memory (#309, #365).
        model_ds = transfer_model.instance  # one load per worker process
        _, _, _, x0, y0 = wgridder_conventions(meta.get("l0", 0.0), meta.get("m0", 0.0))
        rendered = render_model_region(
            model_ds,
            time=meta["time_out"],
            freq_out=float(meta["freq_out"]),
            nx=nx,
            ny=ny,
            cell_rad=cell_rad,
            x0=x0,
            y0=y0,
            # flux-conserving remap onto this grid, refining or coarsening (D52)
            kernel="box",
        )
        planes = _transfer_planes([str(s) for s in model_ds.stokes.values], [str(c) for c in corr])
        model = np.ascontiguousarray(rendered[planes], dtype=real_type)
        del rendered, model_ds
    dirty_sum = np.zeros((ncorr, ny, nx), dtype=real_type)
    residual_sum = np.zeros((ncorr, ny, nx), dtype=real_type) if model is not None else None
    bresidual_sum = np.zeros((ncorr, ny, nx), dtype=real_type) if model is not None else None
    psf_sum = np.zeros((ncorr, ny_psf, nx_psf), dtype=real_type) if do_psf else None
    beam_sum = np.zeros((ncorr, ny, nx), dtype=real_type)
    bdirty_sum = np.zeros((ncorr, ny, nx), dtype=real_type)
    wsum_sum = np.zeros(ncorr, dtype=real_type)
    # float64 copy for the frequency check alone: summed over hundreds of
    # partitions in float32 it drifts to within a factor of 2 of the 1e-6 rtol
    wsum_check = np.zeros(ncorr, dtype=np.float64)
    # worst Re(B) approximation error over this image's partitions (wiki D46)
    beam_imre_max = 0.0
    # float64 regardless of --precision: the tree may be single-precision
    # (wiki D27) and float32 resolves 1e9 Hz only to ~64 Hz
    freq_sum = np.zeros(ncorr, dtype=np.float64)

    for pid, key in enumerate(list(sorted(groups))):
        plist = []
        for path in groups.pop(key):
            piece = _load_piece(dt[path])
            piece.attrs["wsum_img"] = meta["wsum_img"][path]
            plist.append(piece)
        # for a one-piece partition _concat_pieces returns the piece itself
        del piece
        part = _concat_pieces(plist)
        # release the pre-concat originals before gridding (concat copied them)
        del plist
        # place the partition's (mean) field-centred beam on the image grid once,
        # at the offset the partition is gridded with, and fold in 1/n (D22, #365)
        beam = partition_beam(
            part,
            np.array([meta["ra"], meta["dec"]]),
            cell_rad,
            nx,
            ny,
            l0=meta.get("l0", 0.0),
            m0=meta.get("m0", 0.0),
            dtype=real_type,
            nthreads=nthreads,
        )
        part = part.drop_vars(["BEAM_SMALL", "l_beam", "m_beam"], errors="ignore").assign(
            BEAM=(("corr", "y", "x"), beam)
        )
        del beam

        prod = grid_partition(
            part,
            counts,
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            robustness=robustness,
            counts_nx=counts_nx,
            counts_ny=counts_ny,
            counts_cell=counts_cell,
            l0=meta.get("l0", 0.0),
            m0=meta.get("m0", 0.0),
            nthreads=nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            do_psf=do_psf,
            # part is this task's own concat (or loaded piece), and only the
            # imaging weights are stored: reuse its WEIGHT buffer (#339)
            overwrite_weight=True,
            model=model,
        )
        # before anything is written, so a failed prediction leaves no partial node
        _check_prediction("imaging wsum", key, float(prod["WSUM"].astype(np.float64).sum()), wsum_pred[key], 1e-4)

        part_vars = {
            "VIS": (("corr", "row", "chan"), part.VIS.values),
            "WEIGHT": (("corr", "row", "chan"), prod["WEIGHT"]),
            "MASK": (("row", "chan"), part.MASK.values),
            "UVW": (("row", "three"), part.UVW.values),
            "FREQ": (("chan",), part.FREQ.values),
            "BEAM": (("corr", "y", "x"), prod["BEAM"]),
        }
        part_coords = {"corr": corr}
        if do_psf:
            part_vars["PSF"] = (("corr", "y_psf", "x_psf"), prod["PSF"])
            part_vars["PSFHAT"] = (("corr", "y_psf", "xo2"), prod["PSFHAT"])
            part_vars["PSFPARSN"] = (("corr", "bpar"), prod["PSFPARSN"])
            part_coords["bpar"] = bpar
        part_out = xr.Dataset(
            part_vars,
            coords=part_coords,
            attrs={
                "msid": int(key[0]),
                "field_name": key[1],
                "spw_name": key[2],
                "baseline_group": key[3],
                "freq_out": float(part.attrs["freq_out"]),
                "ra": meta["ra"],
                "dec": meta["dec"],
                "ra0": float(part.attrs.get("ra0", meta["ra"])),
                "dec0": float(part.attrs.get("dec0", meta["dec"])),
                "l0": meta.get("l0", 0.0),
                "m0": meta.get("m0", 0.0),
                # stored BEAM = effective image-plane response B/n (D22)
                "beam_includes_n": True,
                # max|Im|/max|Re| of the evaluated beam; non-zero only for the
                # MPM cross group, where the stored BEAM is Re(B) (wiki D46)
                "beam_imre_ratio": float(part.attrs.get("beam_imre_ratio", 0.0)),
                "wsum": prod["WSUM"].tolist(),
            },
        )
        _write_partition(part_out, dt_store, f"{out_name}/part{pid:04d}")
        beam_imre_max = max(beam_imre_max, float(part.attrs.get("beam_imre_ratio", 0.0)))

        if part_fits_dir is not None:
            _partition_fits(
                part_fits_dir,
                out_name,
                pid,
                key[1],
                key[3],
                prod,
                meta,
                float(part.attrs["freq_out"]),
                cell_rad,
                do_psf=do_psf,
                do_beam=part_fits_beam,
            )

        dirty_sum += prod["DIRTY"]
        beam_sum += prod["WSUM"][:, None, None] * prod["BEAM"]
        # exact beam-attenuated dirty sum_p B_p * dirty_p: the model-free term
        # of the deconv gradient (D23), not derivable from the summed DIRTY
        # when partitions carry distinct beams
        bdirty_sum += prod["BEAM"] * prod["DIRTY"]
        if model is not None:
            residual_sum += prod["RESIDUAL"]
            # the outer per-partition beam of the gradient residual (D23)
            bresidual_sum += prod["BEAM"] * prod["RESIDUAL"]
        if do_psf:
            psf_sum += prod["PSF"]
        wsum_sum += prod["WSUM"]
        wsum_check += prod["WSUM"]
        freq_sum += prod["WSUM"].astype(np.float64) * float(part.attrs["freq_out"])
        # Release this partition before the next one loads, and the last one before
        # the band products are assembled (#339). part_vars and part_out hold the
        # same VIS/WEIGHT/MASK/UVW buffers as part; prod holds its image products,
        # which would otherwise live on through the next partition's gridding.
        # The collect: loaded xarray objects sit in reference cycles (see safe_stokes_vis).
        del part, part_vars, part_out, prod
        gc.collect()

    # Level-2 reduction: the band product is the wsum-weighted sum of its
    # partitions, so its effective frequency is the wsum-weighted mean of the
    # partition frequencies -- the identical reduction beam_sum uses below.
    # These are the imaging prod["WSUM"], the same basis _concat_pieces uses
    # one level down, so the whole reduction is the imaging-weighted mean over
    # pieces (wiki D28). Reduced with corr-summed weights and stored scalar, as
    # dt2fits already does for freq_mfs.
    wsum_tot = float(wsum_check.sum())
    freq_gridded = float(freq_sum.sum() / wsum_tot) if wsum_tot > 0 else float(freq_nominal)
    # the driver's value is the one the model was rendered at; this pins that it
    # is the same reduction (to the --precision rounding of WSUM)
    freq_eff = float(meta["freq_out"])
    _check_prediction("effective frequency", out_name, freq_gridded, freq_eff, 1e-6)
    band_attrs = {
        "bandid": meta["bandid"],
        "timeid": meta["timeid"],
        "freq_out": freq_eff,
        "freq_nominal": float(freq_nominal),
        "time_out": meta["time_out"],
        "freq_bounds": meta.get("freq_bounds"),
        "time_bounds": meta.get("time_bounds"),
        "ra": meta["ra"],
        "dec": meta["dec"],
        "l0": meta.get("l0", 0.0),
        "m0": meta.get("m0", 0.0),
        "cell_rad": cell_rad,
        "robustness": robustness,
        "niters": 0,
    }
    band_attrs = {k: v for k, v in band_attrs.items() if v is not None}
    # band-level BEAM: wsum-weighted mean of the partition beams -- the
    # linear-mosaic response (still the effective B/n of wiki D22).
    # RESIDUAL/BRESIDUAL are only computed here with --transfer-model-from
    # (deconv falls back to DIRTY/BDIRTY otherwise).
    with np.errstate(invalid="ignore", divide="ignore"):
        beam_avg = np.where(wsum_sum[:, None, None] > 0, beam_sum / wsum_sum[:, None, None], 0.0)
    band_vars = {
        "DIRTY": (("corr", "y", "x"), dirty_sum),
        "BDIRTY": (("corr", "y", "x"), bdirty_sum),
        "BEAM": (("corr", "y", "x"), beam_avg),
        "WSUM": (("corr",), wsum_sum),
    }
    band_coords = {"corr": corr}
    if do_psf:
        band_vars["PSF"] = (("corr", "y_psf", "x_psf"), psf_sum)
        band_vars["PSFPARSN"] = (
            ("corr", "bpar"),
            # fitcleanbeam returns python floats; keep the tree at --precision
            np.array(fitcleanbeam(psf_sum / wsum_sum[:, None, None], yx_order=True), dtype=real_type),
        )
        band_coords["bpar"] = bpar
    if model is not None:
        band_vars["MODEL"] = (("corr", "y", "x"), model)
        band_vars["RESIDUAL"] = (("corr", "y", "x"), residual_sum)
        band_vars["BRESIDUAL"] = (("corr", "y", "x"), bresidual_sum)
        band_attrs_extra = {"transferred_from": transfer_path}
    else:
        band_attrs_extra = {}
    band_ds = xr.Dataset(
        band_vars,
        coords=band_coords,
        attrs={**band_attrs, **band_attrs_extra},
    )
    band_ds.to_zarr(dt_store, group=out_name, mode="a", consolidated=False)

    # break the reference cycles holding the loaded pieces so a reused Ray
    # worker doesn't accumulate them across tasks (see safe_stokes_vis)
    dt.close()
    del groups, dt
    gc.collect()

    # post-gc memory telemetry (see safe_stokes_vis for interpretation)
    mem = task_memory()
    return {
        "timeid": meta["timeid"],
        "freq_out": freq_eff,
        "wsum": wsum_sum,
        "mem": mem,
        "beam_imre_ratio": beam_imre_max,
    }


def scan_ms(
    ms,
    *,
    partition_columns,
    data_group,
    field_names,
    spw_names,
    scan_names,
    freq_min,
    freq_max,
    check_telescope,
    phase_dir,
    channels_per_image,
):
    """Read what set-up needs from the MSs: the extent of the data and the bands.

    This is the expensive part of set-up (it loads every selected node's UVW),
    and what a reused scratch skips: ``derived`` is stored in its record.

    Returns:
        ``(derived, selected, all_freqs, vis_col, wgt_col)``: ``derived`` is
        JSON-serialisable (``max_blength``, ``max_uvw``, ``max_freq``,
        ``min_chan_width``, ``band_centres``, ``field_centres``, ``radec_new``,
        ``times``); ``selected`` caches each node for the pass-1 dispatch;
        ``all_freqs`` every selected channel (the wizards' frequency check);
        ``vis_col``/``wgt_col`` the data group's column names.
    """
    all_freqs = []
    all_chan_widths = []
    max_blength = 0.0
    # 3D: bounds |uv| after any rephasing, which max_blength (uv only) does not
    max_uvw = 0.0
    vis_col = wgt_col = None
    selected = []  # cached (ims, node, freqs_node, times_node, chan0, field_radec) for the dispatch loop
    for ims, ms_name in enumerate(ms):
        dt_kwargs = get_engine(ms_name, partition_columns)
        if "file://" in ms_name:
            ms_name = ms_name.replace("file://", "")
        # the only open of this MS in the driver: the --baseline-groups telescope
        # check reads the same tree (#365)
        dt = xr.open_datatree(ms_name, **dt_kwargs)
        if check_telescope:
            check_group_telescope(dt, ms_name)
        # Name-based selection, the frequency window and the chan0 offset all
        # come from utils/msv4.select_vis_nodes, shared with degrid so the
        # two front ends cannot drift. It selects channels by matching index
        # rather than by label slice, which is correct for a descending
        # spectral window and gives each node a chan0 on its own axis.
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
            if vis_col is None:
                vis_col = node.ds.attrs["data_groups"][data_group]["correlated_data"]
            if wgt_col is None:
                wgt_col = node.ds.attrs["data_groups"][data_group]["weight"]
            # chan0 indexes the *unsliced* node, which is also what the dispatch
            # loop applies its isel to
            ds = node.ds.isel(frequency=slice(sel.chan0, sel.chan0 + sel.nchan))
            freqs_node = ds.frequency.load().values
            all_freqs.append(freqs_node)
            all_chan_widths.append(ds.frequency.attrs["channel_width"]["data"])
            # xarray-ms establishes a regular grid over irregular or missing data,
            # inserting nans (xarray reads them as missing data)
            uvw = ds.UVW.load().values
            uvw_mask = np.isnan(uvw).all(axis=-1)
            # this forces a reshape (t, bl, 3) -> (row, 3)
            uvw = uvw[~uvw_mask]
            if uvw.size:
                max_blength = max(max_blength, float(np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max()))
                max_uvw = max(max_uvw, float(np.linalg.norm(uvw, axis=-1).max()))
            selected.append((ims, node, freqs_node, ds.time.load().values, sel.chan0, sel.field_radec))

    if not selected:
        log.error_and_raise("Selection matched no data", ValueError)

    # resolve the common phase centre (mosaic tangent point): explicit
    # phase_dir wins; multiple distinct field centres default to their
    # barycentre; a single field keeps its own centre (no rephasing)
    field_centres = np.unique(np.array([np.round(fc, 12) for *_, fc in selected]), axis=0)
    if phase_dir is not None:
        radec_new = parse_sky_coords(phase_dir)
        log.info("Rephasing all data to phase_dir")
    elif field_centres.shape[0] > 1:
        radec_new = radec_barycentre(field_centres)
        log.info("Multiple fields selected; rephasing to their barycentre")
    else:
        radec_new = None

    # guard against irregular channel widths
    cw = np.asarray(all_chan_widths)
    cw = cw[np.isfinite(cw)]
    if cw.size == 0:
        log.error_and_raise("No SPW has a usable channel_width", ValueError)
    min_chan_width = float(np.min(cw))
    if channels_per_image in (0, None, -1):
        nband = len(np.unique([f.tobytes() for f in all_freqs]))  # one per spw
    else:
        flat_freqs = np.concatenate(all_freqs)
        nband = int(np.ceil((flat_freqs.max() - flat_freqs.min()) / (min_chan_width * channels_per_image)))
        nband = max(nband, 1)
    all_freqs = np.unique(np.concatenate(all_freqs))
    band_edges = np.linspace(all_freqs.min() - min_chan_width / 2, all_freqs.max() + min_chan_width / 2, nband + 1)
    # Band-edge midpoints. These define which channels belong to which band and
    # nothing else -- the band's reported frequency is the effective (weighted)
    # one reduced in pass 2 (issue #296, wiki D28).
    band_centres = band_edges[:-1] + (band_edges[1] - band_edges[0]) / 2

    # a --target is resolved at the ends and middle of the track: a named body moves
    times = np.concatenate([np.ravel(t) for *_, t, _, _ in selected])
    times = times[np.isfinite(times)]
    derived = {
        "max_blength": max_blength,
        "max_uvw": max_uvw,
        "max_freq": float(all_freqs.max()),
        "min_chan_width": min_chan_width,
        "band_centres": band_centres.tolist(),
        "field_centres": field_centres.tolist(),
        "radec_new": None if radec_new is None else [float(v) for v in radec_new],
        "times": [float(times.min()), float(times.mean()), float(times.max())],
    }
    return derived, selected, all_freqs, vis_col, wgt_col


def pointing_reach(field_centres, grid_radec, max_fov_rad):
    """How far from a pointing centre the scratch must describe the sky (deg).

    Every image the scratch can serve lies within ``--max-field-of-view`` (a
    radius ``R`` about the image tangent point, the same radius BDA bounds its
    decorrelation over), so a field's beam is needed out to ``R`` plus that
    field's distance from the tangent point. Neither depends on the image grid
    (#354).

    Args:
        field_centres: ``(nfield, 2)`` pointing centres (rad).
        grid_radec: The image tangent point (rad).
        max_fov_rad: ``R`` (rad).

    Returns:
        The largest such distance over the fields (deg).
    """
    dist = max(float(np.hypot(*radec_to_lm(np.asarray(fc), grid_radec))) for fc in field_centres)
    return float(np.rad2deg(dist + max_fov_rad))


def check_image_extent(nx, ny, cell_rad, offsets, max_fov_rad):
    """Refuse an image reaching past ``--max-field-of-view`` from the tangent point.

    Args:
        nx: Image pixels along x.
        ny: Image pixels along y.
        cell_rad: Image cell (rad).
        offsets: ``(l0, m0)`` image-centre offsets to cover (rad).
        max_fov_rad: ``R`` (rad).

    Returns:
        The distance of the farthest image pixel from the tangent point (deg).

    Raises:
        ValueError: If that distance exceeds ``R``.
    """
    half_diag = 0.5 * np.hypot(nx, ny) * cell_rad
    reach = max(float(np.hypot(l0, m0)) for l0, m0 in offsets) + half_diag
    if reach > max_fov_rad * (1 + 1e-12):
        raise ValueError(
            f"the image reaches {np.rad2deg(reach):.3f} deg from the phase centre, beyond "
            f"--max-field-of-view {np.rad2deg(max_fov_rad):.3f} deg (a radius). Reduce the image "
            f"(--field-of-view, --nx/--ny, --cell-size) or the --target offset, or raise --max-field-of-view."
        )
    return float(np.rad2deg(reach))


# Bump when what pass 1 writes changes, so older scratches are refused (#354).
SCRATCH_SCHEMA = 1


def read_scratch_record(scratch_url: str) -> dict | None:
    """The validity record pass 1 left in a scratch's root attrs, or None.

    None means the scratch was written by a pfb-imaging without records, or
    pass 1 did not finish: the record is written last.
    """
    try:
        raw = zarr.open_group(scratch_url, mode="r").attrs.get("pass1")
        return None if raw is None else json.loads(raw)
    except (FileNotFoundError, KeyError, ValueError, zarr.errors.GroupNotFoundError):
        # ValueError covers a truncated or corrupt record (json.JSONDecodeError)
        return None


def record_diffs(record: dict | None, inputs: dict, counts: dict | None = None) -> list[str]:
    """Why a scratch cannot serve this run: one line per differing field.

    Args:
        record: :func:`read_scratch_record`'s result.
        inputs: This run's pass-1 inputs (:func:`pass1_inputs`).
        counts: This run's counts grid, compared once the inputs match.

    Returns:
        Empty when the scratch can be reused.
    """
    if record is None:
        return ["it has no validity record (written by an older pfb-imaging, or pass 1 did not finish)"]
    if record.get("schema") != SCRATCH_SCHEMA:
        return [f"its schema is {record.get('schema')}, this pfb-imaging writes {SCRATCH_SCHEMA}"]
    old = record["inputs"]
    diffs = [f"{k}: {old.get(k)!r} -> {v!r}" for k, v in inputs.items() if old.get(k) != v]
    diffs += [f"{k}: {old[k]!r} (no longer an input)" for k in old if k not in inputs]
    if not diffs and counts is not None and record["counts"] != counts:
        diffs.append(
            f"counts grid: {record['counts']} -> {counts} (under --weight-on-image-grid it follows the image: "
            f"--field-of-view, --cell-size, --nx/--ny, --super-resolution-factor, --min-padding)"
        )
    return diffs


def pass1_inputs(**kwargs) -> dict:
    """Canonical, JSON-comparable values of everything pass 1's output depends on.

    Sequences become lists and MS paths are absolute. A beam model given as an
    object (tests pass BeamWizards) cannot be compared: it gets a fresh uuid, so
    it never matches (an ``id`` can be reused after garbage collection).
    """
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            v = [str(x) if k == "ms" else x for x in v]
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    beam_model = out.get("beam_model")
    if beam_model is not None and not isinstance(beam_model, str):
        out["beam_model"] = f"<{type(beam_model).__name__} object {uuid.uuid4().hex}>"
    return json.loads(json.dumps(out))


def image_geometry(
    max_blength,
    max_uvw,
    max_freq,
    *,
    field_of_view,
    super_resolution_factor,
    cell_size,
    nx,
    ny,
    psf_oversize,
    min_padding,
    weight_on_image_grid,
    max_fov_rad,
):
    """The image grid and the counts grid, from the data's extent and the CLI.

    Returns:
        dict: ``nx``, ``ny``, ``nx_psf``, ``ny_psf``, ``cell_n``, ``cell_rad``,
        ``cell_deg``, ``nx_pad``, ``ny_pad`` and ``counts`` (``{"nx", "ny",
        "cell"}``, the grid pass 1 grids the counts on).
    """
    nx, ny, nx_psf, ny_psf, cell_n, cell_rad, cell_deg = set_image_size(
        max_blength, max_freq, field_of_view, super_resolution_factor, cell_size, nx, ny, psf_oversize
    )
    # the padded uv grid of the image; ideally the wgridder's own padding,
    # which it chooses internally (consider adding that to radiomesh)
    nx_pad = int(np.ceil(min_padding * nx))
    nx_pad += nx_pad % 2
    ny_pad = int(np.ceil(min_padding * ny))
    ny_pad += ny_pad % 2
    if weight_on_image_grid:
        cnx, cny, ccell = nx_pad, ny_pad, cell_rad
    else:
        cnx, cny, ccell = counts_grid(max_uvw, max_freq, max_fov_rad, min_padding)
    return {
        "nx": nx,
        "ny": ny,
        "nx_psf": nx_psf,
        "ny_psf": ny_psf,
        "cell_n": cell_n,
        "cell_rad": cell_rad,
        "cell_deg": cell_deg,
        "nx_pad": nx_pad,
        "ny_pad": ny_pad,
        "counts": {"nx": int(cnx), "ny": int(cny), "cell": float(ccell)},
    }


def counts_grid(max_uvw, max_freq, max_fov_rad, min_padding):
    """The counts grid that serves every image within ``--max-field-of-view`` (#354).

    Its uv cell, ``1 / (min_padding * sqrt(2) * R)``, is the padded image-grid
    cell of the largest square image within ``R`` (side ``sqrt(2) * R``), and
    it reaches one cell past the longest baseline at the highest frequency,
    measured in 3D: rephasing (``--phase-dir``, a mosaic barycentre) rotates
    ``uvw``, so only ``|uvw|`` bounds the ``uv`` pass 1 grids. It
    is returned as the padded uv grid of a virtual image, the form the counts
    kernels take: an even ``n`` puts every cell edge on a multiple of the uv
    cell, so a grid with the same cell bins identically whatever its extent.

    Args:
        max_uvw: Longest baseline, ``max |uvw|`` (m).
        max_freq: Highest frequency gridded (Hz).
        max_fov_rad: ``R`` (rad).
        min_padding: ``--min-padding``.

    Returns:
        ``(n, n, cell)``: grid size and the virtual image cell (rad).
    """
    du = 1.0 / (min_padding * np.sqrt(2.0) * max_fov_rad)
    umax = max_uvw * max_freq / lightspeed
    n = 2 * (int(np.ceil(umax / du)) + 1)
    return n, n, 1.0 / (n * du)


def field_beam_axis(wizards, reach_deg, max_freq):
    """The field-centred beam axis every pass-1 piece evaluates its beam on (#365, #354).

    It reaches ``reach_deg`` from the pointing (:func:`pointing_reach`), at the
    cell ``utils.beam.beam_grid`` sets for the bilinear placement tolerance, but
    no further than the beam cube covers. The rotation average turns the field
    with parallactic angle, so only the cube's largest centred disk is covered
    at every angle, and beyond it the placement error does not converge with the
    cell (wiki D64). The farthest pixel reads bilinear nodes up to one diagonal
    cell further out, and two cube pixels at the edge are kept clear, where the
    cube's cubic spline is pulled towards its zero fill. An image reaching past
    the returned axis is refused by :func:`check_beam_reach`.

    Args:
        wizards: Baseline group (None when ungrouped) -> BeamWizard.
        reach_deg: Distance from a pointing the beam is wanted out to (deg).
        max_freq: Highest frequency gridded (Hz).

    Returns:
        ``(beam_lm, axis_deg, limit_deg)``: the axis (deg), the distance from
        the pointing it serves, and the radius the cube covers less the edge
        margin.
    """
    limit_deg = np.inf
    for wizard in wizards.values():
        x, y = wizard.bds.X.values, wizard.bds.Y.values
        xy = np.concatenate([x, y])
        cover = float(np.abs([xy.min(), xy.max()]).min())
        dx = float(max(np.abs(np.diff(x)).max(), np.abs(np.diff(y)).max()))
        limit_deg = min(limit_deg, cover - 2.0 * dx)
    # the largest dish any of these beams describes bounds the error for all of them
    dish = max(BEAM_DISH_DIAMETER.get(g, BEAM_DISH_DIAMETER["MPMP"]) for g in wizards)
    cell = float(np.diff(beam_grid(max_freq, dish, 0.0))[0])
    axis_deg = min(reach_deg, limit_deg - np.sqrt(2.0) * cell)
    return beam_grid(max_freq, dish, axis_deg), axis_deg, limit_deg


def check_beam_reach(field_centres, grid_radec, offsets, nx, ny, cell_rad, axis_deg, limit_deg):
    """Refuse an image reaching past the beam axis from any pointing (wiki D64).

    Only binds when the beam cube capped the axis (:func:`field_beam_axis`);
    otherwise :func:`check_image_extent` already keeps every image inside it.

    Args:
        field_centres: ``(nfield, 2)`` pointing centres (rad).
        grid_radec: The image tangent point (rad).
        offsets: ``(l0, m0)`` image-centre offsets to cover (rad).
        nx: Image pixels along x.
        ny: Image pixels along y.
        cell_rad: Image cell (rad).
        axis_deg: Distance from a pointing the beam axis serves (deg).
        limit_deg: Radius the beam cube covers less its edge margin (deg).

    Raises:
        ValueError: If the image reaches beyond ``axis_deg`` from a pointing.
    """
    half_diag = 0.5 * np.hypot(nx, ny) * cell_rad
    radius = 0.0
    for fc in field_centres:
        lf, mf = radec_to_lm(np.asarray(fc), grid_radec)
        for l0, m0 in offsets:
            radius = max(radius, float(np.hypot(l0 - lf, m0 - mf)) + half_diag)
    radius_deg = float(np.rad2deg(radius))
    if radius_deg > axis_deg * (1 + 1e-12):
        raise ValueError(
            f"the image reaches {radius_deg:.3f} deg from a pointing centre, beyond the {axis_deg:.3f} deg "
            f"the beam can be placed out to: the beam cube covers {limit_deg:.2f} deg at every parallactic "
            f"angle (less a two-pixel margin at its edge). "
            f"Reduce --field-of-view or the pointing offset, or image without --beam-model."
        )


@pfb_logging.log_exceptions
def imager(
    ms: list[Path],
    output_filename: str,
    scan_names: list[str] | None = None,
    spw_names: list[str] | None = None,
    field_names: list[str] | None = None,
    freq_range: str | None = None,
    overwrite: bool = False,
    rebuild_scratch: bool = False,
    data_column: str = "DATA",
    data_group: str = "base",
    partition_columns: list[str] | None = None,
    weight_column: str | None = None,
    sigma_column: str | None = None,
    flag_column: str = "FLAG",
    gain_table: list[Path] | None = None,
    transfer_model_from: str | Path | None = None,
    integrations_per_image: int = -1,
    channels_per_image: int = -1,
    concat_row: bool = True,
    precision: str = "double",
    bda_decorr: float = 1.0,
    max_field_of_view: float = 3.0,
    beam_model: str | None = None,
    baseline_groups: bool = False,
    antenna_groups: str | None = None,
    phase_dir: str | None = None,
    target: str | None = None,
    chan_average: int = 1,
    progressbar: bool = True,
    log_directory: str | None = None,
    product: str = "I",
    ncpu: int | None = None,
    mem_limit: float | None = None,
    wgt_mode: str = "l2",
    weight_grouping: str = "per-band-time",
    robustness: float | None = None,
    field_of_view: float | None = None,
    super_resolution_factor: float = 2.0,
    cell_size: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    psf_oversize: float = 1.4,
    min_padding: float = 1.7,
    weight_on_image_grid: bool = True,
    filter_counts_level: float = 5.0,
    npix_super: int = 0,
    epsilon: float = 1e-5,
    do_wgridding: bool = True,
    double_accum: bool = True,
    keep_scratch: bool = True,
    psf: bool = True,
    beam: bool = True,
    fits_per_partition: bool = False,
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
    ray_address: str = "local",
    keep_ray_alive: bool = False,  # not used by CLI
):
    """
    Initialise Stokes data products for imaging
    """
    # for logging options
    opts_dict = locals().copy()
    time_start = time.time()
    phases = PhaseClock(log)
    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )
    opts_dict["output_filename"] = output_filename
    opts_dict["fits_output_folder"] = fits_output_folder
    opts_dict["log_directory"] = log_directory

    # before any validation that reads data, so log_exceptions has a file (#348)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/imager_{timestamp}.log"
    pfb_logging.log_to_file(logname)

    # One core budget and one memory budget, shared out afresh by each pass
    # (utils/memplan, D68). Locally they size the cluster; on an existing
    # cluster they are its own CPU and memory resources.
    if ncpu is not None and ncpu < 1:
        log.error_and_raise(f"--ncpu must be at least 1, got {ncpu}", ValueError)
    if mem_limit is not None and mem_limit <= 0:
        log.error_and_raise(f"--mem-limit must be positive, got {mem_limit}", ValueError)
    if ray_address == "local":
        ncpu = ncpu or psutil.cpu_count(logical=True)
        # page cache counts as available: the kernel gives it up
        mem_budget = int(mem_limit * GIB) if mem_limit else int(0.8 * psutil.virtual_memory().available)
    elif mem_limit is not None:
        log.error_and_raise(
            "--mem-limit sizes a local cluster; on an existing one (--ray-address) the cluster's own "
            "memory resource is the budget. Set it with `ray start --memory`, or drop --mem-limit.",
            ValueError,
        )
    else:
        mem_budget = None

    remprod = product.upper().strip("IQUV")
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported", NotImplementedError)

    # africanus' BDA refuses the same range, but only inside pass 1
    if not 0.0 < max_field_of_view <= 90.0:
        log.error_and_raise(f"--max-field-of-view must lie in (0, 90] deg, got {max_field_of_view}", ValueError)
    max_fov_rad = float(np.deg2rad(max_field_of_view))
    if min_padding < 1.0:
        log.error_and_raise(f"--min-padding must be at least 1, got {min_padding}", ValueError)

    # Before init_ray and before pass 1: ducc only refuses an unreachable
    # epsilon when it is handed data, which in the imager is pass 2 -- after the
    # expensive half has run and written the scratch store (#340).
    try:
        check_gridder_epsilon(precision, epsilon, do_wgridding=do_wgridding)
    except ValueError as e:
        log.error_and_raise(str(e), ValueError)

    if (eff := effective_robustness(robustness)) != robustness:
        log.info(f"robustness {robustness} > 2: using natural weighting")
        robustness = eff

    msnames = []
    for ms_path in ms:
        matches = glob_uris(ms_path)
        if not matches:
            log.error_and_raise(f"No MS at {ms_path}", ValueError)
        msnames += matches
    ms = msnames
    opts_dict["ms"] = ms

    antenna_group_override = None
    if baseline_groups:
        antenna_group_override = preflight_baseline_groups(beam_model, antenna_groups)
    elif antenna_groups is not None:
        # Refuse rather than ignore: an ungrouped run with --antenna-groups set
        # is indistinguishable in its output from the grouped run the user asked
        # for, and the spec would not even have parsed the pattern.
        log.error_and_raise(
            "--antenna-groups only has an effect with --baseline-groups, which is not set. "
            "Add --baseline-groups, or drop --antenna-groups.",
            ValueError,
        )

    if gain_table is not None:
        # The MSv4 imager never carried the MSv2 path's gain application across:
        # the option was validated, globbed and logged, and then dropped, so a
        # calibrated-imaging run produced uncalibrated images silently (#333).
        # Refuse until the gains are threaded into `stokes_vis` -- `pfb hci`
        # does apply them, and is the MSv2-era path that still works.
        log.error_and_raise(
            "--gain-table is not implemented for pfb imager: the MSv4 path does not apply "
            "gains, and accepting the option would produce uncalibrated images with no "
            "warning (issue #333). Image a corrected-data column instead, or use pfb hci, "
            "which does apply gains.",
            NotImplementedError,
        )

    transfer_ds = None
    model_bytes = 0
    if transfer_model_from is not None:
        # absolute before any Ray worker (with its own cwd) sees it
        transfer_model_from = str(Path(transfer_model_from).resolve())
        opts_dict["transfer_model_from"] = transfer_model_from
        if not Path(transfer_model_from).exists():
            log.error_and_raise(f"No .mds at {transfer_model_from}", ValueError)
        transfer_ds = open_mds(transfer_model_from)
        try:
            check_stokes_subset(product, model_geometry(transfer_ds)["stokes"])
        except ValueError as e:
            log.error_and_raise(str(e), ValueError)
        # pass 2 applies BEAM to the rendered model, so an apparent model would
        # be attenuated twice; None is a legacy .mds, which deconv wrote intrinsic
        if model_sampling(transfer_ds)["flux_scale"] == "apparent":
            log.error_and_raise(
                f"{transfer_model_from} is an apparent-flux model; --transfer-model-from needs an "
                "intrinsic-flux model, because pass 2 applies the primary beam to it.",
                ValueError,
            )

    log.log_options_dict(opts_dict, title="IMAGER options")

    # the job env carries pass 1's threads; pass 2 overrides them per task
    env_vars = set_envs(P1_THREADS, psutil.cpu_count(logical=False), log=log)

    basename = f"{output_filename}"

    # refused here rather than after pass 1 (#354): a run must not fail halfway
    dt_fs, dt_url = uri_and_fs(f"{basename}.dt")
    if dt_fs.exists(dt_url) and not overwrite:
        log.error_and_raise(f"{basename}.dt exists. Set overwrite to overwrite it.", RuntimeError)

    if freq_range is not None and len(freq_range):
        fmin, fmax = freq_range.strip(" ").split(":")
        if len(fmin) > 0:
            freq_min = float(fmin)
        else:
            freq_min = -np.inf
        if len(fmax) > 0:
            freq_max = float(fmax)
        else:
            freq_max = np.inf
    else:
        freq_min = -np.inf
        freq_max = np.inf

    # Everything pass 1's output depends on (#354). A scratch whose record
    # matches is reused and pass 1 is skipped; the image grid enters through
    # the counts grid, compared below once the record's data extent is known.
    inputs = pass1_inputs(
        ms=ms,
        # selection is by membership, so the order names are given in is irrelevant
        field_names=None if field_names is None else sorted(field_names),
        spw_names=None if spw_names is None else sorted(spw_names),
        scan_names=None if scan_names is None else sorted(scan_names),
        # parsed, so "1.4e9:" and "1400000000:" agree; None for an open end
        freq_range=[None if np.isinf(f) else float(f) for f in (freq_min, freq_max)],
        data_column=data_column,
        data_group=data_group,
        partition_columns=partition_columns,
        weight_column=weight_column,
        sigma_column=sigma_column,
        integrations_per_image=integrations_per_image,
        channels_per_image=channels_per_image,
        product=product,
        precision=precision,
        chan_average=chan_average,
        bda_decorr=bda_decorr,
        max_field_of_view=max_field_of_view,
        beam_model=beam_model,
        baseline_groups=baseline_groups,
        antenna_groups=antenna_groups,
        phase_dir=phase_dir,
        target=target,
        wgt_mode=wgt_mode,
        double_accum=double_accum,
        weight_on_image_grid=weight_on_image_grid,
    )
    geom_kw = dict(
        field_of_view=field_of_view,
        super_resolution_factor=super_resolution_factor,
        cell_size=cell_size,
        nx=nx,
        ny=ny,
        psf_oversize=psf_oversize,
        min_padding=min_padding,
        weight_on_image_grid=weight_on_image_grid,
        max_fov_rad=max_fov_rad,
    )

    # pass-1 fine averaged Stokes pieces are written into a .scratch DataTree
    scratch_fs, scratch_url = uri_and_fs(f"{basename}.scratch")
    reuse = False
    record = None
    if scratch_fs.exists(scratch_url):
        record = read_scratch_record(scratch_url)
        diffs = record_diffs(record, inputs)
        if not diffs:
            d = record["derived"]
            counts = image_geometry(d["max_blength"], d["max_uvw"], d["max_freq"], **geom_kw)["counts"]
            diffs = record_diffs(record, inputs, counts)
        why = "".join(f"\n  {d}" for d in diffs)
        if rebuild_scratch:
            # removed only once every set-up refusal has passed, just before pass 1
            log.info(
                f"Scratch {scratch_url} will be rebuilt (--rebuild-scratch)" + (f", and differs:{why}" if diffs else "")
            )
        elif not diffs:
            reuse = True
            log.info(
                f"Reusing scratch {scratch_url}: its pass-1 inputs match this run, so pass 1 is skipped. "
                f"Run with --rebuild-scratch to run pass 1 again."
            )
            if record.get("pfb-imaging-version") != pfb_version:
                log.warning(
                    f"The scratch was written by pfb-imaging {record.get('pfb-imaging-version')}, this is "
                    f"{pfb_version}. A fix to pass 1 between them is not in it: run with --rebuild-scratch."
                )
        else:
            log.error_and_raise(
                f"{scratch_url} exists but cannot serve this run, because:{why}\n"
                f"Run with --rebuild-scratch to run pass 1 again, or change the options back.",
                RuntimeError,
            )
    # crude column arithmetic
    dc = data_column.replace(" ", "")
    if "+" in dc:
        dc1, dc2 = dc.split("+")
        operator = "+"
    elif "-" in dc:
        dc1, dc2 = dc.split("-")
        operator = "-"
    else:
        dc1 = dc
        dc2 = None
        operator = None

    # The "DATA" sentinel resolves to the data group's correlated_data variable
    # (e.g. "VISIBILITY" for the base group) via the data_groups mechanism rather
    # than a hardcoded name; resolved from the first visibility node below.
    # (sjperkins, PR #252 review.)
    vis_col = None
    # same for the weight column: "WEIGHT_SPECTRUM" is exposed as "WEIGHT" in the datatree
    wgt_col = None

    # ---- the MS metadata, or a reused scratch's record of it ----
    if reuse:
        derived = record["derived"]
        selected = all_freqs = None
    else:
        derived, selected, all_freqs, vis_col, wgt_col = scan_ms(
            ms,
            partition_columns=partition_columns,
            data_group=data_group,
            field_names=field_names,
            spw_names=spw_names,
            scan_names=scan_names,
            freq_min=freq_min,
            freq_max=freq_max,
            check_telescope=bool(baseline_groups and beam_model is not None),
            phase_dir=phase_dir,
            channels_per_image=channels_per_image,
        )
        # map the "DATA" sentinel to the data group's correlated_data variable
        if dc1 == "DATA":
            dc1 = vis_col
        if dc2 == "DATA":
            dc2 = vis_col
        # map the "WEIGHT_SPECTRUM" sentinel to the data group's weight variable
        if weight_column is not None and weight_column == "WEIGHT_SPECTRUM":
            weight_column = wgt_col
    max_blength = float(derived["max_blength"])
    max_uvw = float(derived["max_uvw"])
    max_freq = float(derived["max_freq"])
    min_chan_width = float(derived["min_chan_width"])
    band_centres = np.asarray(derived["band_centres"])
    nband = band_centres.size
    field_centres = np.asarray(derived["field_centres"])
    radec_new = None if derived["radec_new"] is None else np.asarray(derived["radec_new"])
    if radec_new is not None:
        log.info(f"Rephasing to ra={np.rad2deg(radec_new[0]):.8f} deg dec={np.rad2deg(radec_new[1]):.8f} deg")
    log.info(f"Number of output bands determined to be {nband} based on channel width and freq range")
    # tangent point of the output image grid
    grid_radec = radec_new if radec_new is not None else field_centres[0]

    def target_offset(time_out):
        """(l0, m0) of --target w.r.t. the tangent point, at time_out (unix s)."""
        if target is None:
            return 0.0, 0.0
        tmp = target.split(",")
        if len(tmp) == 2:
            tradec = parse_sky_coords(target)
        else:
            # named body known to astropy; get_coordinates expects MJD seconds
            from pfb_imaging.utils.astrometry import get_coordinates

            tradec = np.array(get_coordinates(to_mjd_time(time_out), target=target))
        ell, emm = radec_to_lm(tradec, grid_radec)
        return float(ell), float(emm)

    # shared imaging geometry, and the grid pass 1 grids the counts on
    geom = image_geometry(max_blength, max_uvw, max_freq, **geom_kw)
    nx, ny, nx_psf, ny_psf = geom["nx"], geom["ny"], geom["nx_psf"], geom["ny_psf"]
    cell_rad, nx_pad = geom["cell_rad"], geom["nx_pad"]
    counts_nx, counts_ny, counts_cell = geom["counts"]["nx"], geom["counts"]["ny"], geom["counts"]["cell"]
    log.info(f"Image size (nx={nx}, ny={ny}), cell={np.rad2deg(cell_rad) * 3600:.4e} arcsec")
    if not weight_on_image_grid:
        # < 1: finer than the image's own uv cell, so robust weights lean towards natural
        ratio = (nx_pad * cell_rad) / (counts_nx * counts_cell)
        log.info(
            f"Counts grid {counts_nx}^2 set by --max-field-of-view: its uv cell is {ratio:.2f} times the image's "
            f"(--weight-on-image-grid matches them)"
        )

    # a --target is resolved at the ends and middle of the track: a named body moves
    offsets = {target_offset(t) for t in derived["times"]}
    # before anything is computed: a scratch serves every image within --max-field-of-view
    try:
        image_reach_deg = check_image_extent(nx, ny, cell_rad, offsets, max_fov_rad)
    except ValueError as e:
        log.error_and_raise(str(e), ValueError)
    beam_reach_deg = pointing_reach(field_centres, grid_radec, max_fov_rad)
    log.info(
        f"The image reaches {image_reach_deg:.3f} deg of the {max_field_of_view:.3f} deg --max-field-of-view; "
        f"beams are described out to {beam_reach_deg:.3f} deg from a pointing"
    )

    if transfer_ds is not None:
        try:
            check_transfer_geometry(transfer_ds, grid_radec=grid_radec)
        except ValueError as e:
            log.error_and_raise(str(e), ValueError)
        model_cell = float(transfer_ds.attrs["cell_rad_x"])
        if not np.isclose(cell_rad, model_cell, rtol=1e-9, atol=0.0):
            log.info(
                f"Remapping the transfer model from its {np.rad2deg(model_cell) * 3600:.4e} arcsec cell onto "
                f"{np.rad2deg(cell_rad) * 3600:.4e} arcsec (flux-conserving box kernel)"
            )
        # checked after pass 2, against the frequencies the model is rendered at
        transfer_coverage = transfer_freq_coverage(transfer_ds)
        # each pass-2 worker caches its own copy, so its size enters their estimate
        model_bytes = int(transfer_ds.nbytes)
        # pass-2 workers load their own copy through a Multiton; the driver holds none
        del transfer_ds

    beam_axis = None  # (axis_deg, limit_deg) of a cube-capped wizard axis
    if not reuse:
        # Baseline-group masks, one dict per selected node. Computed here rather
        # than inside the dispatch loop so the set of groups that actually exist is
        # known before any BeamWizard is built -- an eager wizard for an absent
        # group would stage MdV-2026 products for a beam no task consumes.
        # Still strictly per node: different scans/SPWs can carry different antenna
        # subsets, so baseline_id indices are only valid for their own node.
        node_masks_list = []
        for _ims, node, _f, _t, _c, _r in selected:
            if not baseline_groups:
                node_masks_list.append({"all": None})
                continue
            try:
                node_masks_list.append(node_baseline_groups(node, antenna_group_override)[0])
            except ValueError as e:
                log.error_and_raise(str(e), ValueError)
        if baseline_groups:
            present_groups = sorted({g for m in node_masks_list for g in m}, key=GROUP_LABELS.index)
            log.info("Baseline groups present: " + ", ".join(present_groups))
            if beam_model is not None and not isinstance(beam_model, BeamWizard):
                # may repoint MBEAMS_CACHE_DIR, so before the workers inherit it
                missing = resolve_group_beam_cache(str(beam_model).upper(), present_groups)
                if missing:
                    log.error_and_raise(missing, RuntimeError)
                env_vars["MBEAMS_CACHE_DIR"] = os.environ["MBEAMS_CACHE_DIR"]

        # MeerKAT band name -> BeamWizard from the meerkat-beams band cache (the
        # same convention as hci); "katbeam"/None pass through to stokes_vis as is
        beam_refs = None
        beam_lm = None
        wizards = {}
        if beam_model is not None and not isinstance(beam_model, BeamWizard) and beam_model.lower() != "katbeam":
            # Each wizard is built once here, single process and before Ray, so a
            # staging error or a beam that cannot cover the data stops the run
            # before a cluster exists, and concurrent first downloads cannot race.
            # The tasks then get a Multiton recipe rather than the wizard: a worker
            # builds its own once from the warmed cache and keeps it across tasks,
            # and with it the spline prefilters, which a wizard shipped by value
            # arrives without -- ~0.8 s per correlation per task to rebuild (degrid
            # does the same, D38). No image_name: detached mode, pass 1 supplies
            # explicit l/m/times/freq.
            groups = present_groups if baseline_groups else [None]
            if baseline_groups:
                log.info(f"Initialising a BeamWizard for each of {', '.join(present_groups)}")
            else:
                log.info("Assuming MeerKAT data and initialising BeamWizard")
            kwargs = {
                g: {"band": beam_model, "freq_extrapolation": BEAM_FREQ_EXTRAPOLATION}
                | ({} if g is None else {"group": g})
                for g in groups
            }
            wizards = {}
            for g, kw in kwargs.items():
                wizards[g] = BeamWizard(**kw)
                # a piece's beam frequency is a weighted mean of its own channels,
                # so every selected channel bounds what a worker can ask for (D59)
                try:
                    wizards[g].check_freqs(all_freqs)
                except ValueError as e:
                    log.error_and_raise(f"beam {g or beam_model}: {e}", ValueError)
            if baseline_groups:
                beam_refs = {g: Multiton(BeamWizard, **kw) for g, kw in kwargs.items()}
            else:
                beam_model = Multiton(BeamWizard, **kwargs[None])
        elif isinstance(beam_model, BeamWizard):
            wizards = {getattr(beam_model, "group", None): beam_model}
        if wizards:
            beam_lm, axis_deg, limit_deg = field_beam_axis(wizards, beam_reach_deg, max_freq)
            # a reuse checks later images against the same axis
            beam_axis = [float(axis_deg), float(limit_deg)]
            if axis_deg < beam_reach_deg:
                # routine under --baseline-groups (1.81 deg group cubes against the default R)
                log.info(
                    f"The beam cube covers {limit_deg:.2f} deg, less than the {beam_reach_deg:.2f} deg "
                    f"--max-field-of-view and the pointing offsets ask for: images reaching past "
                    f"{axis_deg:.2f} deg from a pointing are refused"
                )
            try:
                check_beam_reach(field_centres, grid_radec, offsets, nx, ny, cell_rad, axis_deg, limit_deg)
            except ValueError as e:
                log.error_and_raise(str(e), ValueError)
            log.info(
                f"Beam grid {beam_lm.size}^2 at {(beam_lm[1] - beam_lm[0]) * 3600:.1f} arcsec, "
                f"out to {axis_deg:.2f} deg (the beam cube allows {limit_deg:.2f} deg)"
            )
            del wizards
        elif isinstance(beam_model, str) and beam_model.lower() == "katbeam":
            # analytic, so no cube limits it (D66)
            beam_lm = beam_grid(max_freq, BEAM_DISH_DIAMETER[None], beam_reach_deg)
            log.info(f"katbeam grid {beam_lm.size}^2 at {(beam_lm[1] - beam_lm[0]) * 3600:.1f} arcsec")
    elif record.get("beam_axis") is not None:
        # the reused scratch's beams reach only as far as its axis did
        try:
            check_beam_reach(field_centres, grid_radec, offsets, nx, ny, cell_rad, *record["beam_axis"])
        except ValueError as e:
            log.error_and_raise(str(e), ValueError)

    # only now: every refusal above, the transfer geometry included, needs the
    # MS metadata but no cluster
    init_ray(
        ncpu,
        ray_address=ray_address,
        runtime_env={
            "env_vars": {**env_vars, **profile_env()},
            "worker_process_setup_hook": setup_ray_worker,
        },
        log=log,
        memory=mem_budget,
    )
    # What the cluster has free is the budget: an existing cluster (or a test
    # session's) has its own size, and other work may hold some of it -- deconv's
    # band actors claim nominal CPUs (D8). A claim beyond what is free waits forever.
    free = ray.available_resources()
    free_cpus = max(1, int(free.get("CPU", 1)))
    ncpu = min(ncpu or free_cpus, free_cpus)
    mem_budget = min(mem_budget or int(free["memory"]), int(free["memory"]))
    node_cpus = int(max(n["Resources"].get("CPU", 0) for n in ray.nodes() if n["Alive"]))
    resize_thread_pool(ncpu)
    log.info(f"Core budget {ncpu} CPUs (largest node {node_cpus}), memory budget {mem_budget / GIB:.1f} GiB")
    # pass 2's per-image terms, fixed before pass 1 writes any data
    itemsize = 4 if precision.lower() == "single" else 8
    p2_geometry = dict(
        npix=nx * ny,
        npsf=nx_psf * ny_psf if psf else 0,
        ncorr=len(set(product.upper())),
        itemsize=itemsize,
        counts_cells=counts_nx * counts_ny if robustness is not None else 0,
        model_bytes=model_bytes,
    )
    p2_floor = pass2_image_bytes([0], **p2_geometry)
    if p2_floor > mem_budget:
        # refused before pass 1, which would otherwise run only for pass 2 to fail (#340)
        log.error_and_raise(
            f"One pass-2 image needs an estimated {p2_floor / GIB:.1f} GiB before any of its data, "
            f"more than the memory budget of {mem_budget / GIB:.1f} GiB. Raise --mem-limit, or make the "
            f"image (--field-of-view, --super-resolution-factor) or the PSF (--psf-oversize) smaller.",
            ValueError,
        )

    phases.lap("Setup")
    if reuse:
        # JSON keys are strings
        band_extent = {int(b): ext for b, ext in record["band_extent"].items()}
        # the previous run's applied counts groups; this run writes its own
        if scratch_fs.exists(f"{scratch_url}/counts"):
            scratch_fs.rm(f"{scratch_url}/counts", recursive=True)
        zarr.consolidate_metadata(scratch_url)
    else:
        # Only now, with every set-up refusal behind it: a refused run must
        # neither delete a scratch nor leave an empty one behind (#354)
        if scratch_fs.exists(scratch_url):
            scratch_fs.rm(scratch_url, recursive=True)
        scratch_fs.makedirs(scratch_url, exist_ok=True)
        log.info(f"Pass-1 scratch products will be stored in {scratch_url}")
        tasks = []
        # (kwargs, estimated peak) per pass-1 task: planned once all are known
        p1_tasks = []
        accum_itemsize = 8 if double_accum or precision.lower() == "double" else 4
        scan_block_to_tid = {}  # (scan_name, block_idx) -> tid
        next_tid = 0
        # Pre-create the band/time parent groups single-threaded so concurrent
        # pass-1 workers only ever create their own distinct leaf piece group. This
        # avoids a check-then-create race (ContainsGroupError) on the shared parent;
        # combined with consolidated=False on the worker writes (consolidation is
        # done once below, in the driver), it removes all shared-mutable-state races
        # on the scratch store. See utils/stokes2vis.stokes_vis.
        scratch_root = zarr.open_group(scratch_url, mode="a")
        created_parents = set()
        # bandid -> [lo, hi]: outer channel edges of every block assigned to the band
        band_extent = {}
        for (ims, node, freqs_node, times_node, chan0, _field_radec), node_masks in zip(selected, node_masks_list):
            scan_name = np.unique(node.ds.scan_name.load().values).item()
            nchan_node = freqs_node.size
            ntimes_node = times_node.size
            if integrations_per_image in (0, None, -1):
                ipi_node = ntimes_node
            else:
                ipi_node = integrations_per_image
            if channels_per_image in (0, None, -1):
                cpi_node = nchan_node
            else:
                cpi_node = channels_per_image
            for tlow in range(0, ntimes_node, ipi_node):
                thigh = min(tlow + ipi_node, ntimes_node)
                t_index = slice(tlow, thigh)
                key = (scan_name, tlow // ipi_node)
                if key not in scan_block_to_tid:
                    scan_block_to_tid[key] = next_tid
                    next_tid += 1
                timeid = scan_block_to_tid[key]
                for flow in range(0, nchan_node, cpi_node):
                    fhigh = min(flow + cpi_node, nchan_node)
                    # flow/fhigh index the freq-range-trimmed axis; isel below acts
                    # on the unsliced node, so shift by the selection offset chan0
                    nu_index = slice(chan0 + flow, chan0 + fhigh)
                    bandid = int(np.argmin(np.abs(band_centres - freqs_node[flow:fhigh].mean())))
                    lo = float(freqs_node[flow:fhigh].min()) - min_chan_width / 2
                    hi = float(freqs_node[flow:fhigh].max()) + min_chan_width / 2
                    extent = band_extent.setdefault(bandid, [lo, hi])
                    extent[:] = [min(extent[0], lo), max(extent[1], hi)]

                    # slice out subset of node
                    subdt = node.isel(time=t_index, frequency=nu_index)

                    # ensure the shared parent group exists before any worker
                    # writes a leaf under it (single-threaded → no creation race)
                    parent = f"band{bandid:04d}_time{timeid:04d}"
                    if parent not in created_parents:
                        scratch_root.require_group(parent)
                        created_parents.add(parent)

                    for bg, bl_idx in node_masks.items():
                        subdt_bg = subdt if bl_idx is None else subdt.isel(baseline_id=bl_idx)
                        # lazy: nbytes comes from shapes and dtypes, nothing is read
                        p1_vars = needed_variables(subdt_bg.ds, dc1, dc2, sigma_column, weight_column)
                        p1_bytes = pass1_task_bytes(
                            int(subdt_bg.ds[p1_vars].nbytes), counts_nx * counts_ny, accum_itemsize
                        )
                        p1_tasks.append(
                            (
                                dict(
                                    dc1=dc1,
                                    dc2=dc2,
                                    operator=operator,
                                    node_dt=subdt_bg,
                                    scratch_store=scratch_url,
                                    bandid=bandid,
                                    timeid=timeid,
                                    msid=ims,
                                    freq_nominal=band_centres[bandid],
                                    precision=precision,
                                    sigma_column=sigma_column,
                                    weight_column=weight_column,
                                    product=product,
                                    chan_average=chan_average,
                                    bda_decorr=bda_decorr,
                                    max_field_of_view=max_field_of_view,
                                    beam_model=beam_model if beam_refs is None else beam_refs[bg],
                                    wgt_mode=wgt_mode,
                                    max_blength=max_blength,
                                    max_freq=max_freq,
                                    counts_nx=counts_nx,
                                    counts_ny=counts_ny,
                                    counts_cell=counts_cell,
                                    baseline_group=bg,
                                    data_group=data_group,
                                    radec_new=radec_new,
                                    target=target,
                                    beam_lm=beam_lm,
                                    double_accum=double_accum,
                                ),
                                p1_bytes,
                            )
                        )

        if p1_tasks:
            try:
                p1_plan = plan_pass(
                    len(p1_tasks), ncpu, node_cpus, [b for _, b in p1_tasks], mem_budget, nthreads=P1_THREADS
                )
            except ValueError as e:
                log.error_and_raise(f"Pass 1: {e}. Raise --mem-limit.", ValueError)
            log.info(
                f"Pass 1: {len(p1_tasks)} tasks, up to {p1_plan.nconcurrent} at once with {p1_plan.nthreads} "
                f"threads each (largest estimate {max(b for _, b in p1_tasks) / GIB:.1f} GiB)"
            )
            for kwargs, p1_bytes in p1_tasks:
                tasks.append(
                    safe_stokes_vis.options(num_cpus=p1_plan.task_cpus, memory=p1_bytes).remote(
                        **kwargs, nthreads=p1_plan.nthreads
                    )
                )
        del p1_tasks

        # submission pickles every task's datatree node, so it is timed on its own
        phases.lap("Pass 1 submission")
        nds = len(tasks)
        ncomplete = 0
        remaining_tasks = tasks.copy()
        bandids_out = []
        timeids_out = []
        while remaining_tasks:
            # Wait for at least 1 task to complete
            ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

            # Process the completed task
            for task in ready:
                result, mem = ray.get(task)
                if result is not None:
                    bandids_out.append(result[0])
                    timeids_out.append(result[1])
                ncomplete += 1
                if progressbar:
                    # post-gc rss ratcheting up for a pid across tasks indicates
                    # retention below Python (C-level caches/arenas); peak is the
                    # worker's lifetime high-water mark
                    log.info(f"Completed: {ncomplete} / {nds} [{format_memory(mem)}]", extra=pfb_logging.NO_MARKUP)

        ntime = len(set(timeids_out))
        nband_out = len(set(bandids_out))

        log.info(f"Pass 1 wrote fine pieces for {nband_out} bands and {ntime} time chunks to {scratch_url}")
        phases.lap("Pass 1")

        # consolidate the scratch metadata once, single-threaded, now that all
        # workers have finished (workers wrote with consolidated=False to avoid
        # racing on the shared root .zmetadata; see stokes_vis)
        zarr.consolidate_metadata(scratch_url)
        # written last, so a scratch whose pass 1 did not finish has no record
        zarr.open_group(scratch_url, mode="a").attrs["pass1"] = json.dumps(
            {
                "schema": SCRATCH_SCHEMA,
                "inputs": inputs,
                "derived": derived,
                "counts": geom["counts"],
                "band_extent": {str(b): ext for b, ext in band_extent.items()},
                "beam_axis": beam_axis,
                "pfb-imaging-version": pfb_version,
            }
        )
        zarr.consolidate_metadata(scratch_url)

    # ---- between passes: stream per-piece counts into the applied grouping ----
    # effective grouping first: time-resolved groupings contradict a
    # time-collapsed image, so concat_row maps each to its band-collapsed
    # analogue (per-band-time -> per-band, per-time -> mfs)
    if concat_row:
        grouping_eff = {"per-band-time": "per-band", "per-time": "mfs"}.get(weight_grouping, weight_grouping)
        if grouping_eff != weight_grouping and robustness is not None:
            log.warning(
                f"concat_row collapses the time axis; using weight_grouping "
                f"'{grouping_eff}' instead of '{weight_grouping}'"
            )
    else:
        grouping_eff = weight_grouping

    def counts_key(bandid, timeid):
        """Applied-weighting group of an output image (concat_row collapses time)."""
        tid = 0 if concat_row else timeid
        if grouping_eff == "per-band-time":
            return (bandid, tid)
        if grouping_eff == "per-band":
            return (bandid,)
        if grouping_eff == "per-time":
            return (tid,)
        return ()  # mfs

    scratch_dt = xr.open_datatree(scratch_url, engine="zarr", chunks=None)
    # per-scratch-node summary: (bandid, timeid, ra, dec, time_out)
    node_info = {}
    # applied-weighting group -> its pieces' node paths; the counts themselves
    # are streamed by write_group_counts below, never held per piece
    counts_sources = {}
    # piece node path -> (wsum_nat, freq_out), for the effective frequency
    # computed below before pass 2; scratch node -> its piece paths
    piece_info = {}
    piece_mem = {}
    node_pieces = {}
    for name in scratch_dt.children:
        if not name.startswith("band"):
            continue
        children = scratch_dt[name].children
        pieces = [child.ds for child in children.values()]
        if not pieces:
            continue
        bandid = int(pieces[0].attrs["bandid"])
        timeid = int(pieces[0].attrs["timeid"])
        # natural weighting (robustness None) never touches counts
        if robustness is not None:
            counts_sources.setdefault(counts_key(bandid, timeid), []).extend(f"{name}/{c}" for c in children)
        for c, ds in zip(children, pieces):
            a = ds.attrs
            piece_info[f"{name}/{c}"] = (float(a["wsum_nat"]), float(a["freq_out"]))
            # partition and in-memory size (as _load_piece loads it), for pass 2's estimate
            piece_mem[f"{name}/{c}"] = (
                (a["msid"], a["field_name"], a["spw_name"], a["baseline_group"]),
                int(ds.drop_vars(["COUNTS_INDEX", "COUNTS_VALUE"], errors="ignore").nbytes),
            )
            node_pieces.setdefault(name, []).append(f"{name}/{c}")
        node_info[name] = {
            "bandid": bandid,
            "timeid": timeid,
            "ra": float(pieces[0].attrs["ra"]),
            "dec": float(pieces[0].attrs["dec"]),
            "time_out": float(np.mean([ds.attrs["time_out"] for ds in pieces])),
            "time_min": float(min(ds.attrs["time_min"] for ds in pieces)),
            "time_max": float(max(ds.attrs["time_max"] for ds in pieces)),
            "time_lo": float(min(ds.attrs["time_lo"] for ds in pieces)),
            "time_hi": float(max(ds.attrs["time_hi"] for ds in pieces)),
        }
    if not node_info:
        log.error_and_raise("Pass 1 produced no output images (all data flagged?)", RuntimeError)

    # build the pass-2 work list: (out_name, src_names, meta)
    work = []
    if concat_row:
        by_band = {}
        for name, info in node_info.items():
            by_band.setdefault(info["bandid"], []).append((name, info))
        for bandid, items in sorted(by_band.items()):
            src_names = [name for name, _ in items]
            infos = [info for _, info in items]
            out_name = f"band{bandid:04d}_time0000"
            # ra/dec: the common tangent point (all nodes agree after phase-centre
            # resolution above); time_out is the mean of the collapsed nodes' times
            time_out = float(np.mean([info["time_out"] for info in infos]))
            l0, m0 = target_offset(time_out)
            work.append(
                (
                    out_name,
                    src_names,
                    {
                        "bandid": bandid,
                        "timeid": 0,
                        "ra": float(grid_radec[0]),
                        "dec": float(grid_radec[1]),
                        "l0": l0,
                        "m0": m0,
                        "time_out": time_out,
                        **_sampling_bounds(infos, *band_extent[bandid]),
                    },
                )
            )
    else:
        for name, info in node_info.items():
            l0, m0 = target_offset(info["time_out"])
            meta = {k: info[k] for k in ("bandid", "timeid", "time_out")}
            meta.update(ra=float(grid_radec[0]), dec=float(grid_radec[1]), l0=l0, m0=m0)
            meta.update(_sampling_bounds([info], *band_extent[info["bandid"]]))
            work.append((name, [name], meta))

    ntime = len({meta["timeid"] for _, _, meta in work})

    # one applied counts grid per weighting group, streamed into the scratch
    # store; pass-2 workers read their group's grid from there (#339)
    counts_groups = {}
    # natural weighting: the imaging weights are the natural ones
    wsum_img = {path: info[0] for path, info in piece_info.items()}
    if robustness is not None:
        counts_groups, wsum_img = write_group_counts(
            scratch_url, counts_sources, robustness, filter_level=filter_counts_level, npix_super=npix_super
        )
        # the counts groups were written after pass 1's consolidation
        zarr.consolidate_metadata(scratch_url)
    log.info(f"Applied uv counts grouping '{grouping_eff}' over {len(counts_groups)} group(s)")

    # every image's effective frequency, before pass 2 (wiki D28): the model is
    # rendered at it, so pass 2 can form residuals while the vis are in memory
    for _, src_names, meta in work:
        pieces = {p: wsum_img[p] for sn in src_names for p in node_pieces[sn]}
        meta["wsum_img"] = pieces
        meta["freq_out"] = effective_frequency(
            [(w, piece_info[p][1]) for p, w in pieces.items()], band_centres[meta["bandid"]]
        )
    if transfer_model_from is not None:
        msg = transfer_coverage_warning([meta["freq_out"] for _, _, meta in work], transfer_coverage)
        if msg is not None:
            log.warning(msg)

    # ---- initialise the .dt store root ----
    dt_fs, dt_url = uri_and_fs(f"{basename}.dt")
    if dt_fs.exists(dt_url):
        if overwrite:
            dt_fs.rm(dt_url, recursive=True)
        else:
            log.error_and_raise(f"{basename}.dt exists. Set overwrite to overwrite it.", RuntimeError)
    dt_fs.makedirs(dt_url, exist_ok=True)
    root_attrs = {
        "pfb-imaging-version": pfb_version,
        "product": product,
        "nband": int(nband),
        "ntime": int(ntime),
        "nx": int(nx),
        "ny": int(ny),
        "nx_psf": int(nx_psf),
        "ny_psf": int(ny_psf),
        "cell_rad": float(cell_rad),
        "max_blength": float(max_blength),
        "max_freq": float(max_freq),
        # the uv grid the imaging weights were computed on (D66)
        "counts_nx": int(counts_nx),
        "counts_ny": int(counts_ny),
        "counts_cell": float(counts_cell),
    }
    xr.Dataset(attrs=root_attrs).to_zarr(dt_url, mode="w")
    log.info(f"Imaging products will be written to {dt_url}")

    # per-partition sanity FITS (field/beam orientation checks); the driver
    # creates the directory single-threaded so workers never race on mkdir
    part_fits_dir = None
    if fits_per_partition:
        part_fits_dir = f"{fits_output_folder}/{oname}_partitions"
        os.makedirs(part_fits_dir, exist_ok=True)
        log.info(f"Per-partition FITS will be written to {part_fits_dir}")

    # ---- pass 2: grid each output image into the .dt tree (parallel over images) ----
    # per-worker cached model: one load per process, not per task (degrid's pattern)
    transfer = Multiton(_load_transfer_model, transfer_model_from) if transfer_model_from else None
    phases.lap("Counts and .dt set-up")
    p2_bytes = []
    for _, src_names, _ in work:
        part_bytes = {}
        for sn in src_names:
            for path in node_pieces[sn]:
                key, nbytes = piece_mem[path]
                part_bytes[key] = part_bytes.get(key, 0) + nbytes
        p2_bytes.append(pass2_image_bytes(list(part_bytes.values()), **p2_geometry))
    del piece_mem
    try:
        p2_plan = plan_pass(len(work), ncpu, node_cpus, p2_bytes, mem_budget, max_concurrent=nband)
    except ValueError as e:
        log.error_and_raise(f"Pass 2: {e}. Raise --mem-limit, or image fewer channels per image.", ValueError)
    log.info(
        f"Pass 2: {len(work)} images, up to {p2_plan.nconcurrent} at once with {p2_plan.nthreads} threads "
        f"each (largest estimate {max(p2_bytes) / GIB:.1f} GiB)"
    )
    # Its own runtime_env: Ray starts fresh workers for pass 2 instead of reusing
    # pass 1's, whose retained memory they would carry, and reaps the idle pass-1
    # ones. The job's env and setup hook still apply; the thread counts are pass 2's.
    p2_env = {"env_vars": thread_env(p2_plan.nthreads)}
    tasks = []
    for (out_name, src_names, meta), image_bytes in zip(work, p2_bytes):
        fut = _grid_image.options(num_cpus=p2_plan.task_cpus, memory=image_bytes, runtime_env=p2_env).remote(
            scratch_url,
            dt_url,
            src_names,
            out_name,
            # None under natural weighting (robustness None); grid_partition
            # ignores counts in that case
            counts_groups.get(counts_key(meta["bandid"], meta["timeid"])),
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            band_centres[meta["bandid"]],
            meta,
            robustness=robustness,
            counts_nx=counts_nx,
            counts_ny=counts_ny,
            counts_cell=counts_cell,
            nthreads=p2_plan.nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            do_psf=psf,
            part_fits_dir=part_fits_dir,
            part_fits_beam=beam,
            transfer_model=transfer,
            transfer_path=transfer_model_from,
        )
        tasks.append(fut)

    phases.lap("Pass 2 submission")
    nds = len(tasks)
    ncomplete = 0
    remaining_tasks = tasks.copy()
    beam_imre_max = 0.0
    while remaining_tasks:
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        for task in ready:
            res = ray.get(task)
            beam_imre_max = max(beam_imre_max, float(res.get("beam_imre_ratio", 0.0)))
            ncomplete += 1
            if progressbar:
                mem = res["mem"]
                log.info(f"Gridded: {ncomplete} / {nds} [{format_memory(mem)}]", extra=pfb_logging.NO_MARKUP)

    if beam_imre_max > 0:
        # The MPM cross-group beam is complex and we store Re(B) (wiki D46).
        # This is the evidence that the approximation holds; it is also on every
        # partition as the beam_imre_ratio attr.
        log.info(
            f"Cross-group beam max|Im|/max|Re| = {beam_imre_max:.3e} "
            f"(stored BEAM is Re(B); see wiki design-decisions D46)"
        )

    # consolidate the .dt metadata once, single-threaded, now that all pass-2
    # workers have finished (they wrote with consolidated=False; see _grid_image)
    zarr.consolidate_metadata(dt_url)
    phases.lap("Pass 2")

    # MFS beam parameters per time chunk, from the wsum-normalised sum of the band
    # PSFs. Read back from the .dt one band at a time rather than returned by the
    # pass-2 tasks, which put an image per task in the object store (#339);
    # summed at the precision the workers gridded in (see _grid_image).
    psf_mfs = {}
    wsum_mfs = {}
    if psf:
        dt_out = xr.open_datatree(dt_url, engine="zarr", chunks=None, cache=False)
        try:
            for out_name, _, meta in work:
                node = dt_out[out_name].ds
                tid = meta["timeid"]
                band_psf = node.PSF.values
                if tid in psf_mfs:
                    psf_mfs[tid] += band_psf
                    wsum_mfs[tid] += node.WSUM.values
                else:
                    psf_mfs[tid] = band_psf
                    wsum_mfs[tid] = node.WSUM.values
                del band_psf, node
        finally:
            dt_out.close()
    psfparsn = {}
    for tid in psf_mfs:
        psfparsn[tid] = np.array(fitcleanbeam(psf_mfs[tid] / wsum_mfs[tid][:, None, None], yx_order=True))
    del psf_mfs
    if psf:
        phases.lap("MFS PSF fit")

    # ---- FITS ----
    if fits_mfs or fits_cubes:
        fits_oname = f"{fits_output_folder}/{oname}"
        log.info(f"Writing fits files to {fits_oname}")
        base_kwargs = dict(
            norm_wsum=True,
            nthreads=p2_plan.nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfparsn if psf else None,
        )
        columns = {"DIRTY": {}}
        if transfer_model_from is not None:
            # the pass-2 residual of the transferred model (#309), Jy/beam like deconv's
            columns["RESIDUAL"] = {}
        if psf:
            columns["PSF"] = {}
        if beam:
            # stored beam is the effective response B/n (wiki D22)
            columns["BEAM"] = dict(
                norm_wsum=False,
                force_unit="",
                psfpars_mfs=None,
                extra_hdr={"BEAMINCN": (True, "beam includes the wgridder n-term (D22)")},
            )
        fits_tasks = [
            rdt2fits.remote(dt_url, column, fits_oname, **{**base_kwargs, **overrides})
            for column, overrides in columns.items()
        ]
        for task in fits_tasks:
            ray.get(task)
        phases.lap("FITS")

    if not keep_scratch:
        log.info(f"Removing scratch store {scratch_url}")
        scratch_fs.rm(scratch_url, recursive=True)
        phases.lap("Scratch removal")

    phases.summary()
    log.info(f"All done after {time.time() - time_start}s")

    dump_timeline(log)
    if not keep_ray_alive:
        ray.shutdown()

    return
