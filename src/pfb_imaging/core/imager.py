import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np
import psutil
import ray
import xarray as xr
import zarr
from ducc0.misc import resize_thread_pool
from meerkat_beams.utils import BeamWizard
from xarray_ms.errors import (
    ColumnShapeImputationWarning,
    FrameConversionWarning,
    IrregularGridWarning,
    MissingMetadataWarning,
)

from pfb_imaging import init_ray, pfb_version, set_envs, setup_ray_worker
from pfb_imaging.operators.gridder import grid_partition
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.baselines import (
    GROUP_LABELS,
    baseline_group_masks,
    check_telescope_is_meerkat,
    classify_antennas,
    parse_antenna_groups,
)
from pfb_imaging.utils.fits import rdt2fits, save_fits, set_wcs
from pfb_imaging.utils.memprof import format_memory, memray_env, memray_task, task_memory
from pfb_imaging.utils.misc import (
    check_gridder_epsilon,
    fitcleanbeam,
    parse_sky_coords,
    radec_barycentre,
    radec_to_lm,
    set_image_size,
    to_mjd_time,
)
from pfb_imaging.utils.msv4 import get_engine, load_detached, select_vis_nodes
from pfb_imaging.utils.naming import glob_uris, set_output_names, uri_and_fs
from pfb_imaging.utils.stokes2vis_msv4 import safe_stokes_vis
from pfb_imaging.utils.weighting import write_group_counts

warnings.filterwarnings("ignore", category=IrregularGridWarning)
warnings.filterwarnings("ignore", category=MissingMetadataWarning)
warnings.filterwarnings("ignore", category=FrameConversionWarning)
warnings.filterwarnings("ignore", category=ColumnShapeImputationWarning)


log = pfb_logging.get_logger("IMAGER")


def _partition_fits(
    fits_dir, out_name, pid, field_name, baseline_group, prod, meta, freq_out, cell_rad, do_psf, do_beam
):
    """Write one partition's sanity-check FITS (dirty [+psf] [+beam]).

    Runs inside the pass-2 worker while the partition products are in memory;
    per-partition DIRTY/PSF are wsum-normalised, the beam is written as stored
    (effective response B/n, wiki D22).
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
    if do_psf:
        with np.errstate(invalid="ignore", divide="ignore"):
            psf = np.where(wsum > 0, prod["PSF"] / wsum, 0.0)
        ny_psf_, nx_psf_ = prod["PSF"].shape[1:]
        save_fits(psf, stem.format(var="psf"), mk_hdr(nx_psf_, ny_psf_, "Jy/beam"), yx_order=True)
    if do_beam:
        hdr = mk_hdr(nx_im, ny_im, "")
        hdr["BEAMINCN"] = (True, "beam includes the wgridder n-term (D22)")
        save_fits(prod["BEAM"], stem.format(var="beam"), hdr, yx_order=True)


def _load_piece(node):
    """Load one scratch piece into memory, minus its COUNTS.

    COUNTS is only consumed by the driver's weight reduction, which has
    already happened, and it is by far the largest piece variable. Loaded
    detached from the open scratch tree (see ``load_detached``), or every piece
    would stay reachable from it -- through its concat, past ``del plist`` and
    into the worker's next task.

    Args:
        node: the piece's node in the scratch DataTree.

    Returns:
        The piece as an in-memory Dataset.
    """
    return load_detached(node.ds.drop_vars("COUNTS", errors="ignore"))


def _concat_pieces(plist):
    """Reduce a partition's scratch pieces to a single Dataset.

    Rows are concatenated; BEAM and the effective frequency are combined as
    ``wsum_nat``-weighted means. Pieces of a partition can differ in both --
    BeamWizard evaluates the beam at the piece's own timestamps, and post-#296
    at its own effective frequency -- so taking piece 0's beam, as this used
    to, silently discarded every other piece (wiki D28).

    The weights are the *natural* wsum because this runs before gridding, where
    the robust imaging weights do not yet exist. Getting them would mean
    passing row offsets into grid_partition and holding every piece's BEAM
    resident through gridding instead of freeing it at the caller's ``del
    plist`` -- ~67 MB per piece at 4096 squared float32 -- against the memory
    discipline in docs/wiki/memory-and-ray.md, for a correction to a beam that
    varies slowly with frequency. The band-level reduction in _grid_image does
    use the imaging weights; see wiki D28 for why the mixed basis is correct.

    Args:
        plist: scratch-piece Datasets of one partition, all sharing a FREQ axis.

    Returns:
        The merged partition Dataset, carrying the weighted ``freq_out`` and
        the summed ``wsum_nat`` in its attrs. A single-piece list is returned
        unchanged.
    """
    if len(plist) == 1:
        return plist[0]

    # rows are only concatenatable when they share the freq axis (same
    # spw + band channel-chunk); guard the invariant before concat
    f0 = plist[0].FREQ.values
    for p in plist[1:]:
        assert np.array_equal(p.FREQ.values, f0), "concat group has mismatched FREQ; cannot concatenate rows"

    w = np.array([float(p.attrs["wsum_nat"]) for p in plist], dtype=np.float64)
    wsum_nat = float(w.sum())
    if wsum_nat > 0:
        freqs = np.array([float(p.attrs["freq_out"]) for p in plist], dtype=np.float64)
        freq_out = float((w * freqs).sum() / wsum_nat)
        # accumulate rather than stack: one buffer, not npiece of them
        beam = np.zeros(plist[0].BEAM.shape, dtype=np.float64)
        for wi, p in zip(w, plist):
            beam += wi * p.BEAM.values
        beam /= wsum_nat
    else:
        freq_out = float(plist[0].attrs["freq_out"])
        beam = plist[0].BEAM.values

    rowvars = ["VIS", "WEIGHT", "MASK", "UVW"]
    cat = xr.concat([p[rowvars] for p in plist], dim="row", coords="minimal", compat="override")
    part = plist[0].drop_vars(rowvars)
    for v in rowvars:
        part[v] = cat[v]
    del cat
    part["BEAM"] = (part.BEAM.dims, beam.astype(part.BEAM.dtype))
    part.attrs["freq_out"] = freq_out
    part.attrs["wsum_nat"] = wsum_nat
    # Re(B) diagnostic (wiki D46): report the WORST piece, not piece 0's, which
    # is what `part = plist[0].drop_vars(...)` would otherwise leave here.
    part.attrs["beam_imre_ratio"] = max(float(p.attrs.get("beam_imre_ratio", 0.0)) for p in plist)
    return part


@ray.remote
def _grid_image(*args, **kwargs):
    """Ray entry point for :func:`_grid_image_body` (opt-in memray tracking)."""
    with memray_task("grid_image"):
        return _grid_image_body(*args, **kwargs)


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
    nx_pad=None,
    ny_pad=None,
    nthreads=1,
    epsilon=1e-5,
    do_wgridding=True,
    double_accum=True,
    do_psf=True,
    part_fits_dir=None,
    part_fits_beam=True,
):
    """Pass-2 worker for one output image.

    Groups the image's fine scratch pieces into partitions ``(msid, field, spw,
    baseline_group)``, concatenates scans along ``row``, grids each partition with
    :func:`grid_partition`, sums the image-space products into the band node, and
    writes the band node + ``part####`` children into the ``.dt`` store. Returns a
    light summary (``timeid``/``wsum``/telemetry); the band PSF goes to the ``.dt``
    only -- returning it would put an image per task in the Ray object store.
    """
    resize_thread_pool(nthreads)
    # cache=False: otherwise the counts read below is memoised on the tree,
    # which outlives the task in a reference cycle until the next gc (#339).
    # Pieces are detached from the tree by _load_piece for the same reason.
    dt = xr.open_datatree(scratch_store, engine="zarr", chunks=None, cache=False)
    groups = {}
    for sn in src_names:
        for _, child in dt[sn].children.items():
            ds = _load_piece(child)
            key = (ds.attrs["msid"], ds.attrs["field_name"], ds.attrs["spw_name"], ds.attrs["baseline_group"])
            groups.setdefault(key, []).append(ds)

    # the applied (filtered, box-summed) group counts, written once by the driver
    # (write_group_counts); grid_partition copies before use
    counts = dt[counts_group].ds.COUNTS.values if robustness is not None else None

    first = next(iter(groups.values()))[0]
    corr = first.corr.values
    ncorr = corr.size
    bpar = ["BMAJ", "BMIN", "BPA"]
    # The whole tree carries the requested --precision: pass 1 stamped it on the
    # scratch pieces, grid_partition's ducc buffers follow the vis dtype (ducc
    # templates every array on one type), and the accumulation below must not
    # upcast it back to f8 (--double-accum is a wgridder-internal control only).
    real_type = first.VIS.values.real.dtype
    dirty_sum = np.zeros((ncorr, ny, nx), dtype=real_type)
    psf_sum = np.zeros((ncorr, ny_psf, nx_psf), dtype=real_type) if do_psf else None
    beam_sum = np.zeros((ncorr, ny, nx), dtype=real_type)
    bdirty_sum = np.zeros((ncorr, ny, nx), dtype=real_type)
    wsum_sum = np.zeros(ncorr, dtype=real_type)
    # worst Re(B) approximation error over this image's partitions (wiki D46)
    beam_imre_max = 0.0
    # float64 regardless of --precision: the tree may be single-precision
    # (wiki D27) and float32 resolves 1e9 Hz only to ~64 Hz
    freq_sum = np.zeros(ncorr, dtype=np.float64)

    for pid, key in enumerate(list(sorted(groups))):
        plist = groups.pop(key)
        part = _concat_pieces(plist)
        # release the pre-concat originals before gridding (concat copied them)
        del plist

        prod = grid_partition(
            part,
            counts,
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            robustness=robustness,
            nx_pad=nx_pad,
            ny_pad=ny_pad,
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
        )

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
                "beam_includes_n": bool(part.attrs.get("beam_includes_n", False)),
                # max|Im|/max|Re| of the evaluated beam; non-zero only for the
                # MPM cross group, where the stored BEAM is Re(B) (wiki D46)
                "beam_imre_ratio": float(part.attrs.get("beam_imre_ratio", 0.0)),
                "wsum": prod["WSUM"].tolist(),
            },
        )
        # consolidated=False: each pass-2 worker owns a distinct image_name node,
        # but they share the store root; the driver consolidates once at the end
        part_out.to_zarr(dt_store, group=f"{out_name}/part{pid:04d}", mode="a", consolidated=False)
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
        if do_psf:
            psf_sum += prod["PSF"]
        wsum_sum += prod["WSUM"]
        freq_sum += prod["WSUM"].astype(np.float64) * float(part.attrs["freq_out"])

    # Level-2 reduction: the band product is the wsum-weighted sum of its
    # partitions, so its effective frequency is the wsum-weighted mean of the
    # partition frequencies -- the identical reduction beam_sum uses below.
    # That is why these weights must be the imaging prod["WSUM"] and not the
    # natural wsum_nat used inside _concat_pieces: they are what actually
    # determines each partition's contribution to the summed image, and BEAM
    # and freq_out must not be weighted by different quantities at the same
    # level (wiki D28). Reduced with corr-summed weights and stored scalar, as
    # dt2fits already does for freq_mfs.
    wsum_tot = float(wsum_sum.astype(np.float64).sum())
    freq_eff = float(freq_sum.sum() / wsum_tot) if wsum_tot > 0 else float(freq_nominal)
    band_attrs = {
        "bandid": meta["bandid"],
        "timeid": meta["timeid"],
        "freq_out": freq_eff,
        "freq_nominal": float(freq_nominal),
        "time_out": meta["time_out"],
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
    # RESIDUAL is only computed once a model input exists (deconv falls back
    # to DIRTY when it is absent).
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
    band_ds = xr.Dataset(
        band_vars,
        coords=band_coords,
        attrs=band_attrs,
    )
    band_ds.to_zarr(dt_store, group=out_name, mode="a", consolidated=False)

    # break the reference cycles holding the loaded pieces so a reused Ray
    # worker doesn't accumulate them across tasks (see safe_stokes_vis)
    dt.close()
    del groups, part, dt
    gc.collect()

    # post-gc memory telemetry (see safe_stokes_vis for interpretation)
    mem = task_memory()
    return {
        "timeid": meta["timeid"],
        "wsum": wsum_sum,
        "mem": mem,
        "beam_imre_ratio": beam_imre_max,
    }


# meerkat-beams serves baseline-group beams for L band only (its
# design-decisions D15 -- no matched MeerKAT counterpart exists for MKE's S3
# product). Keep this in sync with meerkat_beams.cache.
_GROUP_BANDS = ("L",)


def _preflight_baseline_groups(ms, partition_columns, beam_model, antenna_groups):
    """Validate a --baseline-groups request before any Ray cluster exists.

    Opens each MS once to read antenna_xds.attrs["overall_telescope_name"].
    That attr is broadcast from OBSERVATION::TELESCOPE_NAME, so it identifies
    the array (it cannot discriminate dishes -- that is dish diameter's job).

    The telescope and band checks are conditioned on a beam model being
    requested: with no beam model there is no group beam to get wrong, and the
    split degenerates to pure data partitioning whose summation is algebraically
    identical to not splitting. See wiki design-decisions D47.

    Args:
        ms: resolved MS paths.
        partition_columns: MSv2 partition schema override, as passed to imager.
        beam_model: the --beam-model value, or None.
        antenna_groups: the --antenna-groups value, or None.

    Returns:
        The parsed (meerkat_pattern, meerkat_plus_pattern) override, or None.

    Raises:
        ValueError: on a bad override, a non-MeerKAT array, katbeam, or a band
            for which meerkat-beams has no group beams.
    """
    override = None
    if antenna_groups is not None:
        try:
            override = parse_antenna_groups(antenna_groups)
        except ValueError as e:
            log.error_and_raise(str(e), ValueError)

    if beam_model is None:
        return override

    if str(beam_model).lower() == "katbeam":
        log.error_and_raise(
            "--baseline-groups needs per-group MeerKAT beams; katbeam has no MeerKAT+ "
            "model and gives power beams only. Pass --beam-model L instead.",
            ValueError,
        )
    if str(beam_model).upper() not in _GROUP_BANDS:
        log.error_and_raise(
            f"meerkat-beams serves baseline-group beams for {', '.join(_GROUP_BANDS)} band "
            f"only; --beam-model {beam_model!r} has no group products. The MdV-2026 group "
            f"beams also need staging by hand -- see meerkat-beams scripts/stage_group_cache.py.",
            ValueError,
        )

    for ms_name in ms:
        dt_kwargs = get_engine(ms_name, partition_columns)
        path = ms_name.replace("file://", "") if "file://" in ms_name else ms_name
        dt = xr.open_datatree(path, **dt_kwargs)
        try:
            node = next(iter(dt.children.values()))
            telescope = node["antenna_xds"].ds.attrs["overall_telescope_name"]
        finally:
            dt.close()
            del dt
            gc.collect()
        try:
            check_telescope_is_meerkat(telescope, path)
        except ValueError as e:
            log.error_and_raise(str(e), ValueError)

    return override


def imager(
    ms: list[Path],
    output_filename: str,
    scan_names: list[str] | None = None,
    spw_names: list[str] | None = None,
    field_names: list[str] | None = None,
    freq_range: str | None = None,
    overwrite: bool = False,
    data_column: str = "DATA",
    data_group: str = "base",
    partition_columns: list[str] | None = None,
    weight_column: str | None = None,
    sigma_column: str | None = None,
    flag_column: str = "FLAG",
    gain_table: list[Path] | None = None,
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
    nworkers: int = 1,
    nthreads: int | None = None,
    wgt_mode: str = "l2",
    weight_grouping: str = "per-band-time",
    robustness: float | None = None,
    field_of_view: float | None = None,
    super_resolution_factor: float = 2.0,
    cell_size: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    psf_oversize: float = 1.4,
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
    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )
    opts_dict["output_filename"] = output_filename
    opts_dict["fits_output_folder"] = fits_output_folder
    opts_dict["log_directory"] = log_directory

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2
    log.info(f"Using {nworkers} workers with {nthreads} threads per worker")

    remprod = product.upper().strip("IQUV")
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported", NotImplementedError)

    # Before init_ray and before pass 1: ducc only refuses an unreachable
    # epsilon when it is handed data, which in the imager is pass 2 -- after the
    # expensive half has run and written the scratch store (#340).
    try:
        check_gridder_epsilon(precision, epsilon, do_wgridding=do_wgridding)
    except ValueError as e:
        log.error_and_raise(str(e), ValueError)

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
        antenna_group_override = _preflight_baseline_groups(ms, partition_columns, beam_model, antenna_groups)
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

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/imager_{timestamp}.log"
    pfb_logging.log_to_file(logname)

    log.log_options_dict(opts_dict, title="IMAGER options")

    resize_thread_pool(nthreads)
    env_vars = set_envs(nthreads, ncpu, log=log)

    init_ray(
        nworkers,
        ray_address=ray_address,
        runtime_env={
            "env_vars": {**env_vars, **memray_env()},
            "worker_process_setup_hook": setup_ray_worker,
        },
        log=log,
    )

    basename = f"{output_filename}"

    # pass-1 fine averaged Stokes pieces are written into a .scratch DataTree
    scratch_fs, scratch_url = uri_and_fs(f"{basename}.scratch")
    if scratch_fs.exists(scratch_url):
        if overwrite:
            log.info(f"Overwriting {basename}.scratch")
            scratch_fs.rm(scratch_url, recursive=True)
        else:
            log.error_and_raise(f"{basename}.scratch exists. Set overwrite to overwrite it. ", RuntimeError)

    scratch_fs.makedirs(scratch_url, exist_ok=True)

    log.info(f"Pass-1 scratch products will be stored in {scratch_url}")

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

    # figure out where band edges are
    # note mapping currently maps partitions to the band it has most overlap with
    # partitions are not sub-divided
    all_freqs = []
    all_chan_widths = []
    max_blength = 0
    selected = []  # cached (ims, node, freqs_node, times_node, chan0) for the dispatch loop
    for ims, ms_name in enumerate(ms):
        dt_kwargs = get_engine(ms_name, partition_columns)
        if "file://" in ms_name:
            ms_name = ms_name.replace("file://", "")
        dt = xr.open_datatree(
            ms_name,
            **dt_kwargs,
        )
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
            # loop below applies its isel to
            ds = node.ds.isel(frequency=slice(sel.chan0, sel.chan0 + sel.nchan))
            freqs_node = ds.frequency.load().values
            all_freqs.append(freqs_node)
            all_chan_widths.append(ds.frequency.attrs["channel_width"]["data"])
            # xarray-ms establishes a regular grid over irregular or missing data.
            # Nans are inserted in these cases, mostly because xarray interprets nans as missing data.
            # This is different from xarray-kat which inherits katdal's missing data behaviour
            # (zeroed visibilities and weights).
            uvw = ds.UVW.load().values
            uvw_mask = np.isnan(uvw).all(axis=-1)
            # this forces a reshape (t, bl, 3) -> (row, 3)
            uvw = uvw[~uvw_mask]
            if uvw.size:
                max_blength = max(max_blength, np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max())
            # cache the selected node and its loaded coords so the dispatch loop
            # below does not have to re-open and re-filter the datatree
            selected.append((ims, node, freqs_node, ds.time.load().values, sel.chan0, sel.field_radec))

    if not selected:
        log.error_and_raise("Selection matched no data", ValueError)

    # resolve the common phase centre (mosaic tangent point): explicit
    # phase_dir wins; multiple distinct field centres default to their
    # barycentre; a single field keeps its own centre (no rephasing)
    field_centres = np.unique(np.array([np.round(fc, 12) for *_, fc in selected]), axis=0)
    if phase_dir is not None:
        radec_new = parse_sky_coords(phase_dir)
        log.info(
            f"Rephasing all data to phase_dir "
            f"ra={np.rad2deg(radec_new[0]):.8f} deg dec={np.rad2deg(radec_new[1]):.8f} deg"
        )
    elif field_centres.shape[0] > 1:
        radec_new = radec_barycentre(field_centres)
        log.info(
            f"Multiple fields selected; rephasing to their barycentre "
            f"ra={np.rad2deg(radec_new[0]):.8f} deg dec={np.rad2deg(radec_new[1]):.8f} deg"
        )
    else:
        radec_new = None
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

    # map the "DATA" sentinel to the data group's correlated_data variable
    if vis_col is not None:
        if dc1 == "DATA":
            dc1 = vis_col
        if dc2 == "DATA":
            dc2 = vis_col

    # map the "WEIGHT_SPECTRUM" sentinel to the data group's weight variable
    if weight_column is not None and weight_column == "WEIGHT_SPECTRUM":
        weight_column = wgt_col

    # guard against irregular channel widths
    cw = np.asarray(all_chan_widths)
    cw = cw[np.isfinite(cw)]
    if cw.size == 0:
        log.error_and_raise("No SPW has a usable channel_width", ValueError)
    min_chan_width = np.min(cw)
    if channels_per_image in (0, None, -1):
        nband = len(np.unique([f.tobytes() for f in all_freqs]))  # one per spw
    else:
        flat_freqs = np.concatenate(all_freqs)
        nband = int(np.ceil((flat_freqs.max() - flat_freqs.min()) / (min_chan_width * channels_per_image)))
        nband = max(nband, 1)
    all_freqs = np.unique(np.concatenate(all_freqs))
    log.info(f"Number of output bands determined to be {nband} based on channel width and freq range")
    band_edges = np.linspace(all_freqs.min() - min_chan_width / 2, all_freqs.max() + min_chan_width / 2, nband + 1)
    half_band_width = (band_edges[1] - band_edges[0]) / 2
    # Band-edge midpoints. These define which channels belong to which band and
    # nothing else -- the band's reported frequency is the effective (weighted)
    # one reduced in pass 2 (issue #296, wiki D28).
    band_centres = band_edges[0:-1] + half_band_width

    # shared imaging geometry (also fixes the padded uv-grid used for COUNTS)
    max_freq = float(all_freqs.max())
    nx, ny, nx_psf, ny_psf, cell_n, cell_rad, cell_deg = set_image_size(
        max_blength, max_freq, field_of_view, super_resolution_factor, cell_size, nx, ny, psf_oversize
    )
    # TODO - this is currently hard-coded to 1.7 because we don't know what padding the
    # wgridder will choose. It could be exposed as a parameter but ideally should be
    # determined by the gridder (consider adding functionality to radiomesh)
    min_padding = 1.7
    nx_pad = int(np.ceil(min_padding * nx))
    nx_pad += nx_pad % 2
    ny_pad = int(np.ceil(min_padding * ny))
    ny_pad += ny_pad % 2
    log.info(f"Image size (nx={nx}, ny={ny}), cell={np.rad2deg(cell_rad) * 3600:.4e} arcsec")

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
        ant_xds = node["antenna_xds"].ds
        ant_names = ant_xds.antenna_name.values
        diam = ant_xds.ANTENNA_DISH_DIAMETER.values if "ANTENNA_DISH_DIAMETER" in ant_xds.data_vars else None
        try:
            is_ext = classify_antennas(ant_names, diam, override=antenna_group_override)
            node_masks_list.append(
                baseline_group_masks(
                    is_ext,
                    ant_names,
                    node.ds.baseline_antenna1_name.values,
                    node.ds.baseline_antenna2_name.values,
                )
            )
        except ValueError as e:
            log.error_and_raise(str(e), ValueError)
    if baseline_groups:
        present_groups = sorted({g for m in node_masks_list for g in m}, key=GROUP_LABELS.index)
        log.info("Baseline groups present: " + ", ".join(present_groups))

    # MeerKAT band name -> BeamWizard from the meerkat-beams band cache (the
    # same convention as hci); "katbeam"/None pass through to stokes_vis as is
    beam_refs = None
    if beam_model is not None and not isinstance(beam_model, BeamWizard) and beam_model.lower() != "katbeam":
        if baseline_groups:
            # One wizard per baseline group. ray.put is load-bearing, not
            # tidiness: a group wizard holds an IN-MEMORY cross-multiplied
            # dataset (~25 MB at the MdV-2026 grid; nothing group-shaped is
            # file-backed) and pass 1 emits hundreds of tasks, so passing it by
            # value per task would serialise it hundreds of times.
            log.info(f"Initialising a BeamWizard for each of {', '.join(present_groups)}")
            beam_refs = {g: ray.put(BeamWizard(band=beam_model, group=g)) for g in present_groups}
        else:
            log.info("Assuming MeerKAT data and initialising BeamWizard")
            # no image_name: detached mode -- pass 1 supplies explicit l/m/times/freq
            beam_model = BeamWizard(band=beam_model)

    tasks = []
    scan_block_to_tid = {}  # (scan_name, block_idx) -> tid
    next_tid = 0
    # Pre-create the band/time parent groups single-threaded so concurrent
    # pass-1 workers only ever create their own distinct leaf piece group. This
    # avoids a check-then-create race (ContainsGroupError) on the shared parent;
    # combined with consolidated=False on the worker writes (consolidation is
    # done once below, in the driver), it removes all shared-mutable-state races
    # on the scratch store. See utils/stokes2vis_msv4.stokes_vis.
    scratch_root = zarr.open_group(scratch_url, mode="a")
    created_parents = set()
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
                    fut = safe_stokes_vis.remote(
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
                        nx_pad=nx_pad,
                        ny_pad=ny_pad,
                        cell_rad=cell_rad,
                        baseline_group=bg,
                        data_group=data_group,
                        radec_new=radec_new,
                        target=target,
                        nx=nx,
                        ny=ny,
                        nthreads=nthreads,
                    )
                    tasks.append(fut)

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
                print(
                    f"Completed: {ncomplete} / {nds} [{format_memory(mem)}]",
                    end="\n",
                    flush=True,
                )

    ntime = len(set(timeids_out))
    nband_out = len(set(bandids_out))

    log.info(f"Pass 1 wrote fine pieces for {nband_out} bands and {ntime} time chunks to {scratch_url}")
    log.info(f"Pass 1 done after {time.time() - time_start}s")

    # consolidate the scratch metadata once, single-threaded, now that all
    # workers have finished (workers wrote with consolidated=False to avoid
    # racing on the shared root .zmetadata; see stokes_vis)
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
        node_info[name] = {
            "bandid": bandid,
            "timeid": timeid,
            "ra": float(pieces[0].attrs["ra"]),
            "dec": float(pieces[0].attrs["dec"]),
            "time_out": float(np.mean([ds.attrs["time_out"] for ds in pieces])),
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
                    },
                )
            )
    else:
        for name, info in node_info.items():
            l0, m0 = target_offset(info["time_out"])
            meta = {k: info[k] for k in ("bandid", "timeid", "time_out")}
            meta.update(ra=float(grid_radec[0]), dec=float(grid_radec[1]), l0=l0, m0=m0)
            work.append((name, [name], meta))

    ntime = len({meta["timeid"] for _, _, meta in work})

    # one applied counts grid per weighting group, streamed into the scratch
    # store; pass-2 workers read their group's grid from there (#339)
    counts_groups = {}
    if robustness is not None:
        counts_groups = write_group_counts(
            scratch_url, counts_sources, filter_level=filter_counts_level, npix_super=npix_super
        )
        # the counts groups were written after pass 1's consolidation
        zarr.consolidate_metadata(scratch_url)
    log.info(f"Applied uv counts grouping '{grouping_eff}' over {len(counts_groups)} group(s)")

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
    tasks = []
    for out_name, src_names, meta in work:
        fut = _grid_image.remote(
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
            nx_pad=nx_pad,
            ny_pad=ny_pad,
            nthreads=nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            do_psf=psf,
            part_fits_dir=part_fits_dir,
            part_fits_beam=beam,
        )
        tasks.append(fut)

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
                print(
                    f"Gridded: {ncomplete} / {nds} [{format_memory(mem)}]",
                    end="\n",
                    flush=True,
                )

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

    # ---- FITS ----
    if fits_mfs or fits_cubes:
        fits_oname = f"{fits_output_folder}/{oname}"
        log.info(f"Writing fits files to {fits_oname}")
        base_kwargs = dict(
            norm_wsum=True,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfparsn if psf else None,
        )
        columns = {"DIRTY": {}}
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

    if not keep_scratch:
        log.info(f"Removing scratch store {scratch_url}")
        scratch_fs.rm(scratch_url, recursive=True)

    log.info(f"All done after {time.time() - time_start}s")

    if not keep_ray_alive:
        ray.shutdown()

    return
