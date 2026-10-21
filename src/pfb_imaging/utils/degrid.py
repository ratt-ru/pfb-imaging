"""The MSv4 <-> pfb-model-spec seam for `pfb degrid` (issue #278).

Everything here is pure and Ray-free: turning a `(time, frequency)` region of
a visibility node into the arguments `pfb_model_spec.utils.degrid` wants, and
turning its answer back into a minimal Dataset `to_msv2` can write to that
same region. The numerics live in pfb-model-spec; the distribution and the
CLI live in `core/degrid.py`. This mirrors `utils/stokes2vis_msv4.py`,
which plays the same role for the imager's pass 1.

Axis convention: the `.mds` (spec 0.1), pfb-model-spec's rendered images and
astropy `regions` masks are all `(Y, X)`, and visibilities have no image
orientation, so nothing here transposes an image. ducc0's x-major world is
pfb-model-spec's business (`degrid_stokes`).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from msv4_utils import MSv4Backend, infer_backend
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
from pfb_model_spec.utils.degrid import (
    SPECTRAL,
    BeamTerm,
    RegionMask,
    crop_bbox,
    frequency_pieces,
    model_geometry,
    model_to_apparent_vis_for_regions,
    pixel_lm,
    split_at_kinks,
)

from pfb_imaging.utils.beam import BEAM_PA_TOLERANCE_DEG
from pfb_imaging.utils.misc import to_mjd_time

# Canonical MSv4 ordering for a correlated-data variable.
MODEL_DIMS = ("time", "baseline_id", "frequency", "polarization")


def make_column_placeholder(shape: tuple[int, ...], dtype=np.complex64) -> np.ndarray:
    """A zero-strided stand-in used only to declare a column's shape/dtype.

    `sync_msv2` reads `var.dims`, `var.shape` and `var.dtype` and nothing else.
    Allocating the real array -- or `xr.zeros_like(node.VISIBILITY)`, which the
    upstream example uses -- costs a full correlated-data read per node just to
    declare a name. This is 8 bytes.

    Args:
        shape: The variable's shape, in `MODEL_DIMS` order.
        dtype: The column dtype.

    Returns:
        A read-only, zero-strided view of a single zero.
    """
    return np.broadcast_to(np.array(0, dtype=dtype), shape)


def check_writable_backend(ms_path: str) -> None:
    """Refuse a backend degrid cannot write back to.

    `get_engine` resolves zarr and MeerKAT stores happily -- the imager reads
    both -- but degrid's output is a *write*, and only the MSv2/CASA backend
    carries the `common_store_args`/`partition_key` encoding `to_msv2` needs.
    Without this the run degrids a whole chunk first and dies in `sync_msv2`
    with `MissingEncodingError`, which names neither the MS nor the reason.

    Args:
        ms_path: Path to the measurement set.

    Raises:
        ValueError: If `ms_path` is not a CASA measurement set.
    """
    path = ms_path.replace("file://", "") if "file://" in ms_path else ms_path
    backend = infer_backend(path)
    if backend is not MSv4Backend.CASA_TABLE:
        raise ValueError(
            f"{ms_path} is a {backend.name} store; degrid writes model "
            "columns through xarray-ms's MSv2 write support and can only "
            "write to a CASA measurement set. Read-only front ends such as "
            "`pfb imager` accept this store, but degrid cannot."
        )


def ensure_model_columns(
    ms_path: str,
    dt: xr.DataTree,
    columns: Sequence[str],
    *,
    dtype=np.complex64,
) -> None:
    """Create `columns` on the MS backing `dt` if they are not already there.

    Two non-obvious constraints, both of them silent failures otherwise:

    1. The placeholder variable is assigned to **every** correlated node in
       `dt.subtree`, not just the selected ones. `sync_msv2` drops any variable
       whose node count differs from the number of nodes it visits, warning
       rather than raising. This is also semantically right: a CASA column
       belongs to the whole MAIN table, not to a partition.
    2. Creating a column leaves table handles that were already open on this MS
       unable to resync (ska-sa/arcae#241), so the caller must drop any it holds
       -- `core/degrid.degrid` evicts the process-wide table cache once
       after this returns.

    The caller must **close `dt` afterwards** before any other process reads
    the MS: new columns are invisible to other processes until close. It must
    also open `dt` with a **single MAIN instance**
    (`get_engine(..., main_ninstances=1)`): arcae adds the column on instance 0
    but `sync_msv2`'s follow-up `columns()` goes to the least busy instance,
    which intermittently throws "another process changed the number of
    columns" or returns a stale list that trips its own assertion
    (ska-sa/arcae#241).

    Args:
        ms_path: Path to the measurement set (no `file://` prefix).
        dt: An opened MSv4 DataTree over `ms_path`, with one MAIN instance.
        columns: Column names to ensure exist.
        dtype: Column dtype; `complex64` is the MS visibility dtype.

    Raises:
        ValueError: If `dt` holds no visibility datasets, or if its partitions
            have differing `(nchan, ncorr)` shapes.
        ColumnCreationError: From `sync_msv2`, if a column is still absent after
            creation. Canonical MAIN names (`MODEL_DATA`, `CORRECTED_DATA`,
            `DATA`) need xarray-ms >= 0.4.0a8, which fixed
            ratt-ru/xarray-ms#171; before that they were skipped silently.
    """
    # deferred: monkeypatches sync_msv2/to_msv2 onto xarray's Dataset/DataTree
    import xarray_ms  # noqa: F401

    vis_nodes = [n for n in dt.subtree if n.attrs.get("type") in VISIBILITY_XDS_TYPES]
    if not vis_nodes:
        raise ValueError(f"No visibility datasets in {ms_path}")

    # Check this BEFORE touching the table. A CASA column spans the whole MAIN
    # table, so every partition must fit one cell shape. Heterogeneous SPWs
    # would need variably-shaped cells -- which cannot take a partial region
    # write at all (their cells have no array until written whole:
    # `SSMIndColumn::getShape: no array in row 0`). Refuse rather than leave a
    # column behind that half the MS cannot be written into.
    shapes = {(int(n.sizes["frequency"]), int(n.sizes["polarization"])) for n in vis_nodes}
    if len(shapes) > 1:
        raise ValueError(
            f"{ms_path} has visibility partitions with differing "
            f"(nchan, ncorr) shapes {sorted(shapes)}. degrid writes one "
            "fixed-shape column across the whole MAIN table and cannot span "
            "heterogeneous spectral windows. Note --spw-names cannot help: a "
            "CASA column belongs to the whole MAIN table, so this is checked "
            "over every partition regardless of what is selected. Split the "
            "measurement set so each spectral window has its own."
        )

    for node in vis_nodes:
        shape = tuple(node.sizes[d] for d in MODEL_DIMS)
        placeholder = make_column_placeholder(shape, dtype)
        ds = node.ds.assign({c: (MODEL_DIMS, placeholder) for c in columns})
        dt[node.path] = xr.DataTree(ds)

    # identity write_map: promote_write_map overrides MSV4_WRITE_MAP, so a
    # column a user happened to name VISIBILITY is not redirected to DATA
    dt.sync_msv2(write_map={c: c for c in columns})


def check_stokes_subset(product: str, model_stokes: Sequence[str], *, beam: bool = False) -> list[str]:
    """Validate that `product` names Stokes planes the model actually carries.

    Shared by `degrid` and `imager --transfer-model-from`: rendering a model as
    a Stokes product it does not hold produces confidently wrong data, so a
    missing plane is refused rather than zero-filled. With a beam the product
    names apparent planes, which the Mueller block fills from the model's
    planes, so any of IQUV is allowed: naming more is how a user opts into
    beam leakage (D55).

    Args:
        product: The `--product` string, for example `"I"` or `"IQUV"`.
        model_stokes: The model's `stokes` coord, from `model_geometry`.
        beam: Whether a beam maps the model's planes onto the product.

    Returns:
        The requested Stokes labels, in `product` order.

    Raises:
        ValueError: If `product` is empty, has a letter outside IQUV, repeats
            one, or, without a beam, names a plane absent from `model_stokes`.
    """
    requested = list(product.upper().strip())
    if not requested:
        raise ValueError("Product is empty; name at least one Stokes parameter")
    unsupported = [s for s in requested if s not in "IQUV"]
    if unsupported:
        raise ValueError(f"Product {''.join(unsupported)} not yet supported")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Product {product!r} repeats a Stokes parameter")
    missing = [s for s in requested if s not in model_stokes]
    if missing and not beam:
        raise ValueError(
            f"Requested product {product!r} but the model carries Stokes {list(model_stokes)}; "
            f"{''.join(missing)} is missing. Rendering a model as a Stokes product it does not "
            "hold produces confidently wrong data."
        )
    return requested


def build_region_masks(model_ds: xr.Dataset, region_file: str | None) -> list[RegionMask]:
    """Split the model grid into a remainder plus one mask per region (#115).

    Each region's flux is degridded into its own MS column, so the masks must
    partition the grid exactly: mask `i` pairs with column `model_column` for
    `i == 0` (the remainder, i.e. everything outside every region) and
    `f"{model_column}{i}"` thereafter.

    **Orientation.** `to_image((ny, nx))` returns a `(Y, X)` raster, which is
    the `.mds` order since spec 0.1 (pfb-model-spec#20), so no transpose is
    needed.

    Args:
        model_ds: An opened `.mds` dataset.
        region_file: Path to a region file in any format astropy `regions`
            detects, or `None` for a single all-ones mask.

    Returns:
        Float64 masks cropped to their own bounding boxes (see `RegionMask`):
        the remainder first, then one per region. The no-region mask is a
        full-grid, zero-strided all-ones -- see below.

    Raises:
        ValueError: If two regions overlap.
    """
    nx = int(model_ds.npix_x)
    ny = int(model_ds.npix_y)
    if region_file is None:
        # zero-strided, as make_column_placeholder is: the common case has no
        # regions, and a materialised all-ones grid costs nx*ny*8 bytes (361 MB
        # at 6720^2) per holder to multiply the model by 1. Consumers only read
        # .shape and broadcast against it.
        # nothing to crop to: the "region" is the whole grid
        return [RegionMask(np.broadcast_to(np.float64(1.0), (ny, nx)), 0, 0)]

    # deferred: import cycle with utils.fits (load_fits <-> utils.misc)
    from regions import Regions

    from pfb_imaging.utils.fits import set_wcs

    rfile = Regions.read(region_file)  # format auto-detected
    wcs = set_wcs(
        np.rad2deg(model_ds.cell_rad_x),
        np.rad2deg(model_ds.cell_rad_y),
        nx,
        ny,
        (model_ds.ra, model_ds.dec),
        model_ds.freqs.values,
        header=False,
    )
    wcs = wcs.dropaxis(-1).dropaxis(-1)  # drop the freq and stokes axes

    total = np.zeros((ny, nx), dtype=np.float64)
    masks = []
    for i, region in enumerate(rfile):
        # a file in `image`/`physical` coordinates parses straight to a pixel
        # region, which has no to_pixel; only sky regions need converting.
        # The old degrid assumed sky and raised AttributeError on the rest.
        pixel_region = region.to_pixel(wcs) if hasattr(region, "to_pixel") else region
        image = pixel_region.to_mask().to_image((ny, nx))
        if image is None:
            # astropy returns None rather than an empty raster when the region
            # misses the grid entirely; say which region, since the alternative
            # is a TypeError on `total += None` naming nothing
            raise ValueError(
                f"Region {i} of {region_file} does not overlap the model grid "
                f"({nx}x{ny} pixels centred on ra={float(model_ds.ra):.6f}, "
                f"dec={float(model_ds.dec):.6f} rad). Check the region's "
                "coordinates and frame against the model."
            )
        region_mask = image
        total += region_mask
        masks.append(region_mask)

    if (total > 1).any():
        raise ValueError("Overlapping regions are not supported")

    # the remainder (direction-independent component) goes first. Cropping
    # happens only now, so the overlap check above sees whole-grid masks.
    return [crop_bbox(m) for m in [1.0 - total] + masks]


def assert_writable(ds: xr.Dataset) -> None:
    """Fail early if a dataset has lost the encoding `to_msv2` needs.

    `msv2_store_from_dataset` recovers the table handle from
    `ds.encoding["common_store_args"]` and `["partition_key"]`. Those survive
    `isel`, `drop_vars` and `assign`, but not a dataset rebuilt from scratch --
    and the failure would otherwise surface as a `MissingEncodingError` after
    the degridding work is already done.

    Args:
        ds: The dataset about to be written.

    Raises:
        ValueError: If either encoding key is missing.
    """
    missing = [k for k in ("common_store_args", "partition_key") if k not in ds.encoding]
    if missing:
        raise ValueError(
            f"Dataset encoding is missing {missing}; it cannot be written back to "
            "the measurement set. Build the write dataset by dropping and "
            "assigning variables on the opened node, never from scratch."
        )


def _coarse_axes(wizard: Any, l_px: np.ndarray, m_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ascending coarse (l, m) axes covering the model grid at half a beam-cube pixel.

    The beam is smooth on that scale, so evaluating there and interpolating is
    exact to the beam's own resolution while costing a few thousand points
    instead of the model's millions.
    """
    step = 0.5 * np.deg2rad(np.abs(np.diff(np.asarray(wizard.bds.X.values, dtype=float))).min())

    def axis(v: np.ndarray) -> np.ndarray:
        lo, hi = float(v.min()) - step, float(v.max()) + step
        return lo + step * np.arange(max(2, int(np.ceil((hi - lo) / step)) + 1))

    return axis(l_px), axis(m_px)


def chunk_beam_terms(
    wizards: Mapping[str, Any],
    groups: tuple[dict[str, np.ndarray], np.ndarray] | None,
    *,
    nbl: int,
    ntime: int,
    geom: dict,
    radec: tuple[float, float],
    times: np.ndarray,
    freq_out: float,
    stokes_out: Sequence[str],
    nthreads: int = 1,
    chord: tuple[float, float] | None = None,
) -> list[BeamTerm]:
    """Build one work item's beam terms from meerkat-beams wizards.

    Evaluated at the chunk's own times and at `freq_out`, pointing at the
    model's tangent point, which `check_tangent_point` has already required to
    be the field centre. The beam is the bare B from `partition_mueller`: the
    caller degrids with `divide_by_n=True` (D55).

    With `chord`, each term also carries the beam's frequency slope as the
    chord `(M(hi) - M(lo)) / (hi - lo)` over the chunk's extreme channels, the
    linear fit meerkat-beams D19 measured as the most accurate (D59). Its line
    passes through `M(freq_out)`, so the model, the beam and the offsets share
    one expansion point.

    Args:
        wizards: `{"all": wizard}` without groups, else `{label: wizard}`.
        groups: `node_baseline_groups` output for the node, or None.
        nbl: Baselines on the node.
        ntime: Times in the chunk; rows are `(time, baseline)` C-ordered.
        geom: `model_geometry` of the model.
        radec: Model tangent point, radians.
        times: The chunk's unix-second times.
        freq_out: Frequency to evaluate the beam at, Hz.
        stokes_out: Stokes planes predicted, in output order.
        nthreads: Threads for the rotation average; meerkat-beams splits the pixels over them.
        chord: `(lo, hi)` channel frequencies for a beam slope, or None for a
            flat beam. A zero-width chord (one channel) sends no slope.

    Returns:
        One `BeamTerm` per group present (one in all without groups).

    Raises:
        ValueError: If the groups leave a baseline uncovered or name a group
            with no wizard.
    """
    l_px, m_px = pixel_lm(geom)
    field_centre = SkyCoord(ra=radec[0] * u.rad, dec=radec[1] * u.rad)
    # D13: MSv4 time is unix seconds; astropy Time wants MJD days
    t = Time(to_mjd_time(np.unique(times)) / 86400.0, format="mjd")
    if groups is None:
        items = [("all", None, None)]
    else:
        index, conj = groups
        covered = np.zeros(nbl, dtype=bool)
        for idx in index.values():
            covered[idx] = True
        if not covered.all():
            raise ValueError(f"{int((~covered).sum())} baseline(s) belong to no baseline group")
        items = [(label, idx, conj) for label, idx in index.items()]

    terms = []
    for label, idx, conj in items:
        if label not in wizards:
            raise ValueError(f"no beam for baseline group {label!r}")
        wizard = wizards[label]
        l_axis, m_axis = _coarse_axes(wizard, l_px, m_px)

        def mueller_at(freq: float, wizard=wizard, l_axis=l_axis, m_axis=m_axis) -> np.ndarray:
            return np.asarray(
                wizard.partition_mueller(
                    field_centre=field_centre,
                    times=t,
                    freq=float(freq),
                    l=np.rad2deg(l_axis),
                    m=np.rad2deg(m_axis),
                    stokes_out="".join(stokes_out),
                    stokes_in="".join(geom["stokes"]),
                    nthreads=nthreads,
                    pa_tolerance=BEAM_PA_TOLERANCE_DEG,
                )
            )

        block = mueller_at(freq_out)
        dmueller = None
        if chord is not None and chord[1] > chord[0]:
            lo, hi = chord
            dmueller = (mueller_at(hi) - mueller_at(lo)) / (hi - lo)
        rows = conj_rows = None
        if idx is not None:
            mine = np.zeros(nbl, dtype=bool)
            mine[idx] = True
            rows = np.tile(mine, ntime)
            conj_rows = np.tile(conj & mine, ntime)
        terms.append(BeamTerm(block, l_axis, m_axis, rows, conj_rows, dmueller))
    return terms


def piece_anchor(model_ds: xr.Dataset, freq: float) -> float:
    """The weighted anchor of the model piece holding `freq`, the imaging beam's frequency.

    `deconv` solved for the band value against a band beam that is, to first
    order, `B(anchor)` (D28), so a beam held flat across a chunk reproduces
    that gauge only at the anchor (D59).

    Args:
        model_ds: An opened `.mds` dataset.
        freq: A frequency inside the chunk, Hz.

    Returns:
        The piece's anchor in Hz, or `freq` itself for a model without pieces
        (the polynomial bases).
    """
    for piece in frequency_pieces(model_ds):
        if piece.lo <= freq < piece.hi:
            return piece.anchor
    return float(freq)


def degrid_region(
    node_ds: xr.Dataset,
    *,
    region: Mapping[str, slice],
    model_ds: xr.Dataset,
    masks: Sequence[RegionMask],
    columns: Sequence[str],
    corr_types: Sequence[str],
    stokes: Sequence[str],
    accumulate: bool = False,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    nthreads: int = 1,
    wizards: Mapping[str, Any] | None = None,
    groups: tuple[dict[str, np.ndarray], np.ndarray] | None = None,
    spectral: str = "flat",
    beam_slope: bool = True,
) -> xr.Dataset:
    """Degrid the model into one `(time, frequency)` region of a node.

    Args:
        node_ds: The **unsliced** visibility node dataset. Regions index this,
            so a `--freq-range` selection must already be folded into `region`
            (see `SelectedNode.chan0`).
        region: `{"time": slice, "frequency": slice}`. Slices only -- integer
            indexing makes the matching write raise `MismatchedWriteRegion`.
        model_ds: An opened `.mds` dataset.
        masks: Cropped region masks from `build_region_masks`, one per entry
            of `columns` and in the same order.
        columns: Output column names, aligned with `masks`.
        corr_types: Correlation names, i.e. the node's `polarization` values.
        stokes: Stokes planes to degrid, in output order, validated by
            `check_stokes_subset`: a subset of the model's without `wizards`,
            any of IQUV with them (the beam's leakage fills the rest).
        accumulate: Add to what the column already holds in this region.
        epsilon: Gridder accuracy.
        do_wgridding: Perform w-correction via improved w-stacking.
        nthreads: ducc threads.
        wizards: meerkat-beams wizards keyed as `chunk_beam_terms` expects; None applies no beam.
        groups: `node_baseline_groups` output, or None.
        spectral: `"flat"` holds the model and beam at the chunk mean;
            `"linear"` also carries the model's slope, and the region must not
            cross a frequency kink (`frequency_chunks` cuts there; D59).
        beam_slope: Under `"linear"`, carry the beam's chord slope. Off, the
            beam is held at the chunk's piece anchor (`piece_anchor`).

    Returns:
        A dataset carrying only `columns`, sized to `region`, with `node_ds`'s
        encoding intact -- ready for `to_msv2(compute=True, region=region)`.

    Raises:
        ValueError: If `masks` and `columns` differ in length, `spectral` is
            unknown, or a linear region crosses a frequency kink.
    """
    if len(masks) != len(columns):
        raise ValueError(f"{len(masks)} masks for {len(columns)} columns")
    if spectral not in SPECTRAL:
        raise ValueError(f"spectral must be one of {SPECTRAL}, got {spectral!r}")

    ds = node_ds.isel(**region)
    ntime = int(ds.sizes["time"])
    nbl = int(ds.sizes["baseline_id"])
    freq = ds.frequency.values
    nchan = freq.size
    ncorr = len(corr_types)

    # (time, baseline_id, uvw_label) -> (nrow, 3); nrow == ntime * nbl and the
    # inverse reshape below restores the MSv4 layout the write region expects
    uvw = ds.UVW.values.reshape(-1, 3)
    valid = ~np.isnan(uvw).any(axis=-1)
    # xarray-ms lays data on a regular (time, baseline) grid and pads absent
    # cells with NaN UVW. ducc derives its w range from EVERY row it is handed,
    # so an unmasked padded row gives a NaN w extent -- issue #287, guarded by
    # tests/test_nan_padded_rows.py. The mask must be uint8: a bool mask of
    # identical layout raises "incorrect data type" from ducc's pybind layer.
    mask = np.ascontiguousarray(np.broadcast_to(valid[:, None], (uvw.shape[0], nchan)).astype(np.uint8))
    # belt and braces: masked rows are already inert, but zeroing them means
    # correctness does not rest on ducc's undocumented mask ordering
    uvw = np.nan_to_num(uvw, copy=True)

    # Representative time and frequency for the chunk: UNWEIGHTED means, and
    # deliberately not the imager's weight-weighted effective frequency (D28).
    # Degrid reads no weights, and an unweighted mean is reproducible across
    # tools regardless of their flagging, so two consumers cannot disagree
    # about where the model was evaluated. Both axes are unix seconds / Hz,
    # matching the .mds's own coords -- do not convert to MJD (D13). In linear
    # mode the mean is also the expansion point (D59).
    time_out = float(ds.time.values.mean())
    freq_out = float(freq.mean())

    beam_terms = None
    if wizards is not None:
        beam_freq, chord = freq_out, None
        if spectral == "linear":
            if beam_slope:
                chord = (float(freq.min()), float(freq.max()))
            else:
                beam_freq = piece_anchor(model_ds, freq_out)
        beam_terms = chunk_beam_terms(
            wizards,
            groups,
            nbl=nbl,
            ntime=ntime,
            geom=model_geometry(model_ds),
            radec=(float(model_ds.ra), float(model_ds.dec)),
            times=ds.time.values,
            freq_out=beam_freq,
            stokes_out=stokes,
            nthreads=nthreads,
            chord=chord,
        )

    # Render once and degrid each region's cropped box: pfb-model-spec's
    # multi-region entry point (#27). No FLAG is read: the model is predicted on
    # every real row so flagging can be rerun against it; `mask` only hides
    # xarray-ms's NaN-padded cells, which are not MS rows (#287).
    per_region = model_to_apparent_vis_for_regions(
        model_ds,
        uvw=uvw,
        freq=freq,
        corr_types=tuple(corr_types),
        time=time_out,
        freq_out=freq_out,
        region_masks=masks,
        stokes_out=list(stokes),
        mask=mask,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        # 1/n is never folded into a beam on this path, so ducc always applies
        # it, beam or not (D55); the pre-#353 False dropped it off-axis
        divide_by_n=True,
        nthreads=nthreads,
        beam_terms=beam_terms,
        spectral=spectral,
    )
    assigned = {}
    for column, vis in zip(columns, per_region, strict=True):
        vis = vis.astype(np.complex64).reshape(ntime, nbl, nchan, ncorr)
        if accumulate:
            vis += ds[column].values.astype(np.complex64)
        assigned[column] = (MODEL_DIMS, vis)

    # to_msv2 writes every data variable it is handed, so hand it only ours.
    # drop_vars/assign preserve ds.encoding, which the MSv2 store needs.
    return ds.drop_vars(set(ds.data_vars)).assign(assigned)


def default_channels_per_chunk(freq_bounds: np.ndarray, chan_width: float) -> int:
    """Channels per degridding chunk that span one imaging band (#327).

    The `.mds` records band edges in Hz rather than a channel count, so an MS
    with channels half as wide gets twice as many per chunk and every chunk
    covers the bandwidth the model was fitted over.

    Args:
        freq_bounds: The model's `(nband, 2)` band edges in Hz.
        chan_width: The node's channel width in Hz (sign ignored).

    Returns:
        A positive channel count; at least 1 when a channel is wider than a band.
    """
    band_width = float(np.min(freq_bounds[:, 1] - freq_bounds[:, 0]))
    return max(1, int(round(band_width / abs(chan_width))))


def frequency_chunks(freq: np.ndarray, fstep: int, kinks: np.ndarray | None = None) -> list[slice]:
    """Cut a node's selected frequency axis into degridding chunks.

    Without kinks this is the fixed `fstep`-channel blocking degrid has always
    used, remainder and all. With them (`--spectral linear`) the axis is first
    cut at the model's frequency kinks, because a linear chunk must not cross
    one (pfb-model-spec `frequency_kinks`), and each run is then split into
    `ceil(n / fstep)` near-equal parts: a run one channel longer than `fstep`
    would otherwise cost a full gridding pass for a one-channel chunk (D59).

    Args:
        freq: The selected channel frequencies, Hz, ascending or descending.
        fstep: Target channels per chunk.
        kinks: Kink frequencies from `frequency_kinks`, or None for flat blocking.

    Returns:
        Contiguous slices of `freq`, in index order, covering it once.
    """
    n = freq.size
    if kinks is None:
        return [slice(i, min(i + fstep, n)) for i in range(0, n, fstep)]
    if n > 1 and freq[0] > freq[-1]:
        # split_at_kinks wants an ascending axis; map its runs back to index order
        runs = [slice(n - r.stop, n - r.start) for r in reversed(split_at_kinks(freq[::-1], kinks))]
    else:
        runs = split_at_kinks(freq, kinks)
    chunks = []
    for run in runs:
        size = run.stop - run.start
        nsplit = -(-size // fstep)
        bounds = np.cumsum([run.start] + [len(a) for a in np.array_split(np.empty(size), nsplit)])
        chunks += [slice(int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:])]
    return chunks


def flux_scale_warning(flux_scale: str | None) -> str | None:
    """The warning to log when an intrinsic model is degridded without --beam-model.

    Without a beam an intrinsic model yields intrinsic visibilities, correct
    only where the beam is close to 1. Only logged when degrid runs without
    --beam-model.

    Args:
        flux_scale: The `.mds` `flux_scale`, from `model_sampling`.

    Returns:
        The message, or `None` when nothing needs saying.
    """
    if flux_scale != "intrinsic":
        return None
    return (
        "The model is in intrinsic flux and no primary beam is applied, so the model "
        "visibilities are only correct where the beam is close to 1. "
        "Pass --beam-model to predict apparent visibilities."
    )
