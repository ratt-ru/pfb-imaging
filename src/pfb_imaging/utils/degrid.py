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
from typing import NamedTuple

import numpy as np
import xarray as xr
from msv4_utils import MSv4Backend, infer_backend
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
from pfb_model_spec.utils.degrid import (
    degrid_stokes,
    model_geometry,
    render_model_region,
    stokes_vis_to_corr,
)

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


class RegionMask(NamedTuple):
    """A region's weights, cropped to the region's bounding box.

    Cropping is a pure optimisation, and a large one: a region is degridded by
    its own `dirty2vis` call, whose cost scales with the grid it is handed, not
    with the flux on it. Measured on a 6720^2 model with three regions, the
    per-chunk gridder time fell from 3 x 5.29 s to 5.29 + 0.64 + 0.51 s. It
    also shrinks what the driver ships to each replica -- a full-grid float64
    mask is 361 MB at 6720^2, and a small region's is under 1 MB.

    Attributes:
        mask: `(nyc, nxc)` float64 weights, (Y, X) like the model.
        i0: Index along y of the mask's origin in the full model grid.
        j0: Index along x of the mask's origin in the full model grid.
    """

    mask: np.ndarray
    i0: int
    j0: int


def _even_span(lo: int, hi: int, n: int) -> tuple[int, int]:
    """Grow `[lo, hi)` to an even length without leaving `[0, n)`.

    ducc's wgridder asserts `nx_dirty must be even`, so a bounding box of odd
    extent cannot be handed to it. Growing outwards keeps the box a window on
    real model pixels, which padding with zeros would not.
    """
    if (hi - lo) % 2 == 0:
        return lo, hi
    if hi < n:
        return lo, hi + 1
    if lo > 0:
        return lo - 1, hi
    # the full axis is odd, so no even window exists; the uncropped grid would
    # fail the same assertion, and that is the error worth surfacing
    return lo, hi


def crop_bbox(mask: np.ndarray) -> RegionMask:
    """Crop a full-grid mask to the bounding box of its non-zero pixels.

    The box is grown by at most one pixel per axis so that both extents are
    even, which ducc requires (see `_even_span`).

    Args:
        mask: `(ny, nx)` weights, (Y, X) like the model.

    Returns:
        The cropped mask and its origin. An all-zero mask crops to a `(2, 2)`
        zero box at the origin -- `degrid_stokes` skips an empty plane
        outright, so the degenerate box is never gridded.
    """
    ny, nx = mask.shape
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if xs.size == 0 or ys.size == 0:
        return RegionMask(np.zeros((min(2, ny), min(2, nx)), dtype=mask.dtype), 0, 0)
    i0, i1 = _even_span(int(ys[0]), int(ys[-1]) + 1, ny)
    j0, j1 = _even_span(int(xs[0]), int(xs[-1]) + 1, nx)
    return RegionMask(np.ascontiguousarray(mask[i0:i1, j0:j1]), i0, j0)


def crop_phase_centre(
    *,
    x0: float,
    y0: float,
    cell_rad: float,
    flip_u: bool,
    flip_v: bool,
    nx: int,
    ny: int,
    rm: RegionMask,
) -> tuple[float, float]:
    """The `center_x`/`center_y` a cropped sub-grid must be degridded with.

    ducc places image pixel `i` at `(i - nx // 2) * cell` from the image
    centre, so cropping moves the centre by the shift in that origin --
    `j0 + nxc // 2 - nx // 2` pixels along x and `i0 + nyc // 2 - ny // 2`
    along y. The sign is the axis's flip convention: `flip_u`/`flip_v` negate
    the direction the offset is measured in. Getting either sign wrong is not
    a small error -- it puts the model in the wrong place on the sky, and the
    visibilities come back order-unity wrong (`tests/test_degrid.py`, the
    crop-parity test, checks all four flip combinations).

    Args:
        x0: The model's own `center_x`, from `model_geometry`.
        y0: The model's own `center_y`.
        cell_rad: Pixel size in radians (square pixels).
        flip_u: The model's U-axis flip convention.
        flip_v: The model's V-axis flip convention.
        nx: Full model grid size along x.
        ny: Full model grid size along y.
        rm: The cropped mask whose grid the offsets are wanted for.

    Returns:
        `(center_x, center_y)` for `degrid_stokes`.
    """
    nyc, nxc = rm.mask.shape
    sx = -1.0 if flip_u else 1.0
    sy = -1.0 if flip_v else 1.0
    return (
        x0 + sx * (rm.j0 + nxc // 2 - nx // 2) * cell_rad,
        y0 + sy * (rm.i0 + nyc // 2 - ny // 2) * cell_rad,
    )


def check_stokes_subset(product: str, model_stokes: Sequence[str]) -> list[str]:
    """Validate that `product` names Stokes planes the model actually carries.

    Shared by `degrid` and `imager --transfer-model-from`: rendering a model as
    a Stokes product it does not hold produces confidently wrong data, so a
    missing plane is refused rather than zero-filled.

    Args:
        product: The `--product` string, for example `"I"` or `"IQUV"`.
        model_stokes: The model's `stokes` coord, from `model_geometry`.

    Returns:
        The requested Stokes labels, in `product` order.

    Raises:
        ValueError: If `product` is empty, has a letter outside IQUV, repeats
            one, or names a plane absent from `model_stokes`.
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
    if missing:
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
        stokes: Stokes planes to degrid, in output order; a subset of the
            model's, validated by `check_stokes_subset`.
        accumulate: Add to what the column already holds in this region.
        epsilon: Gridder accuracy.
        do_wgridding: Perform w-correction via improved w-stacking.
        nthreads: ducc threads.

    Returns:
        A dataset carrying only `columns`, sized to `region`, with `node_ds`'s
        encoding intact -- ready for `to_msv2(compute=True, region=region)`.

    Raises:
        ValueError: If `masks` and `columns` differ in length.
    """
    if len(masks) != len(columns):
        raise ValueError(f"{len(masks)} masks for {len(columns)} columns")

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
    # matching the .mds's own coords -- do not convert to MJD (D13).
    time_out = float(ds.time.values.mean())
    freq_out = float(freq.mean())

    # Render once for the whole chunk rather than once per region: the model is
    # a function of (time, freq) alone, so every region sees the same image.
    # `model_to_apparent_vis_for_region` renders inside its own per-region call,
    # which is why this composes its primitives instead (pfb-model-spec#27, which
    # would move this loop -- and the crop arithmetic -- behind that seam).
    geom = model_geometry(model_ds)
    planes = [geom["stokes"].index(s) for s in stokes]
    image = render_model_region(model_ds, time=time_out, freq_out=freq_out)
    if planes != list(range(len(geom["stokes"]))):
        # fancy indexing copies the whole grid (361 MB at 6720^2), so only
        # pay for it when the selection is not the identity
        image = image[planes]

    assigned = {}
    for column, rm in zip(columns, masks, strict=True):
        nyc, nxc = rm.mask.shape
        sub = image[:, rm.i0 : rm.i0 + nyc, rm.j0 : rm.j0 + nxc] * rm.mask[None]
        x0, y0 = crop_phase_centre(
            x0=geom["x0"],
            y0=geom["y0"],
            cell_rad=geom["cell_rad"],
            flip_u=geom["flip_u"],
            flip_v=geom["flip_v"],
            nx=geom["nx"],
            ny=geom["ny"],
            rm=rm,
        )
        stokes_vis = degrid_stokes(
            uvw,
            freq,
            np.ascontiguousarray(sub),
            cell_rad=geom["cell_rad"],
            x0=x0,
            y0=y0,
            flip_u=geom["flip_u"],
            flip_v=geom["flip_v"],
            flip_w=geom["flip_w"],
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            # the .mds records no beam, so nothing folds 1/n in (wiki D22)
            divide_by_n=False,
            nthreads=nthreads,
            mask=mask,
        )
        vis = stokes_vis_to_corr(stokes_vis, list(stokes), tuple(corr_types))
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


def flux_scale_warning(flux_scale: str | None) -> str | None:
    """The warning to log for a model's flux scale, if any.

    `degrid` applies no primary beam, so an intrinsic model yields intrinsic
    visibilities -- correct only where the beam is close to 1.

    Args:
        flux_scale: The `.mds` `flux_scale`, from `model_sampling`.

    Returns:
        The message, or `None` when nothing needs saying.
    """
    if flux_scale != "intrinsic":
        return None
    return (
        "The model is in intrinsic flux and degrid applies no primary beam, so the model "
        "visibilities are only correct where the beam is close to 1. "
        "Beam application is tracked in https://github.com/ratt-ru/pfb-imaging/issues/353."
    )
