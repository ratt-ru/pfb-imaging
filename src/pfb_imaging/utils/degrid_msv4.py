"""The MSv4 <-> pfb-model-spec seam for `pfb degrid-msv4` (issue #278).

Everything here is pure and Ray-free: turning a `(time, frequency)` region of
a visibility node into the arguments `pfb_model_spec.utils.degrid` wants, and
turning its answer back into a minimal Dataset `to_msv2` can write to that
same region. The numerics live in pfb-model-spec; the distribution and the
CLI live in `core/degrid_msv4.py`. This mirrors `utils/stokes2vis_msv4.py`,
which plays the same role for the imager's pass 1.

Axis convention: pfb-model-spec and ducc0 are both x-major `(nx, ny)`, and
visibilities have no image orientation, so nothing here transposes an image.
The single exception is `build_region_masks`, which converts an astropy
`regions` mask from `(Y, X)` -- see its docstring.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
from pfb_model_spec.utils.degrid import model_to_apparent_vis_for_region

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
    2. `sync_msv2` will not create a column whose name is in casacore's
       canonical MAIN descriptor but absent from the table -- and `MODEL_DATA`,
       `CORRECTED_DATA` and `DATA` all are. `generate_column_descriptor`
       validates such a name against the canonical descriptor and then falls
       through without emitting one, so `addcols` is never asked for it. We
       therefore verify afterwards and create what is still missing
       (ratt-ru/xarray-ms#171).

    The caller must **close `dt` afterwards** before any other process reads
    the MS: new columns are invisible to other processes until close.

    Args:
        ms_path: Path to the measurement set (no `file://` prefix).
        dt: An opened MSv4 DataTree over `ms_path`.
        columns: Column names to ensure exist.
        dtype: Column dtype; `complex64` is the MS visibility dtype.

    Raises:
        ValueError: If `dt` holds no visibility datasets, if its partitions
            have differing `(nchan, ncorr)` shapes, or if a column is still
            absent after both creation attempts.
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
            f"(nchan, ncorr) shapes {sorted(shapes)}. degrid-msv4 writes one "
            "fixed-shape column across the whole MAIN table and cannot span "
            "heterogeneous spectral windows; select a single spectral window "
            "with --spw-names, or degrid each one into its own measurement set."
        )
    nchan, ncorr = next(iter(shapes))

    for node in vis_nodes:
        shape = tuple(node.sizes[d] for d in MODEL_DIMS)
        placeholder = make_column_placeholder(shape, dtype)
        ds = node.ds.assign({c: (MODEL_DIMS, placeholder) for c in columns})
        dt[node.path] = xr.DataTree(ds)

    # identity write_map: promote_write_map overrides MSV4_WRITE_MAP, so a
    # column a user happened to name VISIBILITY is not redirected to DATA
    dt.sync_msv2(write_map={c: c for c in columns})

    _create_missing_columns(ms_path, columns, nchan=nchan, ncorr=ncorr, dtype=dtype)


def _create_missing_columns(ms_path, columns, *, nchan, ncorr, dtype):
    """Create any column `sync_msv2` declined to, matching what it would build.

    The descriptor is deliberately assembled from xarray-ms's own pieces
    (`fit_tile_shape`, `NUMPY_TO_CASA_MAP`) so the column we create is
    indistinguishable from one it created. Delete this whole function when
    upstream closes the canonical-name gap (ratt-ru/xarray-ms#171).

    A fixed-shape `TiledColumnStMan` column is required, not the canonical
    variable-shape `StandardStMan` descriptor: the latter creates cells with no
    array shape, and a partial region write into an unshaped cell fails with
    `SSMIndColumn::getShape: no array in row 0`.
    """
    # deferred: arcae is the write path's table handle, xarray-ms internals are
    # private and pinned by the write-support pin
    import arcae
    from xarray_ms.backend.msv2.writes import fit_tile_shape
    from xarray_ms.casa_types import NUMPY_TO_CASA_MAP

    with arcae.table(ms_path, readonly=False) as tab:
        missing = [c for c in columns if c not in tab.columns()]
        if not missing:
            return

        # casacore descriptor shapes are FORTRAN ordered -- the reverse of the
        # numpy trailing shape. Getting this backwards does not raise: casacore
        # SIGABRTs and takes the interpreter down with it.
        fixed_shape = (ncorr, nchan)
        value_type = NUMPY_TO_CASA_MAP[np.dtype(dtype).type]

        descs = {}
        dm_groups = []
        for col in missing:
            group = f"{col}_GROUP"
            descs[col] = {
                "valueType": value_type,
                "option": 4,  # FixedShape
                "shape": list(fixed_shape),
                "ndim": len(fixed_shape),
                "dataManagerGroup": group,
                "dataManagerType": "TiledColumnStMan",
            }
            dm_groups.append(
                {
                    "COLUMNS": [col],
                    "NAME": group,
                    "TYPE": "TiledColumnStMan",
                    # descriptors are JSON-serialised by arcae: plain ints only
                    "SPEC": fit_tile_shape(fixed_shape, dtype),
                }
            )

        # addcols takes dminfo positionally and it has no default
        tab.addcols(descs, {f"*{i + 1}": g for i, g in enumerate(dm_groups)})

        still_missing = [c for c in missing if c not in tab.columns()]
        if still_missing:
            raise ValueError(f"Failed to create column(s) {still_missing} in {ms_path}")


def build_region_masks(model_ds: xr.Dataset, region_file: str | None) -> list[np.ndarray]:
    """Split the model grid into a remainder plus one mask per region (#115).

    Each region's flux is degridded into its own MS column, so the masks must
    partition the grid exactly: mask `i` pairs with column `model_column` for
    `i == 0` (the remainder, i.e. everything outside every region) and
    `f"{model_column}{i}"` thereafter.

    **Orientation.** `pixel_region.to_mask().to_image((ny, nx))` returns a
    `(Y, X)` raster, which is astropy's convention and pfb-imaging's own for
    images; the `.mds` and every pfb-model-spec array are x-major `(nx, ny)`.
    The transpose below converts between them and is load-bearing -- it is the
    one place in this module where an image orientation changes. Delete it
    when the `.mds` spec flips (pfb-model-spec#20), not before.

    Args:
        model_ds: An opened `.mds` dataset.
        region_file: Path to a region file in any format astropy `regions`
            detects, or `None` for a single all-ones mask.

    Returns:
        `(nx, ny)` float64 masks: the remainder first, then one per region.

    Raises:
        ValueError: If two regions overlap.
    """
    nx = int(model_ds.npix_x)
    ny = int(model_ds.npix_y)
    if region_file is None:
        return [np.ones((nx, ny), dtype=np.float64)]

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

    total = np.zeros((nx, ny), dtype=np.float64)
    masks = []
    for region in rfile:
        # a file in `image`/`physical` coordinates parses straight to a pixel
        # region, which has no to_pixel; only sky regions need converting.
        # The old degrid assumed sky and raised AttributeError on the rest.
        pixel_region = region.to_pixel(wcs) if hasattr(region, "to_pixel") else region
        # (Y, X) from astropy -> x-major to match the model; see the docstring
        region_mask = pixel_region.to_mask().to_image((ny, nx)).T
        total += region_mask
        masks.append(region_mask)

    if (total > 1).any():
        raise ValueError("Overlapping regions are not supported")

    # the remainder (direction-independent component) goes first
    return [1.0 - total] + masks


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
    masks: Sequence[np.ndarray],
    columns: Sequence[str],
    corr_types: Sequence[str],
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
        masks: `(nx, ny)` region masks from `build_region_masks`, one per entry
            of `columns` and in the same order.
        columns: Output column names, aligned with `masks`.
        corr_types: Correlation names, i.e. the node's `polarization` values.
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

    assigned = {}
    for column, region_mask in zip(columns, masks, strict=True):
        vis = model_to_apparent_vis_for_region(
            model_ds,
            uvw=uvw,
            freq=freq,
            corr_types=tuple(corr_types),
            time=time_out,
            freq_out=freq_out,
            mask=mask,
            region_mask=region_mask,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            # the .mds records no beam, so nothing folds 1/n in (wiki D22)
            divide_by_n=False,
            nthreads=nthreads,
        )
        vis = vis.astype(np.complex64).reshape(ntime, nbl, nchan, ncorr)
        if accumulate:
            vis += ds[column].values.astype(np.complex64)
        assigned[column] = (MODEL_DIMS, vis)

    # to_msv2 writes every data variable it is handed, so hand it only ours.
    # drop_vars/assign preserve ds.encoding, which the MSv2 store needs.
    return ds.drop_vars(set(ds.data_vars)).assign(assigned)
