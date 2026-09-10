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

from collections.abc import Sequence

import numpy as np
import xarray as xr
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

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
       therefore verify afterwards and create what is still missing.

    The caller must **close `dt` afterwards** before any other process reads
    the MS: new columns are invisible to other processes until close.

    Args:
        ms_path: Path to the measurement set (no `file://` prefix).
        dt: An opened MSv4 DataTree over `ms_path`.
        columns: Column names to ensure exist.
        dtype: Column dtype; `complex64` is the MS visibility dtype.

    Raises:
        ValueError: If `dt` holds no visibility datasets, or if a column is
            still absent after both creation attempts.
    """
    # deferred: monkeypatches sync_msv2/to_msv2 onto xarray's Dataset/DataTree
    import xarray_ms  # noqa: F401

    vis_nodes = [n for n in dt.subtree if n.attrs.get("type") in VISIBILITY_XDS_TYPES]
    if not vis_nodes:
        raise ValueError(f"No visibility datasets in {ms_path}")

    for node in vis_nodes:
        shape = tuple(node.sizes[d] for d in MODEL_DIMS)
        placeholder = make_column_placeholder(shape, dtype)
        ds = node.ds.assign({c: (MODEL_DIMS, placeholder) for c in columns})
        dt[node.path] = xr.DataTree(ds)

    # identity write_map: promote_write_map overrides MSV4_WRITE_MAP, so a
    # column a user happened to name VISIBILITY is not redirected to DATA
    dt.sync_msv2(write_map={c: c for c in columns})

    node = vis_nodes[0]
    _create_missing_columns(
        ms_path,
        columns,
        nchan=int(node.sizes["frequency"]),
        ncorr=int(node.sizes["polarization"]),
        dtype=dtype,
    )


def _create_missing_columns(ms_path, columns, *, nchan, ncorr, dtype):
    """Create any column `sync_msv2` declined to, matching what it would build.

    The descriptor is deliberately assembled from xarray-ms's own pieces
    (`fit_tile_shape`, `NUMPY_TO_CASA_MAP`) so the column we create is
    indistinguishable from one it created. Delete this whole function when
    upstream closes the canonical-name gap.

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
