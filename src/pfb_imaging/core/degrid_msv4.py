"""`pfb degrid-msv4`: degrid a `.mds` component model into MSv4 data (#278).

The MSv4 <-> pfb-model-spec seam lives in `utils/degrid_msv4.py` and is pure;
this module owns the guards, the Ray Serve deployment and the driver. It is
the MSv4 replacement for `core/degrid.py`, which is the codebase's last
consumer of `distributed`.
"""

import numpy as np
import xarray as xr
from pfb_model_spec.utils.degrid import model_geometry

from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.msv4 import SelectedNode, wrapped_angle_diff

log = pfb_logging.get_logger("DEGRID_MSV4")


def check_model(model_ds: xr.Dataset, product: str) -> dict:
    """Validate the `.mds` and the requested product before any work starts.

    `model_geometry` already raises on an unknown `spec` and on non-square
    pixels; calling it here means those fire before a Ray cluster is stood up
    and before a column is added to the user's MS.

    The product checks are stricter than the old `degrid`'s and deliberately
    so. The `genesis` spec stores a single Stokes plane, but the old command
    took `len(product)` planes and degridded the same image into each, so
    `--product IQ` emitted `XX = 2I, YY = 0`. Refusing is the fix.

    Args:
        model_ds: An opened `.mds` dataset.
        product: The `--product` string.

    Returns:
        `model_geometry(model_ds)`: nx, ny, cell_rad, x0, y0, flip_u/v/w, stokes.

    Raises:
        ValueError: If the product is not a subset of IQUV, names more than one
            Stokes product, or disagrees with the model's own `stokes` attr; or
            if `model_geometry` rejects the spec or the pixel shape.
    """
    product = product.upper().strip()
    remainder = product.strip("IQUV")
    if remainder:
        raise ValueError(f"Product {remainder} not yet supported")
    if len(product) != 1:
        raise ValueError(
            f"Product {product!r} names {len(product)} Stokes products, but the "
            "'genesis' .mds spec carries a single Stokes plane. Degrid one "
            "product at a time until pfb-model-spec#19 lands."
        )

    geom = model_geometry(model_ds)  # raises on unknown spec / non-square pixels
    if geom["stokes"].upper() != product:
        raise ValueError(
            f"Requested product {product!r} but the model's stokes attr is "
            f"{geom['stokes']!r}. Degridding a model as a different Stokes "
            "product produces confidently wrong correlations."
        )
    return geom


def check_tangent_point(
    model_ds: xr.Dataset,
    nodes: list[SelectedNode],
    tol_rad: float = 1e-9,
) -> None:
    """Refuse to degrid a model against a field it was not made for.

    A model on a different tangent point needs a per-row inverse w-phase to be
    predicted correctly (wiki D21, mosaics). v1 does not do that, so this is a
    refusal rather than a warning.

    Compared as a **wrapped magnitude**: a naive signed difference silently
    accepts half of all real mismatches, and a bare `abs(a - b)` reads two RAs
    either side of zero as ~2*pi apart.

    Args:
        model_ds: An opened `.mds` dataset, carrying `ra`/`dec` attrs.
        nodes: Selected visibility nodes, each with a `field_radec`.
        tol_rad: Tolerance in radians.

    Raises:
        ValueError: If any node's field centre differs from the model's.
    """
    model_radec = np.array([float(model_ds.ra), float(model_ds.dec)])
    for node in nodes:
        sep = wrapped_angle_diff(node.field_radec, model_radec)
        if np.any(sep > tol_rad):
            raise ValueError(
                f"Tangent point mismatch for field {node.field_name!r} in "
                f"{node.path}: field is (ra, dec) = "
                f"{np.rad2deg(node.field_radec)} deg, model is "
                f"{np.rad2deg(model_radec)} deg (separation "
                f"{np.rad2deg(sep)} deg). Degridding a rephased or mosaic "
                "model is not supported in v1 (see wiki D21)."
            )
