import string

import numpy as np
from numba.extending import register_jitable
from radiomesh.generated._stokes_expr import CONVERT_FNS


def jones_to_mueller(gp, gq):
    shape = gp.shape
    rem_shape = shape[2:]
    i0 = string.ascii_lowercase.index("m")
    rem_idx = string.ascii_lowercase[i0 : i0 + len(rem_shape)]
    idxp = "ij" + rem_idx
    idxq = "kl" + rem_idx
    idxo = "ikjl" + rem_idx
    out_shape = (4, 4) + rem_shape
    return np.einsum(f"{idxp},{idxq}->{idxo}", gp, np.conjugate(gq)).reshape(*out_shape)


def mueller_to_stokes(mueller, poltype="linear"):
    """
    Convert a Mueller matrix into a diagonal Stokes matrix.
    """
    if poltype == "linear":
        t_matrix = np.array([[1.0, 1.0, 0, 0], [0, 0, 1.0, 1.0j], [0, 0, 1.0, -1.0j], [1.0, -1.0, 0, 0]])
    elif poltype == "circular":
        t_matrix = np.array([[1.0, 0, 0, 1.0], [0, 1.0, 1.0j, 0], [0, 1.0, -1.0j, 0], [1.0, 0, 0, -1.0]])
    else:
        raise ValueError(f"Unknown poltype {poltype}")
    shape = mueller.shape
    rem_shape = shape[2:]
    i0 = string.ascii_lowercase.index("m")
    rem_idx = string.ascii_lowercase[i0 : i0 + len(rem_shape)]
    idxl = "ij" + rem_idx
    idxr = "ji"
    idxo = "i" + rem_idx
    return np.einsum(f"{idxl},{idxr}->{idxo}", mueller, t_matrix).real


def corr_to_stokes(x, wsum=1.0, axis=0, poltype="linear"):
    """
    x = [I+Q, U+1jV, U-1jV, I-Q]
    out = [I, Q, U, V]
    """
    if poltype.lower() != "linear":
        raise NotImplementedError("Only linear polarisation is implemented")
    if x.shape[axis] != 4:
        raise ValueError(f"Expected 4 polarisation products, got {x.shape[axis]}")
    dirty_i = (np.take(x, 0, axis=axis) + np.take(x, 3, axis=axis)).real / wsum
    dirty_q = (np.take(x, 0, axis=axis) - np.take(x, 3, axis=axis)).real / wsum
    dirty_u = (np.take(x, 1, axis=axis) + np.take(x, 2, axis=axis)).real / wsum
    dirty_v = (np.take(x, 1, axis=axis) - np.take(x, 2, axis=axis)).imag / wsum
    return np.stack((dirty_i, dirty_q, dirty_u, dirty_v), axis=axis)


def stokes_to_corr(x, axis=0, poltype="linear"):
    """
    x = [I, Q, U, V]
    out = [I+Q, U+1jV, U-1jV, I-Q]
    """
    if poltype.lower() != "linear":
        raise NotImplementedError("Only linear polarisation is implemented")
    if x.shape[axis] != 4:
        raise ValueError(f"Expected 4 polarisation products, got {x.shape[axis]}")
    dirty0 = np.take(x, 0, axis=axis) + np.take(x, 1, axis=axis)
    dirty1 = np.take(x, 2, axis=axis) + 1j * np.take(x, 3, axis=axis)
    dirty2 = np.take(x, 2, axis=axis) - 1j * np.take(x, 3, axis=axis)
    dirty3 = np.take(x, 0, axis=axis) - np.take(x, 1, axis=axis)
    return np.stack((dirty0, dirty1, dirty2, dirty3), axis=axis)


# Make every generated expression function callable from nopython code.
# register_jitable returns the original plain function, so CONVERT_FNS
# values stay plain module-level functions: capturing them in an @overload
# impl closure keeps numba's cache key stable across processes (issue #273
# in pfb-imaging) — never replace these with njit dispatchers.
for _fn in set(CONVERT_FNS.values()):
    register_jitable(inline="always")(_fn)


def stokes_expr_funcs(product, pol, nc, wgt_mode, jones_ndim):
    """Select radiomesh per-Stokes expression functions for weight_data.

    Args:
        product: Stokes product string, subset of "IQUV" (any order).
        pol: Polarisation type, "linear" or "circular".
        nc: Number of correlations as a string, "2" or "4".
        wgt_mode: Weighting mode, "l2" or "minvar".
        jones_ndim: 5 for diagonal jones, 6 for full 2x2 jones.

    Returns:
        Tuple of (vis_fns, wgt_fns): equal-length tuples of plain functions
        ordered I, Q, U, V. Diag functions take
        (x00, x01, x10, x11, jp00, jp11, jq00, jq11); full-jones functions
        take (x00, x01, x10, x11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11).

    Raises:
        ValueError: For invalid product/pol/nc/mode/ndim combinations.
        NotImplementedError: For minvar with full 2x2 jones.
    """
    if pol not in ("linear", "circular"):
        raise ValueError(f"Unknown polarisation type {pol}")
    if nc not in ("1", "2", "4"):
        raise ValueError(f"Unsupported number of correlations {nc}")
    if wgt_mode not in ("l2", "minvar"):
        raise ValueError(f"Unknown weighting mode {wgt_mode}")

    # this ensures that outputs are always ordered as [I, Q, U, V]
    stokes = []
    if "I" in product:
        stokes.append("I")
    if "Q" in product:
        if pol == "circular" and nc == "2":
            raise ValueError("Q is not available in circular polarisation with 2 correlations")
        stokes.append("Q")
    if "U" in product:
        if nc == "2":
            raise ValueError(f"U is not available in {pol} polarisation with 2 correlations")
        stokes.append("U")
    if "V" in product:
        if pol == "linear" and nc == "2":
            raise ValueError("V is not available in linear polarisation with 2 correlations")
        stokes.append("V")
    remprod = product.strip("IQUV")
    if len(remprod):
        raise ValueError(f"Unknown polarisation product {remprod}")

    polu = pol.upper()
    if jones_ndim == 6:
        if wgt_mode == "minvar":
            raise NotImplementedError("Minvar weighting not yet implemented for full-Stokes")
        if nc != "4":
            raise ValueError("Full 2x2 jones mode requires 4 correlation data")
        wkey, jmode = "WEIGHT", "JONES"
    elif jones_ndim == 5:
        if nc not in ("2", "4"):
            raise ValueError(
                f"Selected product is only available from 2 or 4 correlation data while you have ncorr={nc}."
            )
        wkey = "WEIGHT_MINVAR" if wgt_mode == "minvar" else "WEIGHT"
        jmode = "DIAGJONES"
    else:
        raise ValueError("Jones term has incorrect number of dimensions")

    vis_fns = tuple(CONVERT_FNS[("VIS", polu, jmode, s)] for s in stokes)
    wgt_fns = tuple(CONVERT_FNS[(wkey, polu, jmode, s)] for s in stokes)
    return vis_fns, wgt_fns
