import string

import numpy as np
import sympy as sm
from numba import njit
from numba.core import types
from sympy.physics.quantum import TensorProduct
from sympy.utilities.lambdify import lambdify


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


def stokes_funcs(data, jones, product, pol, nc):
    if isinstance(product, types.StringLiteral):
        product = product.literal_value
    if isinstance(pol, types.StringLiteral):
        pol = pol.literal_value
    if isinstance(nc, types.StringLiteral):
        nc = nc.literal_value

    # set up symbolic expressions
    gp00, gp10, gp01, gp11 = sm.symbols("gp00 gp10 gp01 gp11", real=False)
    gq00, gq10, gq01, gq11 = sm.symbols("gq00 gq10 gq01 gq11", real=False)
    w0, w1, w2, w3 = sm.symbols("W0 W1 W2 W3", real=True)
    v00, v10, v01, v11 = sm.symbols("v00 v10 v01 v11", real=False)

    # Jones matrices
    gp_matrix = sm.Matrix([[gp00, gp01], [gp10, gp11]])
    gq_matrix = sm.Matrix([[gq00, gq01], [gq10, gq11]])

    # Mueller matrix (row major form)
    mpq = TensorProduct(gp_matrix, gq_matrix.conjugate())
    mpq_inv = TensorProduct(gp_matrix.inv(), gq_matrix.conjugate().inv())

    # inverse noise covariance
    s_inv = sm.Matrix([[w0, 0, 0, 0], [0, w1, 0, 0], [0, 0, w2, 0], [0, 0, 0, w3]])
    s = s_inv.inv()

    # visibilities
    vpq = sm.Matrix([[v00], [v01], [v10], [v11]])

    # Full Stokes to corr operator
    # Is this the only difference between linear and circular pol?
    # What about paralactic angle rotation?
    if pol == "linear":
        t_matrix = sm.Matrix([[1.0, 1.0, 0, 0], [0, 0, 1.0, 1.0j], [0, 0, 1.0, -1.0j], [1.0, -1.0, 0, 0]])
    elif pol == "circular":
        t_matrix = sm.Matrix([[1.0, 0, 0, 1.0], [0, 1.0, 1.0j, 0], [0, 1.0, -1.0j, 0], [1.0, 0, 0, -1.0]])
    t_inv = t_matrix.inv()

    # Full Stokes weights
    w = t_matrix.H * mpq.H * s_inv * mpq * t_matrix
    w_inv = t_inv * mpq_inv * s * mpq_inv.H * t_inv.H

    # Full Stokes coherencies
    c = w_inv * (t_matrix.H * (mpq.H * (s_inv * vpq)))
    # Only keep diagonal of weights
    w = w.diagonal().T  # diagonal() returns row vector

    # this should ensure that outputs are always ordered as
    # [I, Q, U, V]
    i = ()
    if "I" in product:
        i += (0,)

    if "Q" in product:
        i += (1,)
        if pol == "circular" and nc == "2":
            raise ValueError("Q is not available in circular polarisation with 2 correlations")

    if "U" in product:
        i += (2,)
        if pol == "linear" and nc == "2":
            raise ValueError("U is not available in linear polarisation with 2 correlations")
        elif pol == "circular" and nc == "2":
            raise ValueError("U is not available in circular polarisation with 2 correlations")

    if "V" in product:
        i += (3,)
        if pol == "linear" and nc == "2":
            raise ValueError("V is not available in linear polarisation with 2 correlations")

    remprod = product.strip("IQUV")
    if len(remprod):
        raise ValueError(f"Unknown polarisation product {remprod}")

    if jones.ndim == 6:  # Full mode
        w_symb = lambdify((gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3), sm.simplify(w[i, 0]))
        w_jfn = njit(nogil=True, inline="always")(w_symb)

        d_symb = lambdify(
            (gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11), sm.simplify(c[i, 0])
        )
        d_jfn = njit(nogil=True, inline="always")(d_symb)

        @njit(nogil=True, inline="always")
        def wfunc(gp, gq, w):
            gp00 = gp[0, 0]
            gp01 = gp[0, 1]
            gp10 = gp[1, 0]
            gp11 = gp[1, 1]
            gq00 = gq[0, 0]
            gq01 = gq[0, 1]
            gq10 = gq[1, 0]
            gq11 = gq[1, 1]
            w00 = w[0]
            w01 = w[1]
            w10 = w[2]
            w11 = w[3]
            return w_jfn(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w00, w01, w10, w11).real.ravel()

        @njit(nogil=True, inline="always")
        def vfunc(gp, gq, w, v):
            gp00 = gp[0, 0]
            gp01 = gp[0, 1]
            gp10 = gp[1, 0]
            gp11 = gp[1, 1]
            gq00 = gq[0, 0]
            gq01 = gq[0, 1]
            gq10 = gq[1, 0]
            gq11 = gq[1, 1]
            w00 = w[0]
            w01 = w[1]
            w10 = w[2]
            w11 = w[3]
            v00 = v[0]
            v01 = v[1]
            v10 = v[2]
            v11 = v[3]
            return d_jfn(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w00, w01, w10, w11, v00, v01, v10, v11).ravel()

    elif jones.ndim == 5:  # DIAG mode
        w = w.subs(gp10, 0)
        w = w.subs(gp01, 0)
        w = w.subs(gq10, 0)
        w = w.subs(gq01, 0)
        c = c.subs(gp10, 0)
        c = c.subs(gp01, 0)
        c = c.subs(gq10, 0)
        c = c.subs(gq01, 0)

        w_symb = lambdify((gp00, gp11, gq00, gq11, w0, w1, w2, w3), sm.simplify(w[i, 0]))
        w_jfn = njit(nogil=True, inline="always")(w_symb)

        d_symb = lambdify((gp00, gp11, gq00, gq11, w0, w1, w2, w3, v00, v01, v10, v11), sm.simplify(c[i, 0]))
        d_jfn = njit(nogil=True, inline="always")(d_symb)

        if nc == "4":

            @njit(nogil=True, inline="always")
            def wfunc(gp, gq, w):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                w00 = w[0]
                w01 = w[1]
                w10 = w[2]
                w11 = w[3]
                return w_jfn(gp00, gp11, gq00, gq11, w00, w01, w10, w11).real.ravel()

            @njit(nogil=True, inline="always")
            def vfunc(gp, gq, w, v):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                w00 = w[0]
                w01 = w[1]
                w10 = w[2]
                w11 = w[3]
                v00 = v[0]
                v01 = v[1]
                v10 = v[2]
                v11 = v[3]
                return d_jfn(gp00, gp11, gq00, gq11, w00, w01, w10, w11, v00, v01, v10, v11).ravel()
        elif nc == "2":

            @njit(nogil=True, inline="always")
            def wfunc(gp, gq, w):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                w00 = w[0]
                w01 = 1.0
                w10 = 1.0
                w11 = w[-1]
                return w_jfn(gp00, gp11, gq00, gq11, w00, w01, w10, w11).real.ravel()

            @njit(nogil=True, inline="always")
            def vfunc(gp, gq, w, v):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                w00 = w[0]
                w01 = 1.0
                w10 = 1.0
                w11 = w[-1]
                v00 = v[0]
                v01 = 0j
                v10 = 0j
                v11 = v[-1]
                return d_jfn(gp00, gp11, gq00, gq11, w00, w01, w10, w11, v00, v01, v10, v11).ravel()
        else:
            raise ValueError(
                f"Selected product is only available from 2 or 4correlation data while you have ncorr={nc}."
            )

    else:
        raise ValueError("Jones term has incorrect number of dimensions")

    return vfunc, wfunc
