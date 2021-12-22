import numpy as np
from numba import njit, literally
from numba.types import literal
import sympy as sm
from sympy.physics.quantum import TensorProduct
from sympy.utilities.lambdify import lambdify
from africanus.calibration.utils.utils import DIAG_DIAG, DIAG, FULL

def stokes_vis(product, mode, pol='linear'):
    # symbolic variables
    gp00, gp10, gp01, gp11 = sm.symbols("g_{p00} g_{p10} g_{p01} g_{p11}", real=False)
    gq00, gq10, gq01, gq11 = sm.symbols("g_{q00} g_{q10} g_{q01} g_{q11}", real=False)
    w0, w1, w2, w3 = sm.symbols("W_0 W_1 W_2 W_3", real=True)
    v0, v1, v2, v3 = sm.symbols("v_{00} v_{10} v_{01} v_{11}", real=False)
    i, q, u, v = sm.symbols("I Q U V", real=True)

    # Jones matrices
    Gp = sm.Matrix([[gp00, gp01],[gp10, gp11]])
    Gq = sm.Matrix([[gq00, gq01],[gq10, gq11]])

    # Mueller matrix
    Mpq = TensorProduct(Gp, Gq.conjugate())

    # inverse noise covariance (weights)
    W = sm.Matrix([[w0, 0, 0, 0],
                  [0, w1, 0, 0],
                  [0, 0, w2, 0],
                  [0, 0, 0, w3]])

    # visibilities
    Vpq = sm.Matrix([[v0], [v1], [v2], [v3]])

    # Stokes to corr operator
    if True: #pol == 'linear':
        T = sm.Matrix([[1.0, 1.0, 0, 0],
                        [0, 0, 1.0, -1.0j],
                        [0, 0, 1.0, 1.0j],
                        [1, -1, 0, 0]])
    else:

        raise NotImplementedError('Sorry, no circular pol')

    # corr weights
    M = T.H * Mpq.H * W * Mpq * T
    V = T.H * Mpq.H * W * Vpq

    wparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3)
    vparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3, v0, v1, v2, v3)
    if product == literal('I'):
        wfunc = lambdify(wparams, M[0, 0], 'numpy')
        vfunc = lambdify(vparams, V[0], 'numpy')
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[0, 0], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[0], 'numpy'))

    elif product == literal('Q'):
        wfunc = lambdify(wparams, M[1, 1], 'numpy')
        vfunc = lambdify(vparams, V[1], 'numpy')
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[1, 1], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[1], 'numpy'))

    elif product == literal('U'):
        wfunc = lambdify(wparams, M[1, 1], 'numpy')
        vfunc = lambdify(vparams, V[1], 'numpy')
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[2, 2], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[2], 'numpy'))

    elif product == literal('V'):
        wfunc = lambdify(wparams, M[1, 1], 'numpy')
        vfunc = lambdify(vparams, V[1], 'numpy')
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[3, 3], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[3], 'numpy'))
    else:
        raise ValueError(f'Unknown product {product}')

    if mode == DIAG_DIAG:
        # M = M.subs(gp01, 0)
        # M = M.subs(gp10, 0)
        # M = M.subs(gq01, 0)
        # M = M.subs(gq10, 0)
        # V = V.subs(gp01, 0)
        # V = V.subs(gp10, 0)
        # V = V.subs(gq01, 0)
        # V = V.subs(gq10, 0)
        # V = V.subs(v1, 0)
        # V = V.subs(v2, 0)
        # wparams = (gp00, gp11, gq00, gq11, w0, w3)
        # vparams = (gp00, gp11, gq00, gq11, w0, w3, v0, v3)
        # if product == literal('I'):
        #     wfunc = lambdify(wparams, M[0, 0], 'numpy')
        #     vfunc = lambdify(vparams, V[0], 'numpy')

        # elif product == 'Q':
        #     wfunc = lambdify(wparams, M[1, 1], 'numpy')
        #     vfunc = lambdify(vparams, V[1], 'numpy')

        # elif product == 'U':
        #     wfunc = lambdify(wparams, M[2, 2], 'numpy')
        #     vfunc = lambdify(vparams, V[2], 'numpy')

        # elif product == 'V':
        #     wfunc = lambdify(wparams, M[3, 3], 'numpy')
        #     vfunc = lambdify(vparams, V[3], 'numpy')
        # else:
        #     raise ValueError(f'Unknown product {product}')

        # @jit(nogil=True, cache=True, fastmath=True)
        def wgt_func(wgt, gp, gq):
            return wfunc(gp[0], 0j, 0j, gp[1],
                         gq[0], 0j, 0j, gq[1],
                         wgt[0], 0, 0, wgt[1]).real

        # @jit(nogil=True, cache=True, fastmath=True)
        def vis_func(data, wgt, gp, gq):
            return vfunc(gp[0], 0j, 0j, gp[1],
                         gq[0], 0j, 0j, gq[1],
                         wgt[0], 0, 0, wgt[1],
                         data[0], 0j, 0j, data[1])

    elif mode == DIAG:
        # M = M.subs(gp01, 0)
        # M = M.subs(gp10, 0)
        # M = M.subs(gq01, 0)
        # M = M.subs(gq10, 0)
        # V = V.subs(gp01, 0)
        # V = V.subs(gp10, 0)
        # V = V.subs(gq01, 0)
        # V = V.subs(gq10, 0)
        # wparams = (gp00, gp11, gq00, gq11, w0, w1, w2, w3)
        # vparams = (gp00, gp11, gq00, gq11, w0, w1, w2, w3, v0, v1, v2, v3)
        # if product == literal('I'):
        #     wfunc = lambdify(wparams, M[0, 0], 'numpy')
        #     vfunc = lambdify(vparams, V[0], 'numpy')

        # elif product == 'Q':
        #     wfunc = lambdify(wparams, M[1, 1], 'numpy')
        #     vfunc = lambdify(vparams, V[1], 'numpy')

        # elif product == 'U':
        #     wfunc = lambdify(wparams, M[2, 2], 'numpy')
        #     vfunc = lambdify(vparams, V[2], 'numpy')

        # elif product == 'V':
        #     wfunc = lambdify(wparams, M[3, 3], 'numpy')
        #     vfunc = lambdify(vparams, V[3], 'numpy')
        # else:
        #     raise ValueError(f'Unknown product {product}')

        @jit(nogil=True, cache=True, fastmath=True)
        def wgt_func(wgt, gp, gq):
            return wfunc(gp[0], 0j, 0j, gp[1],
                         gq[0], 0j, 0j, gq[1],
                         wgt[0], wgt[1], wgt[2], wgt[3]).real

        @jit(nogil=True, cache=True, fastmath=True)
        def vis_func(data, wgt, gp, gq):
            return vfunc(gp[0], 0j, 0j, gp[1],
                         gq[0], 0j, 0j, gq[1],
                         wgt[0], wgt[1], wgt[2], wgt[3],
                         data[0], data[1], data[2], data[3])

    elif mode == FULL:
        # wparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3)
        # vparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3, v0, v1, v2, v3)
        # if product == 'I':
        #     wfunc = lambdify(wparams, M[0, 0], 'numpy')
        #     vfunc = lambdify(vparams, V[0], 'numpy')

        # elif product == 'Q':
        #     wfunc = lambdify(wparams, M[1, 1], 'numpy')
        #     vfunc = lambdify(vparams, V[1], 'numpy')

        # elif product == 'U':
        #     wfunc = lambdify(wparams, M[2, 2], 'numpy')
        #     vfunc = lambdify(vparams, V[2], 'numpy')

        # elif product == 'V':
        #     wfunc = lambdify(wparams, M[3, 3], 'numpy')
        #     vfunc = lambdify(vparams, V[3], 'numpy')
        # else:
        #     raise ValueError(f'Unknown product {product}')

        @jit(nogil=True, cache=True, fastmath=True)
        def wgt_func(wgt, gp, gq):
            return wfunc(gp[0, 0], gp[1, 0], gp[0, 1], gp[1, 1],
                         gq[0, 0], gq[1, 0], gq[0, 1], gq[1, 1],
                         wgt[0], wgt[1], wgt[2], wgt[3]).real

        @jit(nogil=True, cache=True, fastmath=True)
        def vis_func(data, wgt, gp, gq):
            return vfunc(gp[0, 0], gp[1, 0], gp[0, 1], gp[1, 1],
                         gq[0, 0], gq[1, 0], gq[0, 1], gq[1, 1],
                         wgt[0], wgt[1], wgt[2], wgt[3],
                         data[0], data[1], data[2], data[3])

    return vfunc, wfunc


