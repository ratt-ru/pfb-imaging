import numpy as np
import sympy as sm
from sympy.physics.quantum import TensorProduct
from sympy.utilities.lambdify import lambdify
from numba import njit


def stokes_funcs(data, jones, product, pol, nc):
    # set up symbolic expressions
    gp00, gp10, gp01, gp11 = sm.symbols("gp00 gp10 gp01 gp11", real=False)
    gq00, gq10, gq01, gq11 = sm.symbols("gq00 gq10 gq01 gq11", real=False)
    w0, w1, w2, w3 = sm.symbols("W0 W1 W2 W3", real=True)
    v00, v10, v01, v11 = sm.symbols("v00 v10 v01 v11", real=False)

    # Jones matrices
    Gp = sm.Matrix([[gp00, gp01],[gp10, gp11]])
    Gq = sm.Matrix([[gq00, gq01],[gq10, gq11]])

    # Mueller matrix (row major form)
    Mpq = TensorProduct(Gp, Gq.conjugate())
    Mpqinv = TensorProduct(Gp.inv(), Gq.conjugate().inv())

    # inverse noise covariance
    Sinv = sm.Matrix([[w0, 0, 0, 0],
                      [0, w1, 0, 0],
                      [0, 0, w2, 0],
                      [0, 0, 0, w3]])
    S = Sinv.inv()

    # visibilities
    Vpq = sm.Matrix([[v00], [v01], [v10], [v11]])

    # Full Stokes to corr operator
    # Is this the only difference between linear and circular pol?
    # What about paralactic angle rotation?
    if pol.literal_value == 'linear':
        T = sm.Matrix([[1.0, 1.0, 0, 0],
                       [0, 0, 1.0, 1.0j],
                       [0, 0, 1.0, -1.0j],
                       [1.0, -1.0, 0, 0]])
    elif pol.literal_value == 'circular':
        T = sm.Matrix([[1.0, 0, 0, 1.0],
                       [0, 1.0, 1.0j, 0],
                       [0, 1.0, -1.0j, 0],
                       [1.0, 0, 0, -1.0]])
    Tinv = T.inv()

    # Full Stokes weights
    W = T.H * Mpq.H * Sinv * Mpq * T
    Winv = Tinv * Mpqinv * S * Mpqinv.H * Tinv.H

    # Full Stokes coherencies
    C = Winv * (T.H * (Mpq.H * (Sinv * Vpq)))
    # C = T.H * (Mpq.H * (Sinv * Vpq))

    if product.literal_value == 'I':
        i = 0
    elif product.literal_value == 'Q':
        i = 1
    elif product.literal_value == 'U':
        i = 2
    elif product.literal_value == 'V':
        i = 3
    else:
        raise ValueError(f"Unknown polarisation product {product}")

    if jones.ndim == 6:  # Full mode
        Wsymb = lambdify((gp00, gp01, gp10, gp11,
                          gq00, gq01, gq10, gq11,
                          w0, w1, w2, w3),
                          sm.simplify(sm.expand(W[i,i])))
        Wjfn = njit(nogil=True, inline='always')(Wsymb)


        Dsymb = lambdify((gp00, gp01, gp10, gp11,
                          gq00, gq01, gq10, gq11,
                          w0, w1, w2, w3,
                          v00, v01, v10, v11),
                          sm.simplify(sm.expand(C[i])))
        Djfn = njit(nogil=True, inline='always')(Dsymb)

        @njit(nogil=True, inline='always')
        def wfunc(gp, gq, W):
            gp00 = gp[0,0]
            gp01 = gp[0,1]
            gp10 = gp[1,0]
            gp11 = gp[1,1]
            gq00 = gq[0,0]
            gq01 = gq[0,1]
            gq10 = gq[1,0]
            gq11 = gq[1,1]
            W00 = W[0]
            W01 = W[1]
            W10 = W[2]
            W11 = W[3]
            return Wjfn(gp00, gp01, gp10, gp11,
                        gq00, gq01, gq10, gq11,
                        W00, W01, W10, W11).real

        @njit(nogil=True, inline='always')
        def vfunc(gp, gq, W, V):
            gp00 = gp[0,0]
            gp01 = gp[0,1]
            gp10 = gp[1,0]
            gp11 = gp[1,1]
            gq00 = gq[0,0]
            gq01 = gq[0,1]
            gq10 = gq[1,0]
            gq11 = gq[1,1]
            W00 = W[0]
            W01 = W[1]
            W10 = W[2]
            W11 = W[3]
            V00 = V[0]
            V01 = V[1]
            V10 = V[2]
            V11 = V[3]
            return Djfn(gp00, gp01, gp10, gp11,
                        gq00, gq01, gq10, gq11,
                        W00, W01, W10, W11,
                        V00, V01, V10, V11)

    elif jones.ndim == 5:  # DIAG mode
        W = W.subs(gp10, 0)
        W = W.subs(gp01, 0)
        W = W.subs(gq10, 0)
        W = W.subs(gq01, 0)
        C = C.subs(gp10, 0)
        C = C.subs(gp01, 0)
        C = C.subs(gq10, 0)
        C = C.subs(gq01, 0)

        Wsymb = lambdify((gp00, gp11,
                          gq00, gq11,
                          w0, w1, w2, w3),
                          sm.simplify(sm.expand(W[i,i])))
        Wjfn = njit(nogil=True, inline='always')(Wsymb)


        Dsymb = lambdify((gp00, gp11,
                          gq00, gq11,
                          w0, w1, w2, w3,
                          v00, v01, v10, v11),
                          sm.simplify(sm.expand(C[i])))
        Djfn = njit(nogil=True, inline='always')(Dsymb)

        if nc.literal_value == '4':
            @njit(nogil=True, inline='always')
            def wfunc(gp, gq, W):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                W00 = W[0]
                W01 = W[1]
                W10 = W[2]
                W11 = W[3]
                return Wjfn(gp00, gp11,
                            gq00, gq11,
                            W00, W01, W10, W11).real

            @njit(nogil=True, inline='always')
            def vfunc(gp, gq, W, V):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                W00 = W[0]
                W01 = W[1]
                W10 = W[2]
                W11 = W[3]
                V00 = V[0]
                V01 = V[1]
                V10 = V[2]
                V11 = V[3]
                return Djfn(gp00, gp11,
                            gq00, gq11,
                            W00, W01, W10, W11,
                            V00, V01, V10, V11)
        elif nc.literal_value == '2':
            @njit(nogil=True, inline='always')
            def wfunc(gp, gq, W):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                W00 = W[0]
                W01 = 1.0
                W10 = 1.0
                W11 = W[-1]
                return Wjfn(gp00, gp11,
                            gq00, gq11,
                            W00, W01, W10, W11).real

            @njit(nogil=True, inline='always')
            def vfunc(gp, gq, W, V):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                W00 = W[0]
                W01 = 1.0
                W10 = 1.0
                W11 = W[-1]
                V00 = V[0]
                V01 = 0j
                V10 = 0j
                V11 = V[-1]
                return Djfn(gp00, gp11,
                            gq00, gq11,
                            W00, W01, W10, W11,
                            V00, V01, V10, V11)
        else:
            raise ValueError(f"Selected product is only available from 2 or 4"
                             f"correlation data while you have ncorr={nc}.")


    else:
        raise ValueError(f"Jones term has incorrect number of dimensions")

    # import inspect
    # print(inspect.getsource(Djfn))
    # print(inspect.getsource(Wjfn))

    # quit()

    return vfunc, wfunc
