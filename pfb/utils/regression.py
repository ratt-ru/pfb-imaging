import numpy as np
import numba
from numba import njit
from math import factorial
from scipy.optimize import fmin_l_bfgs_b as fmin
from scipy.special import polygamma
from scipy import linalg
from time import time

def mattern52(xx, sigmaf, l):
    return sigmaf**2*np.exp(-np.sqrt(5)*xx/l)*(1 + np.sqrt(5)*xx/l +
                                               5*xx**2/(3*l**2))


def abs_diff(x, xp):
    try:
        N, D = x.shape
        Np, D = xp.shape
    except Exception:
        N = x.size
        D = 1
        Np = xp.size
        x = np.reshape(x, (N, D))
        xp = np.reshape(xp, (Np, D))
    xD = np.zeros([D, N, Np])
    xpD = np.zeros([D, N, Np])
    for i in range(D):
        xD[i] = np.tile(x[:, i], (Np, 1)).T
        xpD[i] = np.tile(xp[:, i], (N, 1))
    return np.linalg.norm(xD - xpD, axis=0)


@njit(nogil=True, cache=True, inline='always')
def diag_dot(A, B):
    N = A.shape[0]
    C = np.zeros(N)
    for i in range(N):
        for j in range(N):
            C[i] += A[i, j] * B[j, i]
    return C


# @jit(nopython=True, nogil=True, cache=True)
def dZdtheta(theta, xx, y, Sigma):
    '''
    Return log-marginal likelihood and derivs
    '''
    N = xx.shape[0]
    sigmaf = theta[0]
    l = theta[1]
    sigman = theta[2]

    # first the negloglik
    K = mattern52(xx, sigmaf, l)
    Ky = K + np.diag(Sigma) * sigman**2
    # with numba.objmode: # args?
    u, s, v = np.linalg.svd(Ky, hermitian=True)
    logdetK = np.sum(np.log(s))
    Kyinv = u.dot(v/s.reshape(N, 1))
    alpha = Kyinv.dot(y)
    Z = (np.vdot(y, alpha) + logdetK)/2

    # # derivs
    # dZ = np.zeros(theta.size)
    # alpha = alpha.reshape(N, 1)
    # aaT = Kyinv - alpha.dot(alpha.T)

    # # deriv wrt sigmaf
    # dK = 2 * K / sigmaf
    # dZ[0] = np.sum(diag_dot(aaT, dK))/2

    # # deriv wrt l
    # dK = xx * K / l ** 3
    # dZ[1] = np.sum(diag_dot(aaT, dK))/2

    # # deriv wrt sigman
    # dK = np.diag(2*sigman*Sigma)
    # dZ[2] = np.sum(diag_dot(aaT, dK))/2

    return Z  #, dZ

def meanf(xx, xxp, y, Sigma, theta):
    K = mattern52(xx, theta[0], theta[1])
    Kp = mattern52(xxp, theta[0], theta[1])
    Ky = K + np.diag(Sigma) * theta[2]**2
    Kinvy = np.linalg.solve(Ky, y)
    return Kp @ Kinvy

def meancovf(xx, xxp, xxpp, y, Sigma, theta):
    N = xx.shape[0]
    K = mattern52(xx, theta[0], theta[1])
    Kp = mattern52(xxp, theta[0], theta[1])
    Kpp = mattern52(xxpp, theta[0], theta[1])
    Ky = K + np.diag(Sigma) * theta[2]**2
    u, s, v = np.linalg.svd(Ky, hermitian=True)
    Kyinv = u.dot(v/s.reshape(N, 1))
    return Kp.dot(Kyinv.dot(y)), np.diag(Kpp - Kp.T.dot(Kyinv.dot(Kp)))

def gpr(y, x, w, xp, theta0=None, nu=3.0, niter=5):
    # drop entries with zero weights
    idxn0 = w!=0
    x = x[idxn0]
    y = y[idxn0]
    w = w[idxn0]

    N = x.size

    # get matrix of differences
    XX = abs_diff(x, x)
    XXp = abs_diff(x, xp)
    XXpp = abs_diff(xp, xp)

    if theta0 is None:
        theta0 = np.zeros(3)
        theta0[0] = np.std(y)
        theta0[1] = 0.5
        theta0[2] = 1

    # get initial hypers assuming weight scaling
    w = np.ones(N)/np.var(y)
    theta, fval, dinfo = fmin(dZdtheta, theta0, args=(XX, y, 1/w), approx_grad=True,
                              bounds=((1e-5, None), (1e-3, None), (1e-5, 100)))

    mu = meanf(XX, XX, y, 1/w, theta)
    # print(theta)
    # return meancovf(XX, XXp, XXpp, y, 1/w, theta)

    res = y - mu
    for k in range(niter):
        ressq = res**2/theta[-1]**2

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get initial hypers assuming weight scaling
        theta, fval, dinfo = fmin(dZdtheta, theta, args=(XX, y, 1/eta), approx_grad=True,
                                bounds=((1e-5, None), (1e-3, None), (1e-3, 100)))

        if k == niter - 1:
            print(nu, theta)
            return meancovf(XX, XXp, XXpp, y, 1/eta, theta)
        else:
            mu = meanf(XX, XX, y, 1/eta, theta)

        res = y - mu

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((2.0, None),))


def nufunc(nu, meaneta, meanlogeta):
    const = 1 + meanlogeta - meaneta
    val = polygamma(0, nu/2) - np.log(nu/2) - const
    return val*val


# @numba.njit(fastmath=True, inline='always')
def Afunc(delta, m):
    phi = delta**np.arange(m)/list(map(factorial, np.arange(m)))
    A = np.zeros((m,m), dtype=np.float64)
    A[0, :] = phi
    for i in range(1, m):
        A[i, i:] = phi[0:-i]
    return A


# @numba.njit(fastmath=True, inline='always')
def Qfunc(q, delta, m):
    Q = np.zeros((m, m), dtype=np.float64)
    for j in range(1, m+1):
        for k in range(1, m+1):
            tmp = ((m-j) + (m-k) + 1)
            Q[j-1, k-1] = q*delta**tmp/(tmp*factorial(m-j)*factorial(m-k))

    return Q


# @numba.njit
def evidence(theta, y, x, H, Rinv):
    m0 = theta[0]
    dm0 = theta[1]
    P0 = theta[2]
    dP0 = theta[3]
    sigmaf = theta[4]
    sigman = theta[5]
    N = x.size
    delta = x[1:] - x[0:-1]
    M = 2  # cubic spline
    m = np.zeros((M, N), dtype=np.float64)
    P = np.zeros((M, M, N), dtype=np.float64)
    m[0, 0] = m0
    m[1, 0] = dm0
    P[0, 0, 0] = P0
    P[1, 1, 0] = dP0

    Z = 0
    w = Rinv / sigman**2
    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[:, k-1]
        Pp = A @ P[:, :, k-1] @ A.T + Q

        if w[k]:

            v = y[k] - H @ mp
            # Use WMI to write inverse ito weights (not variance)
            Ppinv = np.linalg.inv(Pp)
            tmp = Ppinv + H.T @ (w[k] * H)
            tmpinv = np.linalg.inv(tmp)
            Sinv = w[k] - w[k] * H @ tmpinv @ (H.T * w[k])

            if Sinv <= 0:
                import pdb; pdb.set_trace()
            Z += 0.5*np.log(2*np.pi/Sinv) + 0.5*v*v*Sinv

            K = Pp @ H.T @ Sinv
            m[:, k] = mp + K @ v
            P[:, :, k] = Pp - K @ H @ Pp
        else:
            m[:, k] = mp
            P[:, :, k] = Pp

    return Z


# @numba.njit
def Kfilter(m0, P0, x, y, H, Rinv, sigmaf):
    N = x.size
    delta = x[1:] - x[0:-1]
    M = m0.size
    m = np.zeros((M, N), dtype=np.float64)
    P = np.zeros((M, M, N), dtype=np.float64)
    m[:, 0] = m0
    P[:, :, 0] = P0

    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[:, k-1]
        Pp = A @ P[:, :, k-1] @ A.T + Q

        if Rinv[k]:
            v = y[k] - H @ mp

            # Use WMI to write inverse ito weights (not variance)
            Ppinv = np.linalg.inv(Pp)
            tmp = Ppinv + H.T @ (Rinv[k] * H)
            tmpinv = np.linalg.inv(tmp)
            Sinv = Rinv[k] - Rinv[k] * H @ tmpinv @ (H.T * Rinv[k])
            if Sinv <= 0:
                import pdb; pdb.set_trace()

            K = Pp @ H.T @ Sinv

            m[:, k] = mp + K @ v
            P[:, :, k] = Pp - K @ H @ Pp
        else:
            m[:, k] = mp
            P[:, :, k] = Pp

    return m, P


# @numba.njit
def RTSsmoother(m, P, x, sigmaf):
    K = x.size
    delta = np.zeros(K)
    delta[0] = 100
    delta = x[1:] - x[0:-1]
    M = m[:, 0].size
    ms = np.zeros((M, K), dtype=np.float64)
    Ps = np.zeros((M, M, K), dtype=np.float64)
    ms[:, K-1] = m[:, K-1]
    Ps[:, :, K-1] = P[:, :, K-1]

    for k in range(K-2, -1, -1):
        A = Afunc(delta[k], M)
        Q = Qfunc(sigmaf**2, delta[k], M)

        mp = A @ m[:, k]
        Pp = A @ P[:, :, k] @ A.T + Q
        Pinv = np.linalg.inv(Pp)

        G = P[:, :, k] @ A.T @ Pinv

        ms[:, k] = m[:, k] + G @ (ms[:, k+1] - mp)
        Ps[:, :, k] = P[:, :, k] + G @ (Ps[:, :, k+1] - Pp) @ G.T

    return ms, Ps


def kanterp(x, y, w, niter=5, nu0=2):
    N = x.size
    M = 2  # cubic smoothing spline
    if y[0] == 0:
        theta = np.array([1e-5, 0, 1.0, 0.1, np.sqrt(N), 1.0])
    else:
        theta = np.array([y[0], 0, 1.0, 0.1, np.sqrt(N), 1.0])

    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1

    bnds = ((None, None),
            (None, None),
            (1e-5, None),
            (1e-5, None),
            (1e-5, 2*N),
            (1, 50))

    # theta, fval, dinfo = fmin(evidence, theta, args=(y, x, w),
    #                           approx_grad=True,
    #                           bounds=bnds)



    m0 = np.array([theta[0], theta[1]])
    P0 = np.array([[theta[2], 0], [0, theta[3]]])
    sigmaf = theta[4]
    sigman = theta[5]
    m, P = Kfilter(m0, P0, x, y, H, w/sigman**2, sigmaf)
    ms, Ps = RTSsmoother(m, P, x, sigmaf)

    # print(Z, theta, dinfo)

    # initial residual
    res = y - ms[0]
    nu = nu0
    for k in range(niter):
        ressq = res**2/sigman**2

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get smoothed signal
        theta, fval, dinfo = fmin(evidence, theta, args=(y, x, H, eta),
                                  approx_grad=True,
                                  bounds=bnds)


        m0 = np.array([theta[0], theta[1]])
        P0 = np.array([[theta[2], 0], [0, theta[3]]])
        sigmaf = theta[4]
        sigman = theta[5]
        m, P = Kfilter(m0, P0, x, y, H, eta/sigman**2, sigmaf)
        ms, Ps = RTSsmoother(m, P, x, sigmaf)


        print(fval, theta, dinfo)

        if k == niter - 1:
            return ms, Ps

        # residual
        res = y - ms[0]

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((2.0, None),))


@numba.njit(fastmath=True, cache=True)
def evidence_fast(theta, y, x, Rinv):
    m0 = theta[0]
    m1 = theta[1]
    p00 = theta[2]
    p11 = theta[3]
    p01 = 0
    sigmaf = theta[4]
    sigman = theta[5]
    N = x.size
    delta = x[1:] - x[0:-1]

    Z = 0
    w = Rinv / sigman**2
    q = sigmaf**2
    a00 = a11 = 1
    for k in range(1, N):
        # This can be avoided if the data are on a regular grid
        dlta = delta[k-1]
        a01 = dlta
        qd = q*dlta
        qdd = qd * dlta
        q00 = qdd * dlta / 3
        q01 = qdd/2
        q11 = qd

        mp0 = m0 + dlta * m1
        mp1 = m1
        pp00 = dlta*p01 + dlta*(dlta*p11 + p01) + p00 + q00
        pp01 = dlta*p11 + p01 + q01
        pp11 = p11 + q11

        if w[k]:
            v = y[k] - mp0
            det = pp00 * pp11 - pp01 * pp01

            a2 = pp11/det + w[k]
            b2 = -pp01/det
            c2 = -pp01/det
            d2 = pp00/det
            det2 = a2*d2 - b2 * c2
            Sinv = w[k] - w[k]**2 * d2 / det2

            Z += 0.5*np.log(2*np.pi/Sinv) + 0.5*v*v*Sinv

            m0 = mp0 + pp00 * Sinv * v
            m1 = mp1 + pp01 * Sinv * v
            p00 = -pp00**2*Sinv + pp00
            p01 = -pp00*pp01*Sinv + pp01
            p11 = -pp01*Sinv*pp01 + pp11

        else:
            m0 = mp0
            m1 = mp1
            p00 = pp00
            p01 = pp01
            p11 = pp11

    return Z


@numba.njit
def Kfilter_fast(sigmaf, y, x, Rinv, m0, m1, p00, p01, p11):
    N = x.size
    delta = x[1:] - x[0:-1]
    m = np.zeros((2, N), dtype=np.float64)
    P = np.zeros((2, 2, N), dtype=np.float64)
    m[0, 0] = m0
    m[1, 0] = m1
    P[0, 0, 0] = p00
    P[0, 1, 0] = p01
    P[1, 0, 0] = p01
    P[1, 1, 0] = p11

    q = sigmaf**2
    w = Rinv
    for k in range(1, N):
        # This can be avoided if the data are on a regular grid
        dlta = delta[k-1]
        a01 = dlta
        qd = q*dlta
        qdd = qd * dlta
        q00 = qdd * dlta / 3
        q01 = qdd/2
        q11 = qd

        mp0 = m[0, k-1] + dlta * m[1, k-1]
        mp1 = m[1, k-1]
        pp00 = dlta*P[0, 1, k-1] + dlta*(dlta*P[1, 1, k-1] + P[0, 1, k-1]) + P[0, 0, k-1] + q00
        pp01 = dlta*P[1, 1, k-1] + P[0, 1, k-1] + q01
        pp11 = P[1, 1, k-1] + q11

        if Rinv[k]:
            v = y[k] - mp0
            det = pp00 * pp11 - pp01 * pp01

            a2 = pp11/det + w[k]
            b2 = -pp01/det
            c2 = -pp01/det
            d2 = pp00/det
            det2 = a2*d2 - b2 * c2
            Sinv = w[k] - w[k]**2 * d2 / det2

            m0 = mp0 + pp00 * Sinv * v
            m1 = mp1 + pp01 * Sinv * v
            p00 = -pp00**2*Sinv + pp00
            p01 = -pp00*pp01*Sinv + pp01
            p11 = -pp01*Sinv*pp01 + pp11

            m[0, k] = mp0 + pp00 * Sinv * v
            m[1, k] = mp1 + pp01 * Sinv * v
            P[0, 0, k] = -pp00**2*Sinv + pp00
            P[0, 1, k] = -pp00*pp01*Sinv + pp01
            P[1, 0, k] = P[0, 1, k]
            P[1, 1, k] = -pp01*Sinv*pp01 + pp11
        else:
            m[0, k] = mp0
            m[1, k] = mp1
            P[0, 0, k] = pp00
            P[0, 1, k] = pp01
            P[1, 0, k] = pp01
            P[1, 1, k] = pp11


    return m, P

@numba.njit
def RTSsmoother_fast(m, P, x, sigmaf):
    K = x.size
    delta = np.zeros(K)
    delta = x[1:] - x[0:-1]
    ms = np.zeros((2, K), dtype=np.float64)
    Ps = np.zeros((2, 2, K), dtype=np.float64)
    ms[:, K-1] = m[:, K-1]
    Ps[:, :, K-1] = P[:, :, K-1]

    a00 = a11 = 1
    q = sigmaf**2
    for k in range(K-2, -1, -1):
        dlta = delta[k]

        a01 = dlta
        qd = q*dlta
        qdd = qd * dlta
        q00 = qdd * dlta / 3
        q01 = qdd/2
        q11 = qd

        mp0 = m[0, k] + dlta * m[1, k]
        mp1 = m[1, k]
        pp00 = dlta*P[0, 1, k] + dlta*(dlta*P[1, 1, k] + P[0, 1, k]) + P[0, 0, k] + q00
        pp01 = dlta*P[1, 1, k] + P[0, 1, k] + q01
        pp11 = P[1, 1, k] + q11

        # import pdb; pdb.set_trace()

        det = pp00*pp11 - pp01*pp01

        g00 = (-P[0,1,k]*pp01 + pp11*(dlta*P[0, 1, k] + P[0, 0, k]))/det
        g01 = (P[0,1,k]*pp00 - pp01*(dlta*P[0,1,k] + P[0,0,k]))/det
        g10 = (-P[1,1,k]*pp01 + pp11*(dlta*P[1,1,k] + P[0,1,k]))/det
        g11 = (P[1,1,k]*pp00 - pp01*(dlta*P[1,1,k] + P[0,1,k]))/det

        ms[0, k] = m[0, k] + g00 * (ms[0, k+1] - mp0) + g01 * (ms[1, k+1] - mp1)
        ms[1, k] = m[1, k] + g10 * (ms[0, k+1] - mp0) + g11 * (ms[1, k+1] - mp1)

        Ps[0, 0, k] = P[0, 0, k] - g00*(g00*(pp00 - Ps[0, 0, k+1]) + g01*(pp01 - Ps[0, 1, k+1])) \
                                 - g01*(g00*(pp01 - Ps[0, 1, k+1]) + g01*(pp11 - Ps[1, 1, k+1]))
        Ps[0, 1, k] = P[0, 1, k] - g10*(g00*(pp00 - Ps[0, 0, k+1]) + g01*(pp01 - Ps[0, 1, k+1])) \
                                 - g11*(g00*(pp01 - Ps[0, 1, k+1]) + g01*(pp11 - Ps[1, 1, k+1]))
        Ps[1, 0, k] = P[1, 0, k] - g00*(g10*(pp00 - Ps[0, 0, k+1]) + g11*(pp01 - Ps[0, 1, k+1])) \
                                 - g01*(g10*(pp01 - Ps[0, 1, k+1]) + g11*(pp11 - Ps[1, 1, k+1]))
        Ps[1, 1, k] = P[1, 1, k] - g10*(g10*(pp00 - Ps[0, 0, k+1]) + g11*(pp01 - Ps[0, 1, k+1])) \
                                 - g11*(g10*(pp01 - Ps[0, 1, k+1]) + g11*(pp11 - Ps[1, 1, k+1]))

    return ms, Ps


def kanterp2(x, y, w, niter=5, nu0=2):
    N = x.size
    M = 2  # cubic smoothing spline
    theta = np.array([y[0], 0, 1.0, 1.0, np.sqrt(N), 1.0])

    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1

    bnds = ((1e-5, None),
            (1e-5, None),
            (1e-5, None),
            (1e-5, None),
            (1e-5, 2*N),
            (1, 50))

    # theta, fval, dinfo = fmin(evidence2, theta, args=(y, x, H, w),
    #                           approx_grad=True,
    #                           bounds=bnds)

    sigmaf = theta[4]
    sigman = theta[5]
    # sigmaf, y, x, Rinv, m0, m1, p00, p01, p11, sigman
    m, P = Kfilter_fast(sigmaf, y, x, w/sigman**2, theta[0], theta[1], theta[2], 0, theta[3])
    ms, Ps = RTSsmoother_fast(m, P, x, sigmaf)

    theta = np.array([ms[0, 0], ms[1, 0], 1.0, 1.0, np.sqrt(N), 1.0])

    # initial residual
    res = y - ms[0]
    nu = nu0
    for k in range(niter):
        ressq = res**2/sigman**2

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get smoothed signal
        theta, fval, dinfo = fmin(evidence_fast, theta, args=(y, x, eta),
                                  approx_grad=True,
                                  bounds=bnds)

        m0 = np.array([theta[0], theta[1]])
        P0 = np.array([[theta[2], 0], [0, theta[3]]])
        sigmaf = theta[4]
        sigman = theta[5]
        m, P = Kfilter_fast(sigmaf, y, x, eta/sigman**2, theta[0], theta[1], theta[2], 0, theta[3])
        ms, Ps = RTSsmoother_fast(m, P, x, sigmaf)


        print(fval, theta, dinfo)

        if k == niter - 1:
            return ms, Ps

        # residual
        res = y - ms[0]

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((2.0, None),))



@numba.njit
def evidence3(theta, y, x, Rinv, m0, m1, p00, p01, p11, sigman):
    '''
    Same as evidence fast but theta is just sigmaf
    '''
    N = x.size
    delta = x[1:] - x[0:-1]

    sigmaf = theta[0]

    print(sigmaf, m0, m1, p00, p01, p11, sigman)

    Z = 0
    w = Rinv / sigman**2
    q = sigmaf**2
    a00 = a11 = 1.0
    # import pdb; pdb.set_trace()
    for k in range(1, N):
        # This can be avoided if the data are on a regular grid
        dlta = delta[k-1]
        a01 = dlta
        qd = q*dlta
        qdd = qd * dlta
        q00 = qdd * dlta / 3
        q01 = qdd/2
        q11 = qd

        mp0 = m0 + dlta * m1
        mp1 = m1
        pp00 = dlta*p01 + dlta*(dlta*p11 + p01) + p00 + q00
        pp01 = dlta*p11 + p01 + q01
        pp11 = p11 + q11

        if w[k]:
            v = y[k] - mp0
            det = pp00 * pp11 - pp01 * pp01

            a2 = pp11/det + w[k]
            b2 = -pp01/det
            c2 = -pp01/det
            d2 = pp00/det
            det2 = a2*d2 - b2 * c2
            Sinv = w[k] - w[k]**2 * d2 / det2

            Z += 0.5*np.log(2*np.pi/Sinv) + 0.5*v*v*Sinv

            m0 = mp0 + pp00 * Sinv * v
            m1 = mp1 + pp01 * Sinv * v
            p00 = -pp00**2*Sinv + pp00
            p01 = -pp00*pp01*Sinv + pp01
            p11 = -pp01*Sinv*pp01 + pp11

        else:
            m0 = mp0
            m1 = mp1
            p00 = pp00
            p01 = pp01
            p11 = pp11

    return Z


def kanterp3(x, y, w, niter=5, nu0=2, sigmaf0=None, sigman0=1, verbose=0, window=10):
    N = x.size
    M = 2  # cubic smoothing spline
    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1



    if sigmaf0 is None:
        sigmaf = np.sqrt(N)
    else:
        sigmaf = sigmaf0

    bnds = ((0.1*sigmaf, 10*sigmaf),)
    I = y != 0
    m0 = np.median(y[I][0:window])
    x0 = np.median(x[I][0:window])
    mplus = np.median(y[I][window:2*window])
    xplus = np.median(x[I][window:2*window])
    dm0 = (mplus-m0)/(xplus - x0)
    P0 = np.mean((y[I][0:window] - m0)**2)
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    w /= sigman0**2
    m, P = Kfilter_fast(sigmaf, y, x, w, m0, dm0, P0, 0, P0)
    ms, Ps = RTSsmoother_fast(m, P, x, sigmaf)

    # initial residual
    res = y - ms[0]
    sigman = np.sqrt(np.mean(res**2*w))
    nu = nu0
    for k in range(niter):
        ressq = res**2/sigman**2

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get smoothed signal
        sigmaf, fval, dinfo = fmin(evidence3, np.array(sigmaf),
                                   args=(y, x, eta, ms[0, 0], ms[1, 0], Ps[0, 0, 0], Ps[0, 1, 0], Ps[1, 1, 0], sigman),
                                   approx_grad=True,
                                   bounds=bnds)



        m0 = ms[:, 0]
        P0 = Ps[:, :, 0]
        m, P = Kfilter_fast(sigmaf[0], y, x, eta/sigman**2, ms[0, 0], ms[1, 0], Ps[0, 0, 0], Ps[0, 1, 0], Ps[1, 1, 0])
        ms, Ps = RTSsmoother_fast(m, P, x, sigmaf[0])


        if verbose:
            print(f"Z={fval}, sigmaf={sigmaf[0]}, sigman={sigman}, warning={dinfo['warnflag']}")

        if k == niter - 1:
            return ms, Ps

        # residual
        res = y - ms[0]

        sigman = np.sqrt(np.mean(res**2*eta))

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((2.0, None),))

def func(x):
    return 10*np.sin(20*x**2)*np.exp(-x**2/0.25) + np.exp(x)


if __name__=='__main__':
    # np.random.seed(420)

    N = 1024
    x = np.sort(np.random.random(N))
    xp = np.linspace(0, 1, 100)
    f = func(x)
    ft = func(xp)
    # sigman = np.ones(N)
    sigman = 1 + np.exp(np.random.randn(N))/10
    n = sigman*np.random.randn(N)
    w = 1/sigman**2
    y = f + n

    # add outliers
    # for i in range(int(0.1*N)):
    #     idx = np.random.randint(0, N)
    #     y[idx] += 10 * np.random.randn()
    #     w[idx] = 0.25

    iplot = np.where(w!=0)

    # theta = np.array((6.76914626e-01, 4.04403714e+00, 1.00000000e-05, 3.80339807e+00, 8.19198868e+03, 2.20817999e+00))
    # H = np.zeros((1, 2), dtype=np.float64)
    # H[0, 0] = 1

    # # theta, y, x, H, Rinv
    # ti = time()
    # Z1 = evidence_slow(theta, y, x, H, w)
    # print(time() - ti)

    # ti = time()
    # Z2 = evidence_fast(theta, y, x, w)
    # print(time() - ti)

    # print(Z1 - Z2)

    # quit()
    # ti = time()
    # evidence(theta, y, x, w)
    # print(time() - ti)

    # quit()

    # print(evidence(theta, y, x, w) - evidence2(theta, y, x, H, w))

    # quit()

    # from time import time
    # ti = time()
    # ms, Ps = kanterp(x, y, w, 3, nu0=2)
    # ms, Ps = kanterp2(x, y, w, 10, nu0=2)
    # ms, Ps = kanterp3(x, y, w, 3, nu0=2)
    ti = time()
    ms, Ps, wnew = kanterp3(x, y, w, 100, nu0=2)
    print(time() - ti)
    mu = ms[0, :]
    P = Ps[0, 0, :]
    # # print(time() - ti)

    from scipy.stats import kurtosis, skew

    wres = (y - mu)*np.sqrt(wnew)

    print(skew(wnew), kurtosis(wnew))
    print(skew(n*w), kurtosis(n*w))

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.hist(wres, bins='auto')

    plt.figure(3)
    plt.fill_between(x, mu - np.sqrt(P), mu + np.sqrt(P), color='gray', alpha=0.25)
    plt.plot(x, mu, 'b', alpha=0.5)
    plt.plot(xp, ft, 'k', alpha=0.5)
    plt.errorbar(x[iplot], y[iplot], sigman[iplot], fmt='xr', alpha=0.25)

    plt.show()

    print("Done")






