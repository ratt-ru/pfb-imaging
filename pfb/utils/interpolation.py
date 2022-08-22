import numpy as np
from numba import njit
from math import factorial
from scipy.optimize import fmin_l_bfgs_b as fmin
from scipy.special import polygamma


def Afunc(delta, m):
    phi = delta**np.arange(m)/list(map(factorial, np.arange(m)))
    A = np.zeros((m,m), dtype=np.float64)
    A[0, :] = phi
    for i in range(1, m):
        A[i, i:] = phi[0:-i]
    return A


def Qfunc(q, delta, m):
    Q = np.zeros((m, m), dtype=np.float64)
    for j in range(1, m+1):
        for k in range(1, m+1):
            tmp = ((m-j) + (m-k) + 1)
            Q[j-1, k-1] = q*delta**tmp/(tmp*factorial(m-j)*factorial(m-k))

    return Q


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

        v = y[k] - H @ mp

        # S = H @ Pp @ H.T + R[k]
        # Sinv = np.linalg.inv(S)

        # import pdb; pdb.set_trace()

        # Use WMI to write inverse ito weights (not variance)
        Ppinv = np.linalg.inv(Pp)
        tmp = Ppinv + H.T @ (Rinv[k] * H)
        tmpinv = np.linalg.inv(tmp)
        Sinv = Rinv[k] - Rinv[k] * H @ tmpinv @ (H.T * Rinv[k])

        K = Pp @ H.T @ Sinv

        m[:, k] = mp + K @ v
        P[:, :, k] = Pp - K @ H @ Pp

    return m, P

def RTSsmoother(m, P, x, sigmaf):
    K = x.size
    delta = np.zeros(K)
    delta[0] = 100
    delta = x[1:] - x[0:-1]
    M = m[:, 0].size
    ms = np.zeros((M, K), dtype=np.float64)
    Ps = np.zeros((M, M, K), dtype=np.float64)
    Gout = np.zeros((M, M, K), dtype=np.float64)
    ms[:, K-1] = m[:, K-1]
    Ps[:, :, K-1] = P[:, :, K-1]

    for k in range(K-2, -1, -1):
        A = Afunc(delta[k], M)
        Q = Qfunc(sigmaf**2, delta[k], M)

        mp = A @ m[:, k]
        Pp = A @ P[:, :, k] @ A.T + Q
        Pinv = np.linalg.inv(Pp)

        G = P[:, :, k] @ A.T @ Pinv

        Gout[:, :, k] = G

        ms[:, k] = m[:, k] + G @ (ms[:, k+1] - mp)
        Ps[:, :, k] = P[:, :, k] + G @ (Ps[:, :, k+1] - Pp) @ G.T

    return ms, Ps, Gout

def energy(sigmaf, m0, P0, x, y, H, Rinv):
    m, P = Kfilter(m0, P0, x, y, H, Rinv, sigmaf)
    ms, Ps, G = RTSsmoother(m, P, x, sigmaf)
    chi2_dof = np.mean((y - ms[0, :])**2 * Rinv)
    return (chi2_dof - 1)**2

def nufunc(nu, meaneta, meanlogeta):
    const = 1 + meanlogeta - meaneta
    val = polygamma(0, nu/2) - np.log(nu/2) - const
    return val*val

def kanterp(x, y, w, niter, nu0=5):
    N = x.size
    M = 2  # cubic smoothing spline
    m0 = np.zeros((M), dtype=np.float64)
    m0[0] = y[0]

    P0 = np.eye(M)
    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1

    # start by smoothing signal assuming correct weights
    sigma0 = np.sqrt(N)
    sigma, fval, dinfo = fmin(energy, sigma0, args=(m0, P0, x, y, H, w),
                                approx_grad=True,
                                bounds=((1e-5, None),))

    m, P = Kfilter(m0, P0, x, y, H, w, sigma)
    ms, Ps, G = RTSsmoother(m, P, x, sigma)

    return ms, Ps

    m0 = ms[:, 0]
    P0 = Ps[:, :, 0]

    # initial residual
    res = y - ms[0]
    # w is lam * eta which is just Rinv
    eta = w
    lam = 1
    nu = nu0
    for k in range(niter):
        ressq = res**2*lam

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get smoothed signal
        sigma, fval, dinfo = fmin(energy, sigma0, args=(m0, P0, x, y, H, lam*eta),
                                approx_grad=True,
                                bounds=((1e-5, None),))

        m, P = Kfilter(m0, P0, x, y, H, lam*eta, sigma)
        ms, Ps, G = RTSsmoother(m, P, x, sigma)

        if k == niter - 1:
            return ms, Ps

        # using smoother results as next starting guess
        m0 = ms[:, 0]
        P0 = Ps[:, :, 0]

        # residual
        res = y - ms[0]

        # overall variance factor
        lam = 1/np.mean(res**2*eta)

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((1e-2, None),))


# def func(x):
#     return 10*np.sin(20*x**2)*np.exp(-x**2/0.25) + np.exp(x)

# np.random.seed(420)

# N = 101
# x = np.sort(np.random.random(N))
# xp = np.linspace(0, 1, 100)
# f = func(x)
# ft = func(xp)
# sigman = np.ones(N)  #np.exp(np.random.randn(N)) #/10000
# n = sigman*np.random.randn(N)
# w = 1/sigman**2
# y = f + n

# # add outliers
# for i in range(0):  #int(0.1*N)):
#     idx = np.random.randint(0, N)
#     y[idx] += 10 * np.random.randn()
#     w[idx] = 0
#     sigman[idx] = 1e7

# iplot = np.where(w!=0)

# ms, Ps = kanterp(x, y, w, 1, nu0=5)

# import matplotlib.pyplot as plt

# plt.figure(3)
# plt.fill_between(x, ms[0, :] - np.sqrt(Ps[0, 0, :]), ms[0, :] + np.sqrt(Ps[0, 0, :]), color='gray', alpha=0.25)
# plt.plot(x, ms[0, :], 'b', alpha=0.5)
# plt.plot(xp, ft, 'k', alpha=0.5)
# plt.errorbar(x[iplot], y[iplot], sigman[iplot], fmt='xr', alpha=0.25)

# plt.show()

# print("Done")








