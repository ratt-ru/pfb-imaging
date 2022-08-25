import numpy as np
from numba import njit
from math import factorial
from scipy.optimize import fmin_l_bfgs_b as fmin
from scipy.special import polygamma


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

    # derivs
    dZ = np.zeros(theta.size)
    alpha = alpha.reshape(N, 1)
    aaT = Kyinv - alpha.dot(alpha.T)

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
                        bounds=((1e-2, None),))



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

    Z = 0

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
        try:
            Ppinv = np.linalg.inv(Pp)
        except:
            Ppinv = np.linalg.pinv(Pp)
            import pdb; pdb.set_trace()
        tmp = Ppinv + H.T @ (Rinv[k] * H)
        tmpinv = np.linalg.inv(tmp)
        Sinv = Rinv[k] - Rinv[k] * H @ tmpinv @ (H.T * Rinv[k])


        Z += 0.5*np.log(2*np.pi/Sinv) + v*S*v

        K = Pp @ H.T @ Sinv

        m[:, k] = mp + K @ v
        if np.any(np.isnan(m)):
            import pdb; pdb.set_trace()
        P[:, :, k] = Pp - K @ H @ Pp

    return m, P, Z

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
        try:
            Pinv = np.linalg.inv(Pp)
        except:
            Pinv = np.linalg.pinv(Pp)
            import pdb; pdb.set_trace()

        G = P[:, :, k] @ A.T @ Pinv

        Gout[:, :, k] = G

        ms[:, k] = m[:, k] + G @ (ms[:, k+1] - mp)
        if np.any(np.isnan(ms)):
            import pdb; pdb.set_trace()
        Ps[:, :, k] = P[:, :, k] + G @ (Ps[:, :, k+1] - Pp) @ G.T

    return ms, Ps, Gout

def energy(sigmaf, m0, P0, x, y, H, Rinv):
    m, P, Z = Kfilter(m0, P0, x, y, H, Rinv, sigmaf)
    ms, Ps, G = RTSsmoother(m, P, x, sigmaf)
    chi2_dof = np.mean((y - ms[0, :])**2 * Rinv)
    return (chi2_dof - 1)**2

def nufunc(nu, meaneta, meanlogeta):
    const = 1 + meanlogeta - meaneta
    val = polygamma(0, nu/2) - np.log(nu/2) - const
    return val*val

def kanterp(x, y, w, niter, nu0=2):
    N = x.size
    M = 2  # cubic smoothing spline
    m0 = np.zeros((M), dtype=np.float64)
    m0[0] = y[0]

    sigma0 = np.sqrt(N)
    w = np.ones(N)/np.var(y)

    P0 = np.eye(M) * np.var(y) / 100
    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1

    sigma = np.sqrt(N)
    w = np.ones(N)/np.var(y)
    # sigma, fval, dinfo = fmin(energy, sigma, args=(m0, P0, x, y, H, w),
    #                             approx_grad=True,
    #                             bounds=((1e-5, None),))

    m, P, Z = Kfilter(m0, P0, x, y, H, w, sigma)
    ms, Ps, G = RTSsmoother(m, P, x, sigma)

    print(sigma, fval, dinfo)

    return ms, Ps

    m0 = ms[:, 0]
    P0 = Ps[:, :, 0]

    # initial residual
    res = y - ms[0]
    lam = 1.0/np.mean(res*res)
    nu = nu0
    for k in range(niter):
        ressq = res**2*lam

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get smoothed signal
        # sigma, fval, dinfo = fmin(energy, sigma0, args=(m0, P0, x, y, H, lam*eta),
        #                         approx_grad=True,
        #                         bounds=((1e-5, None),))

        m, P = Kfilter(m0, P0, x, y, H, lam*eta, sigma)
        ms, Ps, G = RTSsmoother(m, P, x, sigma)

        if k == niter - 1:
            print(nu, sigma)
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

# N = 256
# x = np.sort(np.random.random(N))
# xp = np.linspace(0, 1, 100)
# f = func(x)
# ft = func(xp)
# sigman = np.ones(N)
# # sigman = np.exp(np.random.randn(N)) #/10000
# n = sigman*np.random.randn(N)
# w = 1/sigman**2
# y = f + n

# # add outliers
# for i in range(int(0.1*N)):
#     idx = np.random.randint(0, N)
#     y[idx] += 10 * np.random.randn()
#     # w[idx] = 0
#     # sigman[idx] = 1e7

# iplot = np.where(w!=0)

# # ms, Ps = kanterp(x, y, w, 3, nu0=5)
# # mu = ms[0, :]
# # P = Ps[0, 0, :]

# mu, P = gpr(y, x, w, x)

# import matplotlib.pyplot as plt

# plt.figure(3)
# plt.fill_between(x, mu - np.sqrt(P), mu + np.sqrt(P), color='gray', alpha=0.25)
# plt.plot(x, mu, 'b', alpha=0.5)
# plt.plot(xp, ft, 'k', alpha=0.5)
# plt.errorbar(x[iplot], y[iplot], sigman[iplot], fmt='xr', alpha=0.25)

# plt.show()

# print("Done")








