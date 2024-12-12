import numpy as np
from distributed import get_client
from scipy.linalg import norm
import pyscilog
log = pyscilog.get_logger('PM')


def power_method(
        A,
        imsize,
        b0=None,
        tol=1e-5,
        maxit=250,
        verbosity=1,
        report_freq=25):
    if b0 is None:
        b = np.random.randn(*imsize)
        b /= norm(b)
    else:
        b = b0 / norm(b0)
    beta = 1.0
    eps = 1.0
    k = 0
    bp = b.copy()
    while eps > tol and k < maxit:
        b = A(bp)
        bnorm = np.linalg.norm(b)
        betap = beta
        beta = np.vdot(bp, b) / np.vdot(bp, bp)
        b /= bnorm
        # this is a scalar
        eps = np.linalg.norm(beta - betap) / betap
        k += 1
        bp[...] = b[...]

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}", file=log)

    if k == maxit:
        if verbosity:
            print(f"Maximum iterations reached. "
                  f"eps = {eps:.3e}, beta = {beta:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations. "
                  f"beta = {beta:.3e}", file=log)
    return beta, b


def power(A, bp, bnorm, eta):
    bp /= bnorm
    b = A(bp, eta)
    bsumsq = np.sum(b**2)
    beta_num = np.vdot(b, bp)
    beta_den = np.vdot(bp, bp)

    return b, bsumsq, beta_num, beta_den

def sumsq(b):
    return np.sum(b**2)

def bnormf(bsumsq):
    return np.sqrt(np.sum(bsumsq))

def betaf(beta_num, beta_den):
    return np.sum(beta_num)/np.sum(beta_den)

def power_method_dist(actors,
                      nx,
                      ny,
                      nband,
                      tol=1e-4,
                      maxit=200,
                      report_freq=10,
                      verbosity=1,):

    client = get_client()

    bssq = list(map(lambda a: a.init_random(), actors))
    # custom gather?
    bssq = list(map(lambda o: o.result(), bssq))
    bnorm = np.sqrt(np.sum(bssq))
    beta = 1
    for k in range(maxit):
        futures = list(map(lambda a: a.pm_update(bnorm), actors))

        results = list(map(lambda f: f.result(), futures))
        # what is wrong here?
        # bssq = list(map(getitem, results, 0))
        bssq = [r[0] for r in results]
        bnum = [r[1] for r in results]
        bden = [r[2] for r in results]

        bnorm = np.sqrt(np.sum(bssq))
        betap = beta
        beta = np.sum(bnum)/np.sum(bden)

        eps = np.abs(betap - beta)/betap
        if eps < tol:
            break

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}", file=log)

    return beta
