from time import time

import numpy as np
from numba import njit, prange
from scipy.linalg import norm

from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("PM")

_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


@njit(**_FAST_JIT)
def _nb_vdot_pair(bp, b):
    """Compute np.vdot(bp, b) and np.vdot(bp, bp) in a single pass."""
    n = bp.size
    bp_f = bp.ravel()
    b_f = b.ravel()

    num = 0.0
    den = 0.0
    for i in prange(n):
        num += bp_f[i] * b_f[i]
        den += bp_f[i] * bp_f[i]

    return num, den


@njit(**_FAST_JIT)
def _nb_normalize(b, bnorm):
    """b /= bnorm in-place."""
    n = b.size
    b_f = b.ravel()
    inv = 1.0 / bnorm
    for i in prange(n):
        b_f[i] *= inv


def power_method_numba(aop, imsize, b0=None, tol=1e-5, maxit=250, verbosity=1, report_freq=25):
    if b0 is None:
        b = np.random.randn(*imsize)
        b /= norm(b)
    else:
        b = b0 / norm(b0)
    beta = 1.0
    eps = 1.0
    k = 0
    bp = b.copy()
    taop = 0.0
    tnorm = 0.0
    tvdot = 0.0
    tcopy = 0.0
    tii = time()
    while eps > tol and k < maxit:
        ti = time()
        b = aop(bp)
        taop += time() - ti
        ti = time()
        bnorm = np.linalg.norm(b)
        tnorm += time() - ti
        betap = beta
        ti = time()
        beta_num, beta_den = _nb_vdot_pair(bp, b)
        beta = beta_num / beta_den
        _nb_normalize(b, bnorm)
        tvdot += time() - ti
        eps = np.abs(beta - betap) / betap
        k += 1
        ti = time()
        np.copyto(bp, b)
        tcopy += time() - ti

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")
    ttot = time() - tii
    if verbosity > 1:
        print(f"power_method_numba timing breakdown (fraction of {ttot:.3f}s):")
        print(f"  aop:      {taop / ttot:.3f}")
        print(f"  norm:     {tnorm / ttot:.3f}")
        print(f"  vdot+div: {tvdot / ttot:.3f}")
        print(f"  copyto:   {tcopy / ttot:.3f}")
        ttally = taop + tnorm + tvdot + tcopy
        print(f"  accounted:{ttally / ttot:.3f}")

    if k == maxit:
        if verbosity:
            log.info(f"Maximum iterations reached. eps = {eps:.3e}, beta = {beta:.3e}")
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. beta = {beta:.3e}")
    return beta, b


def power_method(aop, imsize, b0=None, tol=1e-5, maxit=250, verbosity=1, report_freq=25):
    if b0 is None:
        b = np.random.randn(*imsize)
        b /= norm(b)
    else:
        b = b0 / norm(b0)
    beta = 1.0
    eps = 1.0
    k = 0
    bp = b.copy()
    taop = 0.0
    tnorm = 0.0
    tvdot = 0.0
    tcopy = 0.0
    tii = time()
    while eps > tol and k < maxit:
        ti = time()
        b = aop(bp)
        taop += time() - ti
        ti = time()
        bnorm = np.linalg.norm(b)
        tnorm += time() - ti
        betap = beta
        ti = time()
        beta = np.vdot(bp, b) / np.vdot(bp, bp)
        b /= bnorm
        tvdot += time() - ti
        # this is a scalar
        eps = np.linalg.norm(beta - betap) / betap
        k += 1
        ti = time()
        bp[...] = b[...]
        tcopy += time() - ti

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")
    ttot = time() - tii
    if verbosity > 1:
        print(f"power_method timing breakdown (fraction of {ttot:.3f}s):")
        print(f"  aop:      {taop / ttot:.3f}")
        print(f"  norm:     {tnorm / ttot:.3f}")
        print(f"  vdot+div: {tvdot / ttot:.3f}")
        print(f"  copyto:   {tcopy / ttot:.3f}")
        ttally = taop + tnorm + tvdot + tcopy
        print(f"  accounted:{ttally / ttot:.3f}")

    if k == maxit:
        if verbosity:
            log.info(f"Maximum iterations reached. eps = {eps:.3e}, beta = {beta:.3e}")
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. beta = {beta:.3e}")
    return beta, b


def power(aop, bp, bnorm, eta):
    bp /= bnorm
    b = aop(bp, eta)
    bsumsq = np.sum(b**2)
    beta_num = np.vdot(b, bp)
    beta_den = np.vdot(bp, bp)

    return b, bsumsq, beta_num, beta_den


def sumsq(b):
    return np.sum(b**2)


def bnormf(bsumsq):
    return np.sqrt(np.sum(bsumsq))


def betaf(beta_num, beta_den):
    return np.sum(beta_num) / np.sum(beta_den)


def power_method_dist(
    actors,
    nx,
    ny,
    nband,
    tol=1e-4,
    maxit=200,
    report_freq=10,
    verbosity=1,
):
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
        beta = np.sum(bnum) / np.sum(bden)

        eps = np.abs(betap - beta) / betap
        if eps < tol:
            break

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")

    return beta
