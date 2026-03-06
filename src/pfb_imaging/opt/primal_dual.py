from time import time

import numexpr as ne
import numpy as np
from numba import njit, prange

from pfb_imaging.prox.prox_21m import dual_update_numba, dual_update_numba_fast
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.misc import norm_diff

log = pfb_logging.get_logger("PD")

_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


@njit(**_FAST_JIT)
def _nb_extrapolate_dual(v, vp):
    """Compute vp = 2*v - vp in-place."""
    n = v.size
    v_f = v.ravel()
    vp_f = vp.ravel()
    for i in prange(n):
        vp_f[i] = 2.0 * v_f[i] - vp_f[i]


@njit(**_FAST_JIT)
def _nb_primal_step(x, xp, xout, tau):
    """Compute x = xp - tau * xout."""
    n = x.size
    x_f = x.ravel()
    xp_f = xp.ravel()
    xo_f = xout.ravel()
    for i in prange(n):
        x_f[i] = xp_f[i] - tau * xo_f[i]


@njit(**_FAST_JIT)
def _nb_positivity(x):
    """Clamp negative values to zero (positivity == 1)."""
    n = x.size
    x_f = x.ravel()
    for i in prange(n):
        if x_f[i] < 0.0:
            x_f[i] = 0.0


@njit(**_FAST_JIT)
def _nb_positivity_band(x):
    """Zero all bands where any band is non-positive (positivity == 2)."""
    nband, nx, ny = x.shape
    for i in prange(nx):
        for j in range(ny):
            for b in range(nband):
                if x[b, i, j] <= 0.0:
                    for bb in range(nband):
                        x[bb, i, j] = 0.0
                    break


@njit(**_FAST_JIT)
def _nb_norm_diff(x, xp):
    """Relative norm of difference: ||x - xp|| / ||x||."""
    n = x.size
    x_f = x.ravel()
    xp_f = xp.ravel()
    num = 0.0
    den = 0.0
    for i in prange(n):
        d = x_f[i] - xp_f[i]
        num += d * d
        den += x_f[i] * x_f[i]
    den = max(den, 1e-12)
    return np.sqrt(num / den)


@njit(nogil=True, cache=True, fastmath=True)
def _nb_any_nonzero(x):
    """Check if any element is nonzero."""
    n = x.size
    x_f = x.ravel()
    for i in range(n):
        if x_f[i] != 0.0:
            return True
    return False


def primal_dual(
    x,  # initial guess for primal variable
    v,  # initial guess for dual variable
    lam,  # regulariser strength
    psi,  # linear operator in dual domain
    psih,  # adjoint of psi
    hessnorm,  # spectral norm of Hessian
    prox,  # prox of regulariser
    grad,  # gradient of smooth term
    nu=1.0,  # spectral norm of psi
    sigma=None,  # step size of dual update
    mask=None,  # regions where mask is False will be masked
    tol=1e-5,
    maxit=1000,
    minit=10,
    positivity=1,
    report_freq=10,
    gamma=1.0,
    verbosity=1,
):
    # initialise
    xp = x.copy()
    vp = v.copy()
    vtilde = np.zeros_like(v)

    # this seems to give a good trade-off between
    # primal and dual problems
    if sigma is None:
        sigma = hessnorm / (2.0 * gamma) / nu

    # stepsize control
    tau = 0.9 / (hessnorm / (2.0 * gamma) + sigma * nu**2)

    # start iterations
    eps = 1.0
    k = 0
    while (eps > tol or k < minit) and k < maxit:
        # tmp prox variable
        vtilde = v + sigma * psih(xp)

        # dual update
        v = vtilde - sigma * prox(vtilde / sigma, lam / sigma)

        # primal update
        x = xp - tau * (psi(2 * v - vp) + grad(xp))
        if positivity == 1:
            x[x < 0.0] = 0.0
        elif positivity == 2:
            msk = np.any(x <= 0, axis=0)
            x[:, msk] = 0.0

        # convergence check
        eps = np.linalg.norm(x - xp) / np.linalg.norm(x)

        # copy contents to avoid allocating new memory
        xp[...] = x[...]
        vp[...] = v[...]

        if np.isnan(eps) or np.isinf(eps):
            import pdb

            pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            # res = xbar-x
            # phi = np.vdot(res, A(res))
            log.info(f"At iteration {k} eps = {eps:.3e}")
        k += 1

    if k == maxit:
        if verbosity:
            log.info(f"Max iters reached. eps = {eps:.3e}")
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations")

    return x, v


def primal_dual_optimised(
    x,  # initial guess for primal variable
    v,  # initial guess for dual variable
    lam,  # regulariser strength
    psih,  # linear operator in dual domain
    psi,  # adjoint of psi
    hessnorm,  # spectral norm of Hessian
    prox,  # prox of regulariser
    l1weight,
    reweighter,
    grad,  # gradient of smooth term
    nu=1.0,  # spectral norm of psi
    sigma=None,  # step size of dual update
    mask=None,  # regions where mask is False will be masked
    tol=1e-5,
    maxit=1000,
    positivity=1,
    report_freq=10,
    gamma=1.0,
    verbosity=1,
    maxreweight=20,
):  # max successive reweights before convergence
    # initialise
    xp = x.copy()
    vp = v.copy()
    xout = np.zeros_like(x)

    # this seems to give a good trade-off between
    # primal and dual problems
    if sigma is None:
        sigma = hessnorm / (2.0 * gamma) / nu

    # stepsize control
    tau = 0.98 / (hessnorm / (2.0 * gamma) + sigma * nu**2)

    # start iterations
    eps = 1.0
    numreweight = 0
    last_reweight_iter = 0
    # for timing
    tpsi = 0.0
    tupdate = 0.0
    teval1 = 0.0
    tpsih = 0.0
    tgrad = 0.0
    teval2 = 0.0
    tpos = 0.0
    tnorm = 0.0
    tcopy = 0.0
    tii = time()
    for k in range(maxit):
        ti = time()
        psi(xp, v)
        tpsi += time() - ti
        ti = time()
        dual_update_numba(vp, v, lam, sigma=sigma, weight=l1weight)
        tupdate += time() - ti
        ti = time()
        ne.evaluate("2.0 * v - vp", out=vp, local_dict={"v": v, "vp": vp})
        teval1 += time() - ti
        ti = time()
        psih(vp, xout)
        tpsih += time() - ti
        ti = time()
        xout += grad(xp)
        tgrad += time() - ti
        ti = time()
        ne.evaluate("xp - tau * xout", out=x, local_dict={"xp": xp, "tau": tau, "xout": xout})
        teval2 += time() - ti

        ti = time()
        if positivity == 1:
            x[x < 0.0] = 0.0
        elif positivity == 2:
            msk = np.any(x <= 0, axis=0)
            x[:, msk] = 0.0
        tpos += time() - ti
        # convergence check
        if x.any():
            ti = time()
            eps = norm_diff(x, xp)
            tnorm += time() - ti
        else:
            import pdb

            pdb.set_trace()
            eps = 1.0
        if eps < tol:
            if reweighter is not None and numreweight < maxreweight:
                # ti = time()
                l1weight = reweighter(x)
                # log.info("reweight = ", time() - ti)
                if k - last_reweight_iter == 1:
                    numreweight += 1
                else:
                    numreweight = 0
                last_reweight_iter = k

            else:
                if numreweight >= maxreweight:
                    log.info("Maximum reweighting steps reached")
                break

        # copy contents to avoid allocating new memory
        ti = time()
        np.copyto(xp, x)
        np.copyto(vp, v)
        tcopy += time() - ti
        if np.isnan(eps) or np.isinf(eps):
            import pdb

            pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")

    ttot = time() - tii
    ttally = tpsi + tpsih + tgrad + tupdate + teval1 + teval2 + tpos + tnorm
    if verbosity > 1:
        log.info("Time taken per step")
        log.info(f"psi = {tpsi / ttot}")
        log.info(f"psih = {tpsih / ttot}")
        log.info(f"grad = {tgrad / ttot}")
        log.info(f"update = {tupdate / ttot}")
        log.info(f"eval1 = {teval1 / ttot}")
        log.info(f"eval2 = {teval2 / ttot}")
        log.info(f"pos = {tpos / ttot}")
        log.info(f"norm = {tnorm / ttot}")
        log.info(f"tally = {ttally / ttot}")

    if k == maxit - 1:
        if verbosity:
            log.info(f"Max iters reached. eps = {eps:.3e}")
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations")

    return x, v


def primal_dual_numba(
    x,  # initial guess for primal variable
    v,  # initial guess for dual variable
    lam,  # regulariser strength
    psih,  # linear operator in dual domain
    psi,  # adjoint of psi
    hessnorm,  # spectral norm of Hessian
    prox,  # prox of regulariser
    l1weight,
    reweighter,
    grad,  # gradient of smooth term
    nu=1.0,  # spectral norm of psi
    sigma=None,  # step size of dual update
    mask=None,  # regions where mask is False will be masked
    tol=1e-5,
    maxit=1000,
    positivity=1,
    report_freq=10,
    gamma=1.0,
    verbosity=1,
    maxreweight=20,
):
    # initialise
    xp = x.copy()
    vp = v.copy()
    xout = np.zeros_like(x)

    if sigma is None:
        sigma = hessnorm / (2.0 * gamma) / nu

    tau = 0.98 / (hessnorm / (2.0 * gamma) + sigma * nu**2)

    eps = 1.0
    numreweight = 0
    last_reweight_iter = 0
    # for timing
    tpsi = 0.0
    tupdate = 0.0
    textrap = 0.0
    tpsih = 0.0
    tgrad = 0.0
    tprimal = 0.0
    tpos = 0.0
    tnorm = 0.0
    tcopy = 0.0
    tii = time()
    for k in range(maxit):
        ti = time()
        psi(xp, v)
        tpsi += time() - ti
        ti = time()
        dual_update_numba_fast(vp, v, lam, sigma=sigma, weight=l1weight)
        tupdate += time() - ti
        ti = time()
        _nb_extrapolate_dual(v, vp)
        textrap += time() - ti
        ti = time()
        psih(vp, xout)
        tpsih += time() - ti
        ti = time()
        xout += grad(xp)
        tgrad += time() - ti
        ti = time()
        _nb_primal_step(x, xp, xout, tau)
        tprimal += time() - ti

        ti = time()
        if positivity == 1:
            _nb_positivity(x)
        elif positivity == 2:
            _nb_positivity_band(x)
        tpos += time() - ti
        # convergence check
        if _nb_any_nonzero(x):
            ti = time()
            eps = _nb_norm_diff(x, xp)
            tnorm += time() - ti
        else:
            import pdb

            pdb.set_trace()
            eps = 1.0
        if eps < tol:
            if reweighter is not None and numreweight < maxreweight:
                l1weight = reweighter(x)
                if k - last_reweight_iter == 1:
                    numreweight += 1
                else:
                    numreweight = 0
                last_reweight_iter = k

            else:
                if numreweight >= maxreweight:
                    log.info("Maximum reweighting steps reached")
                break

        ti = time()
        np.copyto(xp, x)
        np.copyto(vp, v)
        tcopy += time() - ti
        if np.isnan(eps) or np.isinf(eps):
            import pdb

            pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")

    ttot = time() - tii
    ttally = tpsi + tpsih + tgrad + tupdate + textrap + tprimal + tpos + tnorm + tcopy
    if verbosity > 1:
        print(f"primal_dual_numba timing breakdown (fraction of {ttot:.3f}s):")
        print(f"  psi:        {tpsi / ttot:.3f}")
        print(f"  dual_upd:   {tupdate / ttot:.3f}")
        print(f"  extrap:     {textrap / ttot:.3f}")
        print(f"  psih:       {tpsih / ttot:.3f}")
        print(f"  grad:       {tgrad / ttot:.3f}")
        print(f"  primal_stp: {tprimal / ttot:.3f}")
        print(f"  positivity: {tpos / ttot:.3f}")
        print(f"  norm_diff:  {tnorm / ttot:.3f}")
        print(f"  copyto:     {tcopy / ttot:.3f}")
        print(f"  accounted:  {ttally / ttot:.3f}")

    if k == maxit - 1:
        if verbosity:
            log.info(f"Max iters reached. eps = {eps:.3e}")
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations")

    return x, v
