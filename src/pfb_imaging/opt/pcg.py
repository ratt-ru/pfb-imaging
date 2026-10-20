from time import time

import numpy as np
from numba import njit, prange

from pfb_imaging.operators import LinearOperator, require_protocol
from pfb_imaging.utils import inplace
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("CG")

ifftshift = np.fft.ifftshift
fftshift = np.fft.fftshift


_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


def _finish(info, verbosity, k, eps, maxit, stall_count, ttot):
    """Record how a CG solve ended in ``info``, or log it when no ``info`` was asked for.

    A caller passing ``info`` reports the outcome itself: the band workers'
    solves are summarised once on the driver (``log_cg_summary``) instead of
    one line per band on the console.
    """
    if k >= maxit:
        status, msg = "maxit", f"Max iters reached. eps = {eps:.3e}"
    elif stall_count >= 5:
        status, msg = "stalled", f"Stalled after {k} iterations with eps = {eps:.3e}"
    else:
        status, msg = "converged", f"Success, converged after {k} iterations in {ttot:.1f}s"
    if info is not None:
        info.update(iters=k, eps=float(eps), status=status, time=ttot)
    elif verbosity:
        log.info(msg)


def _zero_residual(info):
    if info is not None:
        info.update(iters=0, eps=0.0, status="converged", time=0.0)
    else:
        log.info("Initial residual is zero")


def log_cg_summary(infos, coupled=False):
    """Log one line for a forward solve from its per-solve ``info`` dicts.

    ``infos`` holds one dict per band for the band-parallel solve, or a single
    dict for the band-coupled one. Bands that hit ``maxit`` or stalled get a
    warning, so a CG cut short is visible in the log file.
    """
    iters = [i["iters"] for i in infos]
    eps = max(i["eps"] for i in infos)
    t = max(i["time"] for i in infos)
    if coupled:
        log.info(f"CG (bands coupled): {iters[0]} iterations, eps = {eps:.3e}, {t:.1f}s")
    else:
        log.info(
            f"CG over {len(infos)} bands: {min(iters)}-{max(iters)} iterations, "
            f"max eps = {eps:.3e}, slowest band {t:.1f}s"
        )
    unconverged = [
        f"{'' if coupled else f'band {b} '}{i['status']} after {i['iters']} iterations (eps = {i['eps']:.3e})"
        for b, i in enumerate(infos)
        if i["status"] != "converged"
    ]
    if unconverged:
        log.warning("CG did not converge: " + "; ".join(unconverged))


@njit(**_FAST_JIT)
def _nb_fused_alpha_update(r, y, p, aopp, x, xp, rp):
    """Compute alpha = (r·y)/(p·aopp), then x = xp + alpha*p, r = rp + alpha*aopp."""
    inplace.require_c(x)
    inplace.require_c(r)
    n = r.size
    r_f = r.ravel()
    y_f = y.ravel()
    p_f = p.ravel()
    a_f = aopp.ravel()
    x_f = x.ravel()
    xp_f = xp.ravel()
    rp_f = rp.ravel()

    rnorm = 0.0
    denom = 0.0
    for i in prange(n):
        rnorm += r_f[i] * y_f[i]
        denom += p_f[i] * a_f[i]

    alpha = rnorm / denom

    for i in prange(n):
        x_f[i] = xp_f[i] + alpha * p_f[i]
        r_f[i] = rp_f[i] + alpha * a_f[i]

    return rnorm


@njit(**_FAST_JIT)
def _nb_fused_beta_update(r, y, p, rnorm):
    """Compute beta = (r·y)/rnorm, then p = beta*p - y."""
    inplace.require_c(p)
    n = r.size
    r_f = r.ravel()
    y_f = y.ravel()
    p_f = p.ravel()

    rnorm_next = 0.0
    for i in prange(n):
        rnorm_next += r_f[i] * y_f[i]

    beta = rnorm_next / rnorm

    for i in prange(n):
        p_f[i] = beta * p_f[i] - y_f[i]

    return rnorm_next


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


@njit(**_FAST_JIT)
def _nb_cg_step(r, p, ap, x):
    """``alpha = (r.r)/(p.ap)``, then ``x += alpha p`` and ``r += alpha ap`` in place.

    Returns ``(r.r, alpha, p.p, x.x)``, the last two for the relative step.
    """
    inplace.require_c(x)
    inplace.require_c(r)
    r_f = r.ravel()
    p_f = p.ravel()
    a_f = ap.ravel()
    x_f = x.ravel()
    rnorm = 0.0
    denom = 0.0
    pp = 0.0
    for i in prange(r_f.size):
        rnorm += r_f[i] * r_f[i]
        denom += p_f[i] * a_f[i]
        pp += p_f[i] * p_f[i]
    alpha = rnorm / denom
    xx = 0.0
    for i in prange(r_f.size):
        x_f[i] += alpha * p_f[i]
        r_f[i] += alpha * a_f[i]
        xx += x_f[i] * x_f[i]
    return rnorm, alpha, pp, xx


def cg_inplace(aop, b, x0=None, tol=1e-5, maxit=500, minit=100, verbosity=1, report_freq=10, info=None):
    """Unpreconditioned CG of ``aop(x) = b`` holding three cubes besides ``aop``'s output.

    The iterates of ``pcg_numba`` without a preconditioner, for the driver's
    cube-level solve. ``pcg_numba`` keeps copies of the previous iterate and
    residual only to form ``x = xp + alpha p`` and ``||x - xp||``; updating in
    place and using ``||x - xp|| = |alpha| ||p||`` drops two cubes and two cube
    copies per iteration (D58). Same stopping rules (``tol`` on the relative
    step, ``minit``, ``maxit``, five stalled steps).

    ``aop`` may return a buffer it reuses across calls. When ``x0`` is given it
    is the iterate and is updated **in place** (the returned array IS ``x0``).

    A dict passed as ``info`` receives ``iters``, ``eps``, ``status``
    (``converged``, ``maxit`` or ``stalled``) and ``time`` in seconds, and the
    final status line is then left to the caller.
    """
    if x0 is None:
        x = np.zeros(b.shape, dtype=b.dtype)
        r = np.negative(b)  # aop(0) - b
    else:
        x = x0
        r = np.empty_like(b)
        inplace.axpby(r, 1.0, aop(x), -1.0, b)
    if not np.any(r):
        _zero_residual(info)
        return x
    p = np.negative(r)
    k = 0
    eps = 1.0
    stall_count = 0
    tii = time()
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        rnorm, alpha, pp, xx = _nb_cg_step(r, p, aop(p), x)
        rnorm = _nb_fused_beta_update(r, r, p, rnorm)
        k += 1
        epsp = eps
        eps = np.sqrt(alpha * alpha * pp / max(xx, 1e-12))
        if np.abs(epsp - eps) < 1e-3 * tol:
            stall_count += 1
        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")
    _finish(info, verbosity, k, eps, maxit, stall_count, time() - tii)
    return x


def pcg_numba(
    aop,
    b,
    x0=None,
    precond=None,
    tol=1e-5,
    maxit=500,
    minit=100,
    verbosity=1,
    report_freq=10,
    backtrack=True,
    return_resid=False,
    info=None,
):
    """Legacy CG solver of ``aop(x) = b`` with fused numba update kernels.

    Frozen legacy implementation (the in-worker CG of the deconv band
    workers and the oracle for ``opt.pcg.PCG``); do not change its
    behaviour. See docs/wiki/deconv-primer.md.

    Warning:
        When ``x0`` is given it is bound as the iterate and updated
        **in place** (the returned array IS ``x0``). Callers that must not
        see their ``x0`` mutated pass a copy — Ray callers MUST copy anyway,
        since Ray deserialises task arguments as read-only views and the
        numba kernels crash on them (``_BandWorkerImpl.cg`` is the template).

    ``info`` is as for ``cg_inplace``; it only reports, the iterates are unchanged.
    """
    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype)

    if precond is None:

        def precond(x):
            return x

    r = aop(x0) - b
    y = precond(r)
    if not np.any(y):
        _zero_residual(info)
        return x0
    p = -y
    rnorm = np.vdot(r, y)
    if np.isnan(rnorm) or rnorm == 0.0:
        phi0 = 1.0
    else:
        phi0 = rnorm
    k = 0
    x = x0
    eps = 1.0
    stall_count = 0
    xp = x.copy()
    rp = r.copy()
    tcopy = 0.0
    taop = 0.0
    talpha = 0.0
    tprecond = 0.0
    tbeta = 0.0
    tnorm = 0.0
    tii = time()
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        ti = time()
        inplace.copy(xp, x)
        inplace.copy(rp, r)
        tcopy += time() - ti
        ti = time()
        aopp = aop(p)
        taop += time() - ti
        ti = time()
        rnorm = _nb_fused_alpha_update(r, y, p, aopp, x, xp, rp)
        talpha += time() - ti
        ti = time()
        y = precond(r)
        tprecond += time() - ti
        ti = time()
        rnorm = _nb_fused_beta_update(r, y, p, rnorm)
        tbeta += time() - ti
        k += 1
        epsp = eps
        ti = time()
        eps = _nb_norm_diff(x, xp)
        tnorm += time() - ti
        phi = rnorm / phi0

        if np.abs(epsp - eps) < 1e-3 * tol:
            stall_count += 1

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}, phi = {phi:.3e}")
    ttot = time() - tii
    if verbosity > 1:
        ttally = tcopy + taop + talpha + tprecond + tbeta + tnorm
        log.info(
            f"pcg_numba timing breakdown (fraction of {ttot:.3f}s): "
            f"copyto {tcopy / ttot:.3f}, aop {taop / ttot:.3f}, alpha_update {talpha / ttot:.3f}, "
            f"precond {tprecond / ttot:.3f}, beta_update {tbeta / ttot:.3f}, "
            f"norm_diff {tnorm / ttot:.3f}, accounted {ttally / ttot:.3f}"
        )
    _finish(info, verbosity, k, eps, maxit, stall_count, ttot)
    if not return_resid:
        return x
    else:
        return x, r


class PCG:
    """Conjugate-gradient ForwardSolver: ``update ≈ hess^{-1} residual``.

    Satisfies the ``ForwardSolver`` Protocol.  When the operator exposes a
    ``cg`` method (the distributed per-band fast path of ``HessTreeRay``),
    the solve is delegated to it with this solver's controls; otherwise a
    generic cube-level CG runs over ``hess.dot``.

    Args:
        tol: CG convergence tolerance.
        maxit: Maximum CG iterations.
        minit: Minimum CG iterations.
        verbosity: 0 silent, > 1 per-iteration reporting.
        report_freq: Reporting cadence at verbosity > 1.
    """

    def __init__(
        self,
        tol: float = 1e-3,
        maxit: int = 150,
        minit: int = 1,
        verbosity: int = 0,
        report_freq: int = 10,
    ):
        self.tol = tol
        self.maxit = maxit
        self.minit = minit
        self.verbosity = verbosity
        self.report_freq = report_freq

    def solve(self, hess, residual, x0=None):
        """Solve ``hess @ update = residual`` for update."""
        if hasattr(hess, "cg"):
            return hess.cg(residual, x0=x0, tol=self.tol, maxit=self.maxit, minit=self.minit)
        require_protocol(hess, LinearOperator, "hess")
        return pcg_numba(
            hess.dot,
            residual,
            x0=x0,
            tol=self.tol,
            maxit=self.maxit,
            minit=self.minit,
            verbosity=self.verbosity,
            report_freq=self.report_freq,
        )
