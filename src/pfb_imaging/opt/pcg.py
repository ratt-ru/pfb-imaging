from time import time

import numpy as np
from numba import njit, prange

from pfb_imaging.operators import LinearOperator, require_protocol

ifftshift = np.fft.ifftshift
fftshift = np.fft.fftshift


_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


@njit(**_FAST_JIT)
def _nb_fused_alpha_update(r, y, p, aopp, x, xp, rp):
    """Compute alpha = (r·y)/(p·aopp), then x = xp + alpha*p, r = rp + alpha*aopp."""
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
    """
    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype)

    if precond is None:

        def precond(x):
            return x

    r = aop(x0) - b
    y = precond(r)
    if not np.any(y):
        print("Initial residual is zero")
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
        np.copyto(xp, x)
        np.copyto(rp, r)
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
            print(f"At iteration {k} eps = {eps:.3e}, phi = {phi:.3e}")
    ttot = time() - tii
    if verbosity > 1:
        print(f"pcg_numba timing breakdown (fraction of {ttot:.3f}s):")
        print(f"  copyto:       {tcopy / ttot:.3f}")
        print(f"  aop:          {taop / ttot:.3f}")
        print(f"  alpha_update: {talpha / ttot:.3f}")
        print(f"  precond:      {tprecond / ttot:.3f}")
        print(f"  beta_update:  {tbeta / ttot:.3f}")
        print(f"  norm_diff:    {tnorm / ttot:.3f}")
        ttally = tcopy + taop + talpha + tprecond + tbeta + tnorm
        print(f"  accounted:    {ttally / ttot:.3f}")

    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}")
    elif stall_count >= 5:
        if verbosity:
            print(f"Stalled after {k} iterations with eps = {eps:.3e}")
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations")
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
