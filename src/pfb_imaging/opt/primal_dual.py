from abc import ABC, abstractmethod
from time import time

import numpy as np
from numba import njit, prange

from pfb_imaging.operators import PsiOperatorProtocol
from pfb_imaging.prox.prox_21m import dual_update_numba_fast
from pfb_imaging.utils import logging as pfb_logging

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


class PrimalDual(ABC):
    """Abstract base class for primal-dual splitting algorithms.

    The fixed algorithm loop lives in :meth:`solve`.  Subclasses override
    problem-specific methods (:meth:`dual_step` and :meth:`prox_primal`).

    Operators use the following convention (matching the existing functions):
      - ``psi(x, v)``:    analysis operator, image → coefficients, fills ``v`` in-place
      - ``psih(v, xout)``: synthesis operator, coefficients → image, fills ``xout`` in-place

    Args:
        psi: Linear operator implementing dot (analysis operator) and hdot (synthesis operator) methods.
             The analysis operator ``psi.dot(x, v)`` fills ``v = Ψ†x`` in-place.
             The synthesis operator ``psi.hdot(v, xout)`` fills ``xout = Ψv`` in-place.
        hessnorm: Spectral norm of the data-fidelity Hessian.
        nu: Spectral norm of ``psi`` (default 1.0).
        gamma: Step-size safety factor (default 1.0).
        sigma: Dual step size.  Computed from ``hessnorm``, ``gamma``, ``nu``
            when ``None``.
        tol: Convergence tolerance on relative primal change (default 1e-5).
        maxit: Maximum number of iterations (default 1000).
        report_freq: Log progress every this many iterations at verbosity > 1
            (default 10).
        verbosity: 0 = silent, 1 = convergence message, 2 = per-iter logging
            (default 1).
    """

    def __init__(
        self,
        psi: PsiOperatorProtocol,
        hessnorm: float,
        nu=1.0,
        gamma=1.0,
        sigma=None,
        tol=1e-5,
        maxit=1000,
        report_freq=10,
        verbosity=1,
    ):
        if not isinstance(psi, PsiOperatorProtocol):
            raise TypeError("psi must implement dot() and hdot()")
        self.psi = psi
        self.hessnorm = hessnorm
        self.nu = nu
        self.gamma = gamma
        self.tol = tol
        self.maxit = maxit
        self.report_freq = report_freq
        self.verbosity = verbosity
        self._grad = None

        if sigma is None:
            sigma = hessnorm / (2.0 * gamma) / nu
        self.sigma = sigma
        self.tau = 0.98 / (hessnorm / (2.0 * gamma) + sigma * nu**2)

    def set_grad(self, grad):
        """Set (or replace) the gradient of the smooth data-fidelity term.

        Called before the first :meth:`solve` and between outer iterations
        (e.g. in SARA when the Hessian is re-estimated).

        Args:
            grad: Callable ``grad(x) -> array`` returning the gradient.
        """
        self._grad = grad

    @abstractmethod
    def dual_step(self, xp, v, vp):
        """Dual update: analyse ``xp`` into ``v``, then update ``v`` in-place.

        Should call ``self.psi(xp, v)`` first, then apply the proximal step
        using ``vp`` (the previous dual iterate) to produce the new ``v``.

        Args:
            xp: Previous primal iterate (read-only).
            v: Dual variable array; overwritten with the new dual iterate.
            vp: Previous dual iterate (read-only inside this method; will be
                overwritten by :meth:`extrapolate_dual` immediately after).
        """

    @abstractmethod
    def prox_primal(self, x):
        """Apply the primal proximal operator in-place (e.g. positivity).

        Args:
            x: Primal variable; modified in-place.
        """

    # --- methods with default numba implementations ---

    def extrapolate_dual(self, v, vp):
        """Overwrite ``vp`` with ``2*v - vp`` (over-relaxation step)."""
        _nb_extrapolate_dual(v, vp)

    def primal_step(self, x, xp, xout, tau):
        """Overwrite ``x`` with ``xp - tau * xout``."""
        _nb_primal_step(x, xp, xout, tau)

    def norm_diff(self, x, xp):
        """Return relative norm of change: ``||x - xp|| / ||x||``."""
        return _nb_norm_diff(x, xp)

    def any_nonzero(self, x):
        """Return ``True`` if any element of ``x`` is nonzero."""
        return _nb_any_nonzero(x)

    def on_converge(self, x, k):
        """Hook called when ``eps < tol``.  Return ``True`` to stop.

        Override in subclasses to implement reweighting or other outer-loop
        logic.  The default always returns ``True`` (stop immediately).

        Args:
            x: Current primal iterate.
            k: Current iteration index.

        Returns:
            ``True`` to terminate the solve loop, ``False`` to continue.
        """
        return True

    def solve(self, x, v):
        """Run the primal-dual loop.

        Args:
            x: Initial primal variable (modified in-place and returned).
            v: Initial dual variable (modified in-place and returned).

        Returns:
            Tuple ``(x, v)`` of final primal and dual iterates.

        Raises:
            RuntimeError: If :meth:`set_grad` has not been called.
        """
        if self._grad is None:
            raise RuntimeError("grad not set; call set_grad() before solve()")

        xp = x.copy()
        vp = v.copy()
        xout = np.zeros_like(x)

        eps = 1.0
        tii = time()
        for k in range(self.maxit):
            self.dual_step(xp, v, vp)
            self.extrapolate_dual(v, vp)
            self.psi.hdot(vp, xout)
            xout += self._grad(xp)
            self.primal_step(x, xp, xout, self.tau)
            self.prox_primal(x)

            if self.any_nonzero(x):
                eps = self.norm_diff(x, xp)
            else:
                eps = 1.0

            if eps < self.tol:
                if self.on_converge(x, k):
                    break

            np.copyto(xp, x)
            np.copyto(vp, v)

            if not k % self.report_freq and self.verbosity > 1:
                log.info(f"At iteration {k} eps = {eps:.3e}")

        ttot = time() - tii
        if self.verbosity > 1:
            log.info(f"Total time: {ttot:.3f}s  ({ttot / max(k + 1, 1) * 1e3:.1f} ms/iter)")

        if k == self.maxit - 1:
            if self.verbosity:
                log.info(f"Max iters reached. eps = {eps:.3e}")
        else:
            if self.verbosity:
                log.info(f"Success, converged after {k} iterations")

        return x, v
