from time import time

import numpy as np
from numba import njit, prange

from pfb_imaging.deconv import Regulariser
from pfb_imaging.operators import PsiOperator, require_protocol
from pfb_imaging.prox.positivity import positivity as _positivity_fn
from pfb_imaging.prox.positivity import positivity_band as _positivity_band_fn
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
    """Legacy PDHG solver for min_x F(x) + lam*g(Psi^T x) (validation oracle).

    Frozen legacy implementation kept as the test oracle for the composable
    framework (``opt.primal_dual.PrimalDual`` is the maintained replacement);
    see docs/wiki/deconv-primer.md. Do not change its behaviour.

    Naming trap (differs from ``primal_dual_numba``!): here ``psi`` is the
    SYNTHESIS operator (coefficients -> image, allocating, used in the primal
    step) and ``psih`` is the ANALYSIS operator (image -> coefficients, used
    in the dual step).

    ``nu`` is the squared frame bound ||Psi Psi^T|| — ``nbasis`` for the SARA
    concatenation of orthonormal bases, NOT the tight-frame 1.0. It sets the
    step sizes ``sigma = hessnorm/(2*gamma)/nu`` and
    ``tau = 0.9/(hessnorm/(2*gamma) + sigma*nu**2)``; an underestimated nu
    violates the convergence condition (observed as multi-band divergence).

    The duals ``v`` are warm-started by the caller across major cycles;
    returns ``(x, v)``.
    """
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
    """Legacy numba-kernel PDHG with inner l1 reweighting (validation oracle).

    Frozen legacy implementation kept as the test oracle for
    ``opt.primal_dual.PrimalDual`` + ``prox.l21.L21`` (rdiff < 1e-10 in
    tests/test_primal_dual.py); see docs/wiki/deconv-primer.md. Do not change
    its behaviour.

    Naming trap (INVERTED relative to ``primal_dual``!): here ``psih`` is the
    SYNTHESIS operator and ``psi`` is the ANALYSIS operator — both in-place
    two-argument callables (``psi(image, coeffs_out)``,
    ``psih(coeffs, image_out)``, the ``Psi.dot``/``Psi.hdot`` convention).
    Read the loop body, not the parameter names.

    ``nu`` is ||Psi Psi^T|| = nbasis for the SARA dictionary (see
    ``primal_dual``'s docstring for the step-size consequences). ``prox`` is
    accepted for signature compatibility; the dual update is the fused
    ``dual_update_numba_fast`` with ``l1weight``. ``reweighter`` (or None)
    fires on inner convergence, up to ``maxreweight`` consecutive times.

    Warning:
        Contains ``pdb.set_trace()`` breakpoints on the zero-model and
        NaN-eps paths — an unattended run hangs if those trigger.
    """
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
            _positivity_fn(x)
        elif positivity == 2:
            _positivity_band_fn(x)
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


class PrimalDual:
    """Primal-dual solver for ``min_x f(x) + lam * g(Psi^T x)``.

    Concrete class satisfying the ``BackwardSolver`` Protocol; the
    regulariser arrives via ``setup``.  The dual update prefers the fused
    ``reg.dual_update(vp, v, lam, sigma)`` fast path when the regulariser
    provides one, falling back to the generic Moreau decomposition
    ``v = vtilde - sigma * prox_{(lam/sigma) g}(vtilde/sigma)`` via
    ``reg.prox``.  The dual variable is internal state, warm-started
    across ``solve`` calls; ``reset()`` zeros it.

    Args:
        tol: Convergence tolerance on relative primal change.
        maxit: Maximum iterations.
        report_freq: Log every this many iterations at verbosity > 1.
        verbosity: 0 silent, 1 convergence message, 2 per-iter logging.
        gamma: Step-size safety factor.
        sigma: Dual step size; computed from hessnorm/gamma/nu when None.
        on_converge: Optional ``cb(x, k, eps) -> bool`` fired when
            ``eps < tol``; return False to continue iterating.
        primal_prox: Optional in-place image-domain prox (e.g. positivity).
    """

    def __init__(
        self,
        tol: float = 1e-5,
        maxit: int = 1000,
        report_freq: int = 10,
        verbosity: int = 1,
        gamma: float = 1.0,
        sigma: float | None = None,
        on_converge=None,
        primal_prox=None,
    ):
        self.tol = tol
        self.maxit = maxit
        self.report_freq = report_freq
        self.verbosity = verbosity
        self.gamma = gamma
        self._sigma_opt = sigma
        self.on_converge = on_converge
        self.primal_prox = primal_prox
        self._grad = None
        self._reg = None
        self._v = None

    def setup(self, prox, hessnorm: float) -> None:
        """Bind the regulariser, compute step sizes, allocate the dual."""
        require_protocol(prox, Regulariser, "prox")
        require_protocol(prox.psi, PsiOperator, "prox.psi")
        self._reg = prox
        self.hessnorm = hessnorm
        nu = prox.nu
        sigma = self._sigma_opt
        if sigma is None:
            sigma = hessnorm / (2.0 * self.gamma) / nu
        self.sigma = sigma
        self.tau = 0.98 / (hessnorm / (2.0 * self.gamma) + sigma * nu**2)
        psi = prox.psi
        self._v = np.zeros((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))

    def set_grad(self, grad) -> None:
        """Set the gradient of the smooth data-fidelity term."""
        self._grad = grad

    def reset(self) -> None:
        """Drop the warm-started dual variable."""
        if self._v is not None:
            self._v[...] = 0.0

    def _dual_step(self, xp, v, vp, lam):
        """Analysis + dual proximal update; fused fast path when available."""
        reg = self._reg
        reg.psi.dot(xp, v)
        if hasattr(reg, "dual_update"):
            reg.dual_update(vp, v, lam, sigma=self.sigma)
        else:
            # generic Moreau: v holds Psi^T xp on entry
            vtilde = vp + self.sigma * v
            reg.prox(vtilde, v, lam, sigma=self.sigma)
            np.subtract(vtilde, self.sigma * v, out=v)

    def solve(self, x, lam: float):
        """Run the primal-dual loop; returns the final primal iterate."""
        if self._reg is None:
            raise RuntimeError("regulariser not bound; call setup() before solve()")
        if self._grad is None:
            raise RuntimeError("grad not set; call set_grad() before solve()")

        xp = x.copy()
        v = self._v
        vp = v.copy()
        xout = np.zeros_like(x)

        eps = 1.0
        tii = time()
        for k in range(self.maxit):
            self._dual_step(xp, v, vp, lam)
            _nb_extrapolate_dual(v, vp)
            self._reg.psi.hdot(vp, xout)
            xout += self._grad(xp)
            _nb_primal_step(x, xp, xout, self.tau)
            if self.primal_prox is not None:
                self.primal_prox(x)

            eps = _nb_norm_diff(x, xp) if _nb_any_nonzero(x) else 1.0
            if eps < self.tol:
                if self.on_converge is None or self.on_converge(x, k, eps):
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
        elif self.verbosity:
            log.info(f"Success, converged after {k} iterations")
        return x
