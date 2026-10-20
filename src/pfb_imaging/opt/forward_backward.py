"""Forward-backward splitting (with optional FISTA acceleration).

Concrete solver satisfying the ``BackwardSolver`` Protocol: the regulariser
arrives as data via ``setup`` — no subclassing.  The tight-frame proximal
composition ``x + (1/nu) * Psi(prox_g(Psi^T x) - Psi^T x)`` is implemented
once here, generically for any ``Regulariser``.
"""

import numpy as np

from pfb_imaging.deconv import Regulariser
from pfb_imaging.operators import PsiOperator, require_protocol
from pfb_imaging.opt.primal_dual import _nb_any_nonzero, _nb_norm_diff
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.memprof import SectionClock

log = pfb_logging.get_logger("FB")


class ForwardBackward:
    """Forward-backward solver for ``min_x f(x) + lam * R(x)``.

    At each iteration computes ``x = tight_frame_prox(y - step * grad(y), lam)``
    with optional FISTA momentum on ``y``.

    Args:
        tol: Convergence tolerance on relative primal change.
        maxit: Maximum iterations.
        report_freq: Log every this many iterations at verbosity > 1.
        verbosity: 0 silent, 1 convergence message, 2 per-iter logging.
        gamma: Step-size safety factor; ``step = 2 * gamma / hessnorm``.
        acceleration: Enable FISTA momentum (ISTA when False).
        on_converge: Optional ``cb(x, k, eps) -> bool`` fired when
            ``eps < tol``; return False to continue iterating.
        primal_prox: Optional in-place image-domain prox (e.g. positivity)
            applied after the tight-frame step.
    """

    def __init__(
        self,
        tol: float = 1e-5,
        maxit: int = 1000,
        report_freq: int = 10,
        verbosity: int = 1,
        gamma: float = 1.0,
        acceleration: bool = True,
        on_converge=None,
        primal_prox=None,
    ):
        self.tol = tol
        self.maxit = maxit
        self.report_freq = report_freq
        self.verbosity = verbosity
        self.gamma = gamma
        self.acceleration = acceleration
        self.on_converge = on_converge
        self.primal_prox = primal_prox
        self._grad = None
        self._reg = None

    def setup(self, prox, hessnorm: float) -> None:
        """Bind the regulariser, compute the step size, size buffers."""
        require_protocol(prox, Regulariser, "prox")
        require_protocol(prox.psi, PsiOperator, "prox.psi")
        self._reg = prox
        self.hessnorm = hessnorm
        self.step = 2.0 * self.gamma / hessnorm
        psi = prox.psi
        self._alpha = np.zeros((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))
        self._alpha_buf = np.zeros_like(self._alpha)
        self._xout = np.zeros((psi.nband, psi.nx, psi.ny))

    def set_grad(self, grad) -> None:
        """Set the gradient of the smooth data-fidelity term."""
        self._grad = grad

    def reset(self) -> None:
        """No warm-start state beyond x itself."""

    def _apply_prox(self, x, lam, clock=None):
        """Generic tight-frame prox of ``lam * g(Psi^T x)``, then primal_prox.

        ``clock`` (a ``SectionClock``) is charged per section when given.
        """
        reg = self._reg
        tick = clock.tick if clock is not None else lambda name: None
        reg.psi.dot(x, self._alpha)
        tick("psi_dot")
        reg.prox(self._alpha, self._alpha_buf, self.step * lam, sigma=1.0)
        self._alpha_buf -= self._alpha
        tick("prox")
        reg.psi.hdot(self._alpha_buf, self._xout)
        tick("psi_hdot")
        x += self._xout / reg.nu
        if self.primal_prox is not None:
            self.primal_prox(x)
        tick("primal")
        return x

    def solve(self, x, lam: float):
        """Run the forward-backward loop; returns the final iterate."""
        if self._reg is None:
            raise RuntimeError("regulariser not bound; call setup() before solve()")
        if self._grad is None:
            raise RuntimeError("grad not set; call set_grad() before solve()")

        xp = x.copy()
        y = x.copy()
        t = 1.0
        eps = 1.0
        # always on, as in PrimalDual: separates Ray round trips from driver work
        clock = SectionClock(
            ("grad", "gradient_step", "psi_dot", "prox", "psi_hdot", "primal", "convergence", "reweight", "momentum")
        )
        for k in range(self.maxit):
            clock.start()
            g = self._grad(y)
            clock.tick("grad")
            x = y - self.step * g
            del g
            clock.tick("gradient_step")
            x = self._apply_prox(x, lam, clock)

            eps = _nb_norm_diff(x, xp) if _nb_any_nonzero(x) else 1.0
            clock.tick("convergence")
            if eps < self.tol:
                stop = self.on_converge is None or self.on_converge(x, k, eps)
                clock.tick("reweight")
                if stop:
                    break

            if self.acceleration:
                tp = t
                t = (1.0 + np.sqrt(1.0 + 4.0 * tp**2)) / 2.0
                y = x + (tp - 1.0) / t * (x - xp)
            else:
                np.copyto(y, x)
            np.copyto(xp, x)
            clock.tick("momentum")

            if not k % self.report_freq and self.verbosity > 1:
                log.info(f"At iteration {k} eps = {eps:.3e}")

        self.clock = clock
        if self.verbosity:
            log.info(f"Forward-backward took {clock.summary(k + 1)}")
        if k == self.maxit - 1:
            if self.verbosity:
                log.info(f"Max iters reached. eps = {eps:.3e}")
        elif self.verbosity:
            log.info(f"Success, converged after {k} iterations")
        return x
