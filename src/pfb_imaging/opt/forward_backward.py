"""Forward-backward splitting algorithm (with optional FISTA acceleration)."""

from abc import ABC, abstractmethod
from time import time

import numpy as np

from pfb_imaging.opt.primal_dual import _nb_any_nonzero, _nb_norm_diff
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("FB")


class ForwardBackward(ABC):
    """Abstract base class for forward-backward splitting algorithms.

    The fixed algorithm loop lives in :meth:`solve`.  Subclasses override
    :meth:`prox` with the proximal operator for their specific regulariser.

    The algorithm at each iteration computes::

        x = prox(y - step * grad(y), step)

    With optional FISTA acceleration::

        t' = (1 + sqrt(1 + 4*t^2)) / 2
        y  = x + (t - 1) / t' * (x - xp)

    Args:
        hessnorm: Spectral norm (Lipschitz constant) of the smooth term's
            gradient.  Used to set the step size ``step = 1 / hessnorm``.
        gamma: Safety factor applied to the step size (default 1.0).
            Effective step is ``1 / (hessnorm / (2 * gamma))``.
        tol: Convergence tolerance on relative primal change (default 1e-5).
        maxit: Maximum number of iterations (default 1000).
        report_freq: Log progress every this many iterations at verbosity > 1
            (default 10).
        verbosity: 0 = silent, 1 = convergence message, 2 = per-iter logging
            (default 1).
        acceleration: Enable FISTA acceleration (default True).
    """

    def __init__(
        self,
        hessnorm: float,
        gamma: float = 1.0,
        tol: float = 1e-5,
        maxit: int = 1000,
        report_freq: int = 10,
        verbosity: int = 1,
        acceleration: bool = True,
    ):
        self.hessnorm = hessnorm
        self.gamma = gamma
        self.tol = tol
        self.maxit = maxit
        self.report_freq = report_freq
        self.verbosity = verbosity
        self.acceleration = acceleration
        self._grad = None

        self.step = 2.0 * gamma / hessnorm

    def set_grad(self, grad):
        """Set (or replace) the gradient of the smooth data-fidelity term.

        Args:
            grad: Callable ``grad(x) -> array`` returning the gradient.
        """
        self._grad = grad

    @abstractmethod
    def prox(self, x, step):
        """Apply the proximal operator in-place or return the result.

        Computes ``prox_{step * R}(x)`` where ``R`` is the non-smooth
        regulariser.

        Args:
            x: Input array (may be modified in-place).
            step: Step size scaling the regulariser.

        Returns:
            Result of the proximal operator (may be ``x`` itself if
            modified in-place).
        """

    def on_converge(self, x, k):
        """Hook called when ``eps < tol``.  Return ``True`` to stop.

        Override in subclasses to implement reweighting or other logic.
        The default always returns ``True`` (stop immediately).

        Args:
            x: Current iterate.
            k: Current iteration index.

        Returns:
            ``True`` to terminate, ``False`` to continue.
        """
        return True

    def norm_diff(self, x, xp):
        """Return relative norm of change: ``||x - xp|| / ||x||``."""
        return _nb_norm_diff(x, xp)

    def any_nonzero(self, x):
        """Return ``True`` if any element of ``x`` is nonzero."""
        return _nb_any_nonzero(x)

    def solve(self, x):
        """Run the forward-backward loop.

        Args:
            x: Initial variable (modified in-place and returned).

        Returns:
            The final iterate ``x``.

        Raises:
            RuntimeError: If :meth:`set_grad` has not been called.
        """
        if self._grad is None:
            raise RuntimeError("grad not set; call set_grad() before solve()")

        xp = x.copy()
        y = x.copy()
        t = 1.0

        eps = 1.0
        tii = time()
        for k in range(self.maxit):
            # gradient step
            x = y - self.step * self._grad(y)

            # proximal step
            x = self.prox(x, self.step)

            # convergence check
            if self.any_nonzero(x):
                eps = self.norm_diff(x, xp)
            else:
                eps = 1.0

            if eps < self.tol:
                if self.on_converge(x, k):
                    break

            # FISTA acceleration
            if self.acceleration:
                tp = t
                t = (1.0 + np.sqrt(1.0 + 4.0 * tp**2)) / 2.0
                y = x + (tp - 1.0) / t * (x - xp)
            else:
                np.copyto(y, x)

            np.copyto(xp, x)

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

        return x
