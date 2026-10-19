"""PFBSolver: the concrete DeconvSolver composing (hess, forward, backward, prox)."""

import inspect

import numpy as np

from pfb_imaging.deconv import Regulariser
from pfb_imaging.operators import LinearOperator, require_protocol
from pfb_imaging.opt import BackwardSolver, ForwardSolver
from pfb_imaging.opt.power_method import power_method_numba as power_method
from pfb_imaging.utils import inplace
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("PFB")


def reusing_dot(op, like):
    """``op.dot`` writing into one buffer reused across calls, when ``op`` takes ``out``.

    For loops that consume each result before the next call (gradient, CG,
    power method): a fresh cube per call costs first-touch page faults over
    all of it (D56). Operators without ``out`` fall back to plain ``dot``.
    """
    if "out" not in inspect.signature(op.dot).parameters:
        return op.dot
    buf = np.empty_like(like)
    return lambda x: op.dot(x, out=buf)


class ReweightOnConverge:
    """on_converge callback driving inner l1 reweighting.

    Fired by an iterative BackwardSolver when ``eps < tol``.  While the
    regulariser's reweighting is armed and the consecutive-reweight cap is
    not reached, reweights and returns False (keep iterating); otherwise
    returns True (stop).  Single implementation of the counter formerly
    duplicated in sara_fb.py/sara_pd.py.

    Args:
        regulariser: Object with ``reweight_active`` and ``update_weights``.
        maxreweight: Maximum consecutive reweighting steps.
        verbosity: > 1 logs convergence-event metrics.
    """

    def __init__(self, regulariser, maxreweight: int = 20, verbosity: int = 1):
        self.reg = regulariser
        self.maxreweight = maxreweight
        self.verbosity = verbosity
        self._num = 0
        self._last_iter = 0

    def reset(self) -> None:
        """Clear the consecutive-reweight counter (call before each solve)."""
        self._num = 0
        self._last_iter = 0

    def __call__(self, x, k: int, eps: float) -> bool:
        if self.reg.reweight_active and self._num < self.maxreweight:
            self.reg.update_weights(x)
            if k - self._last_iter == 1:
                self._num += 1
            else:
                self._num = 0
            self._last_iter = k
            if self.verbosity > 1:
                log.info(f"Reweighted at iteration {k}, eps = {eps:.3e}, consecutive = {self._num}")
            return False
        if self._num >= self.maxreweight and self.verbosity:
            log.info("Maximum reweighting steps reached")
        return True


class PFBSolver:
    """Preconditioned forward-backward solver from four composable pieces.

    Satisfies the ``DeconvSolver`` Protocol consumed by the outer major-cycle
    loop.  All wiring between the pieces lives here: the grad closure built
    from ``hess.dot``, the ``backward_alg.setup`` binding, and the
    ``ReweightOnConverge`` installation (plain attribute assignment on the
    concrete solver — the hook is not part of any Protocol).

    Args:
        hess: Data-fidelity Hessian satisfying ``LinearOperator``.
        forward_alg: ``ForwardSolver`` for ``update ≈ hess^{-1} residual``.
        backward_alg: ``BackwardSolver`` for the proximal step.
        prox: ``Regulariser``; may expose the optional reweighting trio.
        model: Initial model image ``(nband, nx, ny)``.
        update: Initial update image ``(nband, nx, ny)``.
        gamma: PFB step size for ``xtilde = model + gamma * update``.
        hessnorm: Spectral norm of ``hess`` (power method when None).
        l1_reweight_from: Arm reweighting after this many major cycles
            (negative disables).
        maxreweight: Cap for ``ReweightOnConverge``.
        pm_tol, pm_maxit, pm_verbose, pm_report_freq: power-method controls.
        verbosity: Logging level.
    """

    def __init__(
        self,
        hess,
        forward_alg,
        backward_alg,
        prox,
        *,
        model: np.ndarray,
        update: np.ndarray,
        gamma: float = 1.0,
        hessnorm: float | None = None,
        l1_reweight_from: int = 5,
        maxreweight: int = 20,
        pm_tol: float = 1e-3,
        pm_maxit: int = 100,
        pm_verbose: int = 1,
        pm_report_freq: int = 25,
        verbosity: int = 1,
    ):
        require_protocol(hess, LinearOperator, "hess")
        require_protocol(forward_alg, ForwardSolver, "forward_alg")
        require_protocol(backward_alg, BackwardSolver, "backward_alg")
        require_protocol(prox, Regulariser, "prox")
        self.hess = hess
        self.forward_alg = forward_alg
        self.backward_alg = backward_alg
        self.reg = prox
        self._model = model.copy()
        self._update = update.copy()
        self._residual = None  # set by first()
        self._xtilde = None  # model + gamma * update, formed by the backward step
        self.model_change = None  # relative model change of the last backward step
        self._gamma = gamma
        self._l1_reweight_from = l1_reweight_from
        self._iter = 0

        if hessnorm is None:
            log.info("Finding spectral norm of Hessian approximation")
            hessnorm, _ = power_method(
                reusing_dot(hess, model),
                model.shape,
                tol=pm_tol,
                maxit=pm_maxit,
                verbosity=pm_verbose,
                report_freq=pm_report_freq,
            )
            hessnorm *= 1.05
        self.hess_norm = hessnorm
        log.info(f"Using hess_norm = {hessnorm:.3e}")

        backward_alg.setup(prox, hessnorm)

        self._reweight_cb = None
        if hasattr(prox, "update_weights") and hasattr(prox, "reweight_active"):
            self._reweight_cb = ReweightOnConverge(prox, maxreweight=maxreweight, verbosity=verbosity)
            if getattr(backward_alg, "on_converge", None) is None:
                backward_alg.on_converge = self._reweight_cb

    # --- DeconvSolver interface ---

    def first(self, residual: np.ndarray) -> None:
        """Cache the beam-attenuated gradient residual (BRESIDUAL/wsum).

        The Hessian applies the beam on both sides, so the forward solve's rhs
        must carry the outer per-partition beam too (legacy sara's
        ``residual *= beam``; wiki D23). The caller passes the attenuated cube;
        the apparent residual never enters the solver.
        """
        self._residual = residual

    def forward(self, residual: np.ndarray) -> np.ndarray:
        """Forward solve: ``update ≈ hess^{-1} residual``.

        Consumes the residual stored by :meth:`first` (the per-iteration
        preprocessing hook of the DeconvSolver contract, e.g. beam
        application); the ``residual`` argument is part of the Protocol
        signature and is not read directly.
        """
        if self._residual is None:
            raise RuntimeError("residual not set; call first() before forward()")
        x0 = self._update if self._update.any() else None
        self._update = self.forward_alg.solve(self.hess, self._residual, x0=x0)
        # consumed: dropping it lets the caller free the gradient's cube
        # while the backward step runs (the next first() supplies a new one)
        self._residual = None
        return self._update

    def _install_grad(self):
        """Give the backward solver ``grad(x) = -hess(xtilde - x) / gamma``.

        Installed by ``backward`` for the length of one solve: its three cubes
        (``xtilde``, the gap and its image under ``hess``) are released when
        the solve returns, so neither the CG of the next forward step nor the
        residual computed in between shares memory with them (D58).
        """
        diff = None
        hdot = reusing_dot(self.hess, self._update)
        gscale = -1.0 / self._gamma

        def grad(x):
            # -hess(xtilde - x) / gamma; the returned cube is reused by the
            # next call, so callers consume it before calling again
            nonlocal diff
            if diff is None:
                diff = np.empty_like(x)
            inplace.axpby(diff, 1.0, self._form_xtilde(), -1.0, x)
            g = hdot(diff)
            inplace.scale(g, gscale)
            return g

        self.backward_alg.set_grad(grad)

    def _form_xtilde(self):
        """``model + gamma * update`` at the start of the backward step.

        Formed on first use, not in ``forward``, so a forward solve that no
        backward step follows (the mop) never allocates it.
        """
        if self._xtilde is None:
            self._xtilde = np.empty_like(self._model)
            inplace.axpby(self._xtilde, 1.0, self._model, self._gamma, self._update)
        return self._xtilde

    def backward(self, lam: float) -> np.ndarray:
        """Backward (proximal) solve; returns the updated model.

        Also sets ``model_change``, ``||model_new - model_old|| / ||model_new||``,
        from ``model_old = xtilde - gamma * update`` so that the driver needs no
        copy of the model to measure it.
        """
        if self._reweight_cb is not None:
            self._reweight_cb.reset()
        self._install_grad()
        xtilde = self._form_xtilde()
        try:
            self._model = self.backward_alg.solve(self._model, lam)
            self.model_change = inplace.rel_diff(self._model, xtilde, self._update, -self._gamma)
        finally:
            self.backward_alg.set_grad(None)
            self._xtilde = None
        self._iter += 1
        return self._model

    @property
    def model(self) -> np.ndarray:
        """The solver's model cube (updated in place by ``backward``)."""
        return self._model

    @property
    def update(self) -> np.ndarray:
        """The latest forward-step update cube."""
        return self._update

    def last(self) -> None:
        """Arm/refresh l1 reweighting once the threshold is reached."""
        if not hasattr(self.reg, "init_reweighting"):
            return
        if self._l1_reweight_from < 0 or self._iter < self._l1_reweight_from:
            return
        log.info("Computing L1 weights")
        self.reg.init_reweighting(self._update)
        self.reg.update_weights(self._model)

    # --- driver sniffing (matches the legacy SARABase extras) ---

    @property
    def reweight_active(self) -> bool:
        """True when the driver should stop at convergence rather than trigger reweighting."""
        if not hasattr(self.reg, "init_reweighting") or self._l1_reweight_from < 0:
            return True  # no reweighting to trigger (plain regulariser, or disabled)
        return self.reg.reweight_active

    def trigger_reweight(self) -> None:
        """Force reweighting to arm at the next :meth:`last` call."""
        self._l1_reweight_from = self._iter
