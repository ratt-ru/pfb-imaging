"""Primal-dual solver for the SARA (l21-regularised) deconvolution problem."""

from pfb_imaging.opt.primal_dual import PrimalDual, _nb_positivity, _nb_positivity_band
from pfb_imaging.prox.prox_21m import dual_update_numba_fast
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("SARA-PD")


class L21PrimalDual(PrimalDual):
    """Primal-dual solver for the l21-regularised imaging problem (SARA).

    Implements the same algorithm as :func:`~pfb_imaging.opt.primal_dual.primal_dual_numba`
    using the :class:`~pfb_imaging.opt.primal_dual.PrimalDual` template.

    Args:
        psi: Analysis operator ``psi(x, v)`` (image → wavelet coefficients).
        psih: Synthesis operator ``psih(v, xout)`` (coefficients → image).
        hessnorm: Spectral norm of the data-fidelity Hessian.
        lam: Regularisation strength.
        l1weight: Per-coefficient l1 weights, shape ``(nbasis, nymax, nxmax)``.
        reweighter: Optional callable ``reweighter(x) -> l1weight`` for
            iterative reweighting.  ``None`` disables reweighting.
        positivity: 0 = no constraint, 1 = per-pixel positivity,
            2 = per-spatial-pixel positivity across all bands (default 1).
        nu: Spectral norm of ``psi`` (default 1.0).
        gamma: Step-size safety factor (default 1.0).
        sigma: Dual step size (computed from other params when ``None``).
        tol: Convergence tolerance (default 1e-5).
        maxit: Maximum iterations (default 1000).
        report_freq: Logging frequency at verbosity > 1 (default 10).
        verbosity: Verbosity level 0–2 (default 1).
        maxreweight: Maximum consecutive reweighting steps before stopping
            (default 20).
    """

    def __init__(
        self,
        psi,
        psih,
        hessnorm,
        lam,
        l1weight,
        reweighter=None,
        positivity=1,
        nu=1.0,
        gamma=1.0,
        sigma=None,
        tol=1e-5,
        maxit=1000,
        report_freq=10,
        verbosity=1,
        maxreweight=20,
    ):
        super().__init__(psi, psih, hessnorm, nu, gamma, sigma, tol, maxit, report_freq, verbosity)
        self.lam = lam
        self.l1weight = l1weight
        self.reweighter = reweighter
        self.positivity = positivity
        self.maxreweight = maxreweight
        self._numreweight = 0
        self._last_reweight_iter = 0

    def dual_step(self, xp, v, vp):
        """Analysis + proximal dual update for the l21 norm."""
        self.psi(xp, v)
        dual_update_numba_fast(vp, v, self.lam, sigma=self.sigma, weight=self.l1weight)

    def prox_primal(self, x):
        """Apply positivity constraint in-place."""
        if self.positivity == 1:
            _nb_positivity(x)
        elif self.positivity == 2:
            _nb_positivity_band(x)

    def on_converge(self, x, k):
        """Perform one reweighting step; return ``True`` when done."""
        if self.reweighter is not None and self._numreweight < self.maxreweight:
            self.l1weight = self.reweighter(x)
            if k - self._last_reweight_iter == 1:
                self._numreweight += 1
            else:
                self._numreweight = 0
            self._last_reweight_iter = k
            return False  # continue iterating
        else:
            if self._numreweight >= self.maxreweight:
                log.info("Maximum reweighting steps reached")
            return True  # stop
