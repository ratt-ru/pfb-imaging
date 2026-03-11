"""SARA deconvolution using l21 regularisation and primal-dual splitting."""

import numpy as np

from pfb_imaging.deconv.sara import SARABase
from pfb_imaging.operators.psi import PsiOperatorProtocol
from pfb_imaging.opt.primal_dual import PrimalDual, _nb_positivity, _nb_positivity_band
from pfb_imaging.prox.prox_21m import dual_update_numba_fast
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("SARA-PD")


class L21PrimalDual(PrimalDual):
    """Primal-dual inner solver for the l21-regularised problem (SARA).

    Args:
        psi: Analysis/synthesis operator satisfying ``PsiOperatorProtocol``.
        hessnorm: Spectral norm of the data-fidelity Hessian.
        lam: Regularisation strength (updated via :meth:`set_lam`).
        l1weight: Per-coefficient l1 weights, shape
            ``(nbasis, nymax, nxmax)``.
        reweighter: Optional callable for iterative reweighting.
        positivity: Positivity constraint mode (0, 1, or 2).
        nu: Spectral norm of ``psi`` (default 1.0).
        gamma: Step-size safety factor (default 1.0).
        sigma: Dual step size (computed when ``None``).
        tol: Convergence tolerance (default 1e-5).
        maxit: Maximum iterations (default 1000).
        report_freq: Logging frequency (default 10).
        verbosity: Verbosity level 0-2 (default 1).
        maxreweight: Maximum consecutive inner reweighting steps (default 20).
    """

    def __init__(
        self,
        psi: PsiOperatorProtocol,
        hessnorm: float,
        lam: float,
        l1weight: np.ndarray,
        reweighter=None,
        positivity: int = 1,
        nu: float = 1.0,
        gamma: float = 1.0,
        sigma: float | None = None,
        tol: float = 1e-5,
        maxit: int = 1000,
        report_freq: int = 10,
        verbosity: int = 1,
        maxreweight: int = 20,
    ):
        super().__init__(psi, hessnorm, nu, gamma, sigma, tol, maxit, report_freq, verbosity)
        self.lam = lam
        self.l1weight = l1weight
        self.reweighter = reweighter
        self.positivity = positivity
        self.maxreweight = maxreweight
        self._numreweight = 0
        self._last_reweight_iter = 0

    def set_lam(self, lam: float) -> None:
        """Update the regularisation strength."""
        self.lam = lam

    def reset(self) -> None:
        """Reset reweighting counters before each outer iteration."""
        self._numreweight = 0
        self._last_reweight_iter = 0

    def dual_step(self, xp, v, vp):
        """Analysis + proximal dual update for the l21 norm."""
        self.psi.dot(xp, v)
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


class SARAPrimalDual(SARABase):
    """SARA deconvolution using the primal-dual opt backend.

    Satisfies the :class:`~pfb_imaging.deconv.DeconvSolver` Protocol.
    Inherits shared SARA setup from :class:`SARABase` and adds the
    primal-dual inner solver and dual variable.

    Args:
        pd_tol: Primal-dual convergence tolerance.
        pd_maxit: Maximum primal-dual iterations.
        pd_verbose: Primal-dual verbosity level.
        pd_report_freq: Primal-dual logging frequency.
        maxreweight: Maximum consecutive inner reweighting steps.

    All other args are forwarded to :class:`SARABase`.
    """

    def __init__(
        self,
        nband: int,
        nx: int,
        ny: int,
        abspsf: np.ndarray,
        beam: np.ndarray,
        wsums: np.ndarray,
        model: np.ndarray,
        update: np.ndarray,
        nthreads: int = 1,
        # precond params
        eta: float = 0.01,
        hess_approx: str = "psf",
        hess_norm: float | None = None,
        cg_tol: float = 1e-3,
        cg_maxit: int = 150,
        cg_verbose: int = 0,
        cg_report_freq: int = 10,
        taper_width: int = 32,
        # power method params
        pm_tol: float = 1e-3,
        pm_maxit: int = 100,
        pm_verbose: int = 1,
        pm_report_freq: int = 25,
        # psi params
        bases: str | list[str] = "self,db1,db2,db3",
        nlevels: int = 2,
        # algo params
        gamma: float = 1.0,
        positivity: int = 1,
        l1_reweight_from: int = 5,
        rmsfactor: float = 1.0,
        alpha: float = 2.0,
        # inner PD params
        pd_tol: float = 3e-4,
        pd_maxit: int = 450,
        pd_verbose: int = 1,
        pd_report_freq: int = 50,
        maxreweight: int = 20,
    ):
        super().__init__(
            nband,
            nx,
            ny,
            abspsf,
            beam,
            wsums,
            model,
            update,
            nthreads,
            eta=eta,
            hess_approx=hess_approx,
            hess_norm=hess_norm,
            cg_tol=cg_tol,
            cg_maxit=cg_maxit,
            cg_verbose=cg_verbose,
            cg_report_freq=cg_report_freq,
            taper_width=taper_width,
            pm_tol=pm_tol,
            pm_maxit=pm_maxit,
            pm_verbose=pm_verbose,
            pm_report_freq=pm_report_freq,
            bases=bases,
            nlevels=nlevels,
            gamma=gamma,
            positivity=positivity,
            l1_reweight_from=l1_reweight_from,
            rmsfactor=rmsfactor,
            alpha=alpha,
        )

        # dual variable
        nbasis = self._psi.nbasis
        nxmax = self._psi.nxmax
        nymax = self._psi.nymax
        self._dual = np.zeros((nband, nbasis, nymax, nxmax), dtype="f8")

        # inner solver
        self._solver = L21PrimalDual(
            self._psi,
            self.hess_norm,
            lam=1.0,  # updated each outer iteration in backward()
            l1weight=self._l1weight,
            reweighter=None,  # set via last() once reweighting is active
            positivity=positivity,
            gamma=gamma,
            tol=pd_tol,
            maxit=pd_maxit,
            verbosity=pd_verbose,
            report_freq=pd_report_freq,
            maxreweight=maxreweight,
        )

    def backward(self, lam: float) -> np.ndarray:
        """Run inner primal-dual solve; return updated model."""
        self._solver.set_lam(lam)
        # sync l1 weights from base (updated by last() on previous iteration)
        self._solver.l1weight = self._l1weight
        self._solver.reweighter = self._reweighter
        self._solver.reset()
        self._model, self._dual = self._solver.solve(self._model, self._dual)
        self._iter += 1
        return self._model
