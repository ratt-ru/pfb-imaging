"""SARA deconvolution using l21 regularisation and forward-backward splitting."""

import numpy as np

from pfb_imaging.deconv.sara import SARABase
from pfb_imaging.operators.psi import PsiOperatorProtocol
from pfb_imaging.opt.forward_backward import ForwardBackward
from pfb_imaging.opt.primal_dual import _nb_positivity, _nb_positivity_band
from pfb_imaging.prox.prox_21m import prox_21m_numba
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("SARA-FB")


class L21ForwardBackward(ForwardBackward):
    """Forward-backward inner solver for the l21-regularised problem (SARA).

    Uses the tight-frame proximal trick to handle composition of
    the l21 norm with the wavelet analysis operator Psi::

        prox(x) = x + (1/nu) * Psi * (soft_thresh(Psi^T x) - Psi^T x)

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
        tol: Convergence tolerance (default 1e-5).
        maxit: Maximum iterations (default 1000).
        report_freq: Logging frequency (default 10).
        verbosity: Verbosity level 0-2 (default 1).
        acceleration: Enable FISTA acceleration (default True).
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
        tol: float = 1e-5,
        maxit: int = 1000,
        report_freq: int = 10,
        verbosity: int = 1,
        acceleration: bool = True,
        maxreweight: int = 20,
    ):
        super().__init__(hessnorm, gamma, tol, maxit, report_freq, verbosity, acceleration)
        self.psi = psi
        self.lam = lam
        self.l1weight = l1weight
        self.reweighter = reweighter
        self.positivity = positivity
        self.nu = nu
        self.maxreweight = maxreweight
        self._numreweight = 0
        self._last_reweight_iter = 0

        # buffers for tight-frame prox
        nband = psi.nband
        nbasis = psi.nbasis
        nymax = psi.nymax
        nxmax = psi.nxmax
        self._alpha = np.zeros((nband, nbasis, nymax, nxmax), dtype="f8")
        self._alpha_buf = np.zeros_like(self._alpha)
        self._xout = np.zeros((nband, psi.nx, psi.ny), dtype="f8")

    def set_lam(self, lam: float) -> None:
        """Update the regularisation strength."""
        self.lam = lam

    def reset(self) -> None:
        """Reset reweighting counters before each outer iteration."""
        self._numreweight = 0
        self._last_reweight_iter = 0

    def prox(self, x, step):
        """Tight-frame proximal operator for lam * ||W * Psi^T x||_{2,1}.

        Followed by positivity constraint.
        """
        # analysis: alpha = Psi^T x
        self.psi.dot(x, self._alpha)

        # soft-threshold in coefficient domain
        prox_21m_numba(
            self._alpha,
            self._alpha_buf,
            step * self.lam,
            sigma=1.0,
            weight=self.l1weight,
        )

        # tight-frame correction: x += (1/nu) * Psi * (prox(alpha) - alpha)
        self._alpha_buf -= self._alpha
        self.psi.hdot(self._alpha_buf, self._xout)
        x += self._xout / self.nu

        # positivity constraint
        if self.positivity == 1:
            _nb_positivity(x)
        elif self.positivity == 2:
            _nb_positivity_band(x)

        return x

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


class SARAForwardBackward(SARABase):
    """SARA deconvolution using the forward-backward opt backend.

    Satisfies the :class:`~pfb_imaging.deconv.DeconvSolver` Protocol.
    Uses FISTA acceleration by default.

    Args:
        fb_tol: Forward-backward convergence tolerance.
        fb_maxit: Maximum forward-backward iterations.
        fb_verbose: Forward-backward verbosity level.
        fb_report_freq: Forward-backward logging frequency.
        acceleration: Enable FISTA acceleration.
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
        # inner FB params
        fb_tol: float = 3e-4,
        fb_maxit: int = 450,
        fb_verbose: int = 1,
        fb_report_freq: int = 50,
        acceleration: bool = True,
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

        # inner solver
        self._solver = L21ForwardBackward(
            self._psi,
            self.hess_norm,
            lam=1.0,  # updated each outer iteration in backward()
            l1weight=self._l1weight,
            reweighter=None,  # set via last() once reweighting is active
            positivity=positivity,
            gamma=gamma,
            tol=fb_tol,
            maxit=fb_maxit,
            verbosity=fb_verbose,
            report_freq=fb_report_freq,
            acceleration=acceleration,
            maxreweight=maxreweight,
        )

    def backward(self, lam: float) -> np.ndarray:
        """Run inner forward-backward solve; return updated model."""
        self._solver.set_lam(lam)
        # sync l1 weights from base (updated by last() on previous iteration)
        self._solver.l1weight = self._l1weight
        self._solver.reweighter = self._reweighter
        self._solver.reset()
        self._model = self._solver.solve(self._model)
        self._iter += 1
        return self._model
