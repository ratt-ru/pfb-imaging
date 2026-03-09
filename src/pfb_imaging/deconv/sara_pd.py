"""SARA deconvolution using l21 regularisation and primal-dual splitting."""

from functools import partial

import numpy as np

from pfb_imaging.operators.hessian import HessPSF
from pfb_imaging.operators.psi import Psi, PsiOperatorProtocol
from pfb_imaging.opt.power_method import power_method_numba as power_method
from pfb_imaging.opt.primal_dual import PrimalDual, _nb_positivity, _nb_positivity_band
from pfb_imaging.prox.prox_21m import dual_update_numba_fast
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.misc import l1reweight_func

log = pfb_logging.get_logger("SARA-PD")


class L21PrimalDual(PrimalDual):
    """Primal-dual solver for the l21-regularised imaging problem (SARA).

    Implements the same algorithm as
    :func:`~pfb_imaging.opt.primal_dual.primal_dual_numba` using the
    :class:`~pfb_imaging.opt.primal_dual.PrimalDual` template.

    Args:
        psi: Analysis/synthesis operator satisfying ``PsiOperatorProtocol``.
        hessnorm: Spectral norm of the data-fidelity Hessian.
        lam: Regularisation strength (updated between outer iterations via
            :meth:`set_lam`).
        l1weight: Per-coefficient l1 weights, shape
            ``(nbasis, nymax, nxmax)``.
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
        super().__init__(hessnorm, psi, nu, gamma, sigma, tol, maxit, report_freq, verbosity)
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


class SARAPrimalDual:
    """SARA deconvolution using the l21 norm and primal-dual splitting.

    Satisfies the :class:`~pfb_imaging.deconv.DeconvSolver` Protocol.
    Constructs :class:`~pfb_imaging.operators.hessian.HessPSF` and
    :class:`~pfb_imaging.operators.psi.Psi` internally so that all
    regulariser-specific setup is encapsulated here.

    Args:
        nband: Number of imaging bands.
        nx: Image size in x.
        ny: Image size in y.
        abspsf: Absolute value of PSF Fourier transform,
            shape ``(nband, nx_psf, nyo2)``.
        beam: Primary beam, shape ``(nband, nx, ny)``.
        wsums: Per-band imaging weights (sum of visibility weights),
            shape ``(nband,)``.  Should be normalised before passing.
        model: Initial model image, shape ``(nband, nx, ny)``.
        update: Initial preconditioned gradient for CG warm-start,
            shape ``(nband, nx, ny)``.
        nthreads: Number of threads for wavelets, FFTs, and CG.
        eta: Diagonal loading factor for the Hessian approximation.
        hess_approx: Mode for :meth:`HessPSF.idot` (``"psf"`` or
            ``"direct"``).
        hess_norm: Spectral norm of the Hessian.  Estimated via the power
            method when ``None``.
        cg_tol: CG convergence tolerance.
        cg_maxit: Maximum CG iterations.
        cg_verbose: CG verbosity level.
        cg_report_freq: CG logging frequency.
        taper_width: Width of the image-space taper in pixels.
        pm_tol: Power-method convergence tolerance.
        pm_maxit: Maximum power-method iterations.
        pm_verbose: Power-method verbosity level.
        pm_report_freq: Power-method logging frequency.
        bases: Comma-separated wavelet basis names.
        nlevels: Number of wavelet decomposition levels.
        gamma: PFB step-size factor (scales the gradient step).
        positivity: Positivity constraint mode (0, 1, or 2).
        l1_reweight_from: Start l1 reweighting after this many outer
            iterations.  Set to 0 to reweight from the first iteration
            (requires a non-zero ``update``).
        rmsfactor: Threshold factor relative to the residual rms used in
            l1 reweighting.
        alpha: Exponent in the l1 reweighting formula.
        pd_tol: Primal-dual convergence tolerance.
        pd_maxit: Maximum primal-dual iterations.
        pd_verbose: Primal-dual verbosity level.
        pd_report_freq: Primal-dual logging frequency.
        maxreweight: Maximum consecutive inner reweighting steps.
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
        bases: str = "self,db1,db2,db3",
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
        if l1_reweight_from == 0 and not update.any():
            raise ValueError("Cannot reweight before any updates: update is all zeros")

        # --- build preconditioner (owns beam for first()) ---
        self._beam = beam
        self._precond = HessPSF(
            nx,
            ny,
            abspsf,
            beam=beam,
            eta=eta * wsums,
            nthreads=nthreads,
            cgtol=cg_tol,
            cgmaxit=cg_maxit,
            cgverbose=cg_verbose,
            cgrf=cg_report_freq,
            taper_width=np.minimum(int(0.1 * nx), taper_width),
        )
        self._hess_approx = hess_approx

        # estimate hess_norm if not provided
        if hess_norm is None:
            log.info("Finding spectral norm of Hessian approximation")
            hess_norm, _ = power_method(
                self._precond.dot,
                (nband, nx, ny),
                tol=pm_tol,
                maxit=pm_maxit,
                verbosity=pm_verbose,
                report_freq=pm_report_freq,
            )
            hess_norm *= 1.05
        self.hess_norm = hess_norm
        log.info(f"Using hess_norm = {hess_norm:.3e}")

        # --- build wavelet dictionary ---
        bases_tuple = tuple(bases.split(","))
        self._psi = Psi(nband, nx, ny, bases_tuple, nlevels, nthreads)
        self._bases_tuple = bases_tuple
        nbasis = self._psi.nbasis
        nxmax = self._psi.nxmax
        nymax = self._psi.nymax
        log.info(f"Using {self._psi.nthreads_per_band} numba threads per band")

        # --- state ---
        self._model = model.copy()
        self._dual = np.zeros((nband, nbasis, nymax, nxmax), dtype="f8")
        self._update = update.copy()
        self._residual = np.zeros_like(model)

        # --- l1 weights and reweighting ---
        self._outvar = np.zeros((nband, nbasis, nymax, nxmax), dtype="f8")
        self._l1weight = np.ones((nbasis, nymax, nxmax), dtype="f8")
        self._reweighter = None
        self._rmsfactor = rmsfactor
        self._alpha = alpha
        self._l1_reweight_from = l1_reweight_from

        # --- inner solver ---
        self._gamma = gamma
        self._solver = L21PrimalDual(
            self._psi,
            hess_norm,
            lam=1.0,  # updated each outer iteration in backward()
            l1weight=self._l1weight,
            reweighter=None,  # set in last() once reweighting is active
            positivity=positivity,
            gamma=gamma,
            tol=pd_tol,
            maxit=pd_maxit,
            verbosity=pd_verbose,
            report_freq=pd_report_freq,
            maxreweight=maxreweight,
        )
        self._iter = 0

    # --- DeconvSolver interface ---

    def first(self, residual: np.ndarray) -> None:
        """Apply beam to residual; store result for forward()."""
        self._residual = residual * self._beam

    def forward(self, residual: np.ndarray) -> np.ndarray:
        """Preconditioned gradient step; sets gradient for backward().

        Args:
            residual: Current residual image, shape ``(nband, nx, ny)``.
                The beam-multiplied version from :meth:`first` is used.

        Returns:
            update: Preconditioned gradient, shape ``(nband, nx, ny)``.
        """
        self._update = self._precond.idot(
            self._residual,
            mode=self._hess_approx,
            x0=self._update if self._update.any() else None,
        )
        xtilde = self._model + self._gamma * self._update

        def grad21(x):
            return -self._precond.dot(xtilde - x) / self._gamma

        self._solver.set_grad(grad21)
        return self._update

    def backward(self, lam: float) -> np.ndarray:
        """Run inner primal-dual solve; return updated model.

        Args:
            lam: Regularisation strength for this outer iteration.

        Returns:
            model: Updated model image, shape ``(nband, nx, ny)``.
        """
        self._solver.set_lam(lam)
        self._solver.reset()
        self._model, self._dual = self._solver.solve(self._model, self._dual)
        self._iter += 1
        return self._model

    def last(self) -> None:
        """Update l1 weights when the reweighting threshold is reached."""
        if self._iter < self._l1_reweight_from:
            return
        log.info("Computing L1 weights")
        self._psi.dot(self._update, self._outvar)
        tmp = np.sum(self._outvar, axis=0)
        rms_comps = np.ones(self._psi.nbasis, dtype=float)
        for i, base in enumerate(self._bases_tuple):
            tmpb = tmp[i]
            rms_comps[i] = np.std(tmpb[tmpb != 0])
            log.info(f"rms_comps for base {base} is {rms_comps[i]:.3e}")
        self._reweighter = partial(
            l1reweight_func,
            psih=self._psi.dot,
            outvar=self._outvar,
            rmsfactor=self._rmsfactor,
            rms_comps=rms_comps,
            alpha=self._alpha,
        )
        self._l1weight = self._reweighter(self._model)
        self._solver.l1weight = self._l1weight
        self._solver.reweighter = self._reweighter

    # --- extra methods (not part of DeconvSolver Protocol) ---

    @property
    def reweight_active(self) -> bool:
        """True once l1 reweighting has been triggered."""
        return self._reweighter is not None

    def trigger_reweight(self) -> None:
        """Force reweighting to start at the next :meth:`last` call.

        Called by the outer loop when convergence is detected before the
        scheduled reweighting iteration.
        """
        self._l1_reweight_from = self._iter
