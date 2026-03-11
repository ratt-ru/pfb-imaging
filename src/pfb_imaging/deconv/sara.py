"""Base class for SARA deconvolution with configurable opt backends."""

from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from pfb_imaging.operators.hessian import HessPSF
from pfb_imaging.operators.psi import Psi
from pfb_imaging.opt.power_method import power_method_numba as power_method
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.misc import l1reweight_func

log = pfb_logging.get_logger("SARA")


class SARABase(ABC):
    """Shared SARA setup for any opt backend.

    Constructs the preconditioner (:class:`HessPSF`), wavelet dictionary
    (:class:`Psi`), and manages l1 reweighting state.  Implements
    ``first``, ``forward``, and ``last`` from the
    :class:`~pfb_imaging.deconv.DeconvSolver` Protocol.

    Subclasses must implement :meth:`backward` and set ``self._solver``
    to an inner solver with ``set_grad`` and ``set_lam`` methods.

    Args:
        nband: Number of imaging bands.
        nx: Image size in x.
        ny: Image size in y.
        abspsf: Absolute value of PSF Fourier transform,
            shape ``(nband, nx_psf, nyo2)``.
        beam: Primary beam, shape ``(nband, nx, ny)``.
        wsums: Per-band imaging weights, shape ``(nband,)``.
        model: Initial model image, shape ``(nband, nx, ny)``.
        update: Initial preconditioned gradient, shape ``(nband, nx, ny)``.
        nthreads: Number of threads for wavelets, FFTs, and CG.
        eta: Diagonal loading factor for the Hessian approximation.
        hess_approx: Mode for ``HessPSF.idot``.
        hess_norm: Spectral norm of the Hessian (estimated when ``None``).
        cg_tol: CG convergence tolerance.
        cg_maxit: Maximum CG iterations.
        cg_verbose: CG verbosity level.
        cg_report_freq: CG logging frequency.
        taper_width: Image-space taper width in pixels.
        pm_tol: Power-method tolerance.
        pm_maxit: Power-method maximum iterations.
        pm_verbose: Power-method verbosity.
        pm_report_freq: Power-method logging frequency.
        bases: Wavelet basis names (comma-separated str or list of str).
        nlevels: Wavelet decomposition levels.
        gamma: PFB step-size factor.
        positivity: Positivity constraint mode (0, 1, or 2).
        l1_reweight_from: Start l1 reweighting after this many outer
            iterations.
        rmsfactor: Threshold factor for l1 reweighting.
        alpha: Exponent in the l1 reweighting formula.
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
    ):
        if l1_reweight_from == 0 and not update.any():
            raise ValueError("Cannot reweight before any updates: update is all zeros")

        # --- build preconditioner ---
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
        if isinstance(bases, str):
            bases_tuple = tuple(bases.split(","))
        else:
            bases_tuple = tuple(bases)
        self._psi = Psi(nband, nx, ny, bases_tuple, nlevels, nthreads)
        self._bases_tuple = bases_tuple
        nbasis = self._psi.nbasis
        nxmax = self._psi.nxmax
        nymax = self._psi.nymax
        log.info(f"Using {self._psi.nthreads_per_band} numba threads per band")

        # --- state ---
        self._model = model.copy()
        self._update = update.copy()
        self._residual = np.zeros_like(model)

        # --- l1 weights and reweighting ---
        self._outvar = np.zeros((nband, nbasis, nymax, nxmax), dtype="f8")
        self._l1weight = np.ones((nbasis, nymax, nxmax), dtype="f8")
        self._reweighter = None
        self._rmsfactor = rmsfactor
        self._alpha = alpha
        self._l1_reweight_from = l1_reweight_from
        self._positivity = positivity

        # --- shared solver state ---
        self._gamma = gamma
        self._iter = 0

    # --- DeconvSolver interface (concrete) ---

    def first(self, residual: np.ndarray) -> None:
        """Apply beam to residual; store result for forward()."""
        self._residual = residual * self._beam

    def forward(self, residual: np.ndarray) -> np.ndarray:
        """Preconditioned gradient step; sets gradient for backward().

        Returns:
            update: Preconditioned gradient, shape ``(nband, nx, ny)``.
        """
        self._update = self._precond.idot(
            self._residual,
            mode=self._hess_approx,
            x0=self._update if self._update.any() else None,
        )
        xtilde = self._model + self._gamma * self._update

        def grad(x):
            return -self._precond.dot(xtilde - x) / self._gamma

        self._solver.set_grad(grad)
        return self._update

    @abstractmethod
    def backward(self, lam: float) -> np.ndarray:
        """Solve the backward (proximal) step.

        Args:
            lam: Regularisation strength for this outer iteration.

        Returns:
            model: Updated model image, shape ``(nband, nx, ny)``.
        """

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

    # --- extra methods (not part of DeconvSolver Protocol) ---

    @property
    def reweight_active(self) -> bool:
        """True once l1 reweighting has been triggered."""
        return self._reweighter is not None

    def trigger_reweight(self) -> None:
        """Force reweighting to start at the next :meth:`last` call."""
        self._l1_reweight_from = self._iter
