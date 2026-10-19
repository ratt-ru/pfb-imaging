"""Weighted l21 regulariser over a wavelet dictionary (the SARA prior)."""

from functools import partial

import numpy as np

from pfb_imaging.operators import PsiOperator, require_protocol
from pfb_imaging.prox.prox_21m import dual_update_numba_fast, l21_dual_scale, prox_21m_numba
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.misc import l1reweight_func

log = pfb_logging.get_logger("L21")


class L21:
    """Satisfies the ``Regulariser`` Protocol; owns the l1-reweighting state.

    R(x) = ||W Psi^T x||_{2,1} with the 2-norm over the band axis.

    Args:
        psi: Operator satisfying ``PsiOperator`` (e.g. ``Psi``/``PsiNocopytRay``).
        bases: Wavelet basis names, one per ``psi.nbasis`` (for logging).
        nu: Spectral norm of ``psi`` (default 1.0; tight frame).
        rmsfactor: Threshold factor in the reweighting formula.
        alpha: Exponent in the reweighting formula.
    """

    def __init__(self, psi, bases, nu: float = 1.0, rmsfactor: float = 1.0, alpha: float = 2.0):
        require_protocol(psi, PsiOperator, "psi")
        self.psi = psi
        self.nu = nu
        self.bases = tuple(bases)
        self.rmsfactor = rmsfactor
        self.alpha = alpha
        self.l1weight = np.ones((psi.nbasis, psi.nymax, psi.nxmax))
        self._reweighter = None
        self._vsum = None

    def prox(self, v, vout, lam, sigma=1.0):
        """vout = prox_{(lam/sigma) ||W .||_{21}}(v/sigma), in-place."""
        prox_21m_numba(v, vout, lam, sigma=sigma, weight=self.l1weight)

    def dual_update(self, vp, v, lam, sigma=1.0):
        """Fused primal-dual dual update (fast path sniffed by PrimalDual)."""
        dual_update_numba_fast(vp, v, lam, sigma=sigma, weight=self.l1weight)

    @property
    def holds_dual(self) -> bool:
        """True when ``psi`` keeps the dual per band in its workers, so ``dual_step`` applies."""
        return hasattr(self.psi, "dual_begin")

    def reset_dual(self):
        """Zero the dual held by ``psi``'s workers and its band sum held here."""
        self.psi.init_dual()
        self._vsum = None

    def dual_step(self, xp, xout, lam, sigma=1.0):
        """The whole dual half of a primal-dual iteration, with the dual held in ``psi``'s workers.

        ``xout = Psi(2 v_new - v)`` with ``v_new = dual_update(v, Psi^T xp)``,
        the same iterate as ``psi.dot`` + ``dual_update`` + extrapolation +
        ``psi.hdot``. No per-band coefficient cube reaches the driver: it holds
        only the band-summed dual, which is all the bands share
        (``prox_21m.l21_dual_scale``).
        """
        c = self.psi.dual_begin(xp, sigma)
        if c.shape != self.l1weight.shape:
            raise ValueError(f"coefficients have shape {c.shape}, l1 weights {self.l1weight.shape}")
        if self._vsum is None:
            self._vsum = np.zeros_like(c)
        l21_dual_scale(self._vsum, c, lam, sigma, self.l1weight)
        self.psi.dual_finish(c, xout)

    @property
    def reweight_active(self) -> bool:
        """True once reweighting has been initialised."""
        return self._reweighter is not None

    def _coeff_sum(self, x):
        """``sum_b Psi^T x_b``: all reweighting needs, and linear in ``x``.

        A ``psi`` with ``dot_sum`` analyses the band sum once, so no
        ``(nband, nbasis, ...)`` coefficient cube is ever allocated (4 cubes at
        4 bases; ~20 GB at 8 bands x 8748^2).
        """
        dot_sum = getattr(self.psi, "dot_sum", None)
        if dot_sum is not None:
            return dot_sum(x)
        out = np.zeros((self.psi.nband, self.psi.nbasis, self.psi.nymax, self.psi.nxmax))
        self.psi.dot(x, out)
        return np.sum(out, axis=0)

    def init_reweighting(self, update):
        """Estimate per-basis component rms from the update and arm reweighting."""
        tmp = self._coeff_sum(update)
        rms_comps = np.ones(self.psi.nbasis, dtype=float)
        for i, base in enumerate(self.bases):
            nonzero = tmp[i][tmp[i] != 0]
            if nonzero.size == 0:
                # np.std([]) is NaN and would poison the weights via
                # l1reweight_func; keep the unit default instead
                log.warning(f"No nonzero coefficients for base {base}; using rms_comps = 1.0")
                continue
            rms_comps[i] = np.std(nonzero)
            log.info(f"rms_comps for base {base} is {rms_comps[i]:.3e}")
        self._reweighter = partial(
            l1reweight_func,
            rmsfactor=self.rmsfactor,
            rms_comps=rms_comps,
            alpha=self.alpha,
        )

    def update_weights(self, x):
        """Recompute l1 weights from the current model/iterate."""
        self.l1weight = self._reweighter(self._coeff_sum(x))
