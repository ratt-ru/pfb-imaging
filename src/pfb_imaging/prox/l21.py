"""Weighted l21 regulariser over a wavelet dictionary (the SARA prior)."""

from functools import partial

import numpy as np

from pfb_imaging.operators import PsiOperator, require_protocol
from pfb_imaging.prox.prox_21m import dual_update_numba_fast, prox_21m_numba
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
        self._outvar = np.zeros((psi.nband, psi.nbasis, psi.nymax, psi.nxmax))
        self._reweighter = None

    def prox(self, v, vout, lam, sigma=1.0):
        """vout = prox_{(lam/sigma) ||W .||_{21}}(v/sigma), in-place."""
        prox_21m_numba(v, vout, lam, sigma=sigma, weight=self.l1weight)

    def dual_update(self, vp, v, lam, sigma=1.0):
        """Fused primal-dual dual update (fast path sniffed by PrimalDual)."""
        dual_update_numba_fast(vp, v, lam, sigma=sigma, weight=self.l1weight)

    @property
    def reweight_active(self) -> bool:
        """True once reweighting has been initialised."""
        return self._reweighter is not None

    def init_reweighting(self, update):
        """Estimate per-basis component rms from the update and arm reweighting."""
        self.psi.dot(update, self._outvar)
        tmp = np.sum(self._outvar, axis=0)
        rms_comps = np.ones(self.psi.nbasis, dtype=float)
        for i, base in enumerate(self.bases):
            tmpb = tmp[i]
            rms_comps[i] = np.std(tmpb[tmpb != 0])
            log.info(f"rms_comps for base {base} is {rms_comps[i]:.3e}")
        self._reweighter = partial(
            l1reweight_func,
            psih=self.psi.dot,
            outvar=self._outvar,
            rmsfactor=self.rmsfactor,
            rms_comps=rms_comps,
            alpha=self.alpha,
        )

    def update_weights(self, x):
        """Recompute l1 weights from the current model/iterate."""
        self.l1weight = self._reweighter(x)
