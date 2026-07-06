"""Weighted l1 regulariser g(alpha) = ||W alpha||_1 (ISTA when psi is IdentityPsi)."""

import numpy as np


class L1:
    """Satisfies the ``Regulariser`` Protocol.

    Args:
        psi: Operator satisfying ``PsiOperator`` (typically ``IdentityPsi``).
        nu: Spectral norm of ``psi`` (default 1.0).
    """

    def __init__(self, psi, nu: float = 1.0):
        self.psi = psi
        self.nu = nu
        self.weight = np.ones((psi.nbasis, psi.nymax, psi.nxmax))

    def prox(self, v, vout, lam, sigma=1.0):
        """vout = prox_{(lam/sigma) ||W .||_1}(v/sigma), in-place."""
        np.divide(v, sigma, out=vout)
        thresh = (lam / sigma) * self.weight  # broadcasts over the band axis
        np.copysign(np.maximum(np.abs(vout) - thresh, 0.0), vout, out=vout)
