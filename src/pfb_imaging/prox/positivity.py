"""Positivity constraints as stock primal proximal operators.

These are indicator-function proxes applied in-place in the image domain;
backward solvers take one as their optional ``primal_prox`` callable.
"""

from numba import njit, prange

_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


@njit(**_FAST_JIT)
def positivity(x):
    """Clamp negative values to zero (mode 1)."""
    n = x.size
    x_f = x.ravel()
    for i in prange(n):
        if x_f[i] < 0.0:
            x_f[i] = 0.0


@njit(**_FAST_JIT)
def positivity_band(x):
    """Zero a pixel in all bands where any band is non-positive (mode 2)."""
    nband, nx, ny = x.shape
    for i in prange(nx):
        for j in range(ny):
            for b in range(nband):
                if x[b, i, j] <= 0.0:
                    for bb in range(nband):
                        x[bb, i, j] = 0.0
                    break


def positivity_prox(mode: int):
    """Map the CLI positivity mode to a primal_prox callable (or None)."""
    if mode == 0:
        return None
    if mode == 1:
        return positivity
    if mode == 2:
        return positivity_band
    raise ValueError(f"Unknown positivity mode {mode}")
