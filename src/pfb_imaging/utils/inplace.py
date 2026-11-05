"""Parallel in-place kernels for the driver's cube arithmetic.

The deconv driver works on ``(nband, ny, nx)`` cubes -- 4.9 GB each at
8 bands x 8748^2 -- several times per Hessian application (the gradient, CG
and power-method updates). Done in numpy, every expression allocates a fresh
cube, paying first-touch page faults over all of it, and runs on one thread:
on com08 that was ~12 s of driver time per primal-dual iteration against a
15 s ``hess_dot``. These kernels write into buffers the caller owns and fan
out over numba's thread pool. All arrays must be C-contiguous and of equal
size; sources may be read-only (Ray results are). Destinations are checked
(``require_c``): numba's ``ravel`` copies any other layout, so writes through it
would vanish without an error.
"""

import numpy as np
from numba import njit, prange

_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


@njit(nogil=True, cache=True)
def require_c(a):
    """Raise unless ``a`` is C-contiguous.

    For any other layout numba's ``ravel`` returns a copy, so a kernel writing
    through it would silently leave ``a`` untouched. Call it on every array a
    kernel writes through ``ravel``; reading a copy is merely slower.
    """
    if not a.flags.c_contiguous:
        raise ValueError("in-place kernel destination must be C-contiguous")


@njit(**_FAST_JIT)
def copy(dst, src):
    """``dst[...] = src``."""
    require_c(dst)
    d = dst.ravel()
    s = src.ravel()
    for i in prange(d.size):
        d[i] = s[i]


@njit(**_FAST_JIT)
def add(dst, src):
    """``dst += src``."""
    require_c(dst)
    d = dst.ravel()
    s = src.ravel()
    for i in prange(d.size):
        d[i] += s[i]


@njit(**_FAST_JIT)
def scale(dst, a):
    """``dst *= a``."""
    require_c(dst)
    d = dst.ravel()
    for i in prange(d.size):
        d[i] *= a


@njit(**_FAST_JIT)
def axpby(dst, a, x, b, y):
    """``dst = a * x + b * y``; ``dst`` may alias ``x`` or ``y``."""
    require_c(dst)
    d = dst.ravel()
    xf = x.ravel()
    yf = y.ravel()
    for i in prange(d.size):
        d[i] = a * xf[i] + b * yf[i]


@njit(**_FAST_JIT)
def rel_diff(x, y, z, b):
    """``||x - (y + b * z)|| / ||x||`` without forming ``y + b * z``."""
    xf = x.ravel()
    yf = y.ravel()
    zf = z.ravel()
    num = 0.0
    den = 0.0
    for i in prange(xf.size):
        d = xf[i] - yf[i] - b * zf[i]
        num += d * d
        den += xf[i] * xf[i]
    return np.sqrt(num / max(den, 1e-30))
