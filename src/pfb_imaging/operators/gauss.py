import numpy as np
from numba import njit, prange

# Pixels per tile. The (b, c) loops re-touch one band slice of ``out`` per band,
# so the tile must be small enough that the slice stays in L2 -- otherwise the
# kernel re-streams ``out`` nband times from memory and loses to numpy at
# production sizes. 2048 pixels is 16 KB per band slice; measured best or tied
# best from 750^2 to 4096^2, and 131072 is ~2x worse everywhere.
_ETA_FREQ_TILE = 2048

_ETA_FREQ_JIT = {"nogil": True, "cache": True, "error_model": "numpy", "fastmath": True, "parallel": True}


@njit(**_ETA_FREQ_JIT)
def _eta_freq_mul_uniform(out, amat, x, tile, nchunk):
    """``out[b, p] += sum_c amat[b, c] * x[c, p]`` on ``(nband, npix)`` arrays."""
    nband, npix = x.shape
    step = (npix + nchunk - 1) // nchunk
    for g in prange(nchunk):
        lo = g * step
        hi = min(lo + step, npix)
        for t0 in range(lo, hi, tile):
            t1 = min(t0 + tile, hi)
            for b in range(nband):
                for c in range(nband):
                    a = amat[b, c]
                    # unit stride in p: this is the loop that vectorises, and the
                    # reason band-major beats a pixel-major gather
                    for p in range(t0, t1):
                        out[b, p] += a * x[c, p]


@njit(**_ETA_FREQ_JIT)
def _eta_freq_mul_profile(out, dcinv, s, x, tile, nchunk):
    """``out[b, p] += s[b, p] * sum_c dcinv[b, c] * s[c, p] * x[c, p]``."""
    nband, npix = x.shape
    step = (npix + nchunk - 1) // nchunk
    for g in prange(nchunk):
        lo = g * step
        hi = min(lo + step, npix)
        for t0 in range(lo, hi, tile):
            t1 = min(t0 + tile, hi)
            for b in range(nband):
                for c in range(nband):
                    a = dcinv[b, c]
                    for p in range(t0, t1):
                        out[b, p] += s[b, p] * a * s[c, p] * x[c, p]


def eta_freq_mul(out, dcinv, s, x, nchunk=1):
    """Fused frequency congruence, accumulated in place: ``out += D^½ dcinv D^½ x``.

    Computes ``out[b] += s[b] * Σ_c dcinv[b, c] * s[c] * x[c]`` over the band
    axis in one pass, replacing the four-pass numpy form

    .. code-block:: python

        np.multiply(x, s, out=buf)
        np.matmul(dcinv, buf.reshape(nband, -1), out=buf2.reshape(nband, -1))
        np.multiply(buf2, s, out=buf2)
        out += buf2

    which is what ``HessTreeRay.dot`` applies for the GP frequency prior (wiki
    D30). Two reasons this exists, both measured:

    * **numpy's ``matmul`` poisons the next Ray round trip.** ``k = nband`` is
      tiny and ``n = npix`` huge, so the call is memory-bound, yet OpenBLAS
      spreads it over every core and those threads then busy-poll for ~100 ms
      (``THREAD_TIMEOUT``). That window covers the following ``ray.get``, during
      which the band workers need the cores: a 3 ms matmul added ~60 ms to a
      70 ms round trip. Numba's TBB pool does not do this -- measured at up to
      22 threads, the next round trip stays at its 0.2-core baseline.
    * **It is faster anyway.** At 8 bands x 4096², numpy takes 287 ms on 11.5
      cores; this takes 83 ms on 15.6 (3.5x), or 118 ms on 6.9 at
      ``nchunk=8``. It also drops the two ``(nband, ny, nx)`` scratch buffers the
      numpy form needs -- 2 GB at that size.

    Args:
        out: ``(nband, ny, nx)`` accumulator, updated in place. Must be
            C-contiguous.
        dcinv: ``(nband, nband)`` frequency coupling, ``C⁻¹ₙ - I`` for the prior.
        s: ``D^½``, either per-band -- shape ``(nband,)`` or ``(nband, 1, 1)``,
            the uniform-eta case -- or a full ``(nband, ny, nx)`` profile.
        x: ``(nband, ny, nx)`` operand. Must be C-contiguous.
        nchunk: Number of pixel chunks to fan out over numba's thread pool.
            ``1`` runs serially. Sets the parallelism without touching the
            process-wide thread count.

    Returns:
        ``out``, for convenience; the update is in place.

    Raises:
        ValueError: On a shape mismatch, a non-contiguous ``out``/``x``, or an
            ``s`` that is neither per-band nor a full profile.
    """
    nband = dcinv.shape[0]
    if dcinv.shape != (nband, nband):
        raise ValueError(f"dcinv must be square, got {dcinv.shape}")
    if out.shape != x.shape or x.shape[0] != nband:
        raise ValueError(f"expected out and x of shape ({nband}, ny, nx), got {out.shape} and {x.shape}")
    # reshape(nband, -1) silently copies a non-contiguous array, which would
    # throw the accumulation away for `out` and read the wrong thing for `x`
    if not out.flags.c_contiguous or not x.flags.c_contiguous:
        raise ValueError("out and x must be C-contiguous")
    nchunk = max(1, int(nchunk))
    out2, x2 = out.reshape(nband, -1), x.reshape(nband, -1)

    s = np.asarray(s)
    if s.size == nband:
        # per-band scalars: the whole congruence folds into one matrix, so the
        # kernel never touches s and does nband^2 fewer multiplies per pixel
        sb = s.reshape(nband)
        _eta_freq_mul_uniform(out2, sb[:, None] * dcinv * sb[None, :], x2, _ETA_FREQ_TILE, nchunk)
    elif s.shape == x.shape:
        if not s.flags.c_contiguous:
            raise ValueError("a full s profile must be C-contiguous")
        _eta_freq_mul_profile(out2, dcinv, s.reshape(nband, -1), x2, _ETA_FREQ_TILE, nchunk)
    else:
        raise ValueError(f"s must have {nband} elements or shape {x.shape}, got {s.shape}")
    return out
