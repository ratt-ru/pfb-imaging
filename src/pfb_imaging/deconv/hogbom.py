import numexpr as ne
import numpy as np

from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("HOGBOM")


def hogbom(dirty, psf, threshold=0, gamma=0.1, pf=0.1, maxit=10000, report_freq=1000, verbosity=1):
    nband, nx, ny = dirty.shape
    _, nx_psf, ny_psf = psf.shape
    nx0 = nx_psf // 2
    ny0 = ny_psf // 2
    x = np.zeros((nband, nx, ny), dtype=dirty.dtype)
    residual = dirty.copy()
    residual_search = np.sum(residual, axis=0) ** 2
    pq = residual_search.argmax()
    p = pq // ny
    q = pq - p * ny
    residual_max = np.sqrt(residual_search[p, q])
    wsums = np.amax(psf, axis=(1, 2))
    fsel = wsums > 0
    tol = np.maximum(pf * residual_max, threshold)
    k = 0
    stall_count = 0
    while residual_max > tol and k < maxit and stall_count < 5:
        xhat = residual[fsel, p, q] / wsums[fsel]
        x[:, p, q] += gamma * xhat
        ne.evaluate(
            "residual - gamma * xhat * psf",
            local_dict={
                "residual": residual,
                "gamma": gamma,
                "xhat": xhat[:, None, None],
                "psf": psf[:, nx0 - p : nx0 + nx - p, ny0 - q : ny0 + ny - q],
            },
            out=residual,
            casting="same_kind",
        )
        residual_search = np.sum(residual, axis=0) ** 2
        pq = residual_search.argmax()
        p = pq // ny
        q = pq - p * ny
        residual_maxp = residual_max
        residual_max = np.sqrt(residual_search[p, q])
        k += 1

        if np.abs(residual_maxp - residual_max) / np.abs(residual_maxp) < 5e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            log.info("At iteration %i max residual = %f" % (k, residual_max))

    residual_mfs = np.sum(residual, axis=0)
    rms = np.std(residual_mfs[~np.any(x, axis=0)])

    if k >= maxit:
        if verbosity:
            log.info(f"Max iters reached. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 1
    elif stall_count >= 5:
        if verbosity:
            log.info(f"Stalled. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 1
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return x, 0


# import jax.numpy as jnp
# from jax import jit
# from jax.ops import index_add
# import jax.lax as lax
# @jit
# def hogbom_jax(dirty, psf, x, gamma=0.1, pf=0.1, maxit=5000):
#     nx, ny = dirty.shape
#     residual = jnp.array(dirty, copy=True)
#     residual_search = jnp.square(residual)
#     pq = jnp.argmax(residual_search)
#     p = pq//ny
#     q = pq - p*ny
#     residual_max = jnp.sqrt(residual_search[p, q])
#     tol = pf*residual_max
#     k = 0

#     def cond_func(inputs):

#         residual_max, residual, residual_search, psf, x, loc, tol, gamma, k = inputs

#         return (k < maxit) & (residual_max > tol)

#     def body_func(inputs):
#         residual_max, residual, residual_search, psf, x, loc, tol, gamma, k = inputs
#         nx, ny = residual.shape
#         p, q = loc
#         xhat = residual[p, q]
#         x = index_add(x, (p, q), gamma * xhat)
#         modconv = lax.dynamic_slice(psf, [nx-p, ny-q], [nx, ny])
#         residual = residual - gamma * xhat * modconv
#         residual_search = jnp.square(residual)
#         pq = residual_search.argmax()
#         p = pq//ny
#         q = pq - p*ny
#         residual_max = jnp.sqrt(residual_search[p, q])
#         return (residual_max, residual, residual_search, psf, x, (p, q), tol, gamma, k+1)

#     init_val = (residual_max, residual, residual_search, psf, x, (p, q), tol, gamma, k)
#     out = lax.while_loop(cond_func, body_func, init_val)

#     return out[4], out[1]
