import numpy as np

from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("FISTA")


def back_track_func(x, xp, gradp, likp, hessnorm):
    df = x - xp
    return likp + np.vdot(gradp, df) + hessnorm * np.vdot(df, df) / 2


def fista(
    x0,
    hessnorm,  # spectral norm of measurement operator
    fprime,  # function returning value and gradient
    prox,  # function implementing prox of regulariser
    tol=1e-3,
    maxit=100,
    report_freq=50,
    verbosity=1,
):
    # start iterations
    t = 1.0
    x = x0.copy()
    y = x0.copy()
    eps = 1.0
    k = 0
    fidp, gradp = fprime(x)
    for k in range(maxit):
        xp = x.copy()

        # gradient update
        x = y - gradp / hessnorm

        # prox w.r.t r(x)
        x = prox(x)

        fidn, gradn = fprime(x)
        # flam = back_track_func(x, xp, gradp, fidp, hessnorm)
        i = 0
        while fidn > fidp and i < 10:
            hessnorm *= 2.0
            if verbosity > 1:
                log.info("Step size too large, adjusting %f" % hessnorm)

            # gradient update
            x = y - gradp / hessnorm

            # prox
            x = prox(x)

            fidn, gradn = fprime(x)
            # flam = back_track_func(x, xp, gradp, fidp, hessnorm)
            i += 1

        if i == 10:
            if verbosity > 1:
                log.info("Stalled")
            k = maxit - 1
            break

        # fista update
        tp = t
        t = (1.0 + np.sqrt(1.0 + 4 * tp**2)) / 2.0
        gamma = (tp - 1) / t
        y = x + gamma * (x - xp)

        # convergence check
        eps = np.linalg.norm(x - xp) / np.linalg.norm(x)
        if eps < tol:
            break

        fidp = fidn
        gradp = gradn

        if not k % report_freq and verbosity > 1:
            log.info("At iteration %i eps = %f" % (k, eps))

    if k == maxit - 1:
        if verbosity:
            log.info("Maximum iterations reached. Relative difference between updates = %f" % eps)
    else:
        if verbosity:
            log.info("Success, converged after %i iterations" % k)

    return x
