import numpy as np
import pyscilog
log = pyscilog.get_logger('FISTA')


def back_track_func(x, xp, gradp, likp, L):
    df = x - xp
    return likp + np.vdot(gradp, df) + L * np.vdot(df, df) / 2


def fista(x0,
          L,       # spectral norm of measurement operator
          fprime,  # function returning value and gradient
          prox,    # function implementing prox of regulariser
          tol=1e-3,
          maxit=100,
          report_freq=50,
          verbosity=1):

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
        x = y - gradp / L

        # prox w.r.t r(x)
        x = prox(x)

        fidn, gradn = fprime(x)
        # flam = back_track_func(x, xp, gradp, fidp, L)
        i = 0
        while fidn > fidp and i < 10:
            L *= 2.0
            if verbosity > 1:
                print("Step size too large, adjusting %f" % L, file=log)

            # gradient update
            x = y - gradp / L

            # prox
            x = prox(x)

            fidn, gradn = fprime(x)
            # flam = back_track_func(x, xp, gradp, fidp, L)
            i += 1

        if i == 10:
            if verbosity > 1:
                print("Stalled", file=log)
            k = maxit - 1
            break

        # fista update
        tp = t
        t = (1. + np.sqrt(1. + 4 * tp**2)) / 2.
        gamma = (tp - 1) / t
        y = x + gamma * (x - xp)

        # convergence check
        eps = np.linalg.norm(x - xp) / np.linalg.norm(x)
        if eps < tol:
            break

        fidp = fidn
        gradp = gradn

        if not k % report_freq and verbosity > 1:
            print("At iteration %i eps = %f" % (k, eps), file=log)

    if k == maxit - 1:
        if verbosity:
            print("Maximum iterations reached. "
                  "Relative difference between updates = %f" %
                  eps, file=log)
    else:
        if verbosity:
            print("Success, converged after %i iterations" % k, file=log)

    return x


fista.__doc__ = r"""
    Algorithm to solve problems of the form

    argmin_x (xbar - x).T A (xbar - x)/2 + lam |x|_21 s.t. x >= 0

    A        - Positive definite Hermitian operator
    xbar     - Data-like term
    x0       - Initial guess
    lam      - Strength of l21 regulariser
    L        - Lipschitz constant of A
    sigma    - l21 step size (set to L/2 by default)
    """
