import numpy as np
import pyscilog
log = pyscilog.get_logger('PCG')


def pcg(
        A,
        b,
        x0,
        M=None,
        tol=1e-5,
        maxit=500,
        minit=100,
        verbosity=1,
        report_freq=10,
        backtrack=True):

    if M is None:
        def M(x): return x

    r = A(x0) - b
    y = M(r)
    p = -y
    rnorm = np.vdot(r, y)
    if np.isnan(rnorm) or rnorm == 0.0:
        eps0 = 1.0
    else:
        eps0 = rnorm
    k = 0
    x = x0
    eps = 1.0
    stall_count = 0
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        rnorm = np.vdot(r, y)
        alpha = rnorm / np.vdot(p, Ap)
        x = xp + alpha * p
        r = rp + alpha * Ap
        y = M(r)
        rnorm_next = np.vdot(r, y)
        while rnorm_next > rnorm and backtrack:  # TODO - better line search
            alpha *= 0.75
            x = xp + alpha * p
            r = rp + alpha * Ap
            y = M(r)
            rnorm_next = np.vdot(r, y)

        beta = rnorm_next / rnorm
        p = beta * p - y
        rnorm = rnorm_next
        k += 1
        epsx = np.linalg.norm(x - xp) / np.linalg.norm(x)
        epsn = rnorm / eps0
        epsp = eps
        eps = np.maximum(epsx, epsn)

        if np.abs(epsp - eps) < tol:
            stall_count += 1

        if not k % report_freq and verbosity > 1:
            print("At iteration %i rnorm = %f" % (k, eps), file=log)

    if k >= maxit:
        if verbosity:
            print("Max iters reached. Norm of residual = %f." %
                  (rnorm / eps0), file=log)
    elif stall_count >= 5:
        if verbosity:
            print("Stalled. Norm of residual = %f." % (rnorm / eps0),
                  file=log)
    else:
        if verbosity:
            print("Success, converged after %i iters" % k, file=log)
    return x
