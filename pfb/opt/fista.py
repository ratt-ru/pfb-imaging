import numpy as np
import pyscilog
log = pyscilog.get_logger('FISTA')

def back_track_func(x, xold, gradp, likp, L):
        df = x - xold
        return likp + np.vdot(gradp, df) + L* np.vdot(df, df)/2

def fista(A, xbar, x0, lam,  L,
          tol=1e-3, maxit=100, report_freq=10):
    """
    Algorithm to solve problems of the form

    argmin_x (xbar - x).T A (xbar - x)/2 + lam |x|_21 s.t. x >= 0

    A        - Positive definite Hermitian operator
    xbar     - Data-like term
    x0       - Initial guess
    lam      - Strength of l21 regulariser
    L        - Lipschitz constant of A
    sigma    - l21 step size (set to L/2 by default)
    """
    # nchan, ncomps = x0.shape
    

    # gradient function
    def fprime(x):
        diff = A(xbar - x)
        return np.vdot(xbar-x, diff), -diff

    def prox(x):
        l2_norm = norm(x, axis=0)  # drops freq axis
        # l2_norm = np.mean(x, axis=0)  # drops freq axis
        l2_soft = np.maximum(l2_norm - lam, 0.0)  # norm is always positive
        mask = l2_norm != 0
        ratio = np.zeros(mask.shape, dtype=x.dtype)
        ratio[mask] = l2_soft[mask] / l2_norm[mask]
        x *= ratio[None, :]  # restore freq axis
        x[x<0] = 0.0  # positivity
        return x 

    # start iterations
    t = 1.0
    x = x0.copy()
    y = x0.copy()
    eps = 1.0
    k = 0
    fidn, gradn = fprime(x)
    objective = np.zeros(maxit)
    fidelity = np.zeros(maxit)
    fidupper = np.zeros(maxit)
    for k in range(maxit):
        xold = x.copy()
        fidp = fidn
        gradp = gradn

        # gradient update
        x = y - gradp/L
        
        # prox w.r.t r(x)
        x = prox(x)

        # check convergence criteria
        normx = norm(x)
        if np.isnan(normx) or normx == 0.0:
            normx = 1.0
        eps = norm(x - xold)/normx
        if eps <= tol:
            break

        fidn, gradn = fprime(x)
        flam = back_track_func(x, xold, gradp, fidp, L)
        fidelity[k] = fidn
        fidupper[k] = flam
        while fidn > flam:
            L *= 2.0
            print("Step size too large, adjusting %f"%L, file=log)

             # gradient update
            x = y - gradp/L

            # prox
            x = prox(x)

            fidn, gradn = fprime(x)
            flam = back_track_func(x, xold, gradp, fidp, L)

        # fista update
        tp = t
        t = (1. + np.sqrt(1. + 4 * tp**2))/2.
        gamma = (tp - 1)/t
        y = x +  gamma * (x - xold)

        if not k%report_freq:
            print("At iteration %i: eps = %f, L = %f, lambda = %f"%(k, eps, L, (tp - 1)/t), file=log)

    if k == maxit-1:
        print("Maximum iterations reached. Relative difference between updates = %f"%eps, file=log)
    else:
        print("Success, converged after %i iterations"%k, file=log)

    return x, fidelity, fidupper
