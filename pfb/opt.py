import numpy as np
from numba import njit, prange
from scipy.linalg import norm, svd

def pcg(A, b, x0, M=None, tol=1e-5, maxit=500, verbosity=1, report_freq=10):
    
    if M is None:
        M = lambda x: x
    
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
    while rnorm/eps0 > tol and k < maxit:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        rnorm = np.vdot(r, y)
        alpha = rnorm/np.vdot(p, Ap)
        x = xp + alpha*p
        r = rp + alpha*Ap
        y = M(r)
        rnorm_next = np.vdot(r, y)
        while rnorm_next > rnorm:  # TODO - better line search
            alpha *= 0.75
            x = xp + alpha*p
            r = rp + alpha*Ap
            y = M(r)
            rnorm_next = np.vdot(r, y)

        beta = rnorm_next/rnorm
        p = beta*p - y
        rnorm = rnorm_next
        k += 1

        if not k%report_freq and verbosity > 1:
            print("At iteration %i rnorm = %f"%(k, rnorm/eps0))

    if k >= maxit:
        if verbosity > 0:
            print("CG - Maximum iterations reached. Norm of residual = %f.  "%(rnorm/eps0))
    else:
        if verbosity > 0:
            print("CG - Success, converged after %i iterations"%k)
    return x

def back_track_func(x, xold, gradp, likp, L):
        df = x - xold
        return likp + np.vdot(gradp, df) + L* np.vdot(df, df)/2

def fista(fprime,
          prox,
          x0, # initial guess for primal and dual variables 
          beta,    # lipschitz constant
          tol=1e-3, maxit=100, report_freq=10):
    """
    Algorithm to solve problems of the form

    argmin_x F(x) +  R(x)

    where F(x) is a smooth data fidelity
    term and R(x) a non-smooth convex regulariser.   

    fprime   - gradient of F
    prox     - sub-gradient of R
    x0       - initial guess
    beta     - Lipschitz constant of F
    """
    nchan, nx, ny = x0.shape
    npix = nx*ny

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
        x = y - gradp/beta
        
        # prox w.r.t R(x)
        x = prox(x)

        # check convergence criteria
        normx = norm(x)
        if np.isnan(normx) or normx == 0.0:
            normx = 1.0
        eps = norm(x - xold)/normx
        if eps <= tol:
            break

        fidn, gradn = fprime(x)
        flam = back_track_func(x, xold, gradp, fidp, beta)
        fidelity[k] = fidn
        fidupper[k] = flam
        while fidn > flam:
            beta *= 1.5
            print("Step size too large, adjusting L", beta)

             # gradient update
            x = y - gradp/beta

            # prox
            x = prox(x)

            fidn, gradn = fprime(x)
            flam = back_track_func(x, xold, gradp, fidp, beta)

        # fista update
        tp = t
        t = (1. + np.sqrt(1. + 4 * tp**2))/2.
        gamma = (tp - 1)/t
        y = x +  gamma * (x - xold)

        if not k%10:
            print("At iteration %i: eps = %f, L = %f, lambda = %f"%(k, eps, beta, (tp - 1)/t))

    if k == maxit-1:
        print("FISTA - Maximum iterations reached. Relative difference between updates = ", eps)
    else:
        print("FISTA - Success, converged after %i iterations"%k)

    return x, fidelity, fidupper


def hpd(A, b, 
        x0, v0,  # initial guess for primal and dual variables 
        L, nu,  # spectral norms
        lam,  # regulariser strength,
        psi,  # orthogonal basis operator (psi.hdot does expansion and psi.dot the reconstruction)
        prox,  # function to compute the prox
        weights,  # weights for l1 thresholding 
        sigma=None,  # step size of dual update
        tol=1e-3, maxit=1000, positivity=True, report_freq=10):
    """
    Algorithm to solve problems of the form

    argmin_x (b - x).T A (b - x) + lam |PSI.H x|_21 s.t. x >= 0

    where x is the image cube and is an PSI an orthogonal basis for the spatial axes 
    Note, the positivity constraint is optional.

    A        - Positive definite Hermitian operator
    L        - Lipschitz constant of A
    lam      - strength of l21 regulariser
    sigma    - l21 step size
    """
    nchan, nx, ny = x0.shape
    npix = nx*ny

    # initialise
    x = x0.copy()
    v = v0.copy()

    # gradient function
    grad_func = lambda x: -A(b-x)

    if sigma is None:
        sigma = L/2.0

    # stepsize control
    tau = 0.9/(L/2.0 + sigma*nu**2)

    # start iterations
    eps = 1.0
    k = 0
    for k in range(maxit):
        xp = x.copy()

        # gradient
        grad = grad_func(x)

        # primal update
        p = x - tau*(v - grad)
        if positivity:
            p[p<0] = 0.0

        # convergence check
        eps = norm(x-xp)/norm(x)
        if eps < tol:
            break

        if not k%report_freq:
            print("At iteration %i eps = %f and current stepsize is %f"%(k, eps, L))

        # tmp prox variable
        vtilde = v + sigma * psi.hdot(2*p - xp)

        # dual update
        v = vtilde - prox(vtilde/sigma, lam/sigma, weights, psi=psi, positivity=positivity)
        
        #
        v21 = v21 + lam_21*(xtmp - ratio[None, :] * r21)

    if k == maxit-1:
        print("HPD - Maximum iterations reached. Relative difference between updates = ", eps)
    else:
        print("HPD - Success, converged after %i iterations"%k)

    return x, v21, vn

def power_method(A, imsize, tol=1e-5, maxit=250):
    b = np.random.randn(*imsize)
    b /= norm(b)
    eps = 1.0
    k = 0
    while eps > tol and k < maxit:
        bp = b
        b = A(bp)
        bnorm = norm(b)
        b /= bnorm
        eps = norm(b - bp)/bnorm
        k += 1
        print(k, eps)

    if k == maxit:
        print("PM - Maximum iterations reached. eps = ", eps)
    else:
        print("PM - Success - convergence after %i iterations"%k)
    return np.vdot(bp, A(bp))/np.vdot(bp, bp)