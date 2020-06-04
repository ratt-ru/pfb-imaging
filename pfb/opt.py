import numpy as np
from numba import njit, prange
from scipy.linalg import norm, svd
from pfb.utils import new_prox_21

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


def simple_pd(A, xbar, 
              x0, v0,  # initial guess for primal and dual variables 
              lam,  # regulariser strength,
              psi,  # orthogonal basis operator
              weights,  # weights for l1 thresholding 
              L, nu=1.0,  # spectral norms
              sigma=None,  # step size of dual update
              tol=1e-5, maxit=1000, positivity=True, report_freq=10):
    """
    Algorithm to solve problems of the form

    argmin_x (xbar - x).T A (xbar - x)/2 + lam |PSI.H x|_21 s.t. x >= 0

    where x is the image cubex is an PSI an orthogonal basis for the spatial axes
    and the positivity constraint is optional.

    A        - Positive definite Hermitian operator
    xbar     - Data-like term
    x0       - Initial guess for primal variable
    v0       - Initial guess for dual variable
    lam      - Strength of l21 regulariser
    psi      - Orthogonal basis operator where 
               psi.hdot() does decomposition and
               psi.dot() the reconstruction
               for all bases and channels
    weights  - Weights for l1 thresholding
    L        - Lipschitz constant of A
    nu       - Spectral norm of psi
    sigma    - l21 step size (set to L/2 by default)

    Note that the primal variable (i.e. x) has shape (nband, nx, ny) where nband is the number of imaging bands
    and the dual variable (i.e. v) has shape (nbasis, nband, ntot) where ntot is the total number of coefficients
    for all bases. It is assumed that all bases are decomposed into the same number of coeffcients. To deal with
    the fact that the Dirac basis does not necessarily have the same number of coefficients as the wavelets it
    is simply padded by zeros. This doe snot effect the algorithm. 
    """
    # initialise
    x = x0.copy()
    v = v0.copy()

    # gradient function
    grad_func = lambda x: -A(xbar - x)

    if sigma is None:
        sigma = L/2.0

    # stepsize control
    tau = 0.9/(L/2.0 + sigma*nu**2)
    print("tau = %f, sigma = %f"%(tau, sigma))

    # start iterations
    eps = 1.0
    k = 0
    for k in range(maxit):
        xp = x.copy()
        vp = v.copy()
        
        # tmp prox variable
        vtilde = v + sigma * psi.hdot(xp)

        # dual update
        v = vtilde - sigma * new_prox_21(vtilde/sigma, lam/sigma, weights)

        # primal update
        x = xp - tau*(psi.dot(2*v - vp) + grad_func(xp))
        if positivity:
            x[x<0] = 0.0

        # convergence check
        eps = norm(x-xp)/norm(x)
        if eps < tol:
            break

        if not k%report_freq:
            print("At iteration %i eps = %f"%(k, eps))

    if k == maxit-1:
        print("PD - Maximum iterations reached. Relative difference between updates = ", eps)
    else:
        print("PD - Success, converged after %i iterations"%k)

    return x, v

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