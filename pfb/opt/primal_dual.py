import numpy as np
from scipy.linalg import norm
from pfb.utils import prox_21

def primal_dual(A, xbar, 
                x0, v0,  # initial guess for primal and dual variables 
                lam,  # regulariser strength,
                psi,  # linear operator in dual domain
                weights,  # weights for l1 thresholding 
                L, nu=1.0,  # spectral norms
                sigma=None,  # step size of dual update
                mask=None,  # regions where mask is False will be masked 
                tol=1e-5, maxit=1000, positivity=True, report_freq=10, axis=1, gamma=1.0, verbosity=1):
    """
    Algorithm to solve problems of the form

    argmin_x (xbar - x).T A (xbar - x)/2 + lam |PSI.H x|_21 s.t. x >= 0

    where x is the image cube and PSI an orthogonal basis for the spatial axes
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
    gamma    - step size during forward step

    Note that the primal variable (i.e. x) has shape (nband, nx, ny) where nband is the number of imaging bands
    and the dual variable (i.e. v) has shape (nbasis, nband, nmax) where nmax is the total number of coefficients
    for all bases. It is assumed that all bases are decomposed into the same number of coefficients. We use zero
    padding to deal with the fact that the basess do not necessarily have the same number of coefficients.
    """
    # initialise
    x = x0.copy()
    v = v0.copy()

    # gradient function
    grad_func = lambda x: -A(xbar - x)/gamma

    if sigma is None:
        sigma = L/2.0

    # stepsize control
    tau = 0.9/(L/2.0 + sigma*nu**2)

    # start iterations
    eps = 1.0
    for k in range(maxit):
        xp = x.copy()
        vp = v.copy()
        
        # tmp prox variable
        vtilde = v + sigma * psi.hdot(xp)

        # dual update
        v = vtilde - sigma * prox_21(vtilde/sigma, lam/sigma, weights, axis=axis)

        # primal update
        x = xp - tau*(psi.dot(2*v - vp) + grad_func(xp))
        if positivity:
            x[x<0] = 0.0

        # apply mask
        if mask is not None:
            if x.ndim == 3:
                x = np.where(mask, x, 0.0)
            elif x.ndim == 4:
                x[1] = np.where(mask, x[1], 0.0)

        # convergence check
        eps = norm(x-xp)/norm(x)
        if eps < tol:
            break

        if not k%report_freq and verbosity > 1:
            print("         At iteration %i eps = %f"%(k, eps))

    if k == maxit-1:
        if verbosity:
            print("         PD - Maximum iterations reached. Relative difference between updates = ", eps)
    else:
        if verbosity:
            print("         PD - Success, converged after %i iterations"%k)

    return x, v