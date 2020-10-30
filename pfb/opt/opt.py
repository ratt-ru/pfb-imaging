import numpy as np
from scipy.linalg import norm
from pfb.utils import prox_21

def pcg(A, b, x0, M=None, tol=1e-5, maxit=500, verbosity=1, report_freq=10, backtrack=True):
    
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
        while rnorm_next > rnorm and backtrack:  # TODO - better line search
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
            print("Step size too large, adjusting L", L)

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
            print("At iteration %i: eps = %f, L = %f, lambda = %f"%(k, eps, L, (tp - 1)/t))

    if k == maxit-1:
        print("FISTA - Maximum iterations reached. Relative difference between updates = ", eps)
    else:
        print("FISTA - Success, converged after %i iterations"%k)

    return x, fidelity, fidupper


def primal_dual(A, xbar, 
                x0, v0,  # initial guess for primal and dual variables 
                lam,  # regulariser strength,
                psi,  # orthogonal basis operator
                weights,  # weights for l1 thresholding 
                L, nu=1.0,  # spectral norms
                sigma=None,  # step size of dual update
                mask=None,  # regions where mask is False will be masked 
                tol=1e-5, maxit=1000, positivity=True, report_freq=10, axis=1):
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

    Note that the primal variable (i.e. x) has shape (nband, nx, ny) where nband is the number of imaging bands
    and the dual variable (i.e. v) has shape (nbasis, nband, nmax) where nmax is the total number of coefficients
    for all bases. It is assumed that all bases are decomposed into the same number of coefficients. We use zero
    padding to deal with the fact that the basess do not necessarily have the same number of coefficients.
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
            x = np.where(mask, x, 0.0)

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
        # print(k, eps)

    if k == maxit:
        print("PM - Maximum iterations reached. eps = ", eps)
    else:
        print("PM - Success - convergence after %i iterations"%k)
    return np.vdot(bp, A(bp))/np.vdot(bp, bp)

# def hogbom(ID, PSF, gamma=0.1, pf=0.1, maxit=10000):
#     from pfb.utils import give_edges
#     nband, nx, ny = ID.shape
#     x = np.zeros((nband, nx, ny), dtype=ID.dtype) 
#     IR = ID.copy()
#     IRmean = np.mean(IR, axis=0)
#     IRmax = IRmean.max()
#     tol = pf*IRmax
#     for i in range(maxit):
#         if IRmax < tol:
#             break
#         p, q = np.argwhere(IRmean == IRmax).squeeze()
#         xhat = IR[:, p, q]
#         Ix, Iy, Ixpsf, Iypsf  = give_edges(p, q, nx, ny)
#         x[:, p, q] += gamma * xhat
#         IR[:, Ix, Iy] -= gamma * xhat[:, None, None] * PSF[:, Ixpsf, Iypsf]
#         IRmean = np.mean(IR, axis=0)
#         IRmax = IRmean.max()
#     return x

def hogbom(ID, PSF, gamma=0.1, pf=0.1, maxit=10000):
    from pfb.utils import give_edges
    nband, nx, ny = ID.shape
    x = np.zeros((nband, nx, ny), dtype=ID.dtype) 
    IR = ID.copy()
    IRmean = np.abs(np.mean(IR, axis=0))
    IRmax = IRmean.max()
    tol = pf*IRmax
    for i in range(maxit):
        if IRmax < tol:
            break
        p, q = np.argwhere(IRmean == IRmax).squeeze()
        xhat = IR[:, p, q]
        x[:, p, q] += gamma * xhat
        IR -= gamma * xhat[:, None, None] * PSF[:, nx-p:2*nx - p, ny-q:2*ny - q]
        IRmean = np.abs(np.mean(IR, axis=0))
        IRmax = IRmean.max()
    return x


        

