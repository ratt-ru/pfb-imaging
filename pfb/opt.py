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


def hpd(fprime, prox, reg, x0, gamma, beta, sig_21, 
        psi=None,
        hess=None,
        cgprecond=None,
        cgtol=1e-3,
        cgmaxit=35,
        cgverbose=0,
        alpha0=0.25,
        alpha_ff=0.2,
        reweight_start=20,
        reweight_freq=5,
        tol=1e-3, 
        maxit=1000,
        report_freq=1,
        verbosity=1):
    """
    Algorithm to solve problems of the form

    argmin_x F(x) + lam_21 |L x|_21

    where x is the image cube and there is an optional positivity 
    constraint. 

    fprime      - function that produces F(x), grad F(x)
    x0           - primal variable
    gamma       - initial step-size
    beta        - Lipschitz constant of F
    sig_21      - strength of l21 regulariser

    """
    nchan, nx, ny = x0.shape
    npix = nx*ny
    real_type = x0.dtype

    # weights
    if psi is None:
        weights_21 = np.ones(npix, dtype=x0.dtype)
    else:
        weights_21 = np.empty(psi.nbasis, dtype=object)
        weights_21[0] = np.ones(nx*ny, dtype=real_type)
        for m in range(1, psi.nbasis):
            weights_21[m] = np.ones(psi.ntot, dtype=real_type)

    # initialise
    x = x0

    # initial fidelity and gradient
    fid, gradn = fprime(x)
    regx = reg(x)
    # initial objective and fidelity funcs
    fidelity = np.zeros(maxit+1)
    objective = np.zeros(maxit+1)
    regulariser = np.zeros(maxit+1)
    fidelity[0] = fid
    regulariser[0] = regx
    objective[0] = fid + regx

    # start iterations
    eps = 1.0
    k = 0
    i = 0  # reweighting counter
    for k in range(maxit):
        xp = x.copy()
        # gradient
        gradp = gradn

        # get update 
        if hess is not None:
            delx = pcg(hess, -gradp, np.zeros(x.shape), M=cgprecond, tol=cgtol, maxit=cgmaxit, verbosity=cgverbose)
        else:
            delx = -gradp/beta
        p = xp + gamma * delx

        x = prox(p, sig_21, weights_21)

        fid, gradn = fprime(x)
        regx = reg(x)
        objective[k+1] = fid + regx
        fidelity[k+1] = fid
        regulariser[k+1] = regx
        while objective[k+1] >= objective[k] and k < reweight_start and k%reweight_freq:
            gamma *= 0.9
            print("Step size too large, adjusting gamma = %f"%gamma)
            
            p = xp + gamma * delx

            x = prox(p, sig_21, weights_21)

            fid, gradn = fprime(x)
            regx = reg(x)
            objective[k+1] = fid + regx
            fidelity[k+1] = fid
            regulariser[k+1] = regx

        # convergence check
        normx = norm(x)
        if np.isnan(normx) or normx == 0.0:
            normx = 1.0
        
        eps = norm(x-xp)/normx
        if eps < tol:
            break

        if k >= reweight_start and not k%reweight_freq:
            alpha = alpha0/(1+i)**alpha_ff
            if psi is None:
                normx = norm(x.reshape(nchan, npix), axis=0)
                weights_21 = 1.0/(normx + alpha)
            else:
                for m in range(nbasis):
                    v = np.zeros((nchan, nx*ny), x.dtype)
                    for l in range(nchan):
                        v[l] = PSIT[m](x[l])
                    l2norm = norm(v, axis=0)
                    weights_21[m] = 1.0/(l2norm + alpha)
            i += 1

        if not k%report_freq and verbosity > 1:
            print("At iteration %i eps = %f, norm21 = %f "%(k, eps, regx))


    if verbosity > 0:
        if k == maxit-1:
            print("HPD - Maximum iterations reached. Relative difference between updates = ", eps)
        else:
            print("HPD - Success, converged after %i iterations"%k)

    return x, objective, fidelity, regulariser

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