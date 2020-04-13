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
          x0, y0,  # initial guess for primal and dual variables 
          beta,    # lipschitz constant
          sig_21, # regulariser strengths
          tol=1e-3, maxit=1000, positivity=True, report_freq=10):
    """
    Algorithm to solve problems of the form

    argmin_x (b - Rx).T (b - Rx) + x.T K^{-1} x +  lam_21 |x|_21

    where x is the image cube, R the measurement operator, K is a
    prior covariance matrix enforcing smoothness in frequency and
    and |x|_21 the l21 norm along the frequency axis. Note we can
    consider the first two terms jointly as a single smooth data
    fidelity term with Lipschitz constant beta. The non-smooth
    l21 norm term is then the regulariser and we can solve the
    optimisation problem via accelerated proximal gradient decent.  

    sig_21   - strength of l21 regulariser
    
    # Note - Spectral norms for both decomposition operators are unity.
    """
    nchan, nx, ny = x0.shape
    npix = nx*ny

    # # fidelity and gradient term
    # def fidgrad(x):
    #     diff = data - R.dot(x)
    #     tmp = K.idot(x)
    #     return 0.5*np.vdot(diff, diff).real + 0.5*np.vdot(x, tmp), -R.hdot(diff) + tmp

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

        objective[k] = fidp + sig_21*np.sum(norm(x.reshape(nchan, nx*ny), axis=0))

        # gradient update
        x = y - gradp/beta
        
        # apply prox
        if positivity:
            x[x<0] = 0.0
        
        # l21
        l2norm = norm(x.reshape(nchan, nx*ny), axis=0)
        l2_soft = np.maximum(l2norm - sig_21, 0.0)
        indices = np.nonzero(l2norm)
        ratio = np.zeros(l2norm.shape, dtype=np.float64)
        ratio[indices] = l2_soft[indices]/l2norm[indices]
        x *= ratio.reshape(1, nx, ny)

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
            if positivity:
                x[x<0.0] = 0.0

            # l21
            l2norm = norm(x.reshape(nchan, nx*ny), axis=0)
            l2_soft = np.maximum(l2norm - sig_21, 0.0)
            indices = np.nonzero(l2norm)
            ratio = np.zeros(l2norm.shape, dtype=np.float64)
            ratio[indices] = l2_soft[indices]/l2norm[indices]
            x *= ratio.reshape(1, nx, ny)

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

    return x, objective, fidelity, fidupper


def hpd(fprime, prox, reg, x0, gamma, beta, sig_21, 
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

    argmin_x F(x) + lam_n |s|_1 + lam_21 |L x|_21

    where x is the image cube and s are the singular values along 
    the frequency dimension and there is an optional positivity 
    constraint. 

    fprime      - function that produces F(x), grad F(x)
    x           - primal variable
    y           - dual variable
    gamma       - initial step-size
    beta        - Lipschitz constant of F
    sig_21      - strength of l21 regulariser

    """
    nchan, nx, ny = x0.shape
    npix = nx*ny

    # weights
    weights_21 = np.ones(npix, dtype=np.float64)

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
            normx = norm(x.reshape(nchan, npix), axis=0)
            weights_21 = 1.0/(normx + alpha)
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