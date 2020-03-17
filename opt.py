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
    xbest = x
    while rnorm/eps0 > tol and k < maxit:
        Ap = A(p)
        rnorm = np.vdot(r, y)
        alpha = rnorm/np.vdot(p, Ap)
        x += alpha*p
        r += alpha*Ap
        y = M(r)
        rnorm_next = np.vdot(r, y)
        beta = rnorm_next/rnorm
        p = beta*p - y
        rnorm = rnorm_next
        k += 1
        if k == 1:
            kbest = k
            epsbest = rnorm
            xbest = x
        elif rnorm < epsbest:
            kbest = k
            epsbest = rnorm
            xbest = x

        if not k%report_freq and verbosity >= 1:
            print("At iteration %i rnorm = %f"%(k, rnorm/eps0))

    if k >= maxit:
        if verbosity >= 1:
            print("CG - Returning best fit after %i iterations. Norm of residual = %f.  "%(kbest, epsbest/eps0))
    else:
        if verbosity >= 1:
            print("CG - Success, converged after %i iterations"%k)
    return xbest    

def back_track_func(x, xold, gradp, likp, L):
        df = x - xold
        return likp + np.vdot(gradp, df) + L* np.vdot(df, df)/2 

def solve_x0(A, b, x0, y0, L, sigma0, tol=1e-4, maxit=500, positivity=False, report_freq=10):
    """
    Proximal gradient algorithm to generate initial soln of

    argmin_x x.T(2 * ID - Ax) + x.T K^{-1} x s.t. x >= 0 

    A - psf.convolve
    ID - dirty image
    x0 - initial guess
    L - max eigenvalue of A.T A
    K - prior covariance matrix

    """
    x = x0.copy()
    y = y0.copy()
    eps = 1.0
    k = 0
    t = 1.0
    gradn = A(x) - b
    likn = np.vdot(x, gradn)
    while eps > tol and k < maxit:
        # gradient decent update
        xold = x.copy()
        gradp = gradn
        likp = likn
        
        # compute prox with line search
        tmp = y - gradp/L 
        if positivity:
            tmp[tmp<0.0] = 0.0
        x = tmp/(1 + sigma0/L)
        gradn = A(x) - b 
        likn = np.vdot(x, gradn)
        flam = back_track_func(x, xold, gradp, likp, L)
        # if likn > flam:
        while likn > flam:
            print("Step size too large, adjusting L", L)
            L *= 1.5
            tmp = y - gradp/L 
            if positivity:
                tmp[tmp<0.0] = 0.0
            x = tmp/(1 + sigma0/L)
            gradn = A(x) - b 
            likn = np.vdot(x, gradn)
            flam = back_track_func(x, xold, gradp, likp, L)
        # else:
        #     L /= 1.2

        # fista update
        tp = t
        t = (1. + np.sqrt(1. + 4 * tp**2))/2.

        # soln update
        y = x + (tp - 1)/t * (x - xold)
        
        # check convergence criteria
        eps = norm(x - xold)/norm(x)

        if not k%report_freq:
            print("At iteration %i eps = %f and current stepsize is %f"%(k, eps, L))

        k += 1

    if k >= maxit:
        print("PG - Maximum iterations reached. Relative difference between updates = ", eps)
    else:
        print("PG - Success, converged after %i iterations"%k)
    return x, y, L

def fista(A, b, x0, L, lam, tol=1e-5, maxit=3000, positivity=True):
    """
    Fast Iterative Shrinkage Thresholding Algorithm for problems of the form

    argmin_x (b - x).T A (b - x) + lam |x|_1

    An additional positivity constraint can also be applied

    """
    t = 1
    x = x0.copy()
    y = x0.copy()
    eps = 1.0
    k = 0
    gradn = A(b - x)
    likn = np.vdot(b-x, gradn)
    while eps > tol and k < maxit:
        # gradient decent update
        xold = x.copy()
        gradp = gradn
        likp = likn

        #z += A(b-x)/L
        
        # l1 threshold
        x = np.maximum(np.abs(y + gradp/L) - lam/L, 0.0) * np.sign(y + gradp/L)
        if positivity:
            x[x<0] = 0.0
        gradn = A(b - x)
        likn = np.vdot(x, gradn)
        flam = back_track_func(x, xold, gradp, likp, L)
        while likn > flam:
            # print("Step size too large, adjusting L")
            L *= 1.5
            x = np.maximum(np.abs(y + gradp/L) - lam/L, 0.0) * np.sign(y + gradp/L)
            if positivity:
                x[x<0] = 0.0
            gradn = A(b - x)
            likn = np.vdot(b-x, gradn)
            flam = back_track_func(x, xold, gradp, likp, L)
            
        # fista update
        tp = t
        t = (1. + np.sqrt(1. + 4 * tp**2))/2.

        # soln update
        y = x + (tp - 1)/t * (x - xold)

        # check convergence criteria
        eps = norm(x - xold)/norm(x)

        if not k%10:
            print("At iteration %i eps = %f and current stepsize is %f"%(k, eps, L))

        k += 1

    if k >= maxit:
        print("FISTA - Maximum iterations reached. Relative difference between updates = ", eps)
    else:
        print("FISTA - Success, converged after %i iterations"%k)
    return x

def hpd(A, b, 
        x0, v210, vn0,  # initial guess for primal and dual variables 
        L, nu21, nun,  # spectral norm 
        sig_21, sig_n, # regulariser strengths
        gamma_21, gamma_n,  # step sizes
        lam_21, lam_n, lam_p,  # extrapolation
        tol=1e-3, maxit=1000, positivity=True, report_freq=10):
    """
    Algorithm to solve problems of the form

    argmin_x (b - x).T A (b - x) + lam_21 |x|_21 + lam_n |s|_1

    where x is the image cube and s are the singular values along 
    the frequency dimension and there is an optional positivity 
    constraint. 

    L        - Lipschitz constant of A
    sig_21   - strength of l21 regulariser
    sig_n    - strength of nuclear norm regulariser
    gamma_21 - l21 step size
    gamma_n  - nuclear step size
    lam_21   - extrapolation parameter for l21 step
    lam_n    - extrapolation parameter for nuclear step
    lam_p    - extrapolation parameter for primal step
    
    # Note - Spectral norms for both decomposition operators are unity.
    """
    nchan, nx, ny = x0.shape
    npix = nx*ny

    # set up dual variables
    v21 = v210
    vn = vn0

    # gradient function
    grad_func = lambda x: -A(b-x)

    # stepsize control
    tau = 0.95/(L/2.0 + gamma_21 + gamma_n)

    # start iterations
    x = x0.copy()
    eps = 1.0
    k = 0
    for k in range(maxit):
        xp = x.copy()

        # gradient
        grad = grad_func(x)

        # primal update
        p = x - tau*(grad + vn.reshape(nchan, nx, ny) + v21.reshape(nchan, nx, ny))
        if positivity:
            p[p<0] = 0.0
        
        x = x + lam_p*(p - xp)

        # convergence check
        eps = norm(x-xp)/norm(x)
        if eps < tol:
            break

        if not k%10:
            print("At iteration %i eps = %f and current stepsize is %f"%(k, eps, L))

        # Vector on which dual operations take place flattened into matrix
        xtmp = (2*p - xp).reshape(nchan, npix)

        # Nuclear dual update
        U, s, Vh = svd(vn + xtmp, full_matrices=False)
        s = np.maximum(np.abs(s) - gamma_n * sig_n, 0.0) * np.sign(s)
        vn = vn + lam_n*(xtmp - U.dot(s[:, None] * Vh))

        # l21 dual update
        r21 = v21 + xtmp
        l2norm = norm(r21, axis=0)
        l2_soft = np.maximum(np.abs(l2norm) - gamma_21 * sig_21, 0.0) * np.sign(l2norm)
        indices = np.nonzero(l2norm)
        ratio = np.zeros(l2norm.shape, dtype=np.float64)
        ratio[indices] = l2_soft[indices]/l2norm[indices]
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