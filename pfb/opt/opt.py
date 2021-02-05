import numpy as np
from scipy.linalg import norm


def power_method(A, imsize, b0=None, tol=1e-5, maxit=250, verbosity=1, report_freq=25):
    if b0 is None:
        b = np.random.randn(*imsize)
        b /= norm(b)
    else:
        b = b0/norm(b0)
    beta = 1.0
    eps = 1.0
    k = 0
    while eps > tol and k < maxit:
        bp = b
        b = A(bp)
        bnorm = np.linalg.norm(b)
        betap = beta
        beta = np.vdot(bp, b)/np.vdot(bp, bp)
        b /= bnorm
        eps = np.linalg.norm(beta - betap)/betap
        k += 1

        if not k%report_freq and verbosity > 1:
            print("         At iteration %i eps = %f"%(k, eps))

    if k == maxit:
        if verbosity:
            print("         PM - Maximum iterations reached. eps = %f, current beta = %f"%(eps, beta))
    else:
        if verbosity:
            print("         PM - Success, converged after %i iterations. beta = %f"%(k, beta))
    return beta, bp

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


        

