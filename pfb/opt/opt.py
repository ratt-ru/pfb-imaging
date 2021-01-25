import numpy as np
from scipy.linalg import norm
from pfb.utils import prox_21

def power_method(A, imsize, b0=None, tol=1e-5, maxit=250):
    if b0 is None:
        b = np.random.randn(*imsize)
        b /= norm(b)
    else:
        b = b0/norm(b0)
    eps = 1.0
    k = 0
    beta = 1.0
    while eps > tol and k < maxit:
        bp = b
        b = A(bp)
        bnorm = norm(b)
        b /= bnorm
        betap = beta
        beta = np.vdot(bp, b)
        eps = np.abs(beta - betap)/betap
        k += 1

    if k == maxit:
        print("PM - Maximum iterations reached. eps = ", eps)
    else:
        print("PM - Success - convergence after %i iterations"%k)
    return beta, b

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


        

