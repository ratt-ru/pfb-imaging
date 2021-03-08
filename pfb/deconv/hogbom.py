import numpy as np
from numba import njit

def hogbom(ID, PSF, gamma=0.1, pf=0.1, maxit=10000, report_freq=1000, verbosity=1):
    nband, nx, ny = ID.shape
    x = np.zeros((nband, nx, ny), dtype=ID.dtype) 
    IR = ID.copy()
    IRsearch = np.abs(np.sum(IR, axis=0))
    IRmax = IRsearch.max()
    tol = pf*IRmax
    k = 0
    while IRmax > tol and k < maxit:
        p, q = np.argwhere(IRsearch == IRmax).squeeze()
        if np.size(p) > 1:
            p = p.squeeze()[0]
            q = q.squeeze()[0]
        xhat = IR[:, p, q]
        x[:, p, q] += gamma * xhat
        IR -= gamma * xhat[:, None, None] * PSF[:, nx-p:2*nx - p, ny-q:2*ny - q]
        IRsearch = np.abs(np.sum(IR, axis=0))
        IRmax = IRsearch.max()
        k += 1

        if not k%report_freq and verbosity > 1:
            print("         Hogbom - At iteration %i max residual = %f"%(k, IRmax))
    
    if k >= maxit:
        if verbosity:
            print("         Hogbom - Maximum iterations reached. Max of residual = %f.  "%(IRmax))
    else:
        if verbosity:
            print("         Hogbom - Success, converged after %i iterations"%k)
    return x

# @njit(nogil=True, fastmath=True, inline='always')
def hogbom_mfs(ID, PSF, gamma=0.1, pf=0.1, maxit=10000, report_freq=1000, verbosity=1):
    nx, ny = ID.shape
    x = np.zeros((nx, ny), dtype=ID.dtype) 
    IR = ID.copy()
    IRsearch = IR**2
    IRmax = IRsearch.max()
    tol = pf*np.sqrt(IRmax)
    k = 0
    while np.sqrt(IRmax) > tol and k < maxit:
        p, q = np.argwhere(IRsearch == IRmax).squeeze()
        if np.size(p) > 1:
            p = p.squeeze()[0]
            q = q.squeeze()[0]
        xhat = IR[p, q]
        x[p, q] += gamma * xhat
        IR -= gamma * xhat[None, None] * PSF[nx-p:2*nx - p, ny-q:2*ny - q]
        IRsearch = IR**2
        IRmax = IRsearch.max()
        k += 1

        if not k%report_freq and verbosity > 1:
            print("         Hogbom - At iteration %i max residual = %f"%(k, np.sqrt(IRmax)))
    
    if k >= maxit:
        if verbosity:
            print("         Hogbom - Maximum iterations reached. Max of residual = %f.  "%(np.sqrt(IRmax)))
    else:
        if verbosity:
            print("         Hogbom - Success, converged after %i iterations"%k)
    return x