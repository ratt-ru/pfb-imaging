import numpy as np
from numba import njit
import pyscilog
log = pyscilog.get_logger('HOGBOM')

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
            print("At iteration %i max residual = %f"%(k, IRmax), file=log)
    
    if k >= maxit:
        if verbosity:
            print("Maximum iterations reached. Max of residual = %f.  "%(IRmax), file=log)
    else:
        if verbosity:
            print("Success, converged after %i iterations"%k, file=log)
    return x