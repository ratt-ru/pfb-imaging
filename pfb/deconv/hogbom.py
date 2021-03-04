import numpy as np

def hogbom(ID, PSF, gamma=0.1, pf=0.1, maxit=10000, report_freq=1000, verbosity=1):
    nband, nx, ny = ID.shape
    x = np.zeros((nband, nx, ny), dtype=ID.dtype) 
    IR = ID.copy()
    IRmean = np.abs(np.mean(IR, axis=0))
    IRmax = IRmean.max()
    tol = pf*IRmax
    for k in range(maxit):
        if IRmax < tol:
            break
        p, q = np.argwhere(IRmean == IRmax).squeeze()
        if np.size(p) > 1:
            p = p.squeeze()[0]
            q = q.squeeze()[0]
        xhat = IR[:, p, q]
        x[:, p, q] += gamma * xhat
        IR -= gamma * xhat[:, None, None] * PSF[:, nx-p:2*nx - p, ny-q:2*ny - q]
        IRmean = np.abs(np.mean(IR, axis=0))
        IRmax = IRmean.max()

        if not k%report_freq and verbosity > 1:
            print("         Hogbom - At iteration %i max residual = %f"%(k, IRmax))
    
    if k >= maxit:
        if verbosity:
            print("         Hogbom - Maximum iterations reached. Max of residual = %f.  "%(IRmax))
    else:
        if verbosity:
            print("         Hogbom - Success, converged after %i iterations"%k)
    return x