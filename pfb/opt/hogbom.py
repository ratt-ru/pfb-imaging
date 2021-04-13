import numpy as np
from numba import njit
import numexpr as ne
import pyscilog
log = pyscilog.get_logger('HOGBOM')


def hogbom(
        ID,
        PSF,
        gamma=0.1,
        pf=0.1,
        maxit=10000,
        report_freq=1000,
        verbosity=1):
    nband, nx, ny = ID.shape
    _, nx_psf, ny_psf = PSF.shape
    nx0 = nx_psf//2
    ny0 = ny_psf//2
    x = np.zeros((nband, nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    IRsearch = np.sum(IR, axis=0)**2
    pq = IRsearch.argmax()
    p = pq//ny
    q = pq - p*ny
    IRmax = np.sqrt(IRsearch[p, q])
    _, nx_psf, ny_psf = PSF.shape
    wsums = np.amax(PSF.reshape(-1, nx_psf*ny_psf), axis=1)
    tol = pf * IRmax
    k = 0
    stall_count = 0
    while IRmax > tol and k < maxit and stall_count < 5:
        xhat = IR[:, p, q] / wsums
        x[:, p, q] += gamma * xhat
        ne.evaluate('IR - gamma * xhat * psf', local_dict={
                    'IR': IR,
                    'gamma': gamma,
                    'xhat': xhat[:, None, None],
                    'psf': PSF[:, nx0 - p:nx0 + nx - p, ny0 - q:ny0 + ny - q]},
                    out=IR, casting='same_kind')
        IRsearch = np.sum(IR, axis=0)**2
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmaxp = IRmax
        IRmax = np.sqrt(IRsearch[p, q])
        k += 1

        if np.abs(IRmaxp - IRmax) < 1e-5:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            print("At iteration %i max residual = %f" % (k, IRmax), file=log)

    if k >= maxit:
        if verbosity:
            print("Maximum iterations reached. Max of residual = %f." %
                (IRmax), file=log)
    elif stall_count >= 5:
        if verbosity:
            print("Stalled. Max of residual = %f." %
                (IRmax), file=log)
    else:
        if verbosity:
            print("Success, converged after %i iterations" % k, file=log)
    return x
