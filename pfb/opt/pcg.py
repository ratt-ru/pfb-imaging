import numpy as np
import dask.array as da
import pyscilog
log = pyscilog.get_logger('PCG')


def pcg(A,
        b,
        x0,
        M=None,
        tol=1e-5,
        maxit=500,
        minit=100,
        verbosity=1,
        report_freq=10,
        backtrack=True):

    if M is None:
        def M(x): return x

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
    eps = 1.0
    stall_count = 0
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        rnorm = np.vdot(r, y)
        alpha = rnorm / np.vdot(p, Ap)
        x = xp + alpha * p
        r = rp + alpha * Ap
        y = M(r)
        rnorm_next = np.vdot(r, y)
        while rnorm_next > rnorm and backtrack:  # TODO - better line search
            alpha *= 0.75
            x = xp + alpha * p
            r = rp + alpha * Ap
            y = M(r)
            rnorm_next = np.vdot(r, y)

        beta = rnorm_next / rnorm
        p = beta * p - y
        rnorm = rnorm_next
        k += 1
        epsx = np.linalg.norm(x - xp) / np.linalg.norm(x)
        epsn = rnorm / eps0
        epsp = eps
        eps = np.maximum(epsx, epsn)

        if np.abs(epsp - eps) < 0.01*tol:
            stall_count += 1

        if not k % report_freq and verbosity > 1:
            print("At iteration %i eps = %f" % (k, eps), file=log)

    if k >= maxit:
        if verbosity:
            print("Max iters reached. eps = %f." % eps, file=log)
    elif stall_count >= 5:
        if verbosity:
            print("Stalled. eps = %f." % eps, file=log)
    else:
        if verbosity:
            print("Success, converged after %i iters" % k, file=log)
    return x

from pfb.operators.psf import _hessian_reg
from functools import partial
def _pcg_psf(psfhat,
             b,
             x0,
             sigma,
             nthreads,
             padding,
             unpad_x,
             unpad_y,
             lastsize,
             tol=1e-5,
             maxit=500,
             minit=100,
             verbosity=1,
             report_freq=10,
             backtrack=True):
    '''
    A specialised distributed version of pcg when the operator implements
    convolution with the psf (+ L2 regularisation by sigma**2)
    '''
    nband, nx, ny = b.shape
    model = np.zeros((nband, nx, ny), dtype=b.dtype)
    sigmasq = sigma**2
    def M(x): return x * sigmasq
    for k in range(nband):
        A = partial(_hessian_reg,
                    psfhat=psfhat[k:k+1],
                    sigmasq=sigmasq,
                    padding=padding,
                    nthreads=nthreads,
                    unpad_x=unpad_x,
                    unpad_y=unpad_y,
                    lastsize=lastsize)
        model[k] = pcg(A, b[k:k+1], x0[k:k+1],
                       M=M, tol=tol, maxit=maxit, minit=minit,
                       verbosity=verbosity, report_freq=report_freq, backtrack=backtrack)

    return model

def pcg_psf_wrapper(psfhat,
                    b,
                    x0,
                    sigma,
                    nthreads,
                    padding,
                    unpad_x,
                    unpad_y,
                    lastsize,
                    tol,
                    maxit,
                    minit,
                    verbosity,
                    report_freq,
                    backtrack):
    return _pcg_psf(psfhat[0][0],
                    b,
                    x0,
                    sigma,
                    nthreads,
                    padding,
                    unpad_x,
                    unpad_y,
                    lastsize,
                    tol,
                    maxit,
                    minit,
                    verbosity,
                    report_freq,
                    backtrack)

def pcg_psf(psfhat,
            b,
            x0,
            sigma,
            nthreads,
            padding,
            unpad_x,
            unpad_y,
            lastsize,
            tol,
            maxit,
            minit,
            verbosity,
            report_freq,
            backtrack):
    model = da.blockwise(pcg_psf_wrapper, ('nband', 'nx', 'ny'),
                         psfhat, ('nband', 'nx_psf', 'ny_psf'),
                         b, ('nband', 'nx', 'ny'),
                         x0, ('nband', 'nx', 'ny'),
                         sigma, None,
                         nthreads, None,
                         padding, None,
                         unpad_x, None,
                         unpad_y, None,
                         lastsize, None,
                         tol, None,
                         maxit, None,
                         minit, None,
                         verbosity, None,
                         report_freq, None,
                         backtrack, None,
                         dtype=b.dtype)
    return model