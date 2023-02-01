import numpy as np
from functools import partial
import dask.array as da
from distributed import wait
from uuid import uuid4
import pyscilog
log = pyscilog.get_logger('PCG')


def cg(A,
       b,
       x0=None,
       tol=1e-5,
       maxit=500,
       verbosity=1,
       report_freq=10):

    if x0 is None:
        x = np.zeros(b.shape, dtype=b.dtype)
    else:
        x = x0.copy()

    # initial residual
    r = A(x) - b
    p = -r
    rnorm = np.vdot(r, r)
    rnorm0 = rnorm
    eps = rnorm
    k = 0
    while eps > tol and k < maxit:
        xp = x
        rp = r
        Ap = A(p)
        alpha = rnorm/np.vdot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        rnorm_next = np.vdot(r, r)
        beta = rnorm_next/rnorm
        p = beta*p - r
        rnorm = rnorm_next
        eps = rnorm #/rnorm0

        k += 1

    if k >= maxit:
        print(f"Max iters reached eps = {eps}")

    return x


def pcg(A,
        b,
        x0=None,
        M=None,
        tol=1e-5,
        maxit=500,
        minit=100,
        verbosity=1,
        report_freq=10,
        backtrack=True,
        return_resid=False):

    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype)

    if M is None:
        def M(x): return x

    r = A(x0) - b
    y = M(r)
    if not np.any(y):
        print(f"Initial residual is zero", file=log)
        return x0
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
        # if p is zero we should stop
        if not np.any(p):
            break
        rnorm = rnorm_next
        k += 1
        epsp = eps
        eps = np.linalg.norm(x - xp) / np.linalg.norm(x)
        # epsn = rnorm / eps0
        # eps = rnorm / eps0
        # eps = np.maximum(epsx, epsn)

        # if np.abs(epsp - eps) < 1e-3*tol:
        #     stall_count += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} epsx = {epsx:.3e}, epsn = {epsn:.3e}",
                  file=log)

    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}", file=log)
    elif stall_count >= 5:
        if verbosity:
            print(f"Stalled after {k} iterations with eps = {eps:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations", file=log)
    if not return_resid:
        return x
    else:
        return x, r

from pfb.operators.hessian import hessian_psf_slice
def _pcg_psf_impl(psfhat,
                  b,
                  x0,
                  beam,
                  lastsize,
                  nthreads,
                  sigmainv,
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
    _, nx_psf, nyo2 = psfhat.shape
    model = np.zeros((nband, nx, ny), dtype=b.dtype, order='C')
    # PCG preconditioner
    if sigmainv > 0:
        def M(x): return x / sigmainv
    else:
        M = None

    for k in range(nband):
        A = partial(hessian_psf_slice,
                    np.empty((nx_psf, lastsize), dtype=b.dtype, order='C'),  # xpad
                    np.empty((nx_psf, nyo2), dtype=psfhat.dtype, order='C'), # xhat
                    np.empty((nx, ny), dtype=b.dtype, order='C'),            # xout
                    psfhat[k],
                    beam[k],
                    lastsize,
                    nthreads=nthreads,
                    sigmainv=sigmainv)

        model[k] = pcg(A, b[k], x0[k],
                       M=M, tol=tol, maxit=maxit, minit=minit,
                       verbosity=verbosity, report_freq=report_freq,
                       backtrack=backtrack)

    return model

def _pcg_psf(psfhat,
             b,
             x0,
             beam,
             lastsize,
             nthreads,
             sigmainv,
             cgopts):
    return _pcg_psf_impl(psfhat,
                         b,
                         x0,
                         beam,
                         lastsize,
                         nthreads,
                         sigmainv,
                         **cgopts)

def pcg_psf(psfhat,
            b,
            x0,
            beam,
            lastsize,
            nthreads,
            sigmainv,
            cgopts,
            compute=True):

    if not isinstance(x0, da.Array):
        x0 = da.from_array(x0, chunks=(1, -1, -1),
                          name="x-" + uuid4().hex)

    if not isinstance(psfhat, da.Array):
        psfhat = da.from_array(psfhat, chunks=(1, -1, -1),
                               name="psfhat-" + uuid4().hex)
    if not isinstance(b, da.Array):
        b = da.from_array(b, chunks=(1, -1, -1),
                          name="psfhat-" + uuid4().hex)

    if beam is None:
        bout = None
    else:
        bout = ('nb', 'nx', 'ny')
        if beam.ndim == 2:
            beam = beam[None]

        if not isinstance(beam, da.Array):
            beam = da.from_array(beam, chunks=(1, -1, -1),
                                 name="beam-" + uuid4().hex)
        if beam.shape[0] == 1:
            beam = da.tile(beam, (psfhat.shape[0], 1, 1))
        elif beam.shape[0] != psfhat.shape[0]:
            raise ValueError('Beam has incorrect shape')

    model = da.blockwise(_pcg_psf, ('nb', 'nx', 'ny'),
                         psfhat, ('nb', 'nx', 'ny'),
                         b, ('nb', 'nx', 'ny'),
                         x0, ('nb', 'nx', 'ny'),
                         beam, bout,
                         lastsize, None,
                         nthreads, None,
                         sigmainv, None,
                         cgopts, None,
                         align_arrays=False,
                         dtype=b.dtype)
    if compute:
        return model.compute()
    else:
        return model


def pcg_dist(A, maxit, minit, tol, sigmainv):
    '''
    kwargs - tol, maxit, minit, nthreads, psf_padding, unpad_x, unpad_y
             sigmainv, lastsize, hessian
    '''

    if hasattr(A, 'residual'):
        b = A.residual/A.wsum
    else:
        b = A.dirty/A.wsum
    x = np.zeros_like(b)

    def M(x):
        return x / sigmainv

    r = A(x) - b
    y = M(r)
    p = -y
    rnorm = np.vdot(r, y)
    if np.isnan(rnorm) or rnorm == 0.0:
        eps0 = 1.0
    else:
        eps0 = rnorm
    k = 0
    eps = 1.0
    stall_count = 0
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        xp = x.copy()
        rp = r.copy()
        epsp = eps
        Ap = A(p)
        rnorm = np.vdot(r, y)
        alpha = rnorm / np.vdot(p, Ap)
        x = xp + alpha * p
        r = rp + alpha * Ap
        y = M(r)
        rnorm_next = np.vdot(r, y)
        while rnorm_next > rnorm:  # TODO - better line search
            alpha *= 0.75
            x = xp + alpha * p
            r = rp + alpha * Ap
            y = M(r)
            rnorm_next = np.vdot(r, y)

        beta = rnorm_next / rnorm
        p = beta * p - y
        # if p is zero we should stop
        if not np.any(p):
            break
        rnorm = rnorm_next
        k += 1
        eps = rnorm / eps0

        if np.abs(eps - epsp) < 1e-3*tol:
            stall_count += 1

    print(f'Band={A.bandid}, iters{k}, eps={eps}', file=log)
    return x


