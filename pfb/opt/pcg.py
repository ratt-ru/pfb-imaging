import numpy as np
from functools import partial
import dask.array as da
from distributed import wait
from uuid import uuid4
from ducc0.misc import make_noncritical
from pfb.utils.misc import norm_diff
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
        eps = norm_diff(x, xp)
        # np.linalg.norm(x - xp) / np.linalg.norm(x)
        # epsn = rnorm / eps0
        # eps = rnorm / eps0
        # eps = np.maximum(epsx, epsn)

        # if np.abs(epsp - eps) < 1e-3*tol:
        #     stall_count += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} epsx = {eps:.3e}",
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


def cg_dct(A,
           b,
           x,
           tol=1e-5,
           maxit=500,
           verbosity=1,
           report_freq=10):
    '''
    A specialised version of the pcg that doesn't need x to live on a
    single grid. As a result x is nested a dictionary keyed on field
    name and then on time and freq. There is currently no support for
    overlapping fields i.e. fields need to be distinct. This simplified
    version does not yet support bactracking or pre-conditioning so it
    will probably only work if A is a very good approximation of Ax=b
    '''

    def residual(Ax, b, r, p):
        for field in Ax.keys():
            for i in Ax[field].keys():
                r[field][i] = Ax[field][i] - b[field][i]
                p[field][i] = -r[field][i]
        return r, p

    def vdot_dct(a, b):
        res = 0.0
        for field in a.keys():
            for i in a[field].keys():
                res += np.vdot(a[field][i], b[field][i])
        return res

    def pluseq_dct(a, b, alpha=1.0):
        # implement a += alpha * b
        for field in a.keys():
            for i in a[field].keys():
                a[field][i] += alpha * b[field][i]
        return a

    def pupdate(p, r, beta=1):
        # implement p = beta * p - r
        for field in p.keys():
            for i in p[field].keys():
                p[field][i] = beta * p[field][i] - r[field][i]
        return p

    def norm_dct(a, b=None):
        norm = 0.0
        for field in a.keys():
            for i in a[field].keys():
                if b is not None:
                    norm += np.linalg.norm(a[field][i] - b[field][i])
                else:
                    norm += np.linalg.norm(a[field][i])
        return norm

    # initial residual
    Ax = A(x)
    r = {}
    p = {}
    for field in Ax.keys():
        r[field] = {}
        p[field] = {}
        for i in Ax[field].keys():
            r[field][i] = Ax[field][i] - b[field][i]
            p[field][i] = -r[field][i]


    rnorm = vdot_dct(r, r)
    rnorm0 = rnorm
    eps = rnorm
    k = 0
    while eps > tol and k < maxit:
        xp = x
        rp = r
        Ap = A(p)
        pAp = vdot_dct(p, Ap)
        alpha = rnorm/pAp  #np.vdot(p, Ap)
        x = pluseq_dct(x, p, alpha=alpha)
        r = pluseq_dct(r, Ap, alpha=alpha)
        # x += alpha * p
        # r += alpha * Ap
        rnorm_next = vdot_dct(r, r)
        beta = rnorm_next/rnorm
        p = pupdate(p, r, beta=beta)
        # p = beta*p - r
        rnorm = rnorm_next
        eps = rnorm
        # eps = norm_dct(x, b=xp) / norm_dct(x)

        k += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}",
                  file=log)

    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations", file=log)
    return x, r


from pfb.operators.hessian import _hessian_psf_slice
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
        xpad = np.empty((nx_psf, lastsize), dtype=b.dtype, order='C')
        xpad = make_noncritical(xpad)
        xhat = np.empty((nx_psf, nyo2), dtype=psfhat.dtype, order='C')
        xhat = make_noncritical(xhat)
        xout = np.empty((nx, ny), dtype=b.dtype, order='C')
        xout = make_noncritical(xout)
        A = partial(_hessian_psf_slice,
                    xpad,
                    xhat,
                    xout,
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


