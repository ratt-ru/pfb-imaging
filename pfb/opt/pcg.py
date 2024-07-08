import numpy as np
from functools import partial
import dask.array as da
from distributed import wait
from uuid import uuid4
from ducc0.misc import make_noncritical
from pfb.utils.misc import norm_diff
from pfb.operators.hessian import _hessian_impl
from ducc0.misc import empty_noncritical
from ducc0.fft import c2c
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

import pyscilog
log = pyscilog.get_logger('PCG')

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
        phi0 = 1.0
    else:
        phi0 = rnorm
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
        phi = rnorm / phi0

        if np.abs(epsp - eps) < 1e-3*tol:
            stall_count += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}, phi = {phi:.3e}",
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


def pcg_dds(ds,
            sigmainvsq,  # regularisation for Hessian approximation
            sigma,       # regularisation for preconditioner
            mask=1.0,
            do_wgridding=True,
            epsilon=5e-4,
            double_accum=True,
            nthreads=1,
            tol=1e-5,
            maxit=500,
            verbosity=1,
            report_freq=10):
    '''
    pcg for fluxmop
    '''
    ds = ds.drop_vars(('PSFHAT','NOISE'))
    ds = dask.persist(ds)[0]

    if 'RESIDUAL' in ds:
        j = ds.RESIDUAL.values
        ds = ds.drop_vars(('DIRTY','RESIDUAL'))
    else:
        j = ds.DIRTY.values
        ds = ds.drop_vars(('DIRTY'))

    j *= mask * ds.BEAM.values

    nx, ny = j.shape

    # set precond if PSF is present
    if 'PSF' in ds:
        psf = ds.PSF.values.astype('c16')
        ds = ds.drop_vars('PSF')
        nx_psf, ny_psf = psf.shape
        nxpadl = (nx_psf - nx)//2
        nxpadr = nx_psf - nx - nxpadl
        nypadl = (ny_psf - ny)//2
        nypadr = ny_psf - ny - nypadl
        if nx_psf != nx:
            unpad_x = slice(nxpadl, -nxpadr)
        else:
            unpad_x = slice(None)
        if ny_psf != ny:
            unpad_y = slice(nypadl, -nypadr)
        else:
            unpad_y = slice(None)
        # TODO - avoid copy
        c2c(Fs(psf, axes=(0,1)), out=psf, forward=True, inorm=0, nthreads=8)
        psf = np.abs(psf)
        xhat = empty_noncritical((nx_psf, ny_psf),
                                 dtype=np.complex128)
        def precond(xhat, x):
            xhat[...] = 0j
            xhat[unpad_x, unpad_y] = x
            c2c(xhat, out=xhat, forward=True, inorm=0, nthreads=8)
            xhat /= (psf + sigma)
            c2c(xhat, out=xhat, forward=False, inorm=2, nthreads=8)
            return xhat[unpad_x, unpad_y].real

        precondo = partial(precond, xhat=xhat)
        x0 = precondo(j)
    else:
        precondo = None
        x0 = np.zeros_like(j)

    hess = partial(_hessian_impl,
                   uvw=ds.UVW.values,
                   weight=ds.WEIGHT.values,
                   vis_mask=ds.MASK.values,
                   freq=ds.FREQ.values,
                   beam=ds.BEAM.values,
                   cell=ds.cell_rad,
                   x0=ds.x0,
                   y0=ds.y0,
                   do_wgridding=do_wgridding,
                   epsilon=epsilon,
                   double_accum=double_accum,
                   nthreads=nthreads)


    x, resid = pcg(hess,
                   j,
                   x0=x0,
                   M=precondo,
                   tol=tol,
                   maxit=maxit,
                   verbosity=verbosity,
                   report_freq=report_freq,
                   backtrack=False,
                   return_resid=True)

    return x, resid, x0


