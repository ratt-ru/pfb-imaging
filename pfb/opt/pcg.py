from time import time
import numpy as np
from numba import njit, prange
from numba.extending import overload
import numexpr as ne
from functools import partial
import dask.array as da
from distributed import wait
from uuid import uuid4
from pfb.utils.misc import norm_diff, fitcleanbeam, Gaussian2D, taperf, JIT_OPTIONS
from pfb.utils.naming import xds_from_list
from ducc0.misc import empty_noncritical
from ducc0.fft import c2c
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

# import pyscilog
# log = pyscilog.get_logger('PCG')

@njit(**JIT_OPTIONS, parallel=True)
def update(x, xp, r, rp, p, Ap, alpha):
    return update_impl(x, xp, r, rp, p, Ap, alpha)


def update_impl(x, xp, r, rp, p, Ap, alpha):
    return NotImplementedError


@overload(update_impl, jit_options=JIT_OPTIONS, parallel=True)
def nb_update_impl(x, xp, r, rp, p, Ap, alpha):
    if x.ndim==3:
        def impl(x, xp, r, rp, p, Ap, alpha):
            nband, nx, ny = x.shape
            for b in range(nband):
                for i in prange(nx):
                    for j in range(ny):
                        x[b, i, j] = xp[b, i, j] + alpha * p[b, i, j]
                        r[b, i, j] = rp[b, i, j] + alpha * Ap[b, i, j]
            return x, r
    elif x.ndim==2:
        def impl(x, xp, r, rp, p, Ap, alpha):
            nx, ny = x.shape
            for i in prange(nx):
                for j in range(ny):
                    x[i, j] = xp[i, j] + alpha * p[i, j]
                    r[i, j] = rp[i, j] + alpha * Ap[i, j]
            return x, r
    else:
        raise ValueError("norm_diff is only implemented for 2D or 3D arrays")

    return impl


@njit(**JIT_OPTIONS, parallel=True)
def alpha_update(r, y, p, Ap):
    return alpha_update_impl(r, y, p, Ap)


def alpha_update_impl(r, y, p, Ap):
    return NotImplementedError


@overload(alpha_update_impl, jit_options=JIT_OPTIONS, parallel=True)
def nb_alpha_update_impl(r, y, p, Ap):
    if r.ndim==2:
        def impl(r, y, p, Ap):
            rnorm = 0.0
            rnorm_den = 0.0
            nx, ny = r.shape
            for i in prange(nx):
                for j in range(ny):
                    rnorm += r[i,j]*y[i,j]
                    rnorm_den += p[i,j]*Ap[i,j]

            alpha = rnorm/rnorm_den
            return rnorm, alpha
    elif r.ndim==3:
        def impl(r, y, p, Ap):
            rnorm = 0.0
            rnorm_den = 0.0
            nband, nx, ny = r.shape
            for b in range(nband):
                for i in prange(nx):
                    for j in range(ny):
                        rnorm += r[b,i,j]*y[b,i,j]
                        rnorm_den += p[b,i,j]*Ap[b,i,j]

            alpha = rnorm/rnorm_den
            return rnorm, alpha
    return impl


@njit(**JIT_OPTIONS, parallel=True)
def alpha_update(r, y, p, Ap):
    return alpha_update_impl(r, y, p, Ap)


def alpha_update_impl(r, y, p, Ap):
    return NotImplementedError


@overload(alpha_update_impl, jit_options=JIT_OPTIONS, parallel=True)
def nb_alpha_update_impl(r, y, p, Ap):
    if r.ndim==2:
        def impl(r, y, p, Ap):
            rnorm = 0.0
            rnorm_den = 0.0
            nx, ny = r.shape
            for i in prange(nx):
                for j in range(ny):
                    rnorm += r[i,j]*y[i,j]
                    rnorm_den += p[i,j]*Ap[i,j]

            alpha = rnorm/rnorm_den
            return rnorm, alpha
    elif r.ndim==3:
        def impl(r, y, p, Ap):
            rnorm = 0.0
            rnorm_den = 0.0
            nband, nx, ny = r.shape
            for b in range(nband):
                for i in prange(nx):
                    for j in range(ny):
                        rnorm += r[b,i,j]*y[b,i,j]
                        rnorm_den += p[b,i,j]*Ap[b,i,j]

            alpha = rnorm/rnorm_den
            return rnorm, alpha
    return impl


@njit(**JIT_OPTIONS, parallel=True)
def beta_update(r, y, p, rnorm):
    return beta_update_impl(r, y, p, rnorm)


def beta_update_impl(r, y, p, rnorm):
    return NotImplementedError


@overload(beta_update_impl, jit_options=JIT_OPTIONS, parallel=True)
def nb_beta_update_impl(r, y, p, rnorm):
    if r.ndim==2:
        def impl(r, y, p, rnorm):
            rnorm_next = 0.0
            nx, ny = r.shape
            for i in prange(nx):
                for j in range(ny):
                    rnorm_next += r[i,j]*y[i,j]

            beta = rnorm_next/rnorm

            for i in prange(nx):
                for j in range(ny):
                    p[i,j] = beta * p[i,j] - y[i,j]
            return rnorm_next, p
    elif r.ndim==3:
        def impl(r, y, p, rnorm):
            rnorm_next = 0.0
            nband, nx, ny = r.shape
            for b in range(nband):
                for i in prange(nx):
                    for j in range(ny):
                        rnorm_next += r[b,i,j]*y[b,i,j]

            beta = rnorm_next/rnorm
            for b in range(nband):
                for i in prange(nx):
                    for j in range(ny):
                        p[b,i,j] = beta * p[b,i,j] - y[b,i,j]
            return rnorm_next, p
    return impl


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
        print(f"Initial residual is zero")  #, file=log)
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
    xp = x.copy()
    rp = r.copy()
    tcopy = 0.0
    tA = 0.0
    tvdot = 0.0
    tupdate = 0.0
    tp = 0.0
    tnorm = 0.0
    tii = time()
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        ti = time()
        np.copyto(xp, x)
        np.copyto(rp, r)
        tcopy += (time() - ti)
        ti = time()
        Ap = A(p)
        tA += (time() - ti)
        ti = time()
        rnorm = np.vdot(r, y)
        alpha = rnorm / np.vdot(p, Ap)
        # import ipdb; ipdb.set_trace()
        # rnorm, alpha = alpha_update(r, y, p, Ap)
        tvdot += (time() - ti)
        ti = time()
        # x = xp + alpha * p
        # r = rp + alpha * Ap
        ne.evaluate('xp + alpha*p',
                    out=x,
                    local_dict={
                        'xp': xp,
                        'alpha': alpha,
                        'p': p},
                    casting='unsafe')
        ne.evaluate('rp + alpha*Ap',
                    out=r,
                    local_dict={
                        'rp': rp,
                        'alpha': alpha,
                        'Ap': Ap},
                    casting='unsafe')
        # x, r = update(x, xp, r, rp, p, Ap, alpha)
        tupdate += (time() - ti)
        y = M(r)

        # while rnorm_next > rnorm and backtrack:  # TODO - better line search
        #     alpha *= 0.75
        #     x = xp + alpha * p
        #     r = rp + alpha * Ap
        #     y = M(r)
        #     rnorm_next = np.vdot(r, y)

        ti = time()
        rnorm_next = np.vdot(r, y)
        beta = rnorm_next / rnorm
        ne.evaluate('beta*p-y',
                    out=p,
                    local_dict={
                        'beta': beta,
                        'p': p,
                        'y': y},
                    casting='unsafe')

        # p = beta * p - y
        # rnorm, p = beta_update(r, y, p, rnorm)
        tp += (time() - ti)
        rnorm = rnorm_next
        k += 1
        epsp = eps
        ti = time()
        eps = norm_diff(x, xp)
        phi = rnorm / phi0
        tnorm += (time() - ti)

        if np.abs(epsp - eps) < 1e-3*tol:
            stall_count += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}, phi = {phi:.3e}")
                #   file=log)
    ttot = time() - tii
    ttally = tcopy + tA + tvdot + tupdate + tp + tnorm
    print('tcopy = ', tcopy/ttot)
    print('tA = ', tA/ttot)
    print('tvdot = ', tvdot/ttot)
    print('tupdate = ', tupdate/ttot)
    print('tp = ', tp/ttot)
    print('tnorm = ', tnorm/ttot)
    print('ttally = ', ttally/ttot)

    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}")  #, file=log)
    elif stall_count >= 5:
        if verbosity:
            print(f"Stalled after {k} iterations with eps = {eps:.3e}")  #, file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations")  #, file=log)
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
                  eta,
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
    if eta > 0:
        def M(x): return x / eta
    else:
        M = None

    for k in range(nband):
        xpad = empty_noncritical((nx_psf, lastsize), dtype=b.dtype)
        xhat = empty_noncritical((nx_psf, nyo2), dtype=psfhat.dtype)
        xout = empty_noncritical((nx, ny), dtype=b.dtype)
        A = partial(_hessian_psf_slice,
                    xpad=xpad,
                    xhat=xhat,
                    xout=xout,
                    abspsf=np.abs(psfhat[k]),
                    beam=beam[k],
                    lastsize=lastsize,
                    nthreads=nthreads,
                    eta=eta)

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
             eta,
             cgopts):
    return _pcg_psf_impl(psfhat,
                         b,
                         x0,
                         beam,
                         lastsize,
                         nthreads,
                         eta,
                         **cgopts)

def pcg_psf(psfhat,
            b,
            x0,
            beam,
            lastsize,
            nthreads,
            eta,
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
                         eta, None,
                         cgopts, None,
                         align_arrays=False,
                         dtype=b.dtype)
    if compute:
        return model.compute()
    else:
        return model


def pcg_dds(ds_name,
            eta,  # regularisation for Hessian approximation
            sigma,       # regularisation for preconditioner
            mask=1.0,
            use_psf=True,
            residual_name='RESIDUAL',
            model_name='MODEL',
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
    # avoid circular import
    from pfb.operators.hessian import _hessian_slice, hess_direct_slice

    # expects a list
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    # drop_vars = ['PSF']
    # if not use_psf:
    #     drop_vars.append('PSFHAT')
    drop_vars = None
    ds = xds_from_list(ds_name, nthreads=nthreads,
                       drop_vars=drop_vars)[0]

    if residual_name in ds:
        j = getattr(ds, residual_name).values * mask * ds.BEAM.values
        ds = ds.drop_vars(residual_name)
    else:
        j = ds.DIRTY.values * mask * ds.BEAM.values

    psf = ds.PSF.values
    nx_psf, py_psf = psf.shape
    nx, ny = j.shape
    wsum = np.sum(ds.WEIGHT.values * ds.MASK.values)
    psf /= wsum
    j /= wsum

    # downweight edges of field compared to center
    # this allows the PCG to downweight the fit to the edges
    # which may be contaminated by edge effects and also
    # stabalises the preconditioner
    width = np.minimum(int(0.1*nx), 32)
    taperxy = taperf((nx, ny), width)
    # eta /= taperxy

    # set precond if PSF is present
    if 'PSFHAT' in ds and use_psf:
        psfhat = np.abs(ds.PSFHAT.values)/wsum
        ds.drop_vars(('PSFHAT'))
        nx_psf, nyo2 = psfhat.shape
        ny_psf = 2*(nyo2-1)  # is this always the case?
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
        xpad = empty_noncritical((nx_psf, ny_psf),
                                 dtype=j.dtype)
        xhat = empty_noncritical((nx_psf, nyo2),
                                 dtype='c16')
        xout = empty_noncritical((nx, ny),
                                 dtype=j.dtype)
        precond = partial(
                    hess_direct_slice,
                    xpad=xpad,
                    xhat=xhat,
                    xout=xout,
                    abspsf=psfhat,
                    taperxy=taperxy,
                    lastsize=ny_psf,
                    nthreads=nthreads,
                    eta=sigma,
                    mode='backward')

        x0 = precond(j)

        # get intrinsic resolution by deconvolving psf
        upsf = precond(psf[unpad_x, unpad_y])
        upsf /= upsf.max()
        gaussparu = fitcleanbeam(upsf[None], level=0.25, pixsize=1.0)[0]
        ds = ds.assign(**{
            'UPSF': (('x', 'y'), upsf)
        })
        ds = ds.assign_attrs(gaussparu=gaussparu)
    else:
        # print('Not using preconditioning')
        precond = None
        x0 = np.zeros_like(j)

    hess = partial(_hessian_slice,
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
                   nthreads=nthreads,
                   eta=eta,
                   wsum=wsum)

    x = pcg(hess,
            j,
            x0=x0.copy(),
            M=precond,
            tol=tol,
            maxit=maxit,
            minit=1,
            verbosity=verbosity,
            report_freq=report_freq,
            backtrack=False,
            return_resid=False)

    if model_name in ds:
        model = getattr(ds, model_name).values + x
    else:
        model = x


    resid = ds.DIRTY.values - _hessian_slice(
                                        model,
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

    ds = ds.assign(**{
        'MODEL_MOPPED': (('x','y'), model),
        'RESIDUAL_MOPPED': (('x','y'), resid),
        'UPDATE': (('x','y'), x),
        'X0': (('x','y'), x0),
    })

    ds.to_zarr(ds_name[0], mode='a')

    return resid, int(ds.bandid)


