from time import time
import numpy as np
import numexpr as ne
from functools import partial
import dask.array as da
from distributed import wait
from uuid import uuid4
from pfb.utils.misc import norm_diff, fitcleanbeam, Gaussian2D, taperf
from pfb.utils.naming import xds_from_list
from ducc0.misc import empty_noncritical
from ducc0.fft import c2c
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

# import pyscilog
# log = pyscilog.get_logger('PCG')

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
        tvdot += (time() - ti)
        ti = time()
        ne.evaluate('xp + alpha*p',
                    out=x)
        ne.evaluate('rp + alpha*Ap',
                    out=r)
        # x = xp + alpha * p
        # r = rp + alpha * Ap
        tupdate += (time() - ti)
        y = M(r)
        rnorm_next = np.vdot(r, y)
        while rnorm_next > rnorm and backtrack:  # TODO - better line search
            alpha *= 0.75
            x = xp + alpha * p
            r = rp + alpha * Ap
            y = M(r)
            rnorm_next = np.vdot(r, y)

        ti = time()
        beta = rnorm_next / rnorm
        ne.evaluate('beta*p-y',
                    out=p)
        # p = beta * p - y
        tp += (time() - ti)
        # if p is zero we should stop
        if not np.any(p):
            break
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
    tcopy /= ttot
    tA /= ttot
    tvdot /= ttot
    tupdate /= ttot
    tp /= ttot
    tnorm /= ttot
    ttally = tcopy + tA + tvdot + tupdate + tp + tnorm
    print('tcopy = ', tcopy)
    print('tA = ', tA)
    print('tvdot = ', tvdot)
    print('tupdate = ', tupdate)
    print('tp = ', tp)
    print('tnorm = ', tnorm)
    print('ttally = ', ttally)

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
        ds = ds.drop_vars(('DIRTY'))

    psf = ds.PSF.values
    nx_psf, py_psf = psf.shape
    nx, ny = j.shape
    wsum = np.sum(ds.WEIGHT.values * ds.MASK.values)
    psf /= wsum
    j /= wsum
    # set sigmas relative to wsum
    # sigma *= wsum
    # eta *= wsum

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
                    psfhat=psfhat,
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
                                        ds.UVW.values,
                                        ds.WEIGHT.values,
                                        ds.MASK.values,
                                        ds.FREQ.values,
                                        ds.BEAM.values,
                                        cell=ds.cell_rad,
                                        x0=ds.x0,
                                        y0=ds.y0,
                                        do_wgridding=do_wgridding,
                                        epsilon=epsilon,
                                        double_accum=double_accum,
                                        nthreads=nthreads,
                                        eta=0)

    ds = ds.assign(**{
        'MODEL_MOPPED': (('x','y'), model),
        'RESIDUAL_MOPPED': (('x','y'), resid),
        'UPDATE': (('x','y'), x),
        'X0': (('x','y'), x0),
    })

    ds.to_zarr(ds_name[0], mode='a')

    return resid, int(ds.bandid)


