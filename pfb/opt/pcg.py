import numpy as np
from functools import partial
import dask.array as da
from distributed import wait
from uuid import uuid4
from ducc0.misc import make_noncritical
from pfb.utils.misc import norm_diff
from pfb.utils.naming import xds_from_list
from pfb.operators.hessian import _hessian_impl
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
            print(f"At iteration {k} eps = {eps:.3e}, phi = {phi:.3e}")
                #   file=log)

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


def taperf(shape, taper_width):
    height, width = shape
    array = np.ones((height, width))

    # Create 1D taper for both dimensions
    taper_x = np.ones(width)
    taper_y = np.ones(height)

    # Apply cosine taper to the edges
    taper_x[:taper_width] = 0.5 * (1 + np.cos(np.linspace(1.1*np.pi, 2*np.pi, taper_width)))
    taper_x[-taper_width:] = 0.5 * (1 + np.cos(np.linspace(0, 0.9*np.pi, taper_width)))

    taper_y[:taper_width] = 0.5 * (1 + np.cos(np.linspace(1.1*np.pi, 2*np.pi, taper_width)))
    taper_y[-taper_width:] = 0.5 * (1 + np.cos(np.linspace(0, 0.9*np.pi, taper_width)))

    return np.outer(taper_y, taper_x)


def pcg_dds(ds_name,
            sigmainvsq,  # regularisation for Hessian approximation
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
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    ds = xds_from_list(ds_name, nthreads=nthreads)[0]

    if residual_name in ds:
        residp = getattr(ds, residual_name).values
        j = residp * mask * ds.BEAM.values
        # ds = ds.drop_vars(residual_name)
    else:
        residp = ds.DIRTY.values
        j = residp * mask * ds.BEAM.values
        # ds = ds.drop_vars(('DIRTY'))

    nx, ny = j.shape
    wsum = np.sum(ds.WEIGHT.values * ds.MASK.values)
    # set sigmas relative to wsum
    sigma *= wsum
    sigmainvsq *= wsum

    # downweight edges of field compared to center
    width = int(0.1*nx)
    taperxy = taperf((nx, ny), width)
    sigmainvsq /= taperxy

    # set precond if PSF is present
    if 'PSF' in ds and use_psf:
        psf = ds.PSF.values.astype('c16')
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
        psf = c2c(Fs(psf, axes=(0,1)), forward=True, inorm=0, nthreads=8)
        psf = np.abs(psf)
        xhat = empty_noncritical((nx_psf, ny_psf),
                                 dtype=np.complex128)

        # downweight long spacings where we don't have data
        # 0.5 based on Nyquist oversampling
        taperuv = taperf((nx_psf, ny_psf), int(0.5*nx_psf))
        taperuv = (1 + np.abs((1 - taperuv)))*sigma/2
        taperuv = Fs(taperuv, axes=(0,1))

        def precond(x, xhat=None, abspsf=None, sigma=None, taperxy=None, taperuv=None):
            xhat[...] = 0j
            xhat[unpad_x, unpad_y] = x  * taperxy
            c2c(xhat, out=xhat, forward=True, inorm=0, nthreads=8)
            xhat /= (abspsf + taperuv)
            c2c(xhat, out=xhat, forward=False, inorm=2, nthreads=8)
            return xhat[unpad_x, unpad_y].real * taperxy

        precondo = partial(precond,
                           xhat=xhat,
                           abspsf=psf,
                           sigma=sigma,
                           taperxy=taperxy,
                           taperuv=taperuv)
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
                   nthreads=nthreads,
                   sigmainvsq=sigmainvsq)

    x = pcg(hess,
            j,
            x0=x0.copy(),
            M=precondo,
            tol=tol,
            maxit=maxit,
            minit=1,
            verbosity=verbosity,
            report_freq=report_freq,
            backtrack=False,
            return_resid=False)

    if model_name in ds:
        modelp = getattr(ds, model_name).values
        model = modelp + x
    else:
        modelp = np.zeros((nx ,ny))
        model = x


    resid = ds.DIRTY.values - _hessian_impl(
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
                                        sigmainvsq=0)

    ds = ds.assign(**{
        'MODEL': (('x','y'), model),
        'RESIDUAL': (('x','y'), resid),
        'MODELP': (('x','y'), modelp),
        'RESIDUALP': (('x','y'), residp),
        'UPDATE': (('x','y'), x),
        'X0': (('x','y'), x0)
    })

    ds.to_zarr(ds_name[0], mode='a')

    return resid, int(ds.bandid)


