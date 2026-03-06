from functools import partial
from time import time
from uuid import uuid4

import dask.array as da
import numexpr as ne
import numpy as np
import ray
from ducc0.misc import empty_noncritical
from numba import njit, prange

from pfb_imaging.utils.misc import norm_diff
from pfb_imaging.utils.naming import xds_from_list

ifftshift = np.fft.ifftshift
fftshift = np.fft.fftshift


_FAST_JIT = {"nogil": True, "cache": True, "parallel": True, "fastmath": True}


@njit(**_FAST_JIT)
def _nb_fused_alpha_update(r, y, p, aopp, x, xp, rp):
    """Compute alpha = (r·y)/(p·aopp), then x = xp + alpha*p, r = rp + alpha*aopp."""
    n = r.size
    r_f = r.ravel()
    y_f = y.ravel()
    p_f = p.ravel()
    a_f = aopp.ravel()
    x_f = x.ravel()
    xp_f = xp.ravel()
    rp_f = rp.ravel()

    rnorm = 0.0
    denom = 0.0
    for i in prange(n):
        rnorm += r_f[i] * y_f[i]
        denom += p_f[i] * a_f[i]

    alpha = rnorm / denom

    for i in prange(n):
        x_f[i] = xp_f[i] + alpha * p_f[i]
        r_f[i] = rp_f[i] + alpha * a_f[i]

    return rnorm


@njit(**_FAST_JIT)
def _nb_fused_beta_update(r, y, p, rnorm):
    """Compute beta = (r·y)/rnorm, then p = beta*p - y."""
    n = r.size
    r_f = r.ravel()
    y_f = y.ravel()
    p_f = p.ravel()

    rnorm_next = 0.0
    for i in prange(n):
        rnorm_next += r_f[i] * y_f[i]

    beta = rnorm_next / rnorm

    for i in prange(n):
        p_f[i] = beta * p_f[i] - y_f[i]

    return rnorm_next


@njit(**_FAST_JIT)
def _nb_norm_diff(x, xp):
    """Relative norm of difference: ||x - xp|| / ||x||."""
    n = x.size
    x_f = x.ravel()
    xp_f = xp.ravel()

    num = 0.0
    den = 0.0
    for i in prange(n):
        d = x_f[i] - xp_f[i]
        num += d * d
        den += x_f[i] * x_f[i]

    den = max(den, 1e-12)
    return np.sqrt(num / den)


def pcg_numba(
    aop,
    b,
    x0=None,
    precond=None,
    tol=1e-5,
    maxit=500,
    minit=100,
    verbosity=1,
    report_freq=10,
    backtrack=True,
    return_resid=False,
):
    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype)

    if precond is None:

        def precond(x):
            return x

    r = aop(x0) - b
    y = precond(r)
    if not np.any(y):
        print("Initial residual is zero")
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
    taop = 0.0
    talpha = 0.0
    tprecond = 0.0
    tbeta = 0.0
    tnorm = 0.0
    tii = time()
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        ti = time()
        np.copyto(xp, x)
        np.copyto(rp, r)
        tcopy += time() - ti
        ti = time()
        aopp = aop(p)
        taop += time() - ti
        ti = time()
        rnorm = _nb_fused_alpha_update(r, y, p, aopp, x, xp, rp)
        talpha += time() - ti
        ti = time()
        y = precond(r)
        tprecond += time() - ti
        ti = time()
        rnorm = _nb_fused_beta_update(r, y, p, rnorm)
        tbeta += time() - ti
        k += 1
        epsp = eps
        ti = time()
        eps = _nb_norm_diff(x, xp)
        tnorm += time() - ti
        phi = rnorm / phi0

        if np.abs(epsp - eps) < 1e-3 * tol:
            stall_count += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}, phi = {phi:.3e}")
    ttot = time() - tii
    if verbosity > 1:
        print(f"pcg_numba timing breakdown (fraction of {ttot:.3f}s):")
        print(f"  copyto:       {tcopy / ttot:.3f}")
        print(f"  aop:          {taop / ttot:.3f}")
        print(f"  alpha_update: {talpha / ttot:.3f}")
        print(f"  precond:      {tprecond / ttot:.3f}")
        print(f"  beta_update:  {tbeta / ttot:.3f}")
        print(f"  norm_diff:    {tnorm / ttot:.3f}")
        ttally = tcopy + taop + talpha + tprecond + tbeta + tnorm
        print(f"  accounted:    {ttally / ttot:.3f}")

    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}")
    elif stall_count >= 5:
        if verbosity:
            print(f"Stalled after {k} iterations with eps = {eps:.3e}")
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations")
    if not return_resid:
        return x
    else:
        return x, r


def pcg(
    aop,
    b,
    x0=None,
    precond=None,
    tol=1e-5,
    maxit=500,
    minit=100,
    verbosity=1,
    report_freq=10,
    backtrack=True,
    return_resid=False,
):
    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype)

    if precond is None:

        def precond(x):
            return x

    r = aop(x0) - b
    y = precond(r)
    if not np.any(y):
        print("Initial residual is zero")
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
    taop = 0.0
    talpha = 0.0
    tprecond = 0.0
    tbeta = 0.0
    tnorm = 0.0
    tii = time()
    while (eps > tol or k < minit) and k < maxit and stall_count < 5:
        ti = time()
        np.copyto(xp, x)
        np.copyto(rp, r)
        tcopy += time() - ti
        ti = time()
        aopp = aop(p)
        taop += time() - ti
        ti = time()
        rnorm = np.vdot(r, y)
        alpha = rnorm / np.vdot(p, aopp)
        ne.evaluate("xp + alpha*p", out=x, local_dict={"xp": xp, "alpha": alpha, "p": p}, casting="unsafe")
        ne.evaluate("rp + alpha*aopp", out=r, local_dict={"rp": rp, "alpha": alpha, "aopp": aopp}, casting="unsafe")
        talpha += time() - ti
        ti = time()
        y = precond(r)
        tprecond += time() - ti
        ti = time()
        rnorm_next = np.vdot(r, y)
        beta = rnorm_next / rnorm
        ne.evaluate("beta*p-y", out=p, local_dict={"beta": beta, "p": p, "y": y}, casting="unsafe")
        tbeta += time() - ti
        rnorm = rnorm_next
        k += 1
        epsp = eps
        ti = time()
        eps = norm_diff(x, xp)
        phi = rnorm / phi0
        tnorm += time() - ti

        if np.abs(epsp - eps) < 1e-3 * tol:
            stall_count += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}, phi = {phi:.3e}")
    ttot = time() - tii
    ttally = tcopy + taop + talpha + tprecond + tbeta + tnorm
    if verbosity > 1:
        print(f"pcg_numexpr timing breakdown (fraction of {ttot:.3f}s):")
        print(f"  copyto:       {tcopy / ttot:.3f}")
        print(f"  aop:          {taop / ttot:.3f}")
        print(f"  alpha_update: {talpha / ttot:.3f}")
        print(f"  precond:      {tprecond / ttot:.3f}")
        print(f"  beta_update:  {tbeta / ttot:.3f}")
        print(f"  norm_diff:    {tnorm / ttot:.3f}")
        ttally = tcopy + taop + talpha + tprecond + tbeta + tnorm
        print(f"  accounted:    {ttally / ttot:.3f}")

    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}")
    elif stall_count >= 5:
        if verbosity:
            print(f"Stalled after {k} iterations with eps = {eps:.3e}")
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations")
    if not return_resid:
        return x
    else:
        return x, r


def _pcg_psf_impl(
    psfhat,
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
    backtrack=True,
):
    """
    A specialised distributed version of pcg when the operator implements
    convolution with the psf (+ L2 regularisation by sigma**2)
    """
    nband, nx, ny = b.shape
    _, nx_psf, nyo2 = psfhat.shape
    model = np.zeros((nband, nx, ny), dtype=b.dtype, order="C")
    # PCG preconditioner
    if eta > 0:

        def precond(x):
            return x / eta
    else:
        precond = None
    from pfb_imaging.operators.hessian import hessian_psf_slice

    for k in range(nband):
        xpad = empty_noncritical((nx_psf, lastsize), dtype=b.dtype)
        xhat = empty_noncritical((nx_psf, nyo2), dtype=psfhat.dtype)
        xout = empty_noncritical((nx, ny), dtype=b.dtype)
        aop = partial(
            hessian_psf_slice,
            xpad=xpad,
            xhat=xhat,
            xout=xout,
            abspsf=np.abs(psfhat[k]),
            beam=beam[k],
            lastsize=lastsize,
            nthreads=nthreads,
            eta=eta[k],
        )

        model[k] = pcg(
            aop,
            b[k],
            x0[k],
            precond=precond,
            tol=tol,
            maxit=maxit,
            minit=minit,
            verbosity=verbosity,
            report_freq=report_freq,
            backtrack=backtrack,
        )

    return model


def _pcg_psf(psfhat, b, x0, beam, lastsize, nthreads, eta, cgopts):
    return _pcg_psf_impl(psfhat, b, x0, beam, lastsize, nthreads, eta, **cgopts)


def pcg_psf(psfhat, b, x0, beam, lastsize, nthreads, eta, cgopts, compute=True):
    if not isinstance(x0, da.Array):
        x0 = da.from_array(x0, chunks=(1, -1, -1), name="x-" + uuid4().hex)

    if not isinstance(psfhat, da.Array):
        psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name="psfhat-" + uuid4().hex)
    if not isinstance(b, da.Array):
        b = da.from_array(b, chunks=(1, -1, -1), name="psfhat-" + uuid4().hex)

    nband = b.shape[0]
    if isinstance(eta, float):
        eta = np.tile(eta, nband)
    else:
        eta = np.array(eta)
        assert eta.size == nband
    eta = da.from_array(eta, chunks=(1,), name="eta-" + uuid4().hex)

    if beam is None:
        bout = None
    else:
        bout = ("nb", "nx", "ny")
        if beam.ndim == 2:
            beam = beam[None]

        if not isinstance(beam, da.Array):
            beam = da.from_array(beam, chunks=(1, -1, -1), name="beam-" + uuid4().hex)
        if beam.shape[0] == 1:
            beam = da.tile(beam, (psfhat.shape[0], 1, 1))
        elif beam.shape[0] != psfhat.shape[0]:
            raise ValueError("Beam has incorrect shape")

    model = da.blockwise(
        _pcg_psf,
        ("nb", "nx", "ny"),
        psfhat,
        ("nb", "nx", "ny"),
        b,
        ("nb", "nx", "ny"),
        x0,
        ("nb", "nx", "ny"),
        beam,
        bout,
        lastsize,
        None,
        nthreads,
        None,
        eta,
        ("nb",),
        cgopts,
        None,
        align_arrays=False,
        dtype=b.dtype,
    )
    if compute:
        return model.compute()
    else:
        return model


@ray.remote
def pcg_dds(
    ds_name,
    eta,  # regularisation for Hessian approximation
    mask=1.0,
    use_psf=True,
    residual_name="RESIDUAL",
    model_name="MODEL",
    do_wgridding=True,
    epsilon=5e-4,
    double_accum=True,
    nthreads=1,
    zero_model_outside_mask=False,
    tol=1e-5,
    maxit=500,
    verbosity=1,
    report_freq=10,
):
    """
    pcg for fluxtractor
    """
    # avoid circular import
    from pfb_imaging.operators.hessian import hessian_slice

    # expects a list
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    drop_vars = ["PSF", "PSFHAT"]
    ds = xds_from_list(ds_name, nthreads=nthreads, drop_vars=drop_vars)[0]
    beam = mask * ds.BEAM.values
    if zero_model_outside_mask:
        if model_name not in ds:
            raise RuntimeError(f"Asked to zero model outside mask but {model_name} not in dds")
        model = getattr(ds, model_name).values
        model = np.where(mask > 0, model, 0.0)
        print("Zeroing model outside mask")
        resid = ds.DIRTY.values - hessian_slice(
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
            nthreads=nthreads,
        )
        j = resid * beam
    else:
        if model_name in ds:
            model = getattr(ds, model_name).values
        else:
            model = np.zeros(mask.shape, dtype=float)

        if residual_name in ds:
            j = getattr(ds, residual_name).values * beam
            ds = ds.drop_vars(residual_name)
        else:
            j = ds.DIRTY.values * beam

    nx, ny = j.shape
    wsum = ds.wsum
    j /= wsum
    precond = None
    if "UPDATE" in ds:
        x0 = ds.UPDATE.values * mask
    else:
        x0 = np.zeros_like(j)

    hess = partial(
        hessian_slice,
        uvw=ds.UVW.values,
        weight=ds.WEIGHT.values,
        vis_mask=ds.MASK.values,
        freq=ds.FREQ.values,
        beam=beam,
        cell=ds.cell_rad,
        x0=ds.x0,
        y0=ds.y0,
        flip_u=ds.flip_u,
        flip_v=ds.flip_v,
        flip_w=ds.flip_w,
        do_wgridding=do_wgridding,
        epsilon=epsilon,
        double_accum=double_accum,
        nthreads=nthreads,
        eta=eta,
        wsum=wsum,
    )

    x = pcg(
        hess,
        j,
        x0=x0.copy(),
        precond=precond,
        tol=tol,
        maxit=maxit,
        minit=1,
        verbosity=verbosity,
        report_freq=report_freq,
        backtrack=False,
        return_resid=False,
    )

    model += x

    resid = ds.DIRTY.values - hessian_slice(
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
        nthreads=nthreads,
    )

    ds = ds.assign(
        **{
            "MODEL_MOPPED": (("x", "y"), model),
            "RESIDUAL_MOPPED": (("x", "y"), resid),
            "UPDATE": (("x", "y"), x),
            "X0": (("x", "y"), x0),
        }
    )

    ds.to_zarr(ds_name[0], mode="a")

    return resid, int(ds.bandid)
