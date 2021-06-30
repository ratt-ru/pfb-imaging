import numpy as np
from functools import partial
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

from pfb.operators.psf import _hessian_reg as hessian_psf
def _pcg_psf(psfhat,
             b,
             x0,
             sigmainv,
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
    sigmainvsq = sigmainv**2
    if sigmainv > 0:
        def M(x): return x / sigmainvsq
    else:
        M = None
    for k in range(nband):
        A = partial(hessian_psf,
                    psfhat=psfhat[k:k+1],
                    sigmainvsq=sigmainvsq,
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


from pfb.operators.hessian import _hessian_reg as hessian_wgt
def _pcg_wgt(uvw,
             weight,
             b,
             x0,
             beam,
             freq,
             freq_bin_idx,
             freq_bin_counts,
             cell,
             wstack,
             epsilon,
             double_accum,
             nthreads,
             sigmainv,
             wsum,
             tol=1e-5,
             maxit=500,
             minit=100,
             verbosity=1,
             report_freq=10,
             backtrack=True):
    '''
    A specialised distributed version of pcg when the operator implements
    the diagonalised hessian (+ L2 regularisation by sigma**2)
    '''
    nband, nx, ny = b.shape
    model = np.zeros((nband, nx, ny), dtype=b.dtype)
    sigmainvsq = sigmainv**2
    if sigmainv > 0:
        def M(x): return x / sigmainvsq
    else:
        M = None

    freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()
    for k in range(nband):
        indl = freq_bin_idx2[k]
        indu = freq_bin_idx2[k] + freq_bin_counts[k]
        A = partial(hessian_wgt,
                    beam=beam[k],
                    uvw=uvw,
                    weight=weight[:, indl:indu],
                    freq=freq[indl:indu],
                    cell=cell,
                    wstack=wstack,
                    epsilon=epsilon,
                    double_accum=double_accum,
                    nthreads=nthreads,
                    sigmainvsq=sigmainvsq,
                    wsum=wsum)
        model[k] = pcg(A,
                       beam[k] * b[k],
                       x0[k],
                       M=M,
                       tol=tol,
                       maxit=maxit,
                       minit=minit,
                       verbosity=verbosity,
                       report_freq=report_freq,
                       backtrack=backtrack)

    return model

def pcg_wgt_wrapper(uvw,
                    weight,
                    b,
                    x0,
                    beam,
                    freq,
                    freq_bin_idx,
                    freq_bin_counts,
                    cell,
                    wstack,
                    epsilon,
                    double_accum,
                    nthreads,
                    sigmainv,
                    wsum,
                    tol=1e-5,
                    maxit=500,
                    minit=100,
                    verbosity=0,
                    report_freq=10,
                    backtrack=True):
    return _pcg_wgt(uvw[0][0],
                    weight[0],
                    b,
                    x0,
                    beam,
                    freq,
                    freq_bin_idx,
                    freq_bin_counts,
                    cell,
                    wstack,
                    epsilon,
                    double_accum,
                    nthreads,
                    sigmainv,
                    wsum,
                    tol,
                    maxit,
                    minit,
                    verbosity,
                    report_freq,
                    backtrack)

def pcg_wgt(uvw,
            weight,
            b,
            x0,
            beam,
            freq,
            freq_bin_idx,
            freq_bin_counts,
            cell,
            wstack,
            epsilon,
            double_accum,
            nthreads,
            sigmainv,
            wsum,
            tol,
            maxit,
            minit,
            verbosity,
            report_freq,
            backtrack):


    return da.blockwise(pcg_wgt_wrapper, ('nchan', 'nx', 'ny'),
                        uvw, ('nrow', 'three'),
                        weight, ('nrow', 'nchan'),
                        b, ('nchan', 'nx', 'ny'),
                        x0, ('nchan', 'nx', 'ny'),
                        beam, ('nchan', 'nx', 'ny'),
                        freq, ('nchan',),
                        freq_bin_idx, ('nchan',),
                        freq_bin_counts, ('nchan',),
                        cell, None,
                        wstack, None,
                        epsilon, None,
                        double_accum, None,
                        nthreads, None,
                        sigmainv, None,
                        wsum, None,
                        tol, None,
                        maxit, None,
                        minit, None,
                        verbosity, None,
                        report_freq, None,
                        backtrack, None,
                        align_arrays=False,
                        adjust_chunks={'nchan': b.chunks[0]},
                        dtype=b.dtype)