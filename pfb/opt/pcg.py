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

from pfb.operators.hessian import _hessian_psf_impl as hessian_psf
def _pcg_psf_impl(psfhat,
                  b,
                  x0,
                  beam,
                  hessopts,
                  waveopts,
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
    nband, nbasis, nmax = b.shape
    model = np.zeros((nband, nbasis, nmax), dtype=b.dtype)
    sigmainvsq = hessopts['sigmainv']**2
    # PCG preconditioner
    if sigmainvsq > 0:
        def M(x): return x / sigmainvsq
    else:
        M = None

    for k in range(nband):
        A = partial(hessian_psf,
                    psfhat=psfhat[k],
                    beam=beam[k],
                    **hessopts,
                    **waveopts)
        model[k] = pcg(A, b[k], x0[k],
                       M=M, tol=tol, maxit=maxit, minit=minit,
                       verbosity=verbosity, report_freq=report_freq,
                       backtrack=backtrack)

    return model

# def _pcg_psf(psfhat,
#              b,
#              x0,
#              beam,
#              hessopts,
#              waveopts,
#              cgopts):
#     return _pcg_psf_impl(psfhat[0][0],
#                          b,
#                          x0,
#                          beam[0][0],
#                          hessopts,
#                          waveopts,
#                          **cgopts)

# def pcg_psf(psfhat,
#             b,
#             x0,
#             beam,
#             hessopts,
#             waveopts,
#             cgopts):

#     # print(hessopts)

#     # quit()
#     model = da.blockwise(_pcg_psf, ('nband', 'nbasis', 'nmax'),
#                          psfhat, ('nband', 'nx_psf', 'ny_psf'),
#                          b, ('nband', 'nbasis', 'nmax'),
#                          x0, ('nband', 'nbasis', 'nmax'),
#                          beam, ('nband', 'nx', 'ny'),
#                          hessopts, None,
#                          waveopts, None,
#                          cgopts, None,
#                          dtype=b.dtype)
#     return model


# from pfb.operators.hessian import _hessian_reg_wgt as hessian_wgt
# def _pcg_wgt_impl(uvw,
#                   weight,
#                   b,
#                   x0,
#                   beam,
#                   freq,
#                   freq_bin_idx,
#                   freq_bin_counts,
#                   hessopts,
#                   waveopts,
#                   tol=1e-5,
#                   maxit=500,
#                   minit=100,
#                   verbosity=1,
#                   report_freq=10,
#                   backtrack=True):
#     '''
#     A specialised distributed version of pcg when the operator implements
#     the diagonalised hessian (+ L2 regularisation by sigma**2)
#     '''
#     nband, nbasis, nmax = b.shape
#     model = np.zeros((nband, nbasis, nmax), dtype=b.dtype)
#     sigmainvsq = hessopts['sigmainv']**2
#     # PCG preconditioner
#     if sigmainvsq > 0:
#         def M(x): return x / sigmainvsq
#     else:
#         M = None

#     freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()
#     for k in range(nband):
#         indl = freq_bin_idx2[k]
#         indu = freq_bin_idx2[k] + freq_bin_counts[k]
#         A = partial(hessian_wgt,
#                     beam=beam[k],
#                     uvw=uvw,
#                     weight=weight[:, indl:indu],
#                     freq=freq[indl:indu],
#                     **hessopts,
#                     **waveopts)


#         model[k] = pcg(A,
#                        b[k],
#                        x0[k],
#                        M=M,
#                        tol=tol,
#                        maxit=maxit,
#                        minit=minit,
#                        verbosity=verbosity,
#                        report_freq=report_freq,
#                        backtrack=backtrack)

#     return model

# def _pcg_wgt(uvw,
#             weight,
#             b,
#             x0,
#             beam,
#             freq,
#             freq_bin_idx,
#             freq_bin_counts,
#             hessopts,
#             waveopts,
#             cgopts):
#     return _pcg_wgt_impl(uvw[0][0],
#                          weight[0],
#                          b,
#                          x0,
#                          beam[0][0],
#                          freq,
#                          freq_bin_idx,
#                          freq_bin_counts,
#                          hessopts,
#                          waveopts,
#                          **cgopts)

# def pcg_wgt(uvw,
#             weight,
#             b,
#             x0,
#             beam,
#             freq,
#             freq_bin_idx,
#             freq_bin_counts,
#             hessopts,
#             waveopts,
#             cgopts):


#     return da.blockwise(_pcg_wgt, ('nchan', 'nbasis', 'nmax'),
#                         uvw, ('nrow', 'three'),
#                         weight, ('nrow', 'nchan'),
#                         b, ('nchan', 'nbasis', 'nmax'),
#                         x0, ('nchan', 'nbasis', 'nmax'),
#                         beam, ('nchan', 'nx', 'ny'),
#                         freq, ('nchan',),
#                         freq_bin_idx, ('nchan',),
#                         freq_bin_counts, ('nchan',),
#                         hessopts, None,
#                         waveopts, None,
#                         cgopts, None,
#                         align_arrays=False,
#                         adjust_chunks={'nchan': b.chunks[0]},
#                         dtype=b.dtype)
