'''
A very aggressive variant of the clean algorithm which:

    - Performs modified Clark clean on the MFS images to generate a mask
    - Uses PCG algorithm to extract flux in the cube
    - Computes the residual cube as dirty - PSF * model
'''

import numpy as np
import numexpr as ne
from functools import partial
from pfb.opt.pcg import pcg
import numba
import pyscilog
log = pyscilog.get_logger('AGROCLEAN')

# @numba.jit(nopython=True, nogil=True, cache=True, inline='always')
# def subtract(A, psf, Ip, Iq, xhat, nxo2, nyo2):
#     '''
#     Subtract psf centered at location of xhat
#     '''
#     # loop over active indices
#     for i in range(Ip.size):
#         pp = nxo2 - Ip[i]
#         qq = nyo2 - Iq[i]
#         A[i] -= xhat * psf[pp, qq]
#     return A

# @numba.jit(parallel=True, nopython=True, nogil=True, cache=True)
# def subminor(A, psf, Ip, Iq, model, gamma=0.1, th=0.0, maxit=500):
#     """
#     Run subminor loop in active set

#     A       - active set (all pixels above subminor threshold)
#     psf     - psf image
#     Ip      - x indices of active set
#     Iq      - y indices of active set
#     model   - current model image
#     gamma   - loop gain
#     th      - threshold to clean down to
#     maxit   - maximum number of iterations

#     Pixels in A map to pixels in the image via

#     p = Ip[pq]
#     q = Iq[pq]

#     where pq is the location of the maximum in A.

#     MFS version of algorithm in clark minor cycle
#     """
#     nx_psf, ny_psf = psf.shape
#     nxo2 = nx_psf//2
#     nyo2 = ny_psf//2
#     Asearch = np.abs(A)
#     pq = Asearch.argmax()
#     p = Ip[pq]
#     q = Iq[pq]
#     Amax = Asearch[pq]
#     k = 0
#     while Amax > th and k < maxit:
#         xhat = A[pq]
#         model[p, q] += gamma * xhat
#         Idelp = p - Ip
#         Idelq = q - Iq
#         mask = (np.abs(Idelp) <= nxo2) & (np.abs(Idelq) <= nyo2)
#         A = subtract(A[mask], psf, Idelp[mask], Idelq[mask],
#                      xhat, nxo2, nyo2)
#         Asearch = np.abs(A)
#         pq = Asearch.argmax()
#         p = Ip[pq]
#         q = Iq[pq]
#         Amax = Asearch[pq]
#         k += 1
#     return model


@numba.jit(parallel=True, nopython=True, nogil=True, cache=True, inline='always')
def subtract(A, psf, Ip, Iq, xhat, nxo2, nyo2):
    '''
    Subtract psf centered at location of xhat
    '''
    # loop over active indices
    nband = xhat.size
    for b in numba.prange(nband):
        for i in range(Ip.size):
            pp = nxo2 - Ip[i]
            qq = nyo2 - Iq[i]
            A[b, i] -= xhat[b] * psf[b, pp, qq]
    return A

@numba.jit(parallel=True, nopython=True, nogil=True, cache=True)
def subminor(A, psf, Ip, Iq, model, wsums, gamma=0.05, th=0.0, maxit=10000):
    """
    Run subminor loop in active set

    A       - active set (all pixels above subminor threshold)
    psf     - psf image
    Ip      - x indices of active set
    Iq      - y indices of active set
    model   - current model image
    gamma   - loop gain
    th      - threshold to clean down to
    maxit   - maximum number of iterations

    Pixels in A map to pixels in the image via

    p = Ip[pq]
    q = Iq[pq]

    where pq is the location of the maximum in A.
    """
    nband, nx_psf, ny_psf = psf.shape
    nxo2 = nx_psf//2
    nyo2 = ny_psf//2
    Asearch = np.sum(A, axis=0)**2
    pq = Asearch.argmax()
    p = Ip[pq]
    q = Iq[pq]
    Amax = np.sqrt(Asearch[pq])
    fsel = wsums > 0
    k = 0
    while Amax > th and k < maxit:
        xhat = A[:, pq]
        model[fsel, p, q] += gamma * xhat[fsel]/wsums[fsel]
        Idelp = p - Ip
        Idelq = q - Iq
        mask = (np.abs(Idelp) <= nxo2) & (np.abs(Idelq) <= nyo2)
        A = subtract(A[:, mask], psf, Idelp[mask], Idelq[mask],
                     xhat, nxo2, nyo2)
        Asearch = np.sum(A, axis=0)**2
        pq = Asearch.argmax()
        p = Ip[pq]
        q = Iq[pq]
        Amax = np.sqrt(Asearch[pq])
        k += 1
    return model


def agroclean(ID,
              PSF,
              psfo,
              gamma=0.1,
              pf=0.15,
              maxit=25,
              subpf=0.75,
              submaxit=500,
              report_freq=1,
              verbosity=1):
    nband, nx, ny = ID.shape
    _, nx_psf, ny_psf = PSF.shape
    wsums = np.amax(PSF, axis=(1,2))
    wsum = wsums.sum()
    # normalise such that sum(ID, axis=0) is in Jy/beam
    ID /= wsum
    PSF /= wsum
    # PSFMFS = np.sum(PSF, axis=0)
    wsums = np.amax(PSF, axis=(1,2))
    nx0 = nx_psf//2
    ny0 = ny_psf//2
    model = np.zeros((nband, nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    IRsearch = np.sum(IR, axis=0)**2
    pq = IRsearch.argmax()
    p = pq//ny
    q = pq - p*ny
    IRmax = np.sqrt(IRsearch[p, q])
    tol = pf * IRmax
    k = 0
    stall_count = 0
    while IRmax > tol and k < maxit and stall_count < 5:
        # identify active set
        subth = subpf * IRmax
        Ip, Iq = np.where(IRsearch > subth**2)
        # run substep in active set
        mmfs = subminor(IR[:, Ip, Iq], PSF, Ip, Iq,
                        model, wsums,
                        gamma=gamma,
                        th=subth,
                        maxit=submaxit)
        mask = np.any(mmfs, axis=0)[None, :, :]
        hess = lambda x: mask * psfo(mask*x) + 1e-6*x
        x = pcg(hess, mask * IR, mmfs - model,
                tol=1e-2, maxit=50, minit=1)
        model += x
        IR = ID - psfo(model)
        IRsearch = np.sum(IR, axis=0)**2
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmaxp = IRmax
        IRmax = np.sqrt(IRsearch[p, q])
        k += 1

        if np.abs(IRmaxp - IRmax) / np.abs(IRmaxp) < 1e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            print("At iteration %i max residual = %f" % (k, IRmax), file=log)

    if k >= maxit:
        if verbosity:
            print("Maximum iterations reached. Max of residual = %f." %
                  (IRmax), file=log)
    elif stall_count >= 5:
        if verbosity:
            print("Stalled. Max of residual = %f." %
                  (IRmax), file=log)
    else:
        if verbosity:
            print("Success, converged after %i iterations" % k, file=log)
    return model
