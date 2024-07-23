import numpy as np
import numexpr as ne
from functools import partial
from numba import njit
import dask.array as da
from pfb.operators.psf import psf_convolve_cube
from ducc0.misc import empty_noncritical
import pyscilog
log = pyscilog.get_logger('CLARK')

@njit(nogil=True, cache=True)  # parallel=True,
def subtract(A, psf, Ip, Iq, xhat, nxo2, nyo2):
    '''
    Subtract psf centered at location of xhat
    '''
    # loop over active indices
    nband = xhat.size
    # for b in numba.prange(nband):
    for b in range(nband):
        for i in range(Ip.size):
            pp = nxo2 - Ip[i]
            qq = nyo2 - Iq[i]
            A[b, i] -= xhat[b] * psf[b, pp, qq]
            # print(b, pp, qq, psf[b, pp, qq])
    return A


@njit(nogil=True, cache=True)  # parallel=True,
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

    # TODO - this does not work unless the PSF is twice the size of the image
    """
    nband, nx_psf, ny_psf = psf.shape
    nxo2 = nx_psf//2
    nyo2 = ny_psf//2
    # _, nx, ny = model.shape
    Asearch = np.sum(A, axis=0)**2
    pq = Asearch.argmax()
    p = Ip[pq]
    q = Iq[pq]
    Amax = np.sqrt(Asearch[pq])
    fsel = wsums > 0
    if fsel.sum() == 0:
        raise ValueError("wsums are all zero")
    k = 0
    while Amax > th and k < maxit:
        xhat = A[:, pq]
        model[fsel, p, q] += gamma * xhat[fsel]/wsums[fsel]
        Idelp = p - Ip
        Idelq = q - Iq
        # # find where PSF overlaps with image
        # left_x =  p - Ip >= 0
        # right_x = p + Ip < nx
        # left_y =  q - Iq >= 0
        # right_y = q + Iq < ny
        # mask = (left_x & right_x) & (left_y & right_y)

        # import ipdb; ipdb.set_trace()
        mask = (np.abs(Idelp) <= nxo2) & (np.abs(Idelq) <= nyo2)
        # Ipp, Iqq = psf[:, nxo2 - Ip[mask], nyo2 - Iq[mask]]
        A = subtract(A[:, mask], psf,
                     Idelp[mask], Idelq[mask],
                     xhat, nxo2, nyo2)

        Asearch = np.sum(A, axis=0)**2
        pq = Asearch.argmax()
        p = Ip[pq]
        q = Iq[pq]
        Amax = np.sqrt(Asearch[pq])
        k += 1
    return model

def clark(ID,
          PSF,
          PSFHAT,
          wsums,
          threshold=0,
          gamma=0.05,
          pf=0.05,
          maxit=50,
          subpf=0.5,
          submaxit=1000,
          report_freq=1,
          verbosity=1,
          psfopts=None,
          sigmathreshold=2,
          nthreads=1):
    nband, nx, ny = ID.shape
    _, nx_psf, ny_psf = PSF.shape
    # we assume that the dirty image and PSF have been normalised by wsum
    # and that we get units of Jy/beam when we take the sum over the frequency
    # axis i.e. the MFS image is in units of Jy/beam
    wsum = wsums.sum()
    assert np.allclose(wsum, 1)
    # ID /= wsum
    # PSF /= wsum
    # PSFHAT /= wsum
    # wsums = wsums/wsum
    model = np.zeros((nband, nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    # pre-allocate arrays for doing FFT's
    xout = empty_noncritical(ID.shape, dtype=ID.dtype)
    xpad = empty_noncritical(PSF.shape, dtype=ID.dtype)
    xhat = empty_noncritical(PSFHAT.shape, dtype=PSFHAT.dtype)
    # square avoids abs of full array
    IRsearch = np.sum(IR, axis=0)**2
    pq = IRsearch.argmax()
    p = pq//ny
    q = pq - p*ny
    IRmax = np.sqrt(IRsearch[p, q])
    tol = np.maximum(pf * IRmax, threshold)
    k = 0
    stall_count = 0
    while IRmax > tol and k < maxit and stall_count < 5:
        # identify active set
        subth = subpf * IRmax
        Ip, Iq = np.where(IRsearch > subth**2)
        # run substep in active set
        model = subminor(IR[:, Ip, Iq], PSF, Ip, Iq, model, wsums,
                         gamma=gamma,
                         th=subth,
                         maxit=submaxit)
        # import ipdb; ipdb.set_trace()
        # subtract from full image (as in major cycle)
        psf_convolve_cube(
                        xpad,
                        xhat,
                        xout,
                        PSFHAT,
                        ny_psf,
                        model,
                        nthreads=nthreads)
        IR = ID - xout
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
            print(f"At iteration {k} max resid = {IRmax}",
                  file=log)

    IRmfs = np.sum(IR, axis=0)
    rms = np.std(IRmfs[~np.any(model, axis=0)])

    if k >= maxit:
        if verbosity:
            print(f"Max iters reached. "
                  f"Max resid = {IRmax:.3e}, rms = {rms:.3e}", file=log)
        return model, 1
    elif stall_count >= 5:
        if verbosity:
            print(f"Stalled. "
                  f"Max resid = {IRmax:.3e}, rms = {rms:.3e}", file=log)
        return model, 1
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations. "
                  f"Max resid = {IRmax:.3e}, rms = {rms:.3e}", file=log)
        return model, 0
