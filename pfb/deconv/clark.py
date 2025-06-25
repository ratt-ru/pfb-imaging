import numpy as np
import numexpr as ne
from functools import partial
from numba import njit, prange
import dask.array as da
from pfb.operators.psf import psf_convolve_cube, psf_convolve_fscube
from ducc0.misc import empty_noncritical
import pyscilog
log = pyscilog.get_logger('CLARK')


@njit(nogil=True, cache=True, parallel=True)
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
        
        for b in prange(nband):
            for i in range(Ip.size):
                pp = nxo2 - (Ip[i] - p)
                qq = nyo2 - (Iq[i] - q)
                if (pp >= 0) & (pp < nx_psf) & (qq >= 0) & (qq < ny_psf):
                    A[b, i] -= gamma * xhat[b] * psf[b, pp, qq]/wsums[b]

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
          mask,
          threshold=0,
          gamma=0.05,
          pf=0.05,
          maxit=50,
          subpf=0.5,
          submaxit=1000,
          report_freq=1,
          verbosity=1,
          nthreads=1):
    nband, nx, ny = ID.shape
    _, nx_psf, ny_psf = PSF.shape
    # we assume that the dirty image and PSF have been normalised by wsum
    # and that we get units of Jy/beam when we take the sum over the frequency
    # axis i.e. the MFS image is in units of Jy/beam
    wsum = wsums.sum()
    assert np.allclose(wsum, 1)
    model = np.zeros((nband, nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    # pre-allocate arrays for doing FFT's
    xout = empty_noncritical(ID.shape, dtype='f8')
    xpad = empty_noncritical(PSF.shape, dtype='f8')
    xhat = empty_noncritical(PSFHAT.shape, dtype='c16')
    # square avoids abs of full array
    IRsearch = np.sum(IR, axis=0)**2 * mask
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
        IRsearch = np.sum(IR, axis=0)**2 * mask
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmaxp = IRmax
        IRmax = np.sqrt(IRsearch[p, q])
        k += 1

        if np.abs(IRmaxp - IRmax) / np.abs(IRmaxp) < 1e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} max resid = {IRmax}")

    IRmfs = np.sum(IR, axis=0)
    rms = np.std(IRmfs[~np.any(model, axis=0)])

    if k >= maxit:
        if verbosity:
            log.info(f"Max iters reached. "
                     f"Max resid = {IRmax:.3e}, rms = {rms:.3e}")
        return model, 1
    elif stall_count >= 5:
        if verbosity:
            log.info(f"Stalled. "
                     f"Max resid = {IRmax:.3e}, rms = {rms:.3e}")
        return model, 1
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. "
                     f"Max resid = {IRmax:.3e}, rms = {rms:.3e}")
        return model, 0


@njit(nogil=True, cache=True, parallel=True)
def fssubminor(A, psf, Ip, Iq, model, wsums, gamma=0.05, th=0.0, maxit=10000):
    """
    Full Stokes subminor loop in active set

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
    nband, ncorr, nx_psf, ny_psf = psf.shape
    nxo2 = nx_psf//2
    nyo2 = ny_psf//2
    # _, nx, ny = model.shape
    # MFS image
    Amfs = np.sum(A, axis=0)
    # total pol image (always positive)
    Asearch = np.sum(Amfs**2, axis=0)
    pq = Asearch.argmax()
    p = Ip[pq]
    q = Iq[pq]
    Amax = np.sqrt(Asearch[pq])
    fsel = wsums > 0
    if fsel.sum() == 0:
        raise ValueError("wsums are all zero")
    k = 0
    while Amax > th and k < maxit:
        xhat = A[:, :, pq]
        model[:, :, p, q] += gamma * xhat[:, :]/wsums[:, :]
        
        for b in prange(nband):
            for c in range(ncorr):
                for i in range(Ip.size):
                    pp = nxo2 - (Ip[i] - p)
                    qq = nyo2 - (Iq[i] - q)
                    if (pp >= 0) & (pp < nx_psf) & (qq >= 0) & (qq < ny_psf):
                        A[b, c, i] -= gamma * xhat[b, c] * psf[b, c, pp, qq]/wsums[b, c]

        Amfs = np.sum(A, axis=0)
        Asearch = np.sum(Amfs**2, axis=0)
        pq = Asearch.argmax()
        p = Ip[pq]
        q = Iq[pq]
        Amax = np.sqrt(Asearch[pq])
        k += 1
    return model


def fsclark(ID,
            PSF,
            PSFHAT,
            wsums,
            mask,
            threshold=0,
            gamma=0.05,
            pf=0.05,
            maxit=50,
            subpf=0.5,
            submaxit=1000,
            report_freq=1,
            verbosity=1,
            nthreads=1):
    nband, ncorr, nx, ny = ID.shape
    _, _, nx_psf, ny_psf = PSF.shape
    # we assume that the dirty image and PSF have been normalised by wsum
    # and that we get units of Jy/beam when we take the sum over the frequency
    # axis i.e. the MFS image is in units of Jy/beam
    wsum = wsums.sum(axis=0)
    assert np.allclose(wsum, 1)
    model = np.zeros((nband, ncorr, nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    # pre-allocate arrays for doing FFT's
    xout = empty_noncritical(ID.shape, dtype='f8')
    xpad = empty_noncritical(PSF.shape, dtype='f8')
    xhat = empty_noncritical(PSFHAT.shape, dtype='c16')
    # MFS image
    IRmfs = np.sum(IR, axis=0)
    # total pol image (always positive)
    # Do we technically need a weighted sum here?
    IRsearch = np.sum(IRmfs**2, axis=0) * mask
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
        model = fssubminor(IR[:, :, Ip, Iq], PSF, Ip, Iq, model, wsums,
                           gamma=gamma,
                           th=subth,
                           maxit=submaxit)

        # subtract from full image (as in major cycle)
        psf_convolve_fscube(
                        xpad,
                        xhat,
                        xout,
                        PSFHAT,
                        ny_psf,
                        model,
                        nthreads=nthreads)
        IR = ID - xout
        IRmfs = np.sum(IR, axis=0)
        IRsearch = np.sum(IRmfs**2, axis=0) * mask
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmaxp = IRmax
        IRmax = np.sqrt(IRsearch[p, q])
        k += 1

        if np.abs(IRmaxp - IRmax) / np.abs(IRmaxp) < 1e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} max resid = {IRmax}")

    IRmfs = np.sum(IR, axis=0)
    rms = np.std(IRmfs, axis=(-2,-1)).max()

    if k >= maxit:
        if verbosity:
            log.info(f"Max iters reached. "
                     f"Max resid = {IRmax:.3e}, rms = {rms:.3e}")
        return model, 1
    elif stall_count >= 5:
        if verbosity:
            log.info(f"Stalled. "
                     f"Max resid = {IRmax:.3e}, rms = {rms:.3e}")
        return model, 1
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. "
                     f"Max resid = {IRmax:.3e}, rms = {rms:.3e}")
        return model, 0