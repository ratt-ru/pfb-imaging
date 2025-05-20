import numpy as np
import numexpr as ne
from functools import partial
from numba import njit, prange, objmode, set_num_threads
import dask.array as da
from pfb.operators.psf import psf_convolve_cube, psf_convolve_fscube
from ducc0.misc import empty_noncritical
import pyscilog
log = pyscilog.get_logger('CLARK')
from time import time


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


@njit(nogil=True, cache=True, parallel=True, error_model='numpy')
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
    ncomp = Ip.size
    # _, nx, ny = model.shape
    # MFS image
    Amfs = np.sum(A, axis=0)
    Aabs = np.abs(Amfs)
    Asearch = np.sum(Aabs, axis=0)
    pq = Asearch.argmax()
    p = Ip[pq]
    q = Iq[pq]
    Amax = Aabs[:, pq]
    if (wsums == 0).any():
        raise ValueError("zero-wsums not supported")
    valid_mask = np.zeros(Ip.size, dtype=np.bool_)
    k = 0
    while (Amax > th).any() and k < maxit:
        pp = nxo2 - (Ip - p)
        qq = nyo2 - (Iq - q)
        valid_mask[:] = (pp >= 0) & (pp < nx_psf) & (qq >= 0) & (qq < ny_psf)
        valid_indices = np.nonzero(valid_mask)[0]
        ppi = pp[valid_mask]
        qqi = qq[valid_mask]
        for b in prange(nband):
            Ab = A[b]
            psfb = psf[b]
            wb = wsums[b]
            modelb = model[b]
            for c in range(ncorr):
                Abc = Ab[c]
                gx = gamma * Abc[pq]/wb[c]
                modelb[c, p, q] += gx      
                psfbc = psfb[c]
                for i, pi, qi in zip(valid_indices, ppi, qqi):
                    Abc[i] -= gx * psfbc[pi, qi]

        Amfs.fill(0.0)
        for b in range(nband):
            for c in range(ncorr):
                for i in prange(ncomp):
                    Amfs[c, i] += A[b, c, i]
            
        Asearch.fill(0.0)
        for c in range(ncorr):
            for i in prange(ncomp):
                Aabs[c, i] = abs(Amfs[c, i])
                Asearch[i] += Aabs[c, i]
        pq = Asearch.argmax()
        p = Ip[pq]
        q = Iq[pq]
        for c in range(ncorr):
            Amax[c] = Aabs[c, pq]
        k += 1
    print(f"fssubminor: {k} iterations")
    return model


@njit(nogil=True, cache=True, parallel=True, error_model='numpy')
def substep(A, psf, model, wsums, gamma, nband, ncorr, pq, p, q, valid_indices):
    for b in prange(nband):
        Ab = A[b]
        psfb = psf[b]
        wb = wsums[b]
        modelb = model[b]
        for c in range(ncorr):
            Abc = Ab[c]
            gx = gamma * Abc[pq]/wb[c]
            modelb[c, p, q] += gx      
            psfbc = psfb[c]
            for i, k in enumerate(valid_indices):
                Abc[k] -= gx * psfbc[i]


def fssubminornp(A, psf, Ip, Iq, model, wsums, gamma=0.05, th=0.0, maxit=10000):
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
    Amfs = np.sum(A, axis=0)
    Aabs = np.abs(Amfs)
    Asearch = np.sum(Aabs, axis=0)
    pq = Asearch.argmax()
    p = Ip[pq]
    q = Iq[pq]
    Amax = Aabs[:, pq]
    if (wsums == 0).any():
        raise ValueError("zero-wsums not supported")
    valid_mask = np.zeros(Ip.size, dtype=np.bool_)
    k = 0
    while (Amax > th).any() and k < maxit:
        pp = nxo2 - (Ip - p)
        qq = nyo2 - (Iq - q)
        valid_mask[:] = (pp >= 0) & (pp < nx_psf) & (qq >= 0) & (qq < ny_psf)
        valid_indices = np.nonzero(valid_mask)[0]
        ppi = pp[valid_mask]
        qqi = qq[valid_mask]
        
        psfpq = np.copy(psf[:,:,ppi,qqi], order='C')
        substep(A, psfpq, model, wsums, gamma, nband, ncorr, pq, p, q, valid_indices)
        
        ne.evaluate('sum(A, axis=0)', out=Amfs)
        ne.evaluate('abs(Amfs)', out=Aabs)
        ne.evaluate('sum(Aabs, axis=0)', out=Asearch)
        pq = Asearch.argmax()
        p = Ip[pq]
        q = Iq[pq]
        Amax[:] = Aabs[:, pq]
        k += 1
    print(f"fssubminornp: {k} iterations")
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
    # set_num_threads(nband)
    # MFS image
    IRmfs = np.sum(IR, axis=0)
    IRabs = np.abs(IRmfs)
    IRsearch = np.sum(IRabs, axis=0) * mask
    pq = IRsearch.argmax()
    p = pq//ny
    q = pq - p*ny
    IRmax = IRabs[: ,p, q]
    tol = np.maximum(pf * IRmax, threshold)
    k = 0
    stall_count = 0
    while (IRmax > tol).any() and k < maxit and stall_count < 5:
        # identify active set
        subth = subpf * IRmax
        tmask = (IRabs > subth[:, None, None]).any(axis=0)
        Ip, Iq = np.where(tmask)
        # run substep in active set
        ti = time()
        model = fssubminor(IR[:, :, Ip, Iq].copy(),
                           PSF, Ip, Iq, model, wsums,
                           gamma=gamma,
                           th=subth,
                           maxit=submaxit)
        print(f'ncomps = {Ip.size}, subminor time = ', time() - ti)
        # subtract from full image (as in major cycle)
        ti = time()
        psf_convolve_fscube(
                        xpad,
                        xhat,
                        xout,
                        PSFHAT,
                        ny_psf,
                        model,
                        nthreads=nthreads)
        print('convolve time = ', time() - ti)
        IR = ID - xout
        IRmfs[...] = np.sum(IR, axis=0)
        IRabs[...] = np.abs(IRmfs)
        IRsearch[...] = np.sum(IRabs, axis=0) * mask
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmaxp = IRmax
        IRmax[...] = IRabs[:, p, q]
        k += 1

        if np.abs(IRmaxp - IRmax).max() / np.abs(IRmaxp).max() < 1e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity:
            ncomp = model.any(axis=(0,1)).sum()
            print(f"Iteration {k} ncomp = {ncomp}", file=log)
            print(f"Max resids = {IRmax}", file=log)

    if k >= maxit:
        print(f"Max iters reached. ", file=log)
        status = 1
    elif stall_count >= 5:
        print(f"Stalled. ", file=log)
        status = 1
    else:
        print(f"Success, converged after {k} iterations. ", file=log)
        status = 0
    
    return model, status