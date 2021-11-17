import numpy as np
import numexpr as ne
from functools import partial
import numba
import pyscilog
log = pyscilog.get_logger('CLARK')

@numba.jit(nopython=True, nogil=True, cache=True, inline='always')
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

@numba.jit(nopython=True, nogil=True, cache=True)
def subminor(A, psf, Ip, Iq, model, gamma=0.05, th=0.0, maxit=10000):
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
    k = 0
    while Amax > th and k < maxit:
        xhat = A[:, pq]
        model[:, p, q] += gamma * xhat
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


def clark(ID,
          PSF,
          psfo=None,
          gamma=0.05,
          pf=0.05,
          maxit=10,
          subpf=0.75,
          submaxit=1000,
          report_freq=1,
          verbosity=1,
          psfopts=None):
    nband, nx, ny = ID.shape
    # initialise the PSF operator if not passed in
    _, nx_psf, ny_psf = PSF.shape
    wsums = np.amax(PSF, axis=(1,2))
    if psfo is None:
        print("Setting up PSF operator", file=log)
        from ducc0.fft import r2c
        iFs = np.fft.ifftshift

        padding = psfopts['padding']
        unpad_x = psfopts['unpad_x']
        unpad_y = psfopts['unpad_y']
        lastsize = psfopts['lastsize']
        nthreads = psfopts['nthreads']
        psf_pad = iFs(PSF, axes=(1, 2))
        psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                     nthreads=nthreads, inorm=0)
        del psf_pad
        psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)

        from pfb.operators.psf import psf_convolve
        psfo = partial(psf_convolve, psfhat=psfhat, psfopts=psfopts)

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
        model = subminor(IR[:, Ip, Iq], PSF, Ip, Iq, model,
                         gamma=gamma,
                         th=subth,
                         maxit=submaxit)
        IR = ID - psfo(model).compute()
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
    return x
