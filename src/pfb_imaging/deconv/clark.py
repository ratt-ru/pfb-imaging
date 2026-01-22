import numpy as np
from ducc0.misc import empty_noncritical
from numba import njit, prange

from pfb_imaging.operators.psf import psf_convolve_cube, psf_convolve_fscube
from pfb_imaging.utils import logging as pfb_logging

log = pfb_logging.get_logger("CLARK")


@njit(nogil=True, cache=True, parallel=True)
def subminor(a_set, psf, p_index, q_index, model, wsums, gamma=0.05, th=0.0, maxit=10000):
    """
    Run subminor loop in active set

    a_set   - active set (all pixels above subminor threshold)
    psf     - psf image
    p_index      - x indices of active set
    q_index      - y indices of active set
    model   - current model image
    gamma   - loop gain
    th      - threshold to clean down to
    maxit   - maximum number of iterations

    Pixels in A map to pixels in the image via

    p = p_index[pq]
    q = q_index[pq]

    where pq is the location of the maximum in A.
    """
    nband, nx_psf, ny_psf = psf.shape
    nxo2 = nx_psf // 2
    nyo2 = ny_psf // 2
    # _, nx, ny = model.shape
    a_search = np.sum(a_set, axis=0) ** 2
    pq = a_search.argmax()
    p = p_index[pq]
    q = q_index[pq]
    a_max = np.sqrt(a_search[pq])
    fsel = wsums > 0
    if fsel.sum() == 0:
        raise ValueError("wsums are all zero")
    k = 0
    while a_max > th and k < maxit:
        xhat = a_set[:, pq]
        model[fsel, p, q] += gamma * xhat[fsel] / wsums[fsel]

        for b in prange(nband):
            for i in range(p_index.size):
                pp = nxo2 - (p_index[i] - p)
                qq = nyo2 - (q_index[i] - q)
                if (pp >= 0) & (pp < nx_psf) & (qq >= 0) & (qq < ny_psf):
                    a_set[b, i] -= gamma * xhat[b] * psf[b, pp, qq] / wsums[b]

        a_search = np.sum(a_set, axis=0) ** 2
        pq = a_search.argmax()
        p = p_index[pq]
        q = q_index[pq]
        a_max = np.sqrt(a_search[pq])
        k += 1
    return model


def clark(
    dirty,
    psf,
    psfhat,
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
    nthreads=1,
):
    nband, nx, ny = dirty.shape
    _, nx_psf, ny_psf = psf.shape
    # we assume that the dirty image and psf have been normalised by wsum
    # and that we get units of Jy/beam when we take the sum over the frequency
    # axis i.e. the MFS image is in units of Jy/beam
    wsum = wsums.sum()
    assert np.allclose(wsum, 1)
    model = np.zeros((nband, nx, ny), dtype=dirty.dtype)
    residual = dirty.copy()
    # pre-allocate arrays for doing FFT's
    xout = empty_noncritical(dirty.shape, dtype="f8")
    xpad = empty_noncritical(psf.shape, dtype="f8")
    xhat = empty_noncritical(psfhat.shape, dtype="c16")
    # square avoids abs of full array
    residual_search = np.sum(residual, axis=0) ** 2 * mask
    pq = residual_search.argmax()
    p = pq // ny
    q = pq - p * ny
    residual_max = np.sqrt(residual_search[p, q])
    tol = np.maximum(pf * residual_max, threshold)
    k = 0
    stall_count = 0
    while residual_max > tol and k < maxit and stall_count < 5:
        # identify active set
        subth = subpf * residual_max
        p_index, q_index = np.where(residual_search > subth**2)
        # run substep in active set
        model = subminor(
            residual[:, p_index, q_index], psf, p_index, q_index, model, wsums, gamma=gamma, th=subth, maxit=submaxit
        )

        # subtract from full image (as in major cycle)
        psf_convolve_cube(xpad, xhat, xout, psfhat, ny_psf, model, nthreads=nthreads)
        residual = dirty - xout
        residual_search = np.sum(residual, axis=0) ** 2 * mask
        pq = residual_search.argmax()
        p = pq // ny
        q = pq - p * ny
        residual_maxp = residual_max
        residual_max = np.sqrt(residual_search[p, q])
        k += 1

        if np.abs(residual_maxp - residual_max) / np.abs(residual_maxp) < 1e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} max resid = {residual_max}")

    residual_mfs = np.sum(residual, axis=0)
    rms = np.std(residual_mfs[~np.any(model, axis=0)])

    if k >= maxit:
        if verbosity:
            log.info(f"Max iters reached. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 1
    elif stall_count >= 5:
        if verbosity:
            log.info(f"Stalled. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 1
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 0


@njit(nogil=True, cache=True, parallel=True)
def fssubminor(a_set, psf, p_index, q_index, model, wsums, gamma=0.05, th=0.0, maxit=10000):
    """
    Full Stokes subminor loop in active set

    a_set       - active set (all pixels above subminor threshold)
    psf     - psf image
    p_index      - x indices of active set
    q_index      - y indices of active set
    model   - current model image
    gamma   - loop gain
    th      - threshold to clean down to
    maxit   - maximum number of iterations

    Pixels in a_set map to pixels in the image via

    p = p_index[pq]
    q = q_index[pq]

    where pq is the location of the maximum in a_set.
    """
    nband, ncorr, nx_psf, ny_psf = psf.shape
    nxo2 = nx_psf // 2
    nyo2 = ny_psf // 2
    # _, nx, ny = model.shape
    # MFS image
    a_mfs = np.sum(a_set, axis=0)
    # total pol image (always positive)
    a_search = np.sum(a_mfs**2, axis=0)
    pq = a_search.argmax()
    p = p_index[pq]
    q = q_index[pq]
    a_max = np.sqrt(a_search[pq])
    fsel = wsums > 0
    if fsel.sum() == 0:
        raise ValueError("wsums are all zero")
    k = 0
    while a_max > th and k < maxit:
        xhat = a_set[:, :, pq]
        model[:, :, p, q] += gamma * xhat[:, :] / wsums[:, :]

        for b in prange(nband):
            for c in range(ncorr):
                for i in range(p_index.size):
                    pp = nxo2 - (p_index[i] - p)
                    qq = nyo2 - (q_index[i] - q)
                    if (pp >= 0) & (pp < nx_psf) & (qq >= 0) & (qq < ny_psf):
                        a_set[b, c, i] -= gamma * xhat[b, c] * psf[b, c, pp, qq] / wsums[b, c]

        a_mfs = np.sum(a_set, axis=0)
        a_search = np.sum(a_mfs**2, axis=0)
        pq = a_search.argmax()
        p = p_index[pq]
        q = q_index[pq]
        a_max = np.sqrt(a_search[pq])
        k += 1
    return model


def fsclark(
    dirty,
    psf,
    psfhat,
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
    nthreads=1,
):
    nband, ncorr, nx, ny = dirty.shape
    _, _, nx_psf, ny_psf = psf.shape
    # we assume that the dirty image and psf have been normalised by wsum
    # and that we get units of Jy/beam when we take the sum over the frequency
    # axis i.e. the MFS image is in units of Jy/beam
    wsum = wsums.sum(axis=0)
    assert np.allclose(wsum, 1)
    model = np.zeros((nband, ncorr, nx, ny), dtype=dirty.dtype)
    residual_mfs = dirty.copy()
    # pre-allocate arrays for doing FFT's
    xout = empty_noncritical(dirty.shape, dtype="f8")
    xpad = empty_noncritical(psf.shape, dtype="f8")
    xhat = empty_noncritical(psfhat.shape, dtype="c16")
    # MFS image
    residual_mfs = np.sum(residual_mfs, axis=0)
    # total pol image (always positive)
    # Do we technically need a weighted sum here?
    residual_search = np.sum(residual_mfs**2, axis=0) * mask
    pq = residual_search.argmax()
    p = pq // ny
    q = pq - p * ny
    residual_max = np.sqrt(residual_search[p, q])
    tol = np.maximum(pf * residual_max, threshold)
    k = 0
    stall_count = 0
    while residual_max > tol and k < maxit and stall_count < 5:
        # identify active set
        subth = subpf * residual_max
        p_index, q_index = np.where(residual_search > subth**2)
        # run substep in active set
        model = fssubminor(
            residual_mfs[:, :, p_index, q_index],
            psf,
            p_index,
            q_index,
            model,
            wsums,
            gamma=gamma,
            th=subth,
            maxit=submaxit,
        )

        # subtract from full image (as in major cycle)
        psf_convolve_fscube(xpad, xhat, xout, psfhat, ny_psf, model, nthreads=nthreads)
        residual_mfs = dirty - xout
        residual_mfs = np.sum(residual_mfs, axis=0)
        residual_search = np.sum(residual_mfs**2, axis=0) * mask
        pq = residual_search.argmax()
        p = pq // ny
        q = pq - p * ny
        residual_maxp = residual_max
        residual_max = np.sqrt(residual_search[p, q])
        k += 1

        if np.abs(residual_maxp - residual_max) / np.abs(residual_maxp) < 1e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} max resid = {residual_max}")

    residual_mfs = np.sum(residual_mfs, axis=0)
    rms = np.std(residual_mfs, axis=(-2, -1)).max()

    if k >= maxit:
        if verbosity:
            log.info(f"Max iters reached. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 1
    elif stall_count >= 5:
        if verbosity:
            log.info(f"Stalled. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 1
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations. Max resid = {residual_max:.3e}, rms = {rms:.3e}")
        return model, 0
