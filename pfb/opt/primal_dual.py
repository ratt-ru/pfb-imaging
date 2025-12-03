import numpy as np
import numexpr as ne
from pfb.utils.dist import l1reweight_func
from pfb.utils.misc import norm_diff
from numba import njit, prange
from uuid import uuid4
from pfb.utils import logging as pfb_logging
from time import time
log = pfb_logging.get_logger('PD')


def primal_dual(
        x,  # initial guess for primal variable
        v,  # initial guess for dual variable
        lam,  # regulariser strength
        psi,  # linear operator in dual domain
        psiH,  # adjoint of psi
        L,  # spectral norm of Hessian
        prox,  # prox of regulariser
        grad,  # gradient of smooth term
        nu=1.0,  # spectral norm of psi
        sigma=None,  # step size of dual update
        mask=None,  # regions where mask is False will be masked
        tol=1e-5,
        maxit=1000,
        minit=10,
        positivity=1,
        report_freq=10,
        gamma=1.0,
        verbosity=1):
    # initialise
    xp = x.copy()
    vp = v.copy()
    vtilde = np.zeros_like(v)
    vout = np.zeros_like(v)

    # this seems to give a good trade-off between
    # primal and dual problems
    if sigma is None:
        sigma = L / (2.0 * gamma) / nu

    # stepsize control
    tau = 0.9 / (L / (2.0 * gamma) + sigma * nu**2)

    # start iterations
    eps = 1.0
    k = 0
    while (eps > tol or k < minit) and k < maxit:
        # tmp prox variable
        vtilde = v + sigma * psiH(xp)

        # dual update
        v = vtilde - sigma * prox(vtilde / sigma, lam / sigma)

        # primal update
        x = xp - tau * (psi(2 * v - vp) + grad(xp))
        if positivity == 1:
            x[x < 0.0] = 0.0
        elif positivity == 2:
            msk = np.any(x<=0, axis=0)
            x[:, msk] = 0.0

        # convergence check
        eps = np.linalg.norm(x - xp) / np.linalg.norm(x)

        # copy contents to avoid allocating new memory
        xp[...] = x[...]
        vp[...] = v[...]

        if np.isnan(eps) or np.isinf(eps):
            import pdb; pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            # res = xbar-x
            # phi = np.vdot(res, A(res))
            log.info(f"At iteration {k} eps = {eps:.3e}")
        k += 1

    if k == maxit:
        if verbosity:
            log.info(f"Max iters reached. eps = {eps:.3e}")
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations")

    return x, v


from pfb.prox.prox_21m import dual_update_numba
def primal_dual_optimised(
        x,  # initial guess for primal variable
        v,  # initial guess for dual variable
        lam,  # regulariser strength
        psiH,  # linear operator in dual domain
        psi,  # adjoint of psi
        L,  # spectral norm of Hessian
        prox,  # prox of regulariser
        l1weight,
        reweighter,
        grad,  # gradient of smooth term
        nu=1.0,  # spectral norm of psi
        sigma=None,  # step size of dual update
        mask=None,  # regions where mask is False will be masked
        tol=1e-5,
        maxit=1000,
        positivity=1,
        report_freq=10,
        gamma=1.0,
        verbosity=1,
        maxreweight=20):  # max successive reweights before convergence

    # initialise
    xp = x.copy()
    vp = v.copy()
    xout = np.zeros_like(x)

    # this seems to give a good trade-off between
    # primal and dual problems
    if sigma is None:
        sigma = L / (2.0 * gamma) / nu

    # stepsize control
    tau = 0.98 / (L / (2.0 * gamma) + sigma * nu**2)

    # start iterations
    eps = 1.0
    numreweight = 0
    last_reweight_iter = 0
    # for timing
    tpsi = 0.0
    tupdate = 0.0
    teval1 = 0.0
    tpsiH = 0.0
    tgrad = 0.0
    teval2 = 0.0
    tpos = 0.0
    tnorm = 0.0
    tcopy = 0.0
    tii = time()
    for k in range(maxit):
        ti = time()
        psi(xp, v)
        tpsi += time() - ti
        ti = time()
        dual_update_numba(vp, v, lam,
                          sigma=sigma, weight=l1weight)
        tupdate += time() - ti
        ti = time()
        ne.evaluate('2.0 * v - vp', out=vp)  #, casting='same_kind')
        teval1 += time() - ti
        ti = time()
        psiH(vp, xout)
        tpsiH += time() - ti
        ti = time()
        xout += grad(xp)
        tgrad += time() - ti
        ti = time()
        ne.evaluate('xp - tau * xout', out=x)  #, casting='same_kind')
        teval2 += time() - ti

        ti = time()
        if positivity == 1:
            x[x < 0.0] = 0.0
        elif positivity == 2:
            msk = np.any(x<=0, axis=0)
            x[:, msk] = 0.0
        tpos += time() - ti
        # convergence check
        if x.any():
            ti = time()
            eps = norm_diff(x, xp)
            tnorm += time() - ti
        else:
            import pdb; pdb.set_trace()
            eps = 1.0
        if eps < tol:
            if reweighter is not None and numreweight < maxreweight:
                # ti = time()
                l1weight = reweighter(x)
                # log.info("reweight = ", time() - ti)
                if k-last_reweight_iter==1:
                    numreweight += 1
                else:
                    numreweight = 0
                last_reweight_iter = k

            else:
                if numreweight >= maxreweight:
                    log.info("Maximum reweighting steps reached")
                break

        # copy contents to avoid allocating new memory
        ti = time()
        np.copyto(xp, x)
        np.copyto(vp, v)
        tcopy += time() - ti
        if np.isnan(eps) or np.isinf(eps):
            import pdb; pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")

    ttot = time() - tii
    ttally = tpsi + tpsiH + tgrad + tupdate + teval1 + teval2 + tpos + tnorm
    if verbosity > 1:
        log.info('Time taken per step')
        log.info(f'psi = {tpsi/ttot}')
        log.info(f'psiH = {tpsiH/ttot}')
        log.info(f'grad = {tgrad/ttot}')
        log.info(f'update = {tupdate/ttot}')
        log.info(f'eval1 = {teval1/ttot}')
        log.info(f'eval2 = {teval2/ttot}')
        log.info(f'pos = {tpos/ttot}')
        log.info(f'norm = {tnorm/ttot}')
        log.info(f'tally = {ttally/ttot}')

    if k == maxit-1:
        if verbosity:
            log.info(f"Max iters reached. eps = {eps:.3e}")
    else:
        if verbosity:
            log.info(f"Success, converged after {k} iterations")

    return x, v

from distributed import as_completed

def primal_dual_dist(
            actors,
            lam,  # strength of regulariser
            L,    # spectral norm of Hessian
            l1weight,
            rmsfactor,
            rms_comps,
            alpha,
            nu=1.0,  # spectral norm of psi
            sigma=None,  # step size of dual update
            mask=None,  # regions where mask is False will be masked
            tol=1e-5,
            maxit=1000,
            positivity=1,
            report_freq=10,
            gamma=1.0,
            verbosity=1,
            maxreweight=50):
    '''
    Distributed primal dual algorithm.
    Distribution is over datasets in ddsf.

    Inputs:

    actors      - list of band actors
    lam         - strength of regulariser
    L           - spectral norm of hessian approximation
    l1weight    - array of L1 weights
    reweighter  - function to compute L1 reweights
    '''

    # this seems to give a good trade-off between
    # primal and dual problems
    if sigma is None:
        sigma = L / (2.0 * gamma) / nu

    # stepsize control
    tau = 0.9 / (L / (2.0 * gamma) + sigma * nu**2)

    # we need to do this upfront only at the outset
    futures = list(map(lambda a: a.init_pd_params(L, nu), actors))
    # we don't want to allocate at each iteration
    eps_num = [1.0]*len(futures)
    eps_den = [1.0]*len(futures)
    vtilde = np.zeros((len(actors), *l1weight.shape), dtype=l1weight.dtype)
    for fut in as_completed(futures):
        tmp, b = fut.result()
        vtilde[b] = tmp
    # vtilde = list(map(lambda f: f.result(), futures))
    numreweight = 0
    do_reweight = ~(l1weight == 1.0).all()  # reweighting active
    ratio = np.zeros(l1weight.shape, dtype=l1weight.dtype)
    for k in range(maxit):
        ti = time()
        get_ratio(vtilde, l1weight, sigma, lam, ratio)
        # get_ratio(np.array(vtilde), l1weight, sigma, lam, ratio)
        log.info('ratio - ', time() - ti)

        ti = time()
        # do on individual workers
        futures = list(map(lambda a: a.pd_update(ratio), actors))
        for fut in as_completed(futures):
            tmp, epsn, epsd, b = fut.result()
            vtilde[b] = tmp
            eps_num[b] = epsn
            eps_den[b] = epsd

        # results = list(map(lambda f: f.result(), futures))
        log.info('update - ', time() - ti)

        # vtilde = [r[0] for r in results]
        # eps_num = [r[1] for r in results]
        # eps_den = [r[2] for r in results]
        eps = np.sqrt(np.sum(eps_num)/np.sum(eps_den))

        if np.isnan(eps):
            log.error_and_raise("eps is nan",
                                ValueError)


        if not k % report_freq and verbosity > 1:
            log.info(f"At iteration {k} eps = {eps:.3e}")

        if eps < tol:
            if do_reweight and numreweight < maxreweight:
                l1weight = l1reweight_func(actors,
                                           rmsfactor,
                                           rms_comps=rms_comps,
                                           alpha=alpha)
                numreweight += 1
                log.info(f"Reweighting iter {numreweight}")
            else:
                if numreweight >= maxreweight:
                    log.info("Maximum reweighting steps reached")
                break

    if k >= maxit-1:
        log.info(f'Maximum iterations reached. eps={eps:.3e}')
    else:
        log.info(f'Success converged after {k} iterations')

    return


@njit(nogil=True, cache=True, parallel=True)
def get_ratio(vtilde, l1weight, sigma, lam, ratio):
    nband, nbasis, nymax, nxmax = vtilde.shape
    for b in range(nbasis):
        for i in prange(nymax):  # WTF without the prange it segfaults when parallel=True
            vtildebi = vtilde[:, b, i]
            weightbi = l1weight[b, i]
            for j in range(nxmax):
                vtildebij = vtildebi[:, j]
                weightbij = weightbi[j]
                absvbisum = np.abs(np.sum(vtildebij)/sigma)  # sum over band axis
                softvbisum = absvbisum - lam*weightbij/sigma
                if absvbisum > 0 and softvbisum > 0:
                    ratio[b, i, j] = softvbisum/absvbisum
                else:
                    ratio[b, i, j] = 0.0



primal_dual.__doc__ = r"""
    Algorithm to solve problems of the form

    argmin_x (xbar - x).T A (xbar - x)/2 + lam |PSI.H x|_21 s.t. x >= 0

    where x is the image cube and PSI an orthogonal basis for the spatial axes
    and the positivity constraint is optional.

    A        - Positive definite Hermitian operator
    xbar     - Data-like term
    x0       - Initial guess for primal variable
    v0       - Initial guess for dual variable
    lam      - Strength of l21 regulariser
    psi      - Orthogonal basis operator where
               psi.hdot() does decomposition and
               psi.dot() the reconstruction
               for all bases and channels
    weights  - Weights for l1 thresholding
    L        - Lipschitz constant of A
    nu       - Spectral norm of psi
    sigma    - l21 step size (set to L/2 by default)
    gamma    - step size during forward step

    Note that the primal variable (i.e. x) has shape (nband, nx, ny) where
    nband is the number of imaging bands and the dual variable (i.e. v) has
    shape (nbasis, nband, nmax) where nmax is the total number of coefficients
    for all bases. It is assumed that all bases are decomposed into the same
    number of coefficients. We use zero padding to deal with the fact that the
    basess do not necessarily have the same number of coefficients.
    """
