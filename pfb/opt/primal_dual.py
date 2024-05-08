import numpy as np
import numexpr as ne
import dask.array as da
from distributed import wait, get_client, as_completed
from operator import getitem
from ducc0.misc import make_noncritical
from pfb.utils.misc import norm_diff
from uuid import uuid4
import pyscilog
log = pyscilog.get_logger('PD')


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
            print(f"At iteration {k} eps = {eps:.3e}",  # and phi = {phi:.3e}",
                  file=log)
        k += 1

    if k == maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations", file=log)

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
        maxreweight=50):
    # TODO - we can't use make_noncritical because it sometimes returns an
    # array that is not explicitly c-contiguous. How does this impact
    # performance?
    # initialise
    xp = x.copy()
    # xp = make_noncritical(xp)
    vp = v.copy()
    xout = np.zeros_like(x)
    # xout = make_noncritical(xout)


    # this seems to give a good trade-off between
    # primal and dual problems
    if sigma is None:
        sigma = L / (2.0 * gamma) / nu

    # stepsize control
    tau = 0.9 / (L / (2.0 * gamma) + sigma * nu**2)

    # start iterations
    eps = 1.0
    numreweight = 0
    for k in range(maxit):
        psi(xp, v)
        dual_update_numba(vp, v, lam, sigma=sigma, weight=l1weight)
        ne.evaluate('2.0 * v - vp', out=vp)  #, casting='same_kind')
        psiH(vp, xout)
        xout += grad(xp)
        ne.evaluate('xp - tau * xout', out=x)  #, casting='same_kind')

        if positivity == 1:
            x[x < 0.0] = 0.0
        elif positivity == 2:
            msk = np.any(x<=0, axis=0)
            x[:, msk] = 0.0

        # convergence check
        if x.any():
            eps = norm_diff(x, xp)
        else:
            import pdb; pdb.set_trace()
            eps = 1.0
        if eps < tol:
            if reweighter is not None and numreweight < maxreweight:
                l1weight = reweighter(x)
                numreweight += 1
            else:
                if numreweight >= maxreweight:
                    print("Maximum reweighting steps reached", file=log)
                break

        # copy contents to avoid allocating new memory
        xp[...] = x[...]
        vp[...] = v[...]

        if np.isnan(eps) or np.isinf(eps):
            import pdb; pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}", file=log)

    if k == maxit-1:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations", file=log)

    return x, v


def vtilde_update(ds, sigma, psi):
    vtilde = ds.DUAL.values + sigma * psi.dot(ds.MODEL.values)
    model = ds.MODEL.values
    dual = ds.DUAL.values
    return vtilde, model, dual

def update(gradf, vtilde, ratio, x, v, sigma, lam, tau, gamma, psi, positivity):
    xp = x.copy()
    vp = v.copy()

    # dual
    v = vtilde - vtilde * ratio

    # primal
    grad = gradf(xp)
    x = xp - tau * (psi.hdot(2 * v - vp) + grad)
    if positivity:
        x[x < 0] = 0.0

    vtilde = v + sigma * psi.dot(x)

    eps = np.linalg.norm(x-xp)/np.linalg.norm(x)

    return x, v, vtilde, eps


def primal_dual_dist(
            ddsf,
            psif,
            gradf,
            lam,  # strength of regulariser
            L,    # spectral norm of Hessian
            l1weight,
            rmsfactor,
            rmc_comps,
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

    ddsf        - dict of futures to dds
    psif        - dict of futures to per band psi operators with dot anf hdot methods
    gradf       - dict of futures to gradient operators
    lam         - strength of regulariser
    L           - spectral norm of hessian approximation
    l1weight    - array of L1 weights
    reweighter  - function to compute L1 reweights
    '''

    client = get_client()

    # this seems to give a good trade-off between
    # primal and dual problems
    if sigma is None:
        sigma = L / (2.0 * gamma) / nu

    # stepsize control
    tau = 0.9 / (L / (2.0 * gamma) + sigma * nu**2)

    # we need to do this upfront only at the outset
    vtildef = {}
    modelf = {}
    dualf = {}
    for wname, ds in ddsf.items():
        fut = client.submit(vtilde_update,
                                       ds,
                                       sigma,
                                       psif[wname],
                                       workers=wname,
                                       key='vtilde-'+uuid4().hex)
        vtildef[wname] = client.submit(getitem, fut, 0, workers=wname)
        modelf[wname] = client.submit(getitem, fut, 1, workers=wname)
        dualf[wname] = client.submit(getitem, fut, 2, workers=wname)

    epsf = {}
    for k in range(maxit):
        # done on runner since need to combine over freq
        # vtilde = client.gather(list(vtildef.values()))
        vmfs = np.sum(client.gather(list(vtildef.values())), axis=0)/sigma
        vsoft = np.maximum(np.abs(vmfs) - lam*l1weight/sigma, 0.0) * np.sign(vmfs)
        mask = vmfs != 0
        ratio = np.zeros(mask.shape, dtype=l1weight.dtype)
        ratio[mask] = vsoft[mask] / vmfs[mask]

        # do on individual workers
        for wname, ds in ddsf.items():
            future = client.submit(update,
                                   gradf[wname],
                                   vtildef[wname],
                                   ratio,
                                   modelf[wname],
                                   dualf[wname],
                                   sigma,
                                   lam,
                                   tau,
                                   gamma,
                                   psif[wname],
                                   positivity,
                                   workers=wname,
                                   pure=False)
            modelf[wname] = client.submit(getitem, future, 0, workers=wname)
            dualf[wname] = client.submit(getitem, future, 1, workers=wname)
            vtildef[wname] = client.submit(getitem, future, 2, workers=wname)
            epsf[wname] = client.submit(getitem, future, 3, workers=wname)

        eps = client.gather(list(epsf.values()))

        eps = np.array(eps)
        epsmax = eps.max()

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {epsmax:.3e}", file=log)

        if epsmax < tol:
            break



    if k >= maxit-1:
        print(f'Maximum iterations reached. eps={eps}', file=log)
    else:
        print(f'Success after {k} iterations', file=log)

    return modelf, dualf


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
