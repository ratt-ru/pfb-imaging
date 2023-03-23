import numpy as np
import numexpr as ne
import dask.array as da
from distributed import wait, get_client, as_completed
from operator import getitem
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
        positivity=True,
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
        if positivity:
            x[x < 0.0] = 0.0

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


def primal_dual_optimised(
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
        positivity=True,
        report_freq=10,
        gamma=1.0,
        verbosity=1):
    # initialise
    xp = x.copy()
    xp = make_noncritical(xp)
    vp = v.copy()
    vp = make_noncritical(vp)
    vtilde = np.zeros_like(v)
    vtilde = make_noncritical(vtilde)
    vout = np.zeros_like(v)
    vout = make_noncritical(vout)
    xout = np.zeros_like(x)
    xout = make_noncritical(xout)

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
        # vtilde = v + sigma * psiH(xp)
        psiH(xp, vtilde)
        ne.evaluate('v + sigma * vtilde', out=vtilde)  #, casting='same_kind')

        # dual update
        # v = vtilde - sigma * prox(vtilde / sigma, lam / sigma)
        prox(vtilde, vout, lam, sigma)
        ne.evaluate('vtilde - sigma*vout', out=v)  #, casting='same_kind')

        # primal update
        # x = xp - tau * (psi(2 * v - vp) + grad(xp))
        ne.evaluate('2.0 * v - vp', out=vout)  #, casting='same_kind')
        psi(vout, xout)
        xout += grad(xp)
        ne.evaluate('xp - tau * xout', out=x)  #, casting='same_kind')

        if positivity:
            x[x < 0.0] = 0.0

        # convergence check
        eps = np.linalg.norm(x - xp) / np.linalg.norm(x)

        # copy contents to avoid allocating new memory
        # this is not faster but it allows us to see where the time is spent
        np.copyto(xp, x)
        np.copyto(vp, v)
        # xp[...] = x[...]
        # vp[...] = v[...]

        if np.isnan(eps) or np.isinf(eps):
            import pdb; pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}", file=log)
        k += 1

    if k == maxit:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations", file=log)

    return x, v


def vtilde_update(A, sigma, psiH):
    return A.dual + sigma * psiH(A.model)


def get_ratio(vtildes, lam, sigma, l1weights):
    vmfs = np.zeros(l1weights.shape)
    nband = 0
    for wid, v in vtildes.items():
        # l2_norm += (v/sigma)**2
        vmfs += v/sigma
    # l2_norm = np.sqrt(l2_norm)
    vsoft = np.maximum(np.abs(vmfs) - lam*l1weights/sigma, 0.0)  # norm is positive
    # l2_soft = np.where(np.abs(l2_norm) >= lam/sigma, l2_norm, 0.0)
    mask = vmfs != 0
    ratio = np.zeros(mask.shape, dtype=l1weights.dtype)
    ratio[mask] = vsoft[mask] / vmfs[mask]
    return ratio

def update(A, y, vtilde, ratio, xp, vp, sigma, lam, tau, gamma, psi, psiH, positivity):
    # dual
    v = vtilde - vtilde * ratio

    # primal
    grad = -A(y - xp)/gamma
    x = xp - tau * (psi(2 * v - vp) + grad)
    if positivity:
        x[x < 0] = 0.0

    vtilde = v + sigma * psiH(x)

    eps = np.linalg.norm(x-xp)/np.linalg.norm(x)

    return x, v, vtilde, eps


def sety(A, x, gamma):
    if hasattr(A, 'model'):
        return A.model + gamma * x
    else:
        return gamma * x


def primal_dual_dist(
            Afs,
            xfs,
            psi,
            psiH,
            lam,  # regulariser strength,
            L,  # spectral norm of Hessian
            l1weight,
            nu=1.0,  # spectral norm of dictionary
            sigma=None,  # step size of dual update
            tol=1e-5,
            maxit=100,
            positivity=True,
            gamma=1.0,
            verbosity=1):

    client = get_client()

    names = [w['name'] for w in client.scheduler_info()['workers'].values()]

    if sigma is None:
        sigma = L / (2.0 * gamma)

    # stepsize control
    tau = 0.9 / (L / (2.0 * gamma) + sigma * nu**2)

    # we need to do this upfront only at the outset
    yfs = {}
    vtildefs = {}
    modelfs = {}
    dualfs = {}
    for wid, A in Afs.items():
        yf = client.submit(sety, A, xfs[wid], gamma, workers={wid})
        yfs[wid] = yf
        vtildef = client.submit(vtilde_update, A, sigma, psiH, workers={wid})
        vtildefs[wid] = vtildef
        modelf = client.submit(getattr, A, 'model', workers={wid})
        modelfs[wid] = modelf
        dualf = client.submit(getattr, A, 'dual', workers={wid})
        dualfs[wid] = dualf

    epsfs = {}
    for k in range(maxit):
        # TODO - should get one ratio per basis
        # split over different workers
        ratio = client.submit(get_ratio,
                              vtildefs, lam, sigma, l1weight,
                              workers={names[0]})

        wait([ratio])
        for wid, A in Afs.items():
            future = client.submit(update,
                                   A, yfs[wid], vtildefs[wid], ratio,
                                   modelfs[wid], dualfs[wid],
                                   sigma,
                                   lam,
                                   tau,
                                   gamma,
                                   psi,
                                   psiH,
                                   positivity,
                                   workers={wid})
            modelfs[wid] = client.submit(getitem, future, 0, workers={wid})
            dualfs[wid] = client.submit(getitem, future, 1, workers={wid})
            vtildefs[wid] = client.submit(getitem, future, 2, workers={wid})
            epsfs[wid] = client.submit(getitem, future, 3, workers={wid})

        wait(list(epsfs.values()))

        eps = []
        for wid, epsf in epsfs.items():
            eps.append(epsf.result())
        eps = np.array(eps)
        if eps.max() < tol:
            break

    if k >= maxit-1:
        print(f'Maximum iterations reached. eps={eps}', file=log)
    else:
        print(f'Success after {k} iterations', file=log)

    return modelfs, dualfs


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
