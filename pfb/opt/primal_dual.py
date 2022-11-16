import numpy as np
from distributed import wait, get_client, as_completed
from operator import getitem
import pyscilog
log = pyscilog.get_logger('PD')


def primal_dual(
        A,
        xbar,
        x0,
        v0,  # initial guess for primal and dual variables
        lam,  # regulariser strength,
        psi,  # linear operator in dual domain
        psiH,  # adjoint of psi
        l1weights,  # weights for l1 thresholding
        L,  # spectral norm of Hessian
        prox,  # prox of regulariser
        nu=1.0,  # spectral norm of dictionary
        sigma=None,  # step size of dual update
        mask=None,  # regions where mask is False will be masked
        tol=1e-5,
        maxit=1000,
        positivity=True,
        report_freq=10,
        gamma=1.0,
        verbosity=1):
    # initialise
    x = x0.copy()
    v = v0.copy()

    # gradient function
    def grad_func(x):
        return -A(xbar - x) / gamma

    if sigma is None:
        sigma = L / (2.0 * gamma)

    # stepsize control
    tau = 0.9 / (L / (2.0 * gamma) + sigma * nu**2)

    # start iterations
    eps = 1.0
    for k in range(maxit):
        xp = x.copy()
        vp = v.copy()

        # tmp prox variable
        vtilde = v + sigma * psiH(xp)

        # dual update
        v = vtilde - sigma * prox(vtilde / sigma, lam / sigma,
                                  l1weights)

        # primal update
        x = xp - tau * (psi(2 * v - vp) + grad_func(xp))
        if positivity:
            x[x < 0] = 0.0

        # convergence check
        eps = np.linalg.norm(x - xp) / np.linalg.norm(x)
        if eps < tol:
            break

        if np.isnan(eps) or np.isinf(eps):
            import pdb; pdb.set_trace()

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}", file=log)

    if k == maxit - 1:
        if verbosity:
            print(f"Max iters reached. eps = {eps:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations", file=log)

    return x, v



def psiH(x, nx, ny):
    return x.reshape(1, nx*ny)

def psi(v, nx, ny):
    return v.reshape(nx, ny)

def vtilde_update(ds, **kwargs):
    nbasis, ntot = ds.DUAL.shape
    nx, ny = ds.MODEL.shape
    sigma = kwargs['sigma']
    return ds.DUAL.values + sigma * psiH(ds.MODEL.values, nx, ny)


def get_ratio(vtildes, lam, sigma, l1weights):
    l2_norm = np.zeros(l1weights.shape)
    nbasis = 0
    for v in vtildes:
        # l2_norm += (v/sigma)**2
        l2_norm += v/sigma
        nbasis += 1
    # l2_norm = np.sqrt(l2_norm)
    l2_norm /= nbasis
    l2_soft = np.maximum(np.abs(l2_norm) - lam*l1weights/sigma, 0.0)  # norm is positive
    # l2_soft = np.where(np.abs(l2_norm) >= lam/sigma, l2_norm, 0.0)
    mask = l2_norm != 0
    ratio = np.zeros(mask.shape, dtype=l1weights.dtype)
    ratio[mask] = l2_soft[mask] / l2_norm[mask]
    return ratio

def update(ds, A, y, vtilde, ratio, **kwargs):
    sigma = kwargs['sigma']
    lam = kwargs['lam']
    tau = kwargs['tau']
    gamma = kwargs['gamma']

    xp = ds.MODEL.values
    nx, ny = ds.MODEL.shape
    vp = ds.DUAL.values

    # dual
    v = vtilde - vtilde * ratio

    # primal
    grad = -A(y - xp)/gamma
    x = xp - tau * (psi(2 * v - vp, nx, ny) + grad)
    if kwargs['positivity']:
        x[x < 0] = 0.0

    vtilde = v + sigma * psiH(x, nx, ny)

    eps = np.linalg.norm(x-xp)/np.linalg.norm(x)

    ds_out = ds.assign(**{'MODEL': (('x','y'), x),
                          'DUAL': (('b', 'c'), v)})

    return ds_out, vtilde, eps


def sety(ds, **kwargs):
    gamma = kwargs['gamma']
    if 'MODEL' in ds:
        return ds.MODEL.values + gamma * ds.UPDATE.values
    else:
        return gamma * ds.UPDATE.values


def primal_dual_dist(
            ddsf,
            Af,
            lam,  # regulariser strength,
            L,  # spectral norm of Hessian
            wsum,
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

    yf = client.map(sety, ddsf, gamma=gamma)

    # we need to do this upfront only at the outset
    vtildes = client.map(vtilde_update, ddsf, sigma=sigma)

    eps = 1.0
    for k in range(maxit):
        # TODO - should get one ratio per basis
        # split over different workers
        ratio = client.submit(get_ratio,
                              vtildes, lam, sigma, l1weight,
                              workers=[names[0]], pure=False)

        wait([ratio])

        future = client.map(update,
                            ddsf, Af, yf, vtildes, [ratio]*len(ddsf),
                            pure=False,
                            wsum=wsum,
                            sigma=sigma,
                            tau=tau,
                            lam=lam,
                            gamma=gamma,
                            positivity=positivity)

        wait(future)

        ddsf = client.map(getitem, future, [0]*len(future),
                          pure=False)
        vtildes = client.map(getitem, future, [1]*len(future),
                             pure=False)
        epsf = client.map(getitem, future, [2]*len(future),
                          pure=False)

        eps = []
        for f in as_completed(epsf):
            eps.append(f.result())
        eps = np.array(eps)
        if eps.max() < tol:
            break

    if k >= maxit:
        print(f'Maximum iterations reached. eps={eps}', file=log)

    return ddsf


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
