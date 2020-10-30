
import numpy as np
from pfb.operators import Dirac, Prior, PSF
from ducc0.wgridder import ms2dirty, dirty2ms
import matplotlib.pyplot as plt
from africanus.constants import c as lightspeed

def pcg(A, b, x0, M=None, tol=1e-5, maxit=500, verbosity=1, report_freq=10):
    
    if M is None:
        M = lambda x: x
    
    r = A(x0) - b
    y = M(r)
    p = -y
    rnorm = np.vdot(r, y)
    if np.isnan(rnorm) or rnorm == 0.0:
        eps0 = 1.0
    else:
        eps0 = rnorm
    k = 0
    x = x0
    while rnorm/eps0 > tol and k < maxit:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        rnorm = np.vdot(r, y)
        alpha = rnorm/np.vdot(p, Ap)
        x = xp + alpha*p
        r = rp + alpha*Ap
        y = M(r)
        rnorm_next = np.vdot(r, y)
        # while rnorm_next > rnorm:  # TODO - better line search
        #     alpha *= 0.75
        #     x = xp + alpha*p
        #     r = rp + alpha*Ap
        #     y = M(r)
        #     rnorm_next = np.vdot(r, y)

        beta = rnorm_next/rnorm
        p = beta*p - y
        rnorm = rnorm_next
        k += 1

        if not k%report_freq and verbosity > 1:
            print("At iteration %i rnorm = %f"%(k, rnorm/eps0))

    if k >= maxit:
        if verbosity > 0:
            print("CG - Maximum iterations reached. Norm of residual = %f.  "%(rnorm/eps0))
    else:
        if verbosity > 0:
            print("CG - Success, converged after %i iterations"%k)
    return x


def X_func(x, Xdesign):
    return Xdesign.dot(x)


def XH_func(x, Xdesign):
    nband, npix = x.shape
    ncomps = Xdesign.shape[-1]
    res = np.zeros((ncomps, npix), dtype=np.float64)
    XdesH = Xdesign.T
    for i in range(npix):
        res[:, i] = XdesH.dot(x[:, i])
    return res


if __name__=="__main__":
    npix = 27
    nband = 8
    order = 4

    # random coefficients
    comps = np.random.randn(order, npix)

    # freqs
    w = np.linspace(0.5, 1.4, nband)[:, None]
    
    # design matrix
    Xdesign = np.tile(w, order) ** np.arange(0, order)

    # image
    model = Xdesign.dot(comps)


    # add some noise
    data = model

    # dirty
    dirty = Xdesign.T.dot(data)


    # hessian
    hess = Xdesign.T.dot(Xdesign)

    comp_rec = np.linalg.solve(hess, dirty)

    print(np.abs(comps-comp_rec).max())

    model_rec = Xdesign.dot(comp_rec)

    print(np.abs(model-model_rec).max())


