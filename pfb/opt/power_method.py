import numpy as np
import dask.array as da
from operator import getitem
from distributed import wait, get_client, as_completed
from scipy.linalg import norm
from copy import deepcopy
import pyscilog
log = pyscilog.get_logger('PM')


def power_method(
        A,
        imsize,
        b0=None,
        tol=1e-5,
        maxit=250,
        verbosity=1,
        report_freq=25):
    if b0 is None:
        b = np.random.randn(*imsize)
        b /= norm(b)
    else:
        b = b0 / norm(b0)
    beta = 1.0
    eps = 1.0
    k = 0
    while eps > tol and k < maxit:
        bp = b
        b = A(bp)
        bnorm = np.linalg.norm(b)
        betap = beta
        beta = np.vdot(bp, b) / np.vdot(bp, bp)
        b /= bnorm
        eps = np.linalg.norm(beta - betap) / betap
        k += 1

        if not k % report_freq and verbosity > 1:
            print(f"At iteration {k} eps = {eps:.3e}", file=log)

    if k == maxit:
        if verbosity:
            print(f"Maximum iterations reached. "
                  f"eps = {eps:.3e}, beta = {beta:.3e}", file=log)
    else:
        if verbosity:
            print(f"Success, converged after {k} iterations. "
                  f"beta = {beta:.3e}", file=log)
    return beta, b


def power(A, bp, **kwargs):
    bp /= kwargs['bnorm']
    b = A(bp)
    bsumsq = np.sum(b**2)
    beta_num = np.vdot(b, bp)
    beta_den = np.vdot(bp, bp)

    return b, bsumsq, beta_num, beta_den

def sumsq(b):
    return np.sum(b**2)

def bnormf(bsumsq):
    return np.sqrt(np.sum(bsumsq))

def betaf(beta_num, beta_den):
    return np.sum(beta_num)/np.sum(beta_den)

def power_method_dist(Af,
                      nx,
                      ny,
                      nband,
                      tol=1e-5,
                      maxit=200):

    client = get_client()

    names = [w['name'] for w in client.scheduler_info()['workers'].values()]

    b = []
    for _ in range(nband):
        f = client.submit(np.random.randn, nx, ny)
        b.append(f)

    bssq = client.map(sumsq, b)
    bnorm = client.submit(bnormf, bssq,
                          workers=[names[0]], pure=False).result()
    for k in range(maxit):
        fut = client.map(power, Af, b,
                         pure=False,
                         bnorm=bnorm)

        b = client.map(getitem, fut, [0]*len(fut), pure=False)
        bssq = client.map(getitem, fut, [1]*len(fut), pure=False)
        bnum = client.map(getitem, fut, [2]*len(fut), pure=False)
        bden = client.map(getitem, fut, [3]*len(fut), pure=False)

        bnorm = client.submit(bnormf, bssq,
                              workers=[names[0]], pure=False)

        beta = client.submit(betaf, bnum, bden,
                             workers=[names[1]], pure=False)

        wait([b, bnorm, beta])

    return beta

from pfb.operators.hessian import hessian_psf_slice_dask
def power_method_persist(ddsf,
                         Af,
                         nx,
                         ny,
                         nband,
                         tol=1e-5,
                         maxit=200):

    client = get_client()
    b = []
    bssq = []
    for ds in ddsf:
        wid = ds.worker
        tmp = client.persist(da.random.normal(0, 1, (nx, ny)),
                             workers={wid})
        b.append(tmp)
        bssq.append(da.sum(tmp**2))

    bssq = da.stack(bssq)
    bnorm = da.sqrt(da.sum(bssq))
    bp = deepcopy(b)
    beta_num = [da.array(1) for _ in range(nband)]
    beta_den = [da.array(1) for _ in range(nband)]
    beta = 1
    for k in range(maxit):
        for i, (ds, A) in enumerate(zip(ddsf, Af)):
            bp[i] = b[i]/bnorm
            b[i] = A(bp[i])
            bssq[i] = da.sum(b[i]**2)
            beta_num[i] = da.vdot(b[i], bp[i])
            beta_den[i] = da.vdot(bp[i], bp[i])
        bnorm = da.sqrt(da.sum(bssq))
        betap = beta
        beta = (beta_num.sum()/beta_den.sum()).compute()
        eps = np.linalg.norm(beta - betap) / betap

        if eps < tol:
            break

    if k == maxit:
        print(f"Maximum iterations reached. "
                f"eps = {eps:.3e}, beta = {beta:.3e}", file=log)
    else:
        print(f"Success, converged after {k} iterations. "
                f"beta = {beta:.3e}", file=log)
    return beta







