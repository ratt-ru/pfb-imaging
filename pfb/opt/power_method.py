import numpy as np
import dask.array as da
from operator import getitem
from distributed import wait, get_client, as_completed
from scipy.linalg import norm
from copy import deepcopy
from pfb.operators.hessian import hessian_psf_slice
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


def power(A, bp, bnorm):
    bp /= bnorm
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

def power_method_dist(Afs,
                      nx,
                      ny,
                      nband,
                      tol=1e-5,
                      maxit=200):

    client = get_client()

    b = []
    bssq = []
    bnum = []
    bden = []
    for i, (wid, A) in enumerate(Afs.items()):
        b.append(client.submit(np.random.randn, nx, ny,
                          workers={wid}))
        bssq.append(client.submit(sumsq, b[i],
                             workers={wid}))
        # this just initialises the lists required below
        bnum.append(1)
        bden.append(1)
    # wid corresponds to last worker
    bnorm = client.submit(bnormf, bssq,
                          workers={wid}).result()
    for k in range(maxit):
        for i, (wid, A) in enumerate(Afs.items()):
            fut = client.submit(power, A, b[i], bnorm,
                                workers={wid})

            b[i] = client.submit(getitem, fut, 0, workers={wid})
            bssq[i] = client.submit(getitem, fut, 1, workers={wid})
            bnum[i] = client.submit(getitem, fut, 2, workers={wid})
            bden[i] = client.submit(getitem, fut, 3, workers={wid})

        bnorm = client.submit(bnormf, bssq,
                              workers={wid})

        beta = client.submit(betaf, bnum, bden,
                             workers={wid})

        wait([b, bnorm, beta])

    return beta
