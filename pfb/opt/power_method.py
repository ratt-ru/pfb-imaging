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
    bp = b.copy()
    while eps > tol and k < maxit:
        b = A(bp)
        bnorm = np.linalg.norm(b)
        betap = beta
        beta = np.vdot(bp, b) / np.vdot(bp, bp)
        b /= bnorm
        eps = np.linalg.norm(beta - betap) / betap
        k += 1
        bp[...] = b[...]

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


def power(A, bp, bnorm, sigmainv):
    bp /= bnorm
    b = A(bp, sigmainv)
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

def power_method_dist(hess_psfs,
                      nx,
                      ny,
                      nband,
                      sigmainv=0,
                      tol=1e-5,
                      maxit=200):

    client = get_client()

    b = []
    bssq = []
    bnum = []
    bden = []
    for i, (wname, hess) in enumerate(hess_psfs.items()):
        b.append(client.submit(np.random.randn, nx, ny,
                               workers=wname))
        bssq.append(client.submit(sumsq, b[i],
                                  workers=wname))
        # this just initialises the lists required below
        bnum.append(1)
        bden.append(1)

    bssq = client.gather(bssq)
    bnorm = np.sqrt(np.sum(bssq))
    beta = 1
    for k in range(maxit):
        for i, (wname, A) in enumerate(hess_psfs.items()):
            fut = client.submit(power, A, b[i], bnorm, sigmainv,
                                workers=wname)

            b[i] = client.submit(getitem, fut, 0, workers=wname)
            bssq[i] = client.submit(getitem, fut, 1, workers=wname)
            bnum[i] = client.submit(getitem, fut, 2, workers=wname)
            bden[i] = client.submit(getitem, fut, 3, workers=wname)

        bssq = client.gather(bssq)
        bnorm = np.sqrt(np.sum(bssq))
        betap = beta
        bnum = client.gather(bnum)
        bden = client.gather(bden)
        beta = np.sum(bnum)/np.sum(bden)

        eps = np.abs(betap - beta)/betap
        if eps < tol:
            break

        # if not k % 10:
        #     print(f"At iteration {k} eps = {eps:.3e}", file=log)
        #     from pympler import summary, muppy
        #     all_objects = muppy.get_objects()
        #     # bytearrays = [obj for obj in all_objects if isinstance(obj, bytearray)]
        #     # print(summary.print_(summary.summarize(bytearrays)))
        #     print(summary.print_(summary.summarize(all_objects)))

    return beta
