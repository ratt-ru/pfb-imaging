import numpy as np
from distributed import wait
from scipy.linalg import norm
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


def power(ds, bp, **kwargs):
    A = partial(_hessian,
                psfhat=ds.PSFHAT.values/kwargs['wsum'],
                nthreads=kwargs['nthreads'],
                sigmainv=kwargs['sigmainv'],
                padding=kwargs['psf_padding'],
                unpad_x=kwargs['unpad_x'],
                unpad_y=kwargs['unpad_y'],
                lastsize=kwargs['lastsize'])
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

def betaf(beta_num, beta_det):
    return np.sum(beta_num)/np.sum(beta_det)

def power_method_dist(ddsf,
                 nx,
                 ny,
                 nband,
                 nthreads,
                 psf_padding,
                 unpad_x,
                 unpad_y,
                 lastsize,
                 sigmainv,
                 wsum,
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
        fut = client.map(power, ddsf, b,
                         pure=False,
                         wsum=wsum,
                         bnorm=bnorm,
                         nthreads=nthreads,
                         psf_padding=psf_padding,
                         unpad_x=unpad_x,
                         unpad_y=unpad_y,
                         lastsize=lastsize,
                         sigmainv=sigmainv)

        b = client.map(getitem, fut, [0]*len(fut), pure=False)
        bssq = client.map(getitem, fut, [1]*len(fut), pure=False)
        bnum = client.map(getitem, fut, [2]*len(fut), pure=False)
        bdet = client.map(getitem, fut, [3]*len(fut), pure=False)

        bnorm = client.submit(bnormf, bssq,
                              workers=[names[0]], pure=False)

        beta = client.submit(betaf, bnum, bdet,
                             workers=[names[1]], pure=False)

        wait(b, bnorm, beta)

    return beta
