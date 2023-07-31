import numpy as np
import dask.array as da
from pfb.utils.misc import fitcleanbeam

# submit on these
def get_resid_and_stats(Afs, wsum):
    dirty = list(Afs.values())[0].dirty
    nx, ny = dirty.shape
    residual_mfs = np.zeros((nx, ny), dtype=dirty.dtype)
    for wid, A in Afs.items():
        if hasattr(A, 'residual'):
            residual_mfs += A.residual
        else:
            residual_mfs += A.dirty
    residual_mfs /= wsum
    rms = np.std(residual_mfs)
    rmax = np.sqrt((residual_mfs**2).max())
    return residual_mfs, rms, rmax

def accum_wsums(As):
    wsum = 0
    for wid, A in As.items():
        wsum += A.wsumb
    return wsum

def get_epsb(xp, x):
    return np.sum((x-xp)**2), np.sum(x**2)

def get_eps(num, den):
    return np.sqrt(np.sum(num)/np.sum(den))

def l1reweight(model, residual, l1weight, psiH, wsum, pix_per_beam, alpha=2):
    # get norm of model and residual
    l2mod = np.zeros(l1weight.shape, l1weight.dtype)
    l2res = np.zeros(l1weight.shape, l1weight.dtype)
    nbasis = l1weight.shape[0]
    nband = 0
    for wid, mod in model.items():
        l2mod += psiH(mod)
        # Jy/beam -> Jy/pixel
        l2res += psiH(residual[wid]/wsum/pix_per_beam)
        nband += 1
    l2mod /= nband
    rms = np.std(l2res, axis=-1)
    return alpha/(1 + (l2mod/rms[:, None])**2)


def set_l1weight(ds):
    return ds.L1WEIGHT.values


def get_cbeam_area(As, wsum):
    psf_mfs = np.zeros(As[0].psf.shape, dtype=As[0].psf.dtype)
    for wid, A in As.items():
        psf_mfs += A.psf
    psf_mfs /= wsum
    # beam pars in pixel units
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    return GaussPar[0]*GaussPar[1]*np.pi/4


def update_results(A, ds, model, dual, residual):
    A.model[...] = model
    A.dual[...] = dual
    A.residual[...] = residual

    ds_out = ds.assign(**{'MODEL': (('x','y'), da.from_array(model, chunks=(4096, 4096))),
                          'DUAL': (('b', 'c'), da.from_array(dual, chunks=(1, 4096**2))),
                          'RESIDUAL': (('x', 'y'), da.from_array(residual, chunks=(4096, 4096)))})
    return ds_out


from pfb.operators.hessian import _hessian_impl
def compute_residual(A, model):
    return A.compute_residual(model)


def set_wsum(A, wsum):
    A.wsum = wsum
    return
