import numpy as np
import dask.array as da
from pfb.utils.misc import fitcleanbeam

# submit on these
def get_resid_and_stats(dds, wsum):
    nx, ny = dds[0].nx, dds[0].ny
    residual_mfs = np.zeros((nx, ny), dtype=dds[0].DIRTY.dtype)
    for ds in dds:
        if 'RESIDUAL' in ds:
            residual_mfs += ds.RESIDUAL.values
        else:
            residual_mfs += ds.DIRTY.values
    residual_mfs /= wsum
    rms = np.std(residual_mfs)
    rmax = np.sqrt((residual_mfs**2).max())
    return residual_mfs, rms, rmax

def accum_wsums(dds):
    wsum = 0
    for ds in dds:
        wsum += ds.WSUM.values[0]
    return wsum

def get_eps(modelp, dds):
    eps = []
    for xp, ds in zip(modelp, dds):
        x = ds.MODEL.values
        eps.append(np.linalg.norm(x-xp)/np.linalg.norm(x))
    eps = np.array(eps)
    return eps.max()

def l1reweight(dds, l1weight, psiH, wsum, pix_per_beam, alpha=2):
    # get norm of model and residual
    l2mod = np.zeros(l1weight.shape, l1weight.dtype)
    l2res = np.zeros(l1weight.shape, l1weight.dtype)
    nbasis = l1weight.shape[0]
    nband = 0
    for ds in dds:
        l2mod += psiH(ds.MODEL.values)
        # Jy/beam -> Jy/pixel
        l2res += psiH(ds.RESIDUAL.values/wsum/pix_per_beam)
        nband += 1
    l2mod /= nband
    rms = np.std(l2res, axis=-1)
    return alpha/(1 + (l2mod/rms[:, None])**2)

def get_cbeam_area(dds, wsum):
    psf_mfs = np.zeros(dds[0].PSF.shape, dtype=dds[0].PSF.dtype)
    for ds in dds:
        psf_mfs += ds.PSF.values/wsum
    # beam pars in pixel units
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    return GaussPar[0]*GaussPar[1]*np.pi/4


# map on everything below
def init_dual_and_model(ds, **kwargs):
    dct = {}
    if 'MODEL' not in ds:
        model = np.zeros((kwargs['nx'], kwargs['ny']))
        dct['MODEL'] = (('x', 'y'), da.from_array(model, chunks=(-1, -1)))
        # dct['MODEL'] = (('x', 'y'), model)
    if 'DUAL' not in ds:
        dual = np.zeros((kwargs['nbasis'], kwargs['nmax']))
        dct['DUAL'] = (('b', 'c'), da.from_array(dual, chunks=(-1, -1)))
        # dct['DUAL'] = (('b', 'c'), dual)
    ds_out = ds.assign(**dct)
    return ds_out

from pfb.operators.hessian import _hessian_impl
def compute_residual(ds, **kwargs):
    dirty = ds.DIRTY.values
    wgt = ds.WEIGHT.values
    uvw = ds.UVW.values
    freq = ds.FREQ.values
    beam = ds.BEAM.values
    vis_mask = ds.MASK.values
    # we only want to apply the beam once here
    residual = dirty - _hessian_impl(beam * ds.MODEL.values,
                                     uvw,
                                     wgt,
                                     vis_mask,
                                     freq,
                                     None,
                                     cell=kwargs['cell'],
                                     wstack=kwargs['wstack'],
                                     epsilon=kwargs['epsilon'],
                                     double_accum=kwargs['double_accum'],
                                     nthreads=kwargs['nthreads'])
    ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), da.from_array(residual))})
    return ds_out
