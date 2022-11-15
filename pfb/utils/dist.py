import numpy as np

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

# map on everything below
def init_dual_and_model(ds, **kwargs):
    dct = {}
    if 'MODEL' not in ds:
        model = np.zeros((kwargs['nx'], kwargs['ny']))
        dct['MODEL'] = (('x', 'y'), model)
    if 'DUAL' not in ds:
        dual = np.zeros((1, kwargs['nx']*kwargs['ny']))
        dct['DUAL'] = (('b', 'c'), dual)
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
    ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), residual)})
    return ds_out
