import numpy as np
import dask.array as da
from distributed import get_client, worker_client, wait
from pfb.utils.misc import fitcleanbeam
from pfb.operators.hessian import _hessian_impl
from uuid import uuid4

# submit on these
def accum_wsums(dds):
    return np.sum([ds.WSUM.values for ds in dds])


def get_cbeam_area(dds, wsum):
    psf_mfs = np.stack([ds.PSF.values for ds in dds]).sum(axis=0)/wsum
    # beam pars in pixel units
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    return GaussPar[0]*GaussPar[1]*np.pi/4


def get_resids(ds, wsum, hessopts):
    dirty = ds.DIRTY.values
    nx, ny = dirty.shape
    if ds.MODEL.values.any():
        resid = dirty - _hessian_impl(ds.MODEL.values,
                                      ds.UVW.values,
                                      ds.WEIGHT.values,
                                      ds.MASK.values,
                                      ds.FREQ.values,
                                      ds.BEAM.values,
                                      x0=0.0,
                                      y0=0.0,
                                      **hessopts)
    else:
        resid = ds.DIRTY.values.copy()

    # dso = ds.assign(**{
    #     'RESIDUAL': (('x', 'y'), resid)
    # })

    return resid/wsum


def get_mfs_and_stats(resids):
    # resids should aready be normalised by wsum
    residual_mfs = np.sum(resids, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.sqrt((residual_mfs**2).max())
    return residual_mfs, rms, rmax


class almost_grad(object):
    def __init__(self, model, psfo, residual):
        self.psfo = psfo
        if model.any():
            self.data = residual + self.psfo(model, 0.0)
        else:
            self.data = residual.copy()

    def __call__(self, model):
        return self.psfo(model, 0.0) - self.data

    def update_data(self, model, psfo, residual):
        self.data = residual + self.psfo(model, 0.0)





def get_epsb(xp, x):
    return np.sum((x-xp)**2), np.sum(x**2)

def get_eps(num, den):
    return np.sqrt(np.sum(num)/np.sum(den))


def psi_dot(psib, model):
    return psib.dot(model)


def l1reweight_func(psif, ddsf, rmsfactor, rms_comps, alpha=4):
    '''
    The logic here is that weights should remain the same for model
    components that are rmsfactor times larger than the rms.
    High SNR values should experience relatively small thresholding
    whereas small values should be strongly thresholded
    '''
    client = get_client()
    outvar = []
    for wname, ds in ddsf.items():
        outvar.append(client.submit(psi_dot,
                                    psif[wname],
                                    ds.MODEL.values,
                                    key='outvar-'+uuid4().hex,
                                    workers=wname))

    outvar = client.gather(outvar)
    mcomps = np.abs(np.sum(outvar, axis=0))
    # the **alpha here results in more agressive reweighting
    return (1 + rmsfactor)/(1 + mcomps**alpha/rms_comps**alpha)





def update_results(ds, model, dual, residual):
    ds_out = ds.assign(**{'MODEL': (('x','y'), da.from_array(model, chunks=(4096, 4096))),
                          'DUAL': (('b', 'c'), da.from_array(dual, chunks=(1, 4096**2))),
                          'RESIDUAL': (('x', 'y'), da.from_array(residual, chunks=(4096, 4096)))})
    return ds_out

