import numpy as np
from pfb_imaging.utils.misc import convolve2gaussres
from pfb_imaging.utils.naming import xds_from_list
import ray

@ray.remote
def rrestore_image(*args, **kwargs):
    return restore_image(*args, **kwargs)

def restore_image(ds_name,
                  model_name,
                  residual_name,
                  gaussparf=None,  # final desired resolution
                  nthreads=1):
    '''
    Writes restored image to dds. 
    Final resolution is specified by gaussparf.
    The intrinsic resolution used by default
    '''
    # convolve residual
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    drop_all_but = [model_name, residual_name, 'PSFPARSN', 'WSUM']
    dds = xds_from_list(ds_name, nthreads=nthreads,
                        drop_all_but=drop_all_but)
    if len(dds) > 1:
        raise RuntimeError('Some thing went wrong. '
                           'This should return a single dataset')
    ds = dds[0]
    if model_name not in ds:
        raise ValueError(f'Could not find {model_name} in dds')
    if residual_name not in ds:
        raise ValueError(f'Could not find {residual_name} in dds')
    wsum = ds.WSUM.values
    model = ds.get(model_name).values
    ncorr, nx, ny = model.shape
    l = -(nx//2) + np.arange(nx)
    m = -(ny//2) + np.arange(ny)
    xx, yy = np.meshgrid(l, m, indexing='ij')
    residual = ds.get(residual_name).values/wsum[:, None, None]

    if gaussparf is not None:
        rconv = convolve2gaussres(residual, xx, yy, gaussparf,
                              nthreads=nthreads,
                              gausspari=ds.PSFPARSN.values,
                              pfrac=0.2,
                              norm_kernel=False)    
    else:
        gaussparf = ds.PSFPARSN.values
        rconv = residual
    mconv = convolve2gaussres(model, xx, yy, gaussparf,
                              nthreads=nthreads,
                              pfrac=0.2,
                              norm_kernel=False)


    

    mconv += rconv

    ds = ds.assign({'IMAGE': (('corr', 'x', 'y'), mconv),
                    'PSFPARSF': (('corr', 'bpar'), gaussparf)})
    # only write updates
    ds = ds[['IMAGE', 'PSFPARSF']]
    ds.to_zarr(ds_name[0], mode='a')
