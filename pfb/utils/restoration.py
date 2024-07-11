from functools import partial
import numpy as np
from pfb.utils.misc import Gaussian2D, convolve2gaussres, fitcleanbeam
from pfb.utils.naming import xds_from_list
from pfb.operators.hessian import _hessian_impl
from pfb.opt.pcg import pcg


def restore_ds(ds_name,
               outputs,
               residual_name='RESIDUAL',
               model_name='MODEL',
               nthreads=1,
               gaussparf=None):
    '''
    Create a restored data products from dataset.
    If gaussparf is provided the images will be convolved to this resolution.
    '''
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    ds = xds_from_list(ds_name)[0]

    try:
        residual = getattr(ds, residual_name).values
    except:
        raise ValueError(f'Could not find {residual_name} in ds')
    try:
        model = getattr(ds, model_name).values
    except:
        raise ValueError(f'Could not find {model_name} in ds')
    psf = ds.PSF.values
    wsum = ds.WSUM.values[0]
    psf /= wsum
    residual /= wsum

    cell_rad = ds.cell_rad
    cell_deg = np.rad2deg(cell_rad)
    cell_asec = cell_deg * 3600

    # intrinsic to target resolution (the old way)
    gausspari = fitcleanbeam(psf[None], level=0.5, pixsize=1.0)[0]
    # lm in pixel coordinates since gausspar in pixel coords
    nx = ds.x.size
    ny = ds.y.size
    l = -(nx//2) + np.arange(nx)
    m = -(ny//2) + np.arange(ny)
    xx, yy = np.meshgrid(l, m, indexing='ij')
    if gaussparf is None:
        gaussparf = gausspari
        gausspari = None
    else:
        residual = convolve2gaussres(residual,
                                     xx, yy,
                                     gaussparf,
                                     nthreads,
                                     gausspari=gausspari,
                                     norm_kernel=True)

    image = convolve2gaussres(model,
                              xx, yy,
                              gaussparf,
                              nthreads,
                              norm_kernel=False,
                              gausspari=gausspari)

    image += residual

    # get natural resolution
    nx_psf, ny_psf = psf.shape
    # use 128 x 128 box
    slicex = slice(nx_psf//2 - 64, nx_psf//2 + 64)
    slicey = slice(ny_psf//2 - 64, ny_psf//2 + 64)
    j = psf[slicex,
            slicey] * wsum
    l = -64 + np.arange(128)
    m = -64 + np.arange(128)
    xx, yy = np.meshgrid(l, m, indexing='ij')
    x0 = Gaussian2D(xx, yy, GaussPar=gausspari, normalise=True, nsigma=3)
    hess = partial(_hessian_impl,
                   uvw=ds.UVW.values,
                   weight=ds.WEIGHT.values,
                   vis_mask=ds.MASK.values,
                   freq=ds.FREQ.values,
                   beam=None,
                   cell=ds.cell_rad,
                   x0=ds.x0,
                   y0=ds.y0,
                   do_wgridding=False,
                   epsilon=5e-4,
                   double_accum=True,
                   nthreads=nthreads,
                   sigmainvsq=1e-5*wsum)

    upsf = pcg(hess,
            j,
            x0=x0.copy(),
            tol=5e-3,
            maxit=100,
            minit=1,
            verbosity=0,
            report_freq=100,
            backtrack=False,
            return_resid=False)

    upsf /= upsf.max()
    gaussparu = fitcleanbeam(upsf[None], level=0.5, pixsize=1.0)[0]

    print(gaussparu)
    print(gausspari)
    print(gaussparf)

    import ipdb; ipdb.set_trace()

    return


