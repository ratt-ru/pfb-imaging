from functools import partial
import numpy as np
from pfb.utils.misc import Gaussian2D, convolve2gaussres, fitcleanbeam
from pfb.utils.naming import xds_from_list
from pfb.operators.hessian import _hessian_impl
from pfb.opt.pcg import pcg


def restore_ds(ds_name,
               residual_name='RESIDUAL',
               model_name='MODEL',
               nthreads=1,
               gaussparf=None):
    '''
    Create restored image from dataset.
    If gaussparf is provided the images will be convolved to this resolution.
    Otherwise they will be convolved to the native resolution obtained by
    fitting a Gaussian to the main lobe of the PSF.
    '''
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    ds = xds_from_list(ds_name)[0]
    wsum = ds.WSUM.values[0]
    if not wsum:
        return

    try:
        residual = getattr(ds, residual_name).values
    except:
        raise ValueError(f'Could not find {residual_name} in ds')
    try:
        model = getattr(ds, model_name).values
    except:
        raise ValueError(f'Could not find {model_name} in ds')
    psf = ds.PSF.values
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
                              norm_kernel=False)

    image += residual

    ds['IMAGE'] = (('x','y'), image)
    ds.to_zarr(ds_name, mode='a')

    return image, int(ds.bandid), gaussparf

def ublurr_ds(ds_name,
              gaussparf=None,
              model_name='MODEL',
              nthreads=1,
              npix_psf_box=128):
    '''
    Create uniformly blurred images from a dataset.
    model_name ideally refers to the mopped model.
    The intrinsic resolution is obtained by computing the point source
    response after correcting for the uv-sampling density.
    If gaussparf is provided the images will be convolved to this resolution.
    '''
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    ds = xds_from_list(ds_name)[0]
    wsum = ds.WSUM.values[0]
    if not wsum:
        return

    try:
        model = getattr(ds, model_name).values
    except:
        raise ValueError(f'Could not find {model_name} in ds')
    psf = ds.PSF.values
    psf /= wsum

    cell_rad = ds.cell_rad
    cell_deg = np.rad2deg(cell_rad)
    cell_asec = cell_deg * 3600

    # get intrinsic resolution
    nx_psf, ny_psf = psf.shape
    npo2 = npix_psf_box//2
    slicex = slice(nx_psf//2 - npo2, nx_psf//2 + npo2)
    slicey = slice(ny_psf//2 - npo2, ny_psf//2 + npo2)
    l = -npo2 + np.arange(npix_psf_box)
    m = -npo2 + np.arange(npix_psf_box)
    xx, yy = np.meshgrid(l, m, indexing='ij')
    gausspari = fitcleanbeam(psf[None, slicex, slicey], level=0.5, pixsize=1.0)[0]
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
    j = psf[slicex, slicey] * wsum
    x0 = Gaussian2D(xx, yy, GaussPar=gausspari, normalise=True, nsigma=3)
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
    gausspari = fitcleanbeam(upsf[None], level=0.5, pixsize=1.0)[0]

    # first call to get intrinsic resolution
    if gaussparf is None:
        return gausspari

    # lm in pixel coordinates since gausspar in pixel coords
    nx = ds.x.size
    ny = ds.y.size
    l = -(nx//2) + np.arange(nx)
    m = -(ny//2) + np.arange(ny)
    xx, yy = np.meshgrid(l, m, indexing='ij')
    image = convolve2gaussres(model,
                              xx, yy,
                              gaussparf,
                              nthreads,
                              norm_kernel=False,
                              gausspari=gausspari)

    ds['UIMAGE'] = (('x','y'), image)
    ds.to_zarr(ds_name, mode='a')

    return image, int(ds.bandid)

