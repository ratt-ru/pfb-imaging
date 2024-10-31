from functools import partial
import numpy as np
from pfb.utils.misc import Gaussian2D, convolve2gaussres, fitcleanbeam
from pfb.utils.fits import set_wcs, add_beampars, save_fits
from pfb.utils.naming import xds_from_list
from pfb.opt.pcg import pcg


def restore_cube(ds_name,
                 output_name,
                 model_name,
                 residual_name,
                 gausspari,  # intrinsic resolution of the residual
                 gausspari_mfs,  # intrinsic resolution of the MFS residual
                 gaussparf,  # final desired resolution
                 gaussparf_mfs,  # final desired resolution
                 gaussparm=None,  # intrinsic resolution of the model
                 gaussparm_mfs=None,  # intrinsic resolution of the model
                 nthreads=1,
                 unit='Jy/beam',
                 output_dtype='f4'):
    '''
    Function to create science data products.

    '''
    # convolve residual
    if not isinstance(ds_name, list):
        ds_name = [ds_name]

    if unit.lower() == 'jy/beam':
        norm_kernel = False
        unit = 'Jy/beam'  # for consistency
    else:
        norm_kernel = True
        unit = 'Jy/pixel'  # for consistency

    drop_all_but = [model_name, residual_name]
    dds = xds_from_list(ds_name, nthreads=nthreads,
                        drop_all_but=drop_all_but)
    if model_name not in dds[0]:
        raise ValueError(f'Could not find {model_name} in dds')
    if residual_name not in dds[0]:
        raise ValueError(f'Could not find {residual_name} in dds')
    wsums = np.array([ds.wsum for ds in dds])
    fsel = wsums > 0
    wsum = np.sum(wsums)
    nx = dds[0].x.size
    ny = dds[0].y.size
    l = -(nx//2) + np.arange(nx)
    m = -(ny//2) + np.arange(ny)
    xx, yy = np.meshgrid(l, m, indexing='ij')

    mcube = np.stack([getattr(ds, model_name).values for ds in dds],
                        axis=0)
    nband = mcube.shape[0]
    mcube_mfs = np.mean(mcube[fsel], axis=0)
    if gaussparm is not None and len(gaussparm) != nband:
        raise ValueError(f'Got inconsistent GaussPars for {model_name}')
    mcube = convolve2gaussres(mcube, xx, yy, gaussparf,
                              nthreads=nthreads,
                              gausspari=gaussparm,
                              pfrac=0.2,
                              norm_kernel=norm_kernel)
    if gaussparm_mfs is not None:
        gaussparm_mfs = (gaussparm_mfs,)
    mcube_mfs = convolve2gaussres(mcube_mfs[None], xx, yy, gaussparf_mfs,
                                  nthreads=nthreads,
                                  gausspari=gaussparm_mfs,
                                  pfrac=0.2,
                                  norm_kernel=norm_kernel)

    if len(gausspari) != nband:
        raise ValueError(f'Got inconsistent GaussPars for {residual_name}')
    rcube = np.stack([getattr(ds, residual_name).values for ds in dds],
                        axis=0)
    if unit == 'Jy/beam':
        rcube_mfs = np.sum(rcube, axis=0)/wsum
        rcube[fsel] /= wsums[fsel, None, None]
    else:
        rcube_mfs = np.sum(rcube[fsel]*wsums[fsel, None, None], axis=0)/wsum
    rcube = convolve2gaussres(rcube, xx, yy, gaussparf,
                              nthreads=nthreads,
                              gausspari=gausspari,
                              pfrac=0.2,
                              norm_kernel=False)
    rcube_mfs = convolve2gaussres(rcube_mfs[None], xx, yy, gaussparf_mfs,
                                  nthreads=nthreads,
                                  gausspari=(gausspari_mfs,),
                                  pfrac=0.2,
                                  norm_kernel=False)

    mcube += rcube
    mcube_mfs += rcube_mfs

    radec = (dds[0].ra, dds[0].dec)
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    cell_asec = cell_deg * 3600
    freq_out = np.array([ds.freq_out for ds in dds])
    freq_out = np.unique(freq_out)
    time_out = np.array([ds.time_out for ds in dds])
    time_out = np.unique(time_out)
    if time_out.size > 1:
        raise ValueError('Got multiple times to restore')
    else:
        time_out = time_out[0]
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out,
                  unit=unit, ms_time=time_out)
    if gaussparf is not None:
        # ref_freq is at the center of the band so we provide beam_pars there
        add_beampars(hdr, gaussparf[nband//2], GaussPars=gaussparf, unit2deg=cell_deg)

    save_fits(mcube, output_name + '.fits', hdr, overwrite=True,
              dtype=output_dtype)

    ref_freq = np.sum(freq_out[fsel]*wsums[fsel, None, None])/wsum
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq,
                  unit=unit, ms_time=time_out)
    if gaussparf_mfs is not None:
        add_beampars(hdr, gaussparf_mfs, unit2deg=cell_deg)

    save_fits(mcube_mfs, output_name + '_mfs.fits', hdr, overwrite=True,
              dtype=output_dtype)

