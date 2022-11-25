# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('CLEAN')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.clean["inputs"].keys():
    defaults[key] = schema.clean["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.clean)
def clean(**kw):
    '''
    Modified single-scale clean.
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'clean_{timestamp}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _clean(**opts)

def _clean(**kw):
    opts = OmegaConf.create(kw)
    # always combine over ds during cleaning
    opts['mean_ds'] = True
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import xarray as xr
    import numexpr as ne
    import dask
    import dask.array as da
    from dask.distributed import performance_report
    from pfb.utils.fits import (set_wcs, save_fits, dds2fits,
                                dds2fits_mfs, load_fits)
    from pfb.utils.misc import dds2cubes
    from pfb.deconv.hogbom import hogbom
    from pfb.deconv.clark import clark
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.operators.hessian import hessian
    from pfb.opt.pcg import pcg, pcg_psf
    from pfb.operators.hessian import hessian_xds
    from pfb.operators.psf import psf_convolve_cube, _hessian_reg_psf
    from scipy import ndimage
    from copy import copy

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}{opts.postfix}.dds.zarr'
    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan':-1})
    if opts.memory_greedy:
        dds = dask.persist(dds)[0]

    nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf

    # stitch dirty/psf in apparent scale
    output_type = dds[0].DIRTY.dtype
    dirty, model, residual, psf, psfhat, _, wsums = dds2cubes(
                                                            dds,
                                                            opts,
                                                            apparent=True,
                                                            log=log)
    wsum = np.sum(wsums)
    psf_mfs = np.sum(psf, axis=0)
    assert (psf_mfs.max() - 1.0) < 2*opts.epsilon
    dirty_mfs = np.sum(dirty, axis=0)
    if residual is None:
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()
    else:
        residual_mfs = np.sum(residual, axis=0)

    # for intermediary results (not currently written)
    freq_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
    freq_out = np.unique(np.array(freq_out))
    nx = dds[0].nx
    ny = dds[0].ny
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if opts.mask is not None:
        mask = load_fits(mask, dtype=output_type).squeeze()
        assert mask.shape == (nx, ny)
        mask = mask.astype(output_type)
        print('Using provided fits mask', file=log)
    else:
        mask = np.ones((nx, ny), dtype=output_type)

    # set up vis space Hessian
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads
    # always clean in apparent scale so no beam
    # mask is applied to residual after hessian application
    hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=0,
                   mask=np.ones((nx, ny), dtype=output_type),
                   compute=True, use_beam=False)

    # set up image space Hessian
    npad_xl = (nx_psf - nx)//2
    npad_xr = nx_psf - nx - npad_xl
    npad_yl = (ny_psf - ny)//2
    npad_yr = ny_psf - ny - npad_yl
    padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    lastsize = ny + np.sum(padding[-1])
    psfopts = {}
    psfopts['nthreads'] = opts.nvthreads
    psfopts['padding'] = padding
    psfopts['unpad_x'] = unpad_x
    psfopts['unpad_y'] = unpad_y
    psfopts['lastsize'] = lastsize
    # psfo = partial(psf_convolve_cube,
    #                psfhat=da.from_array(psfhat, chunks=(1, -1, -1)),
    #                beam=None,
    #                psfopts=psfopts,
    #                wsum=1,  # psf is normalised to sum to one
    #                sigmainv=0,
    #                compute=True)

    hess2opts = copy(psfopts)
    hess2opts['sigmainv'] = 1e-8

    padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    psfo = partial(_hessian_reg_psf, beam=mask[None, :, :], psfhat=psfhat,
                    nthreads=opts.nthreads, sigmainv=0,
                    padding=padding, unpad_x=unpad_x, unpad_y=unpad_y,
                    lastsize = lastsize)



    cgopts = {}
    cgopts['tol'] = opts.cg_tol
    cgopts['maxit'] = opts.cg_maxit
    cgopts['minit'] = opts.cg_minit
    cgopts['verbosity'] = opts.cg_verbose
    cgopts['report_freq'] = opts.cg_report_freq
    cgopts['backtrack'] = opts.backtrack


    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    if opts.threshold is None:
        threshold = opts.sigmathreshold * rms
    else:
        threshold = opts.threshold

    print(f"Iter 0: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    for k in range(opts.nmiter):
        if opts.algo.lower() == 'clark':
            print("Running Clark", file=log)
            # import cProfile
            # with cProfile.Profile() as pr:
            x, status = clark(mask*residual, psf, psfo,
                              threshold=threshold,
                              gamma=opts.gamma,
                              pf=opts.peak_factor,
                              maxit=opts.clark_maxit,
                              subpf=opts.sub_peak_factor,
                              submaxit=opts.sub_maxit,
                              verbosity=opts.verbose,
                              report_freq=opts.report_freq,
                              sigmathreshold=opts.sigmathreshold)
            # pr.print_stats(sort='cumtime')
            # quit()

        elif opts.algo.lower() == 'hogbom':
            print("Running Hogbom", file=log)
            x, status = hogbom(residual, psf,
                               threshold=threshold,
                               gamma=opts.gamma,
                               pf=opts.peak_factor,
                               maxit=opts.hogbom_maxit,
                               verbosity=opts.verbose,
                               report_freq=opts.report_freq)
        else:
            raise ValueError(f'{opts.algo} is not a valid algo option')

        model += x

        print("Getting residual", file=log)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        # report rms where there aren;t any model components
        tmp_mask = ~np.any(model, axis=0)
        rms = np.std(residual_mfs[tmp_mask])
        rmax = np.abs(residual_mfs).max()

        if opts.threshold is None:
            threshold = opts.sigmathreshold * rms
        else:
            threshold = opts.threshold

        # trigger flux mop if clean has stalled, not converged or
        # we have reached the final iteration/threshold
        status |= k == opts.nmiter-1
        status |= rmax <= threshold
        if opts.mop_flux and status:
            print(f"Mopping flux at iter {k+1}", file=log)
            mopmask = np.any(model, axis=0)
            if opts.dirosion:
                struct = ndimage.generate_binary_structure(2, opts.dirosion)
                mopmask = ndimage.binary_dilation(mopmask, structure=struct)
                mopmask = ndimage.binary_erosion(mopmask, structure=struct)
            # hess2opts['sigmainv'] = 1e-8
            x0 = np.zeros_like(x)
            x0[:, mopmask] = residual_mfs[mopmask]
            mopmask = mopmask[None, :, :].astype(residual.dtype)
            x = pcg_psf(psfhat, mopmask*residual, x0,
                        mopmask, hess2opts, cgopts)

            model += x

            save_fits(f'{basename}_premop{k}_resid_mfs.fits', residual_mfs, hdr_mfs)

            print("Getting residual", file=log)
            convimage = hess(model)
            ne.evaluate('dirty - convimage', out=residual,
                        casting='same_kind')
            ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                        casting='same_kind')

            save_fits(f'{basename}_postmop{k}_resid_mfs.fits', residual_mfs, hdr_mfs)

            tmp_mask = ~np.any(model, axis=0)
            rms = np.std(residual_mfs[tmp_mask])
            rmax = np.abs(residual_mfs).max()

            if opts.threshold is None:
                threshold = opts.sigmathreshold * rms
            else:
                threshold = opts.threshold

        print(f"Iter {k+1}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
              file=log)

        print("Updating results", file=log)
        dds_out = []
        for ds in dds:
            b = ds.bandid
            r = da.from_array(residual[b]*wsum)
            m = da.from_array(model[b])
            ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                                  'MODEL': (('x', 'y'), m)})
            dds_out.append(ds_out)
        writes = xds_to_zarr(dds_out, dds_name,
                             columns=('RESIDUAL', 'MODEL'), rechunk=True)
        dask.compute(writes)

        if rmax <= threshold:
            print("Terminating because final threshold has been reached",
                  file=log)
            break

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})

    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', basename, norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds, 'MODEL', basename, norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds, 'RESIDUAL', basename, norm_wsum=True))
        fitsout.append(dds2fits(dds, 'MODEL', basename, norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here.", file=log)
