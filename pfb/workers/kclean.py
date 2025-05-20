# flake8: noqa
from pfb.workers.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('kclean')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.kclean)
def kclean(**kw):
    '''
    Modified single-scale clean.
    '''
    opts = OmegaConf.create(kw)

    from pfb.utils.naming import set_output_names
    opts, basedir, oname = set_output_names(opts)

    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        opts.nthreads = nthreads//2
        ncpu = ncpu//2

    OmegaConf.set_struct(opts, True)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/kclean_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    from pfb.utils.naming import xds_from_url, get_opts

    basename = f'{basedir}/{oname}'
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_name = f'{basename}_{opts.suffix}.dds'

    ti = time.time()
    _kclean(**opts)

    dds, dds_list = xds_from_url(dds_name)

    if opts.fits_mfs or opts.fits:
        from daskms.fsspec_store import DaskMSStore
        from pfb.utils.fits import dds2fits
        # get the psfpars for the mfs cube
        dds_store = DaskMSStore(dds_name)
        if '://' in dds_store.url:
            protocol = dds_store.url.split('://')[0]
        else:
            protocol = 'file'
        psfpars_mfs = get_opts(dds_store.url,
                               protocol,
                               name='psfparsn_mfs.pkl')
        print(f"Writing fits files to {fits_oname}_{opts.suffix}", file=log)
        dds2fits(dds_list,
                 'RESIDUAL',
                 f'{fits_oname}_{opts.suffix}',
                 norm_wsum=True,
                 do_mfs=opts.fits_mfs,
                 do_cube=opts.fits_cubes,
                 psfpars_mfs=psfpars_mfs)
        dds2fits(dds_list,
                 'MODEL',
                 f'{fits_oname}_{opts.suffix}',
                 norm_wsum=False,
                 do_mfs=opts.fits_mfs,
                 do_cube=opts.fits_cubes,
                 psfpars_mfs=psfpars_mfs)

    print(f"All done after {time.time() - ti}s", file=log)


# def _kclean(**kw):
#     opts = OmegaConf.create(kw)
#     OmegaConf.set_struct(opts, True)

#     from functools import partial
#     import numpy as np
#     import xarray as xr
#     from pfb.utils.fits import set_wcs, save_fits, load_fits
#     from pfb.deconv.clark import clark
#     from pfb.utils.naming import xds_from_url
#     from pfb.opt.pcg import pcg, pcg_psf
#     from pfb.operators.gridder import compute_residual
#     from scipy import ndimage
#     from pfb.utils.misc import fit_image_cube

#     basename = opts.output_filename
#     if opts.fits_output_folder is not None:
#         fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
#     else:
#         fits_oname = basename

#     dds_name = f'{basename}_{opts.suffix}.dds'
#     dds, dds_list = xds_from_url(dds_name)

#     if dds[0].corr.size > 1:
#         raise NotImplementedError("Joint polarisation deconvolution not "
#                                   "yet supported for kclean algorithm")

#     nx, ny = dds[0].x.size, dds[0].y.size
#     nx_psf, ny_psf = dds[0].x_psf.size, dds[0].y_psf.size
#     lastsize = ny_psf
#     freq_out = []
#     time_out = []
#     for ds in dds:
#         freq_out.append(ds.freq_out)
#         time_out.append(ds.time_out)
#     freq_out = np.unique(np.array(freq_out))
#     time_out = np.unique(np.array(time_out))

#     nband = freq_out.size

#     # stitch dirty/psf in apparent scale
#     # drop_vars to avoid duplicates in memory
#     if 'RESIDUAL' in dds[0]:
#         residual = np.stack([ds.RESIDUAL.values[0] for ds in dds], axis=0)
#         dds = [ds.drop_vars('RESIDUAL') for ds in dds]
#     else:
#         residual = np.stack([ds.DIRTY.values[0] for ds in dds], axis=0)
#         dds = [ds.drop_vars('DIRTY') for ds in dds]
#     if 'MODEL' in dds[0]:
#         model = np.stack([ds.MODEL.values[0] for ds in dds], axis=0)
#         dds = [ds.drop_vars('MODEL') for ds in dds]
#     else:
#         model = np.zeros((nband, nx, ny))
#     psf = np.stack([ds.PSF.values[0] for ds in dds], axis=0)
#     psfhat = np.stack([ds.PSFHAT.values[0] for ds in dds], axis=0)
#     dds = [ds.drop_vars('PSF') for ds in dds]
#     wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
#     fsel = wsums > 0  # keep track of empty bands


#     wsum = np.sum(wsums)
#     psf /= wsum
#     psfhat /= wsum
#     residual /= wsum
#     psf_mfs = np.sum(psf, axis=0)
#     residual_mfs = np.sum(residual, axis=0)

#     # for intermediary results
#     nx = dds[0].x.size
#     ny = dds[0].y.size
#     ra = dds[0].ra
#     dec = dds[0].dec
#     radec = [ra, dec]
#     cell_rad = dds[0].cell_rad
#     cell_deg = np.rad2deg(cell_rad)
#     ref_freq = np.mean(freq_out)
#     hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq, casambm=False)
#     if 'niters' in dds[0].attrs:
#         iter0 = dds[0].niters
#     else:
#         iter0 = 0

#     # TODO - check coordinates match
#     # Add option to interp onto coordinates?
#     if opts.mask is not None:
#         mask = load_fits(opts.mask, dtype=residual.dtype).squeeze()
#         assert mask.shape == (nx, ny)
#         print('Using provided fits mask', file=log)
#     else:
#         mask = np.ones((nx, ny), dtype=residual.dtype)

#     # PCG related options for flux mop
#     cgopts = {}
#     cgopts['tol'] = opts.cg_tol
#     cgopts['maxit'] = opts.cg_maxit
#     cgopts['minit'] = opts.cg_minit
#     cgopts['verbosity'] = opts.cg_verbose
#     cgopts['report_freq'] = opts.cg_report_freq
#     cgopts['backtrack'] = opts.backtrack


#     rms = np.std(residual_mfs)
#     rmax = np.abs(residual_mfs).max()
#     best_rms = rms
#     best_rmax = rmax
#     best_model = model.copy()
#     diverge_count = 0
#     if opts.threshold is None:
#         threshold = opts.rmsfactor * rms
#     else:
#         threshold = opts.threshold

#     print(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
#           file=log)
#     for k in range(iter0, iter0 + opts.niter):
#         print("Cleaning", file=log)
#         x, status = clark(residual, psf, psfhat, wsums/wsum, mask,
#                           threshold=threshold,
#                           gamma=opts.gamma,
#                           pf=opts.peak_factor,
#                           maxit=opts.minor_maxit,
#                           subpf=opts.sub_peak_factor,
#                           submaxit=opts.subminor_maxit,
#                           verbosity=opts.verbose,
#                           report_freq=opts.report_freq,
#                           nthreads=opts.nthreads)
#         model += x

#         # write component model
#         print(f"Writing model at iter {k+1} to "
#               f"{basename}_{opts.suffix}_model.mds", file=log)
#         try:
#             coeffs, Ix, Iy, expr, params, texpr, fexpr = \
#                 fit_image_cube(time_out, freq_out[fsel], model[None, fsel, :, :],
#                                wgt=wsums[None, fsel],
#                                nbasisf=int(np.sum(fsel)),
#                                method='Legendre')
#             # save interpolated dataset
#             data_vars = {
#                 'coefficients': (('par', 'comps'), coeffs),
#             }
#             coords = {
#                 'location_x': (('x',), Ix),
#                 'location_y': (('y',), Iy),
#                 'params': (('par',), params),  # already converted to list
#                 'times': (('t',), time_out),  # to allow rendering to original grid
#                 'freqs': (('f',), freq_out)
#             }
#             attrs = {
#                 'spec': 'genesis',
#                 'cell_rad_x': cell_rad,
#                 'cell_rad_y': cell_rad,
#                 'npix_x': nx,
#                 'npix_y': ny,
#                 'texpr': texpr,
#                 'fexpr': fexpr,
#                 'center_x': dds[0].x0,
#                 'center_y': dds[0].y0,
#                 'flip_u': dds[0].flip_u,
#                 'flip_v': dds[0].flip_v,
#                 'flip_w': dds[0].flip_w,
#                 'ra': dds[0].ra,
#                 'dec': dds[0].dec,
#                 'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
#                 'parametrisation': expr  # already converted to str
#             }

#             coeff_dataset = xr.Dataset(data_vars=data_vars,
#                                        coords=coords,
#                                        attrs=attrs)
#             coeff_dataset.to_zarr(f"{basename}_{opts.suffix}_model.mds",
#                                   mode='w')
#         except Exception as e:
#             print(f"Exception {e} raised during model fit .", file=log)

#         save_fits(np.mean(model[fsel], axis=0),
#                   fits_oname + f'_{opts.suffix}_model_{k+1}.fits',
#                   hdr_mfs)

#         print(f'Computing residual', file=log)
#         for ds_name, ds in zip(dds_list, dds):
#             b = int(ds.bandid)
#             resid, _ = compute_residual(ds_name,
#                                      nx, ny,
#                                      cell_rad, cell_rad,
#                                      ds_name,
#                                      model[b][None, :, :],  # add back corr axis
#                                      nthreads=opts.nthreads,
#                                      epsilon=opts.epsilon,
#                                      do_wgridding=opts.do_wgridding,
#                                      double_accum=opts.double_accum)
#             residual[b] = resid[0]  # remove corr axis
#         residual /= wsum
#         residual_mfs = np.sum(residual, axis=0)
#         save_fits(residual_mfs,
#                   fits_oname + f'_{opts.suffix}_residual_{k+1}.fits',
#                   hdr_mfs)

#         # report rms and rmax inside mask
#         rmsp = rms
#         # tmp_mask = ~np.any(model, axis=0)
#         rms = np.std(residual_mfs[mask>0])
#         rmax = np.abs(residual_mfs[mask>0]).max()

#         # base this on rmax?
#         if rms < best_rms:
#             best_rms = rms
#             best_rmax = rmax
#             best_model = model.copy()

#         # these are not updated in compute_residual
#         for ds_name, ds in zip(dds_list, dds):
#             b = int(ds.bandid)
#             ds = ds.assign(**{
#                 'MODEL': (('corr', 'x', 'y'), model[b][None, :, :]),
#                 'MODEL_BEST': (('corr', 'x', 'y'), best_model[b][None, :, :]),
#             })
#             attrs = {}
#             attrs['rms'] = best_rms
#             attrs['rmax'] = best_rmax
#             attrs['niters'] = k+1
#             ds = ds.assign_attrs(**attrs)
#             ds.to_zarr(ds_name, mode='a')

#         if opts.threshold is None:
#             threshold = opts.rmsfactor * rms
#         else:
#             threshold = opts.threshold

#         # trigger flux mop if clean has stalled, not converged or
#         # we have reached the final iteration/threshold
#         status |= k == iter0 + opts.niter-1
#         status |= rmax <= threshold
#         if opts.mop_flux and status:
#             print(f"Mopping flux at iter {k+1}", file=log)
#             mopmask = np.any(model, axis=0)
#             if opts.dirosion:
#                 struct = ndimage.generate_binary_structure(2, opts.dirosion)
#                 mopmask = ndimage.binary_dilation(mopmask, structure=struct)
#                 mopmask = ndimage.binary_erosion(mopmask, structure=struct)
#             x0 = np.zeros_like(x)
#             # x0[:, mopmask] = residual_mfs[mopmask]
#             # TODO - applying mask as beam is wasteful
#             mopmask = (mopmask.astype(residual.dtype) * mask)[None, :, :]
#             x = pcg_psf(psfhat,
#                         mopmask*residual,
#                         x0,
#                         mopmask,
#                         lastsize,
#                         opts.nthreads,
#                         rmax,  # used as eta
#                         cgopts)

#             model += opts.mop_gamma*x

#             print(f'Computing residual', file=log)
#             for ds_name, ds in zip(dds_list, dds):
#                 b = int(ds.bandid)
#                 resid, _ = compute_residual(ds_name,
#                                         nx, ny,
#                                         cell_rad, cell_rad,
#                                         ds_name,
#                                         model[b][None, :, :],  # add back corr axis
#                                         nthreads=opts.nthreads,
#                                         epsilon=opts.epsilon,
#                                         do_wgridding=opts.do_wgridding,
#                                         double_accum=opts.double_accum)
#                 residual[b] = resid[0]  # remove corr axis
#             residual /= wsum
#             residual_mfs = np.sum(residual, axis=0)

#             save_fits(residual_mfs,
#                       f'{fits_oname}_{opts.suffix}_postmop{k+1}_residual_mfs.fits',
#                       hdr_mfs)

#             save_fits(np.mean(model[fsel], axis=0),
#                       f'{fits_oname}_{opts.suffix}_postmop{k+1}_model_mfs.fits',
#                       hdr_mfs)

#             rmsp = rms
#             # tmp_mask = ~np.any(model, axis=0)
#             rms = np.std(residual_mfs[mask>0])
#             rmax = np.abs(residual_mfs[mask>0]).max()

#             # base this on rmax?
#             if rms < best_rms:
#                 best_rms = rms
#                 best_rmax = rmax
#                 best_model = model.copy()
#                 for ds_name, ds in zip(dds_list, dds):
#                     b = int(ds.bandid)
#                     ds = ds.assign(**{
#                         'MODEL_BEST': (('corr', 'x', 'y'), best_model[b][None, :, :])
#                     })

#                     ds.to_zarr(ds_name, mode='a')

#             if opts.threshold is None:
#                 threshold = opts.rmsfactor * rms
#             else:
#                 threshold = opts.threshold

#         print(f"Iter {k+1}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
#               file=log)

#         if rmax <= threshold:
#             print("Terminating because final threshold has been reached",
#                   file=log)
#             break

#         if rms > rmsp:
#             diverge_count += 1
#             if diverge_count > 3:
#                 print("Algorithm is diverging. Terminating.", file=log)
#                 break

#         # keep track of total number of iterations
#         for ds_name, ds in zip(dds_list, dds):
#             b = int(ds.bandid)
#             ds = ds.assign_attrs(**{
#                 'niter': k,
#             })

#             ds.to_zarr(ds_name, mode='a')

#     return


def _kclean(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from functools import partial
    import numpy as np
    import xarray as xr
    from pfb.utils.fits import set_wcs, save_fits, load_fits
    from pfb.deconv.clark import fsclark
    from pfb.utils.naming import xds_from_url, xds_from_list
    from pfb.operators.gridder import compute_residual
    from pfb.operators.hessian import fshessian_jax
    from scipy import ndimage
    from pfb.utils.misc import fit_image_fscube
    from jax.scipy.sparse.linalg import cg
    from pfb.operators.energy import stokes_energy
    from pfb.opt.fista import fista
    from pfb.prox.prox_21m import prox_21m_jax

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds, dds_list = xds_from_url(dds_name)

    nx, ny = dds[0].x.size, dds[0].y.size
    nx_psf, ny_psf = dds[0].x_psf.size, dds[0].y_psf.size
    lastsize = ny_psf
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))

    nband = freq_out.size
    ncorr = dds[0].corr.size
    corrs = dds[0].corr.values

    # stitch dirty/psf in apparent scale
    # drop_vars to avoid duplicates in memory
    print("Loading image space data products", file=log)
    dds = xds_from_list(dds_list, nthreads=opts.nthreads,
                        drop_all_but=['RESIDUAL', 'DIRTY', 'MODEL', 'PSF',
                                      'PSFHAT','BEAM', 'WSUM'])
    if 'RESIDUAL' in dds[0]:
        residual = np.stack([ds.RESIDUAL.values for ds in dds], axis=0)
        dds = [ds.drop_vars('RESIDUAL') for ds in dds]
    else:
        residual = np.stack([ds.DIRTY.values for ds in dds], axis=0)
        dds = [ds.drop_vars('DIRTY') for ds in dds]
    if 'MODEL' in dds[0]:
        model = np.stack([ds.MODEL.values for ds in dds], axis=0)
        dds = [ds.drop_vars('MODEL') for ds in dds]
    else:
        model = np.zeros((nband, ncorr, nx, ny))
    psf = np.stack([ds.PSF.values for ds in dds], axis=0)
    psfhat = np.stack([ds.PSFHAT.values for ds in dds], axis=0)
    beam = np.stack([ds.BEAM.values for ds in dds], axis=0)
    dds = [ds.drop_vars(['PSF', 'PSFHAT', 'BEAM']) for ds in dds]
    wsums = np.stack([ds.WSUM.values for ds in dds], axis=0)

    print("Normalising by wsum", file=log)
    wsum = np.sum(wsums, axis=0)
    # ensure wsums sum to one for each correlation
    wsums /= wsum[None, :]
    psf /= wsum[None, :, None, None]
    psfhat /= wsum[None, :, None, None]
    residual /= wsum[None, :, None, None]
    residual_mfs = np.sum(residual, axis=0)

    # for intermediary results
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq, casambm=False)
    if 'niters' in dds[0].attrs:
        iter0 = dds[0].niters
    else:
        iter0 = 0

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if opts.mask is not None:
        mask = load_fits(opts.mask, dtype=residual.dtype).squeeze()
        assert mask.shape == (nx, ny)
        print('Using provided fits mask', file=log)
    else:
        mask = np.ones((nx, ny), dtype=residual.dtype)

    rms = np.std(residual_mfs, axis=(-2,-1))
    rmax = np.abs(residual_mfs).max(axis=(-2,-1))
    diverge_count = 0
    if opts.threshold is None:
        threshold = opts.rmsfactor * rms
    else:
        threshold = opts.threshold * np.ones(corrs.size)

    weights21 = np.ones((ncorr, nx, ny), dtype=residual.dtype)
    for i, c in enumerate(corrs):
        print(f"Iter {iter0}: peak {c} residual = {rmax[i]:.3e}, rms = {rms[i]:.3e}",
            file=log)
    for k in range(iter0, iter0 + opts.niter):
        print("Cleaning", file=log)
        for i, c in enumerate(corrs):
            print(f"Threshold {c} = {threshold[i]:.3e}", file=log)
        # x, status = fsclark(residual, psf, psfhat,
        #                     wsums, mask,
        #                     threshold=threshold,
        #                     gamma=opts.clean_gamma,
        #                     pf=opts.peak_factor,
        #                     maxit=opts.minor_maxit,
        #                     subpf=opts.sub_peak_factor,
        #                     submaxit=opts.subminor_maxit,
        #                     verbosity=opts.verbose,
        #                     report_freq=opts.report_freq,
        #                     nthreads=opts.nthreads)
        status = 1
        x = np.zeros_like(residual)

        print("CG step", file=log)
        mopmask = np.any(model + x, axis=(0,1))
        if opts.dirosion:
            struct = ndimage.generate_binary_structure(2, opts.dirosion)
            mopmask = ndimage.binary_dilation(mopmask, structure=struct)
            mopmask = ndimage.binary_erosion(mopmask, structure=struct)
        mopmask = (mopmask.astype(residual.dtype) * mask)
        mbeam = beam * mopmask[None, None, :, :]
        A = partial(fshessian_jax, nx, ny, nx_psf, ny_psf,
                    rmax.max(), mbeam, np.abs(psfhat))
        x, _ = cg(A,
                residual*mbeam,
                x0=x,
                tol=opts.cg_tol,
                maxiter=opts.cg_maxit)
    
    
        print("FISTA step", file=log)
        y = model + opts.gamma*np.array(x)
        value_and_grad = partial(stokes_energy, A, y)
        prox = partial(prox_21m_jax, weights21, 0)
        model, _ = fista(value_and_grad,
                        prox, 
                        model,
                        threshold,
                        maxit=opts.fista_maxit,
                        tol=opts.fista_tol,
                        L0=10,
                        report_freq=opts.report_freq)


        # write component model
        print(f"Writing model at iter {k+1} to "
              f"{basename}_{opts.suffix}_model.mds", file=log)
        try:
            coeffs, Ix, Iy, expr, params, fexpr = \
                fit_image_fscube(freq_out, model,
                                 wgt=wsums,
                                 nbasisf=int(nband),
                                 method='Legendre')
            # save interpolated dataset
            data_vars = {
                'coefficients': (('corr', 'par', 'comps'), coeffs),
            }
            coords = {
                'location_x': (('x',), Ix),
                'location_y': (('y',), Iy),
                'params': (('par',), params),  # already converted to list
                'times': (('t',), time_out),  # to allow rendering to original grid
                'freqs': (('f',), freq_out),
                'corr': (('corr',), corrs),
            }
            attrs = {
                'spec': 'genesis',
                'cell_rad_x': cell_rad,
                'cell_rad_y': cell_rad,
                'npix_x': nx,
                'npix_y': ny,
                'fexpr': fexpr,
                'center_x': dds[0].x0,
                'center_y': dds[0].y0,
                'flip_u': dds[0].flip_u,
                'flip_v': dds[0].flip_v,
                'flip_w': dds[0].flip_w,
                'ra': dds[0].ra,
                'dec': dds[0].dec,
                'parametrisation': expr  # already converted to str
            }

            coeff_dataset = xr.Dataset(data_vars=data_vars,
                                       coords=coords,
                                       attrs=attrs)
            coeff_dataset.to_zarr(f"{basename}_{opts.suffix}_model.mds",
                                  mode='w')
        except Exception as e:
            print(f"Exception {e} raised during model fit .", file=log)

        save_fits(np.mean(model, axis=0),
                  fits_oname + f'_{opts.suffix}_model_{k+1}.fits',
                  hdr_mfs)

        print(f'Computing residual', file=log)
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            resid, _ = compute_residual(ds_name,
                                     nx, ny,
                                     cell_rad, cell_rad,
                                     ds_name,
                                     model[b],
                                     nthreads=opts.nthreads,
                                     epsilon=opts.epsilon,
                                     do_wgridding=opts.do_wgridding,
                                     double_accum=opts.double_accum)
            residual[b] = resid
        residual /= wsum[None, :, None, None]
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs,
                  fits_oname + f'_{opts.suffix}_residual_{k+1}.fits',
                  hdr_mfs)

        # report rms and rmax inside mask
        rmsp = rms
        rms = np.std(residual_mfs, axis=(-2,-1))
        rmax = np.abs(residual_mfs).max(axis=(-2,-1))

        # these are not updated in compute_residual
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            attrs = {}
            attrs['rms'] = rms
            attrs['rmax'] = rmax
            attrs['niters'] = k+1
            ds = ds.assign_attrs(**attrs)
            ds.to_zarr(ds_name, mode='a', compute=True)

        if opts.threshold is None:
            threshold = opts.rmsfactor * rms

        for i, c in enumerate(corrs):
            print(f"Iter {k}: peak {c} residual = {rmax[i]:.3e}, rms = {rms[i]:.3e}",
                file=log)
        
        if (rmax <= threshold).all():
            print("Terminating because final threshold has been reached",
                  file=log)
            break

        if (rms > rmsp).any():
            diverge_count += 1
            if diverge_count > 3:
                print("Algorithm is diverging. Terminating.", file=log)
                break

    return