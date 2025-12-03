# flake8: noqa
import concurrent.futures as cf
import click
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema
from pfb.opt.ista_fb import ISTA

log = pfb_logging.get_logger('ISTA')


@click.command(context_settings={'show_default': True})
@clickify_parameters(schema.ista)
def ista(**kw):
    '''
    Deconvolution using ista regularisation
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
    logname = f'{str(opts.log_directory)}/ista{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    pfb_logging.log_options_dict(log, opts)

    from pfb.utils.naming import xds_from_url, get_opts

    basename = opts.output_filename
    print(basename)
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_name = f'{basename}_{opts.suffix}.dds'

    ti = time.time()
    _ista(**opts)

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
        log.info(f"Writing fits files to {fits_oname}_{opts.suffix}")

        dds2fits(dds_list,
                'RESIDUAL',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=True,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes,
                psfpars_mfs=psfpars_mfs)
        log.info('Done writing RESIDUAL')
        dds2fits(dds_list,
                'MODEL',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=False,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes,
                psfpars_mfs=psfpars_mfs)
        log.info('Done writing MODEL')
        dds2fits(dds_list,
                'UPDATE',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=False,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes,
                psfpars_mfs=psfpars_mfs)
        log.info('Done writing UPDATE')

    from numba import threading_layer
    log.info(f"Numba use the {threading_layer()} threading layer")
    log.info(f"All done after {time.time() - ti}s")


def _ista(**kw):
    """
    # gradient step (major cycle)
    nabla f(x) = I^D - R.H W R x  # needs full Hessian, R = degrid, R.H = grid, W = weights

    # forward step (invert Hessian using PCG)
    tilde{x} = x_k + gamma delta,   where   delta = -U^{-1} nabla f(x)

    # backward step (primal dual)
    U = A.H Z.H F.H \hat{I} F Z A + sigma**2 I
    x_{k+1} = prox^U_{gamma r}(x) = argmin_x r(x) + frac{1}{2\gamma}(tilde{x} - x) U (tilde{x} - x)

    """
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from functools import partial
    import numpy as np
    import xarray as xr
    import numexpr as ne
    from pfb.utils.fits import set_wcs, save_fits
    from pfb.utils.naming import xds_from_url
    from pfb.opt.power_method import power_method
    from pfb.opt.primal_dual import primal_dual_optimised as primal_dual
    from pfb.utils.misc import l1reweight_func
    from pfb.operators.hessian import hess_psf
    from pfb.operators.psi import Psi
    from pfb.operators.gridder import compute_residual
    from copy import deepcopy
    from ducc0.misc import thread_pool_size
    # replace with ista prox
    # from pfb.prox.prox_21m import prox_21m_numba as prox_21
    from pfb.utils.modelspec import fit_image_cube, eval_coeffs_to_slice
    
    ## Opening the DDS

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds, dds_list = xds_from_url(dds_name)

    if dds[0].corr.size > 1:
        log.error_and_raise("Joint polarisation deconvolution not "
                            "yet supported for ista algorithm",
                            NotImplementedError)
    ## Reading the image size
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
    log.info(freq_out)

    if time_out.size > 1:
        log.error_and_raise('Only static models currently supported',
                            NotImplementedError)

    nband = freq_out.size # number of independant 

    # drop_vars after access to avoid duplicates in memory
    # and avoid unintentional side effects?
    if 'RESIDUAL' in dds[0]:
        residual = np.stack([ds.RESIDUAL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars('DIRTY') for ds in dds]
        dds = [ds.drop_vars('RESIDUAL') for ds in dds]
    else:
        residual = np.stack([ds.DIRTY.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars('DIRTY') for ds in dds]
    if 'MODEL' in dds[0]:
        model = np.stack([ds.MODEL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars('MODEL') for ds in dds]
    else:
        model = np.zeros((nband, nx, ny))
    if 'UPDATE' in dds[0]:
        update = np.stack([ds.UPDATE.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars('UPDATE') for ds in dds]
    else:
        update = np.zeros((nband, nx, ny))
    abspsf = np.stack([np.abs(ds.PSFHAT.values[0]) for ds in dds], axis=0)
    dds = [ds.drop_vars('PSFHAT') for ds in dds]
    beam = np.stack([ds.BEAM.values[0] for ds in dds], axis=0)
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands

    wsum = np.sum(wsums)
    wsums /= wsum
    abspsf /= wsum
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)
    model_mfs = np.mean(model[fsel], axis=0)

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

    # image space hessian
    precond = hess_psf(nx, ny, abspsf,
                       beam=beam,
                       eta=opts.eta*wsums,
                       nthreads=opts.nthreads,
                       cgtol=opts.cg_tol,
                       cgmaxit=opts.cg_maxit,
                       cgverbose=opts.cg_verbose,
                       cgrf=opts.cg_report_freq,
                       taper_width=np.minimum(int(0.1*nx), 32))

    if opts.hess_norm is None:
        # if the grid worker had been rerun hess_norm won't be in attrs
        if 'hess_norm' in dds[0].attrs:
            hess_norm = dds[0].hess_norm
            log.info(f"Using previously estimated hess_norm of {hess_norm:.3e}")
        else:
            log.info("Finding spectral norm of Hessian approximation")
            hess_norm, hessbeta = power_method(
                                            precond.dot, (nband, nx, ny),
                                            tol=opts.pm_tol,
                                            maxit=opts.pm_maxit,
                                            verbosity=opts.pm_verbose,
                                            report_freq=opts.pm_report_freq)
            # inflate slightly for stability
            hess_norm *= 1.05
    else:
        hess_norm = opts.hess_norm
        log.info(f"Using provided hess-norm of = {hess_norm:.3e}")

    log.info(f"Using {thread_pool_size()} threads for gridding")
    # number of frequency basis functions
    if opts.nbasisf is None:
        nbasisf = int(np.sum(fsel))
    else:
        nbasisf = opts.nbasisf
    log.info(f"Using {nbasisf} frequency basis functions")

    if opts.rms_outside_model and model.any():
        rms_mask = model_mfs == 0
        rms = np.std(residual_mfs[rms_mask])
    else:
        rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count = 0
    eps = 1.0
    write_futures = None
    log.info(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}")
    if opts.skip_model:
        mrange = []
    else:
        mrange = range(iter0, iter0 + opts.niter)
    for k in mrange:
        log.info('Solving for update')
        residual *= beam  # avoid copy
        update = precond.idot(residual,
                              mode=opts.hess_approx,
                              x0=update if update.any() else None)
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs,
                  fits_oname + f'_{opts.suffix}_update_{k+1}.fits',
                  hdr_mfs)

        modelp = deepcopy(model)
        xtilde = model + opts.gamma * update
        if iter0 == 0:
            lam = opts.init_factor * opts.rmsfactor * rms
        else:
            lam = opts.rmsfactor*rms
        log.info(f'Solving for model with lambda = {lam}')
        prox_solver = ISTA(lmbda=5e-7,
                            gamma=hess_norm,
                            max_iter=20,
                            step_size=1,
                            precond=precond,
        )

        model = prox_solver(xtilde)

        # write component model
        log.info(f"Writing model to {basename}_{opts.suffix}_model.mds")
        try:
            coeffs, Ix, Iy, expr, params, texpr, fexpr = \
                fit_image_cube(time_out,
                               freq_out[fsel],
                               model[None, fsel, :, :],
                               wgt=wsums[None, fsel],
                               nbasisf=nbasisf,
                               method='Legendre',
                               sigmasq=1e-6)
            # save interpolated dataset
            data_vars = {
                'coefficients': (('par', 'comps'), coeffs),
            }
            coords = {
                'location_x': (('x',), Ix),
                'location_y': (('y',), Iy),
                # 'shape_x':,
                'params': (('par',), params),  # already converted to list
                'times': (('t',), time_out),  # to allow rendering to original grid
                'freqs': (('f',), freq_out)
            }
            mattrs = {
                'spec': 'genesis',
                'cell_rad_x': cell_rad,
                'cell_rad_y': cell_rad,
                'npix_x': nx,
                'npix_y': ny,
                'texpr': texpr,
                'fexpr': fexpr,
                'center_x': dds[0].x0,
                'center_y': dds[0].y0,
                'flip_u': dds[0].flip_u,
                'flip_v': dds[0].flip_v,
                'flip_w': dds[0].flip_w,
                'ra': dds[0].ra,
                'dec': dds[0].dec,
                'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
                'parametrisation': expr  # already converted to str
            }
            for key, val in opts.items():
                if key == 'pd_tol':
                    mattrs[key] = pd_tolf
                else:
                    mattrs[key] = val

            coeff_dataset = xr.Dataset(data_vars=data_vars,
                               coords=coords,
                               attrs=mattrs)
            coeff_dataset.to_zarr(f"{basename}_{opts.suffix}_model.mds",
                                  mode='w')

            for b in range(nband):
                model[b] = eval_coeffs_to_slice(
                        time_out[0],
                        freq_out[b],
                        coeffs,
                        Ix, Iy,
                        expr,
                        params,
                        texpr,
                        fexpr,
                        nx, ny,
                        cell_rad, cell_rad,
                        dds[0].x0, dds[0].y0,
                        nx, ny,
                        cell_rad, cell_rad,
                        dds[0].x0, dds[0].y0
                )
        except Exception as e:
            log.info(f"Exception {e} raised during model fit .")
        
        model = model[np.newaxis, ...] if len(model.shape) == 2 else model
        model_mfs = np.mean(model[fsel], axis=0) 
        save_fits(model_mfs,
                  fits_oname + f'_{opts.suffix}_model_{k+1}.fits',
                  hdr_mfs)

        # make sure write futures have finished
        if write_futures is not None:
            cf.wait(write_futures)

        log.info(f'Computing residual')
        write_futures = []
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            resid, fut = compute_residual(
                                    ds_name,
                                    nx, ny,
                                    cell_rad, cell_rad,
                                    ds_name,
                                    model[b][None, :, :],  # add corr axis
                                    nthreads=opts.nthreads,
                                    epsilon=opts.epsilon,
                                    do_wgridding=opts.do_wgridding,
                                    double_accum=opts.double_accum,
                                    verbosity=opts.verbosity)
            write_futures.append(fut)
            residual[b] = resid[0]  # remove corr axis

        residual /= wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs,
                  fits_oname + f'_{opts.suffix}_residual_{k+1}.fits',
                  hdr_mfs)
        rmsp = rms
        if opts.rms_outside_model:
            rms_mask = model_mfs == 0
            rms = np.std(residual_mfs[rms_mask])
        else:
            rms = np.std(residual_mfs)
        rmaxp = rmax
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        # what to base this on?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        # these are not updated in compute_residual
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            ds['UPDATE'] = (('corr', 'x', 'y'), update[b][None, :, :])
            # don't write unecessarily
            for var in ds.data_vars:
                if var != 'UPDATE':
                    ds = ds.drop_vars(var)

            if (model==best_model).all():
                ds['MODEL_BEST'] = (('corr', 'x', 'y'), best_model[b][None, :, :])

            attrs = {}
            attrs['rms'] = best_rms
            attrs['rmax'] = best_rmax
            attrs['niters'] = k+1
            attrs['hess_norm'] = hess_norm
            ds = ds.assign_attrs(**attrs)


            with cf.ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(ds.to_zarr, ds_name, mode='a')
                write_futures.append(fut)


        log.info(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}")


        if (rms > rmsp) and (rmax > rmaxp):
            diverge_count += 1
            if diverge_count > opts.diverge_count:
                log.info("Algorithm is diverging. Terminating.")
                break

    # make sure write futures have finished
    cf.wait(write_futures)

    return
