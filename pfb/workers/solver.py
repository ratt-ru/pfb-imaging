# flake8: noqa
import concurrent.futures as cf
from pfb.workers.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('SARA')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.sara)
def sara(**kw):
    '''
    Deconvolution using some type of convex regularizer sara.

    '''

    # Import options in OmegaConf object from kwargs dict
    opts = OmegaConf.create(kw)

    # Make sure base folders exist and set default naming conventions
    from pfb.utils.naming import set_output_names
    opts, _ , oname = set_output_names(opts)


    # Instantiate nthreads and ncpu
    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        opts.nthreads = nthreads//2
        ncpu = ncpu//2

    
    # No more opts can be added by writting into opts.
    OmegaConf.set_struct(opts, True)

    
    # Environment variables setup (mostly for parralelization using numba)
    from pfb import set_envs
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)


    # Initialize log file and timestamps
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/sara_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)
    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)


    # Running sara from DDS
    from pfb.utils.naming import xds_from_url
    basename = opts.output_filename
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_name = f'{basename}_{opts.suffix}.dds'

    ti = time.time()
    _sara(**opts)

    _, dds_list = xds_from_url(dds_name)


    # Writing FITS Files
    if opts.fits_mfs or opts.fits:
        from pfb.utils.fits import dds2fits
        print(f"Writing fits files to {fits_oname}_{opts.suffix}",
                file=log)

        dds2fits(dds_list,
                'RESIDUAL',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=True,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes)
        print('Done writing RESIDUAL', file=log)
        dds2fits(dds_list,
                'MODEL',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=False,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes)
        print('Done writing MODEL', file=log)
        dds2fits(dds_list,
                'UPDATE',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=False,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes)
        print('Done writing UPDATE', file=log)
        # try:
        #     dds2fits(dds_list,
        #          'MPSF',
        #          f'{fits_oname}_{opts.suffix}',
        #          norm_wsum=False,
        #          nthreads=opts.nthreads,
        #          do_mfs=opts.fits_mfs,
        #          do_cube=opts.fits_cubes)
        #     print('Done writing MPSF', file=log)
        # except Exception as e:
        #     print(e)

    print(f"All done after {time.time() - ti}s", file=log)


def _sara(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    # Imports
    from functools import partial
    import numpy as np
    import xarray as xr
    import numexpr as ne
    from pfb.utils.fits import set_wcs, save_fits, load_fits
    from pfb.utils.naming import xds_from_url, xds_from_list
    from pfb.opt.power_method import power_method
    from pfb.opt.pcg import pcg
    from pfb.opt.primal_dual import primal_dual_optimised as primal_dual
    from pfb.utils.misc import l1reweight_func
    from pfb.operators.hessian import hess_psf
    from pfb.operators.psi import Psi
    from pfb.operators.gridder import compute_residual
    from copy import copy, deepcopy
    from ducc0.misc import empty_noncritical, thread_pool_size
    from pfb.prox.prox_21m import prox_21m_numba as prox_21
    # from pfb.prox.prox_21 import prox_21
    from pfb.utils.misc import fitcleanbeam, fit_image_cube, eval_coeffs_to_slice
    from daskms.fsspec_store import DaskMSStore
    from ducc0.fft import c2c

    # Reading DDS
    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds, dds_list = xds_from_url(dds_name)

    
    # Reading image size (nx,ny)
    nx, ny = dds[0].x.size, dds[0].y.size

    # Getting all times and freqs from all bands
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)

    # Looking at unique values to get number of bands
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))

    if time_out.size > 1:
        raise NotImplementedError('Only static models currently supported')
    nband = freq_out.size




    # drop_vars after access to avoid duplicates in memory
    # and avoid unintentional side effects?

    # Reading RESIDUAL, MODEL and UPDATE if exist else setting default.
    if 'RESIDUAL' in dds[0]:
        residual = np.stack([ds.RESIDUAL.values for ds in dds], axis=0)
        dds = [ds.drop_vars('DIRTY') for ds in dds]
        dds = [ds.drop_vars('RESIDUAL') for ds in dds]
    else:
        residual = np.stack([ds.DIRTY.values for ds in dds], axis=0)
        dds = [ds.drop_vars('DIRTY') for ds in dds]

    if 'MODEL' in dds[0]:
        model = np.stack([ds.MODEL.values for ds in dds], axis=0)
        dds = [ds.drop_vars('MODEL') for ds in dds]
    else:
        model = np.zeros((nband, nx, ny))


    if 'UPDATE' in dds[0]:
        update = np.stack([ds.UPDATE.values for ds in dds], axis=0)
        dds = [ds.drop_vars('UPDATE') for ds in dds]
    else:
        update = np.zeros((nband, nx, ny))


    # Reading PSF, BEAM and current weigths.
    abspsf = np.stack([np.abs(ds.PSFHAT.values) for ds in dds], axis=0)
    dds = [ds.drop_vars('PSFHAT') for ds in dds]
    beam = np.stack([ds.BEAM.values for ds in dds], axis=0)
    

    # Applying weights to residual, model and psf. 
    wsums = np.stack([ds.wsum for ds in dds], axis=0) # sum of weights per bands
    fsel = wsums > 0  # keep track of empty bands
    wsum = np.sum(wsums) # global sum of weights 
    wsums /= wsum # ?
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)
    model_mfs = np.mean(model[fsel], axis=0)
    abspsf /= wsum 


    # Reading from first band : RA, DEC, cell sizes and ref frequencies. 
    # Setting FITS header for intermediary results
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq) 

    # If from previous run, get current niter else starting from 0.
    if 'niters' in dds[0].attrs:
        iter0 = dds[0].niters
    else:
        iter0 = 0

    # Allow calling with pd_tol as float,
    # (pd_tol is either a float or a list of float giving the primal dual tolerance for convergence per major iteration).
    try:
        ntol = len(opts.pd_tol)
        pd_tol = opts.pd_tol
    except TypeError:
        assert isinstance(opts.pd_tol, float)
        ntol = 1
        pd_tol = [opts.pd_tol]


    niters = opts.niter
    if ntol <= niters:
        pd_tolf = pd_tol[-1] # pd_tolf is the final tolerance.
        pd_tol += [pd_tolf]*niters  # no harm in too many.

    # Image space hessian.
    # Pre-allocate arrays for doing FFT's.
    real_type = 'f8'

    # Defining the preconditionner.
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
            print(f"Using previously estimated hess_norm of {hess_norm:.3e}",
                  file=log)
        else:
            print("Finding spectral norm of Hessian approximation", file=log)
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
        print(f"Using provided hess-norm of beta = {hess_norm:.3e}", file=log)



    # Setting up the dictionary.
    print("Setting up dictionary", file=log)
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)
    psi = Psi(nband, nx, ny, bases, opts.nlevels, opts.nthreads)
    Nxmax = psi.Nxmax
    Nymax = psi.Nymax

    print(f"Using {psi.nthreads_per_band} numba threads for each band", file=log)
    print(f"Using {thread_pool_size()} threads for gridding", file=log)


    # Setting up the number of frequency basis functions.
    if opts.nbasisf is None:
        nbasisf = int(np.sum(fsel))
    else:
        nbasisf = opts.nbasisf
    print(f"Using {nbasisf} frequency basis functions", file=log)


    # Setting up L1 reweighting.
    ### a value less than zero turns L1 reweighting off
    ### we'll start on convergence or at the iteration
    ### indicated by l1-reweight-from, whichever comes first
    ### we need an array to put the components in for reweighting
    l1_reweight_from = opts.l1_reweight_from
    l1reweight_active = False
    outvar = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=real_type)
    if l1_reweight_from == 0:
        print('Initialising with L1 reweighted', file=log)
        if not update.any():
            raise ValueError("Cannot reweight before any updates have been performed")
        psi.dot(update, outvar)
        tmp = np.sum(outvar, axis=0)
        # exclude zeros from padding DWT's
        # rms_comps = np.std(tmp[tmp!=0])
        # print(f'rms_comps updated to {rms_comps}', file=log)
        # per basis rms_comps
        rms_comps = np.ones((nbasis,), dtype=float)
        for i, base in enumerate(bases):
            tmpb = tmp[i]
            rms_comps[i] = np.std(tmpb[tmpb!=0])
            print(f'rms_comps for base {base} is {rms_comps[i]}',
                    file=log)
        reweighter = partial(l1reweight_func,
                             psiH=psi.dot,
                             outvar=outvar,
                             rmsfactor=opts.rmsfactor,
                             rms_comps=rms_comps,
                             alpha=opts.alpha)
        l1weight = reweighter(model)
        l1reweight_active = True
    else:
        l1weight = np.ones((nbasis, Nymax, Nxmax), dtype=residual.dtype)
        reweighter = None
        l1reweight_active = False


    # Compute the RMS from the RESIDUAL. 
    ### If rms_outside_model will compute std by using exclusively pixels where model is 0.
    ### Computing max RMS as well.
    if opts.rms_outside_model and model.any():
        rms_mask = model_mfs == 0
        rms = np.std(residual_mfs[rms_mask])
    else:
        rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()

    # Initializing hyper parameters.
    diverge_count = 0
    eps = 1.0
    write_futures = None
    print(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    if opts.skip_model:
        mrange = []
    else: # If worker ran after iter0 iteration already done.
        mrange = range(iter0, iter0 + opts.niter)


    # Initializing dual variable.
    dual = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=residual.dtype)

    # Major loop.
    for k in mrange:

    ## Compute regularizer hyperparameter.
        if iter0 == 0:
            lam = opts.init_factor * opts.rmsfactor * rms
        else:
            lam = opts.rmsfactor*rms
        print(f'Solving for model with lambda = {lam}', file=log)
    

    ## Compute update (gradient descent on data fitting term).
        print('Solving for update', file=log)
        residual *= beam  # avoid copy
        update = precond.idot(residual,
                              mode=opts.hess_approx,
                              x0=update if update.any() else None)
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs,
                  fits_oname + f'_{opts.suffix}_update_{k+1}.fits',
                  hdr_mfs)


    ## Compute Prox of regularizer using a primal dual algorithm.
        modelp = deepcopy(model)
        xtilde = model + opts.gamma * update
        grad21 = lambda x: -precond.dot(xtilde - x)/opts.gamma
        model, dual = primal_dual(model,
                                  dual,
                                  lam,
                                  psi.hdot,
                                  psi.dot,
                                  hess_norm,
                                  prox_21,
                                  l1weight,
                                  reweighter,
                                  grad21,
                                  nu=nbasis,
                                  positivity=opts.positivity,
                                  tol=pd_tol[k-iter0],
                                  maxit=opts.pd_maxit,
                                  verbosity=opts.pd_verbose,
                                  report_freq=opts.pd_report_freq,
                                  gamma=opts.gamma)

    ## Interpolate Multi frequency model, compute residual and write in MDS and FITS.
        print(f"Writing model to {basename}_{opts.suffix}_model.mds",
              file=log)
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
            print(f"Exception {e} raised during model fit .", file=log)


        model_mfs = np.mean(model[fsel], axis=0)
        save_fits(model_mfs,
                  fits_oname + f'_{opts.suffix}_model_{k+1}.fits',
                  hdr_mfs)


        # make sure write futures have finished
        if write_futures is not None:
            cf.wait(write_futures)

        print(f'Computing residual', file=log)
        write_futures = []
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            residual[b], fut = compute_residual(
                                    ds_name,
                                    nx, ny,
                                    cell_rad, cell_rad,
                                    ds_name,
                                    model[b],
                                    nthreads=opts.nthreads,
                                    epsilon=opts.epsilon,
                                    do_wgridding=opts.do_wgridding,
                                    double_accum=opts.double_accum,
                                    verbosity=opts.verbosity)
            write_futures.append(fut)

        residual /= wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs,
                  fits_oname + f'_{opts.suffix}_residual_{k+1}.fits',
                  hdr_mfs)
        
    ## Compute new RMS and check convergence (divergence) of algorithm.
        rmsp = rms
        if opts.rms_outside_model:
            rms_mask = model_mfs == 0
            rms = np.std(residual_mfs[rms_mask])
        else:
            rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        # what to base this on?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        print(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)
        
        # these are not updated in compute_residual
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            ds['UPDATE'] = (('x', 'y'), update[b])
            # don't write unecessarily
            for var in ds.data_vars:
                if var != 'UPDATE':
                    ds = ds.drop_vars(var)

            if (model==best_model).all():
                ds['MODEL_BEST'] = (('x', 'y'), best_model[b])

            attrs = {}
            attrs['rms'] = best_rms
            attrs['rmax'] = best_rmax
            attrs['niters'] = k+1
            attrs['hess_norm'] = hess_norm
            ds = ds.assign_attrs(**attrs)

            with cf.ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(ds.to_zarr, ds_name, mode='a')
                write_futures.append(fut)



        if eps < opts.tol:
            # do not converge prematurely
            if l1_reweight_from > 0 and not l1reweight_active:  # only happens once
                # start reweighting
                l1_reweight_from = k+1 - iter0
                l1reweight_active = True
            else:
                print(f"Converged after {k+1} iterations.", file=log)
                break

        if (k+1 - iter0 >= l1_reweight_from) and (k+1 - iter0 < opts.niter):
            print('Computing L1 weights', file=log)
            psi.dot(update, outvar)
            tmp = np.sum(outvar, axis=0)
            # exclude zeros from padding DWT's
            # rms_comps = np.std(tmp[tmp!=0])
            rms_comps = np.ones((nbasis,), dtype=float)
            # print(f'rms_comps updated to {rms_comps}', file=log)
            for i, base in enumerate(bases):
                tmpb = tmp[i]
                rms_comps[i] = np.std(tmpb[tmpb!=0])
                print(f'rms_comps for base {base} is {rms_comps[i]}',
                      file=log)
            reweighter = partial(l1reweight_func,
                                 psiH=psi.dot,
                                 outvar=outvar,
                                 rmsfactor=opts.rmsfactor,
                                 rms_comps=rms_comps,
                                 alpha=opts.alpha)
            l1weight = reweighter(model)
            l1reweight_active = True

        if rms > rmsp:
            diverge_count += 1
            if diverge_count > opts.diverge_count:
                print("Algorithm is diverging. Terminating.", file=log)
                break

    # make sure write futures have finished
    cf.wait(write_futures)

    return
