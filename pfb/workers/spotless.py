# flake8: noqa
from contextlib import ExitStack
import click
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
pfb_logging.init('pfb')
log = pfb_logging.get_logger('SPOTLESS')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@click.command(context_settings={'show_default': True})
@clickify_parameters(schema.spotless)
def spotless(**kw):
    '''
    Distributed spotless algorithm.
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

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import xds_from_url

    basename = f'{basedir}/{oname}'
    if opts.xds is not None:
        xds_store = DaskMSStore(opts.xds.rstrip('/'))
        xds_name = opts.xds
    else:
        xds_name = f'{basename}.xds'
        xds_store = DaskMSStore(xds_name)
        opts.xds = xds_name
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_name = f'{basename}_{opts.suffix}.dds'

    OmegaConf.set_struct(opts, True)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{opts.log_directory}/spotless_{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    # TODO - prettier config printing
    log.info('Input Options:')
    for key in opts.keys():
        log.info('     %25s = %s' % (key, opts[key]))

    with ExitStack() as stack:
        from pfb import set_client
        if opts.nworkers > 1:
            client = set_client(opts.nworkers, log, stack=stack,
                                direct_to_workers=opts.direct_to_workers,
                                client_log_level=opts.log_level)
            from distributed import as_completed
        else:
            log.info("Faking client")
            from pfb.utils.dist import fake_client
            client = fake_client()
            names = [0]
            as_completed = lambda x: x

        ti = time.time()
        _spotless(**opts)

        dds, dds_list = xds_from_url(dds_name)

        # convert to fits files
        futures = []
        if opts.fits_mfs or opts.fits_cubes:
            from pfb.utils.fits import dds2fits
            log.info(f"Writing fits files to {fits_oname}_{opts.suffix}")
            fut = client.submit(dds2fits,
                                dds_list,
                                'MODEL',
                                f'{fits_oname}_{opts.suffix}',
                                norm_wsum=False,
                                nthreads=opts.nthreads,
                                do_mfs=opts.fits_mfs,
                                do_cube=opts.fits_cubes)
            futures.append(fut)

            fut = client.submit(dds2fits,
                                dds_list,
                                'UPDATE',
                                f'{fits_oname}_{opts.suffix}',
                                norm_wsum=False,
                                nthreads=opts.nthreads,
                                do_mfs=opts.fits_mfs,
                                do_cube=opts.fits_cubes)
            futures.append(fut)

            fut = client.submit(dds2fits,
                                dds_list,
                                'RESIDUAL',
                                f'{fits_oname}_{opts.suffix}',
                                norm_wsum=True,
                                nthreads=opts.nthreads,
                                do_mfs=opts.fits_mfs,
                                do_cube=opts.fits_cubes)
            futures.append(fut)

            for fut in as_completed(futures):
                column = fut.result()
                log.info(f'Done writing {column}')

    log.info(f"All done after {time.time() - ti}s.")


def _spotless(**kw):
    '''
    Distributed spotless algorithm.

    The key inputs to the algorithm are the xds and mds.
    The xds contains the Stokes coherencies created with the init worker.
    There can be multiple datasets in the xds, each corresponding to a specific frequency and time range.
    These datasets are persisted on separate workers.
    The mds contains a continuous representation of the model in compressed format.
    The mds is just a single dataset which gets evaluated into a discretised cube on the runner node.
    The frequency resolution of the model is the same as the frequency resoltuion of the xds.

    '''
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from itertools import cycle
    import numpy as np
    import numba
    import xarray as xr
    from distributed import get_client
    from pfb.opt.power_method import power_method_dist as power_method
    from pfb.opt.primal_dual import primal_dual_dist as primal_dual
    from pfb.utils.naming import xds_from_url, get_opts, cache_opts
    from pfb.utils.fits import load_fits, dds2fits, set_wcs, save_fits
    from pfb.utils.dist import l1reweight_func
    from itertools import cycle
    from uuid import uuid4
    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
    import fsspec
    import pickle
    from pfb.utils.dist import band_actor

    # the runner should use all available resources
    import psutil
    ncpu = psutil.cpu_count(logical=False)
    numba.set_num_threads(ncpu)

    real_type = np.float64
    complex_type = np.complex128

    try:
        client = get_client()
        names = list(client.scheduler_info()['workers'].keys())
    except:
        from pfb.utils.dist import fake_client
        client = fake_client()
        names = [0]
        as_completed = lambda x: x

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    # xds contains vis products, no imaging weights applied
    xds_name = f'{basename}.xds' if opts.xds is None else opts.xds
    xds_store = DaskMSStore(xds_name)
    try:
        assert xds_store.exists()
    except Exception as e:
        log.error_and_raise(f"There must be a dataset at {xds_store.url}",
                            ValueError)

    xds, xds_list = xds_from_url(xds_store.url)

    # create dds and cache
    dds_name = opts.output_filename + f'_{opts.suffix}' + '.dds'
    dds_store = DaskMSStore(dds_name)
    if '://' in dds_store.url:
        protocol = xds_store.url.split('://')[0]
    else:
        protocol = 'file'

    if dds_store.exists() and opts.overwrite:
        log.info(f"Removing {dds_store.url}")
        dds_store.rm(recursive=True)

    optsp_name = dds_store.url + '/opts.pkl'
    fs = fsspec.filesystem(protocol)
    if dds_store.exists() and not opts.overwrite:
        # get opts from previous run
        optsp = get_opts(dds_store.url,
                         protocol,
                         name='opts.pkl')

        # check if we need to remake the data products
        verify_attrs = ['epsilon',
                        'do_wgridding', 'double_accum',
                        'field_of_view', 'super_resolution_factor']
        try:
            for attr in verify_attrs:
                assert optsp[attr] == opts[attr]

            from_cache = True
            log.info("Initialising from cached data products")
            dds, dds_list = xds_from_url(dds_store.url)
            iter0 = dds[0].niters
        except Exception as e:
            log.info(f'Cache verification failed on {attr}. '
                  'Will remake image data products')
            dds_store.rm(recursive=True)
            fs.makedirs(dds_store.url, exist_ok=True)
            # dump opts to validate cache on rerun
            cache_opts(opts,
                       dds_store.url,
                       protocol,
                       name='opts.pkl')
            from_cache = False
            iter0 = 0
    else:
        fs.makedirs(dds_store.url, exist_ok=True)
        cache_opts(opts,
                   dds_store.url,
                   protocol,
                   name='opts.pkl')
        from_cache = False
        log.info("Initialising from scratch.")
        iter0 = 0

    log.info(f"Image data products will be cached in {dds_store.url}")

    uv_max = 0
    max_freq = 0
    for ds in xds:
        uv_max = np.maximum(uv_max, ds.uv_max)
        max_freq = np.maximum(max_freq, ds.max_freq)

    # filter datasets by band
    xdsb = {}
    for ds in xds_list:
        idx = ds.find('band') + 4
        bid = int(ds[idx:idx+4])
        xdsb.setdefault(bid, [])
        xdsb[bid].append(ds)

    nband = len(xdsb.keys())

    # set up band actors
    log.info("Setting up actors")
    futures = []
    for wname, (bandid, dsl) in zip(cycle(names), xdsb.items()):
        f = client.submit(band_actor,
                          dsl,
                          opts,
                          bandid,
                          dds_store.url,
                          uv_max,
                          max_freq,
                          workers=wname,
                          key='actor-'+uuid4().hex,
                          actor=True,
                          pure=False)
        futures.append(f)
    actors = list(map(lambda f: f.result(), futures))
    futures = list(map(lambda a: a.get_image_info(), actors))
    results = list(map(lambda f: f.result(), futures))
    nx, ny, nymax, nxmax, cell_rad, ra, dec, x0, y0, freq_out, time_out = results[0]
    freq_out = [freq_out]
    for result in results[1:]:
        assert result[0] == nx
        assert result[1] == ny
        assert result[2] == nymax
        assert result[3] == nxmax
        assert result[4] == cell_rad
        assert result[5] == ra
        assert result[6] == dec
        assert result[7] == x0
        assert result[8] == y0
        freq_out.append(result[9])

    log.info(f"Image size set to ({nx}, {ny})")

    radec = [ra, dec]
    cell_deg = np.rad2deg(cell_rad)
    freq_out = np.array(freq_out)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    time_out = np.array((time_out,))

    # get size of dual domain
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)

    if opts.mds is not None:
        mds_name = opts.mds
    else:
        mds_name = opts.output_filename + f'_{opts.suffix}' + '.mds'

    mds_store = DaskMSStore(mds_name)

    # initialise model (assumed constant in time)
    if mds_store.exists():
        log.info(f"Loading model from {mds_store.url}")
        mds = xr.open_zarr(mds_store.url, chunks=None)

        # we only want to load these once
        model_coeffs = mds.coefficients.values
        locx = mds.location_x.values
        locy = mds.location_y.values
        params = mds.params.values
        coeffs = mds.coefficients.values

        model = np.zeros((nband, nx, ny))
        for i in range(nband):
            model[i] = eval_coeffs_to_slice(
                time_out[0],
                freq_out[i],
                model_coeffs,
                locx, locy,
                mds.parametrisation,
                params,
                mds.texpr,
                mds.fexpr,
                mds.npix_x, mds.npix_y,
                mds.cell_rad_x, mds.cell_rad_y,
                mds.center_x, mds.center_y,
                nx, ny,
                cell_rad, cell_rad,
                x0, y0
            )

    else:
        model = np.zeros((nband, nx, ny))

    log.info("Computing image data products")
    futures = []
    for b in range(nband):
        fut = actors[b].set_image_data_products(model[b],
                                                iter0,
                                                from_cache=from_cache)
        futures.append(fut)


    results = list(map(lambda f: f.result(), futures))
    residual_mfs = np.sum([r[0] for r in results], axis=0)
    wsums = np.array([r[1] for r in results])
    wsum = np.sum(wsums)
    fsel = wsums > 0
    futures = list(map(lambda a: a.set_wsum(wsum), actors))
    results = list(map(lambda f: f.result(), futures))

    residual_mfs /= wsum
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    if iter0 == 0:
        save_fits(residual_mfs,
                  fits_oname + f'_{opts.suffix}_dirty_mfs.fits',
                  hdr_mfs)

    if opts.hess_norm is None:
        log.info('Getting spectral norm of Hessian approximation')
        hess_norm = power_method(actors, nx, ny, nband,
                                tol=opts.pm_tol,
                                maxit=opts.pm_maxit,
                                report_freq=opts.pm_report_freq,
                                verbosity=opts.pm_verbose)
        # inflate so we don't have to recompute after each L2 reweight
        hess_norm *= 1.05
    else:
        hess_norm = opts.hess_norm
    log.info(f'hess-norm = {hess_norm:.3e}')

    # a value less than zero turns L1 reweighting off
    # we'll start on convergence or at the iteration
    # indicated by l1_reweight_from, whichever comes first
    l1_reweight_from = opts.l1_reweight_from
    l1reweight_active = False

    if l1_reweight_from == 0:
        log.info(f'Initialising with L1 reweighted')
        l1weight, rms_comps = l1reweight_func(actors,
                                   opts.rmsfactor,
                                   alpha=opts.alpha)
        l1reweight_active = True
    else:
        rms_comps = rms  # this should not happen
        l1weight = np.ones((nbasis, nymax, nxmax), dtype=real_type)
        l1reweight_active = False

    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count = 0
    l2_reweights = 0
    log.info(f"It {iter0}: max resid = {rmax:.3e}, rms = {rms:.3e}")
    for k in range(iter0, iter0 + opts.niter):
        log.info('Solving for update')
        futures = list(map(lambda a: a.cg_update(), actors))
        update_mfs = np.sum(list(map(lambda f: f.result(), futures)), axis=0)
        update_mfs /= np.sum(fsel)

        save_fits(update_mfs,
                  fits_oname + f'_{opts.suffix}_update_{k+1}.fits',
                  hdr_mfs)

        log.info('Solving for model')
        primal_dual(actors,
                    rms*opts.rmsfactor,
                    hess_norm,
                    l1weight,
                    opts.rmsfactor,
                    rms_comps,
                    opts.alpha,
                    nu=len(bases),
                    tol=opts.pd_tol,
                    maxit=opts.pd_maxit,
                    positivity=opts.positivity,
                    gamma=opts.gamma,
                    verbosity=opts.pd_verbose)

        log.info('Updating model')
        futures = list(map(lambda a: a.give_model(), actors))
        results = list(map(lambda f: f.result(), futures))

        bandids = [r[1] for r in results]
        modelp = model.copy()
        for i, b in enumerate(bandids):
            model[b] = results[i][0]

        if np.isnan(model).any():
            log.error_and_raise('Model is nan', RuntimeError)

        save_fits(np.mean(model[fsel], axis=0),
                  fits_oname + f'_{opts.suffix}_model_{k+1}.fits',
                  hdr_mfs)

        log.info(f"Writing model to {mds_store.url}")
        coeffs, Ix, Iy, expr, params, texpr, fexpr = \
            fit_image_cube(time_out, freq_out[fsel], model[None, fsel, :, :],
                            wgt=wsums[None, fsel],
                            method='Legendre')

        data_vars = {
            'coefficients': (('par', 'comps'), coeffs),
        }
        coords = {
            'location_x': (('x',), Ix),
            'location_y': (('y',), Iy),
            'params': (('par',), params),  # already converted to list
            'times': (('t',), time_out),  # to allow rendering to original grid
            'freqs': (('f',), freq_out)
        }
        attrs = {
            'spec': 'genesis',
            'cell_rad_x': cell_rad,
            'cell_rad_y': cell_rad,
            'npix_x': nx,
            'npix_y': ny,
            'texpr': texpr,
            'fexpr': fexpr,
            'center_x': x0,
            'center_y': y0,
            'ra': ra,
            'dec': dec,
            'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
            'parametrisation': expr,  # already converted to str
        }

        coeff_dataset = xr.Dataset(data_vars=data_vars,
                            coords=coords,
                            attrs=attrs)
        coeff_dataset.to_zarr(f"{mds_store.url}", mode='w')

        # convergence check
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)
        if eps < opts.tol:
            # do not converge prematurely
            if l1_reweight_from > 0 and not l1reweight_active:  # only happens once
                # start reweighting
                l1_reweight_from = k+1 - iter0
                # don't start L2 reweighting before L1 reweighting
                # if it is enabled
                dof = None
                l1reweight_active = True
            elif l2_reweights < opts.max_l2_reweight:
                # L1 reweighting has already kicked in and we have
                # converged again so perform L2 reweight
                dof = opts.l2_reweight_dof
            else:
                # we have reached max L2 reweights so we are done
                log.info(f"Converged after {k+1} iterations.")
                break
        else:
            # do not perform an L2 reweight unless the M step has converged
            dof = None

        if k+1 - iter0 >= opts.l2_reweight_from and l2_reweights < opts.max_l2_reweight:
            dof = opts.l2_reweight_dof

        if dof is not None:
            log.info('Recomputing image data products since L2 reweight '
                  'is required.')
            futures = []
            for b in range(nband):
                fut = actors[b].set_image_data_products(model[b],
                                                        k+1,
                                                        dof=dof)
                futures.append(fut)


            results = list(map(lambda f: f.result(), futures))
            residual_mfs = np.sum([r[0] for r in results], axis=0)
            wsums = np.array([r[1] for r in results])
            wsum = np.sum(wsums)
            futures = list(map(lambda a: a.set_wsum(wsum), actors))
            results = list(map(lambda f: f.result(), futures))

            residual_mfs /= wsum

            # don't keep the cubes on the runner
            del results

            # # TODO - how much does hess-norm change after L2 reweight?
            # log.info('Getting spectral norm of Hessian approximation')
            # hess_norm = power_method(actors, nx, ny, nband)
            # log.info(f'hess-norm = {hess_norm:.3e}')

            l2_reweights += 1
        else:
            # compute normal residual, no need to redo PSF etc
            log.info('Computing residual')
            futures = list(map(lambda a: a.set_residual(k+1), actors))
            resids = list(map(lambda f: f.result(), futures))
            # we never resids by wsum inside the worker
            residual_mfs = np.sum(resids, axis=0)/wsum


        save_fits(residual_mfs,
                  fits_oname + f'_{opts.suffix}_residual_{k+1}.fits',
                  hdr_mfs)

        rmsp = rms
        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()

        # base this on rmax?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        log.info(f"It {k+1}: max resid = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}")

        if k+1 - iter0 >= l1_reweight_from:
            log.info('L1 reweighting')
            l1weight, rms_comps = l1reweight_func(actors,
                                       opts.rmsfactor,
                                       alpha=opts.alpha)

        if rms > rmsp:
            diverge_count += 1
            if diverge_count > opts.diverge_count:
                log.info("Algorithm is diverging. Terminating.")
                break

    return

