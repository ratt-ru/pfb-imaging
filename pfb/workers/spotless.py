# flake8: noqa
import os
import sys
from pathlib import Path
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('SPOTLESS')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.spotless["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.spotless["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.spotless)
def spotless(**kw):
    '''
    Distributed spotless algorithm.
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    OmegaConf.set_struct(opts, True)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{str(ldir)}/spotless_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/spotless_{timestamp}.log', file=log)
    basedir = Path(opts.output_filename).resolve().parent
    basedir.mkdir(parents=True, exist_ok=True)
    basename = f'{opts.output_filename}_{opts.product.upper()}'
    dds_name = f'{basename}_{opts.suffix}.dds'

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        os.environ["OMP_NUM_THREADS"] = str(opts.nthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nthreads)
        os.environ["MKL_NUM_THREADS"] = str(opts.nthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nthreads)
        paths = sys.path
        ppath = [paths[i] for i in range(len(paths)) if 'pfb/bin' in paths[i]]
        if len(ppath):
            ldpath = ppath[0].replace('bin', 'lib')
            ldcurrent = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ["LD_LIBRARY_PATH"] = f'{ldpath}:{ldcurrent}'
            # TODO - should we fall over in else?
        os.environ["NUMBA_NUM_THREADS"] = str(opts.nthreads)

        import numexpr as ne
        max_cores = ne.detect_number_of_cores()
        ne_threads = min(max_cores, opts.nthreads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)
        import dask
        dask.config.set(**{'array.slicing.split_large_chunks': False})

        # set up client
        host_address = opts.host_address or os.environ.get("DASK_SCHEDULER_ADDRESS")
        if host_address is not None:
            from distributed import Client
            print("Initialising distributed client.", file=log)
            client = stack.enter_context(Client(host_address))
        else:
            from dask.distributed import Client, LocalCluster
            print("Initialising client with LocalCluster.", file=log)
            cluster = LocalCluster(processes=True,
                                    n_workers=opts.nworkers,
                                    threads_per_worker=opts.nthreads_dask,
                                    memory_limit=0,  # str(mem_limit/nworkers)+'GB'
                                    asynchronous=False)
            cluster = stack.enter_context(cluster)
            client = stack.enter_context(Client(cluster,
                                                direct_to_workers=True))

        client.wait_for_workers(opts.nworkers)
        dashboard_url = client.dashboard_link
        print(f"Dask Dashboard URL at {dashboard_url}", file=log)
        client.amm.stop()
        _spotless(**opts)

    print("All done here.", file=log)


def _spotless(**kw):
    '''
    Distributed spotless algorithm.

    The key inputs to the algorithm are the xds and mds.
    The xds contains the Stokes coeherencies created with the imit worker.
    There can be multiple datasets in the xds, each corresponding to a specific frequency and time range.
    These datasets are persisted on separate workers.
    The mds contains a continuous representation of the model in compressed format.
    The mds is just a single dataset which gets evaluated into a discretised cube on the runner node.
    For now the frequency resolution of the model is the same as the frequency resoltuion of the xds.
    It is assumed that the model is static in time so a per band model potentially speaks to many datasets in the xds.
    This makes it possible to approximately incorporate time variable instrumental effects during deconvolution.
    Thus the measurement model for each band potentially consists of different data terms i.e.

    [d1, d2, d3, ...] = [R1, R2, R3, ...] x + [eps1, eps2, eps3, ...]

    The likelihood is therefore a sum over data terms

    chi2 = sum_i (di - Ri x).H Wi (di - Ri x)

    The gradient therefore takes the form

    Dx chi2 = - 2 sum_i Ri.H Wi (di - Ri x) = 2 sum_i Ri.H Wi Ri x - IDi




    '''
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import xarray as xr
    import dask
    import dask.array as da
    from distributed import get_client
    from pfb.opt.power_method import power_method_dist as power_method
    from pfb.opt.primal_dual import primal_dual_dist as primal_dual
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, dds2fits, dds2fits_mfs, set_wcs, save_fits
    from pfb.utils.dist import l1reweight_func
    from copy import deepcopy
    from operator import getitem
    from itertools import cycle
    from uuid import uuid4
    from pfb.utils.misc import fitcleanbeam
    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.misc import eval_coeffs_to_slice, fit_image_cube
    import pywt
    from pfb.wavelets.wavelets_jsk import get_buffer_size
    import fsspec
    import pickle
    from pfb.utils.dist import band_actor

    real_type = np.float64
    complex_type = np.complex128

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    xds_name = f'{basename}.xds'
    xds_store = DaskMSStore(xds_name)
    try:
        assert xds_store.exists()
    except Exception as e:
        raise ValueError(f'No xds at {xds_name}')

    client = get_client()
    names = list(client.scheduler_info()['workers'].keys())

    try:
        assert len(names) == opts.nband
    except Exception as e:
        raise ValueError("You must initialise a worker per imaging band")

    ds_list = xds_store.fs.glob(f'{xds_store.url}/*')

    url_prepend = xds_store.protocol + '://'
    ds_list = list(map(lambda x: url_prepend + x, ds_list))

    mds_name = f'{basename}.mds'
    mdsstore = DaskMSStore(mds_name)

    # create an actor cache dir
    actor_cache_dir = xds_store.url.rstrip('.xds') + '.dds'
    ac_store = DaskMSStore(actor_cache_dir)

    if ac_store.exists() and opts.reset_cache:
        print(f"Removing {ac_store.url}", file=log)
        ac_store.rm(recursive=True)

    optsp_name = ac_store.url + '/opts.pkl'
    # does this use the correct protocol?
    fs = fsspec.filesystem(ac_store.protocol)
    if ac_store.exists() and not opts.reset_cache:
        # get opts from previous run
        with fs.open(optsp_name, 'rb') as f:
            optsp = pickle.load(f)

        # check if we need to remake the data products
        verify_attrs = ['epsilon',
                        'do_wgridding', 'double_accum',
                        'field_of_view', 'super_resolution_factor']
        try:
            for attr in verify_attrs:
                assert optsp[attr] == opts[attr]

            from_cache = True
            print("Initialising from cached data products", file=log)
        except Exception as e:
            print(f'Cache verification failed on {attr}. '
                  'Will remake image data products', file=log)
            ac_store.rm(recursive=True)
            fs.makedirs(ac_store.url, exist_ok=True)
            # dump opts to validate cache on rerun
            with fs.open(optsp_name, 'wb') as f:
                pickle.dump(opts, f)
            from_cache = False
    else:
        fs.makedirs(ac_store.url, exist_ok=True)
        # dump opts to validate cache on rerun
        with fs.open(optsp_name, 'wb') as f:
            pickle.dump(opts, f)
        from_cache = False
        print("Initialising from scratch.", file=log)

    print(f"Data products will be cached in {ac_store.url}", file=log)

    # filter datasets by band
    dsb = {}
    for ds in ds_list:
        idx = ds.find('band') + 4
        bid = int(ds[idx:idx+4])
        dsb.setdefault(bid, [])
        dsb[bid].append(ds)

    nband = len(dsb.keys())
    try:
        assert len(names) == nband
    except Exception as e:
        raise ValueError(f"You must initialise {nband} workers. "
                         "One for each imaging band.")


    # set up band actors
    print("Setting up actors", file=log)
    futures = []
    for wname, (bandid, dsl) in zip(names, dsb.items()):
        f = client.submit(band_actor,
                          dsl,
                          opts,
                          bandid,
                          ac_store.url,
                          workers=wname,
                          key='actor-'+uuid4().hex,
                          actor=True,
                          pure=False)
        futures.append(f)
    actors = list(map(lambda f: f.result(), futures))

    futures = list(map(lambda a: a.get_image_info(), actors))
    results = list(map(lambda f: f.result(), futures))
    nx, ny, nmax, cell_rad, ra, dec, x0, y0, freq_out, time_out, wsum0 = results[0]
    freq_out = [freq_out]
    wsums = [wsum0]
    for result in results[1:]:
        assert result[0] == nx
        assert result[1] == ny
        assert result[2] == nmax
        assert result[3] == cell_rad
        assert result[4] == ra
        assert result[5] == dec
        assert result[6] == x0
        assert result[7] == y0
        freq_out.append(result[8])
        wsums.append(result[9])

    print(f"Image size set to ({nx}, {ny})", file=log)
    # need to set wsum after setting image products
    wsums = np.array(wsums)
    fsel = wsums > 0
    wsum = np.sum(wsums)


    radec = [ra, dec]
    cell_deg = np.rad2deg(cell_rad)
    freq_out = np.array(freq_out)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    time_out = np.array((time_out,))


    # get size of dual domain
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)

    # initialise model (assumed constant in time)
    if mdsstore.exists():
        print(f"Loading model from {mdsstore.url}", file=log)
        mds = xr.open_zarr(mdsstore.url)

        # we only want to load these once
        model_coeffs = mds.coefficients.values
        locx = mds.location_x.values
        locy = mds.location_y.values
        params = mds.params.values
        coeffs = mds.coefficients.values
        iter0 = mds.niters

        model = np.zeros((nband, nx, ny))
        for i in range(opts.nband):
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
        iter0 = 0

    print("Computing image data products", file=log)
    futures = []
    for b in range(nband):
        fut = actors[b].set_image_data_products(model[b],
                                                from_cache=from_cache)
        futures.append(fut)


    results = list(map(lambda f: f.result(), futures))
    residual_mfs = np.sum([r[0] for r in results], axis=0)
    wsums = np.array([r[1] for r in results])
    wsum = np.sum(wsums)
    futures = list(map(lambda a: a.set_wsum(wsum), actors))
    results = list(map(lambda f: f.result(), futures))

    residual_mfs /= wsum
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    if iter0 == 0:
        save_fits(residual_mfs,
                  basename + f'_{opts.suffix}_dirty_mfs.fits',
                  hdr_mfs)

    if opts.hess_norm is None:
        print('Getting spectral norm of Hessian approximation', file=log)
        hess_norm = power_method(actors, nx, ny, nband,
                                tol=opts.pm_tol,
                                maxit=opts.pm_maxit,
                                report_freq=opts.pm_report_freq,
                                verbosity=opts.pm_verbose)
        # inflate so we don't have to recompute after each L2 reweight
        hess_norm *= 1.05
    else:
        hess_norm = opts.hess_norm
    print(f'hess-norm = {hess_norm:.3e}', file=log)

    # a value less than zero turns L1 reweighting off
    # we'll start on convergence or at the iteration
    # indicated by l1reweight_from, whichever comes first
    l1reweight_from = opts.l1reweight_from

    if l1reweight_from == 0:
        # this is an approximation
        rms_comps = np.std(residual_mfs)*nband/pix_per_beam
        l1weight = l1reweight_func(actors,
                                   opts.rmsfactor,
                                   rms_comps,
                                   alpha=opts.alpha)
    else:
        rms_comps = 1.0
        l1weight = np.ones((nbasis, nmax), dtype=real_type)
        reweighter = None

    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count = 0
    l2reweights = 0
    print(f"It {iter0}: max resid = {rmax:.3e}, rms = {rms:.3e}", file=log)
    for k in range(iter0, iter0 + opts.niter):
        print('Solving for update', file=log)
        futures = list(map(lambda a: a.cg_update(), actors))
        updates = list(map(lambda f: f.result(), futures))
        updates = np.array(updates)
        update_mfs = np.mean(updates[fsel], axis=0)

        print('Solving for model', file=log)
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

        print('Updating model', file=log)
        futures = list(map(lambda a: a.give_model(), actors))
        results = list(map(lambda f: f.result(), futures))

        bandids = [r[1] for r in results]
        modelp = model.copy()
        for i, b in enumerate(bandids):
            model[b] = results[i][0]

        if np.isnan(model).any():
            import ipdb; ipdb.set_trace()

            raise ValueError('Model is nan')
        # save model and residual
        save_fits(update_mfs,
                  basename + f'_{opts.suffix}_update_{k+1}.fits',
                  hdr_mfs)

        save_fits(np.mean(model[fsel], axis=0),
                  basename + f'_{opts.suffix}_model_{k+1}.fits',
                  hdr_mfs)

        print(f"Writing model to {mdsstore.url}",
              file=log)
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
            'niters': k+1  # keep track in mds
        }

        coeff_dataset = xr.Dataset(data_vars=data_vars,
                            coords=coords,
                            attrs=attrs)
        coeff_dataset.to_zarr(f"{mdsstore.url}", mode='w')

        # convergence check
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)
        if eps < opts.tol:
            # do not converge prematurely
            if k+1 - iter0 < l1reweight_from:  # only happens once
                # start reweighting
                l1reweight_from = k+1 - iter0
                # don't start L2 reweighting before L1 reweighting
                # if it is enabled
                dof = None
            elif l2reweights < opts.max_l2_reweight:
                # L1 reweighting has already kicked in and we have
                # converged again so perform L2 reweight
                dof = opts.l2_reweight_dof
            else:
                # we have reached max L2 reweights so we are done
                print(f"Converged after {k+1} iterations.", file=log)
                break
        else:
            # do not perform an L2 reweight unless the M step has converged
            dof = None

        if k+1 - iter0 >= opts.l2_reweight_from and l2reweights < opts.max_l2_reweight:
            dof = opts.l2_reweight_dof

        if dof is not None:
            print('Recomputing image data products since L2 reweight '
                  'is required.', file=log)
            futures = []
            for b in range(nband):
                fut = actors[b].set_image_data_products(model[b],
                                                        from_cache=from_cache)
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
            # print('Getting spectral norm of Hessian approximation', file=log)
            # hess_norm = power_method(actors, nx, ny, nband)
            # print(f'hess-norm = {hess_norm:.3e}', file=log)

            l2reweights += 1
        else:
            # compute normal residual, no need to redo PSF etc
            print('Computing residual', file=log)
            futures = list(map(lambda a: a.set_residual(), actors))
            resids = list(map(lambda f: f.result(), futures))
            # we never resids by wsum inside the worker
            residual_mfs = np.sum(resids, axis=0)/wsum


        save_fits(residual_mfs,
                  basename + f'_{opts.suffix}_residual_{k+1}.fits',
                  hdr_mfs)

        rmsp = rms
        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()

        # base this on rmax?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        print(f"It {k+1}: max resid = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        if k+1 - iter0 >= l1reweight_from:
            print('L1 reweighting', file=log)
            rms_comps = np.std(update_mfs)
            l1weight = l1reweight_func(actors,
                                       opts.rmsfactor,
                                       rms_comps,
                                       alpha=opts.alpha)

        if rms > rmsp:
            diverge_count += 1
            if diverge_count > opts.diverge_count:
                print("Algorithm is diverging. Terminating.", file=log)
                break

    # # convert to fits files
    # fitsout = []
    # if opts.fits_mfs:
    #     fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', basename, norm_wsum=True))
    #     fitsout.append(dds2fits_mfs(dds, 'MODEL', basename, norm_wsum=False))

    # if opts.fits_cubes:
    #     fitsout.append(dds2fits(dds, 'RESIDUAL', basename, norm_wsum=True))
    #     fitsout.append(dds2fits(dds, 'MODEL', basename, norm_wsum=False))

    # if len(fitsout):
    #     print("Writing fits", file=log)
    #     dask.compute(fitsout)

    return

