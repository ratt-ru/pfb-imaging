# flake8: noqa
import os
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('SPOTLESS')

from numba import njit
@njit
def showtys(iy):
    print(len(iy), iy)
    # for n in range(1, len(tys)):
    #     for k, v in tys[n].items():
    #         print(k, v)
    return

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.spotless["inputs"].keys():
    defaults[key] = schema.spotless["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.spotless)
def spotless(**kw):
    '''
    Spotless algorithm
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'spotless_{timestamp}.log')

    if opts.scheduler=='distributed':
        # total number of threads
        if opts.nthreads is None:
            if opts.host_address is not None:
                raise ValueError("You have to specify nthreads in the "
                                 "distributed case")  # not
            import multiprocessing
            nthreads = multiprocessing.cpu_count()
            opts.nthreads = nthreads

        if opts.nworkers is None:
            opts.nworkers = opts.nband

        if opts.nthreads_per_worker is None:
            nthreads_per_worker = 1
            opts.nthreads_per_worker = nthreads_per_worker

        nthreads_dask = opts.nworkers * opts.nthreads_per_worker

        if opts.nvthreads is None:
            if opts.scheduler in ['single-threaded', 'sync']:
                nvthreads = nthreads
            elif opts.host_address is not None:
                nvthreads = max(nthreads//nthreads_per_worker, 1)
            else:
                nvthreads = max(nthreads//nthreads_dask, 1)
            opts.nvthreads = nvthreads

        OmegaConf.set_struct(opts, True)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        with ExitStack() as stack:
            os.environ["OMP_NUM_THREADS"] = str(opts.nvthreads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nvthreads)
            os.environ["MKL_NUM_THREADS"] = str(opts.nvthreads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nvthreads)
            os.environ["NUMBA_NUM_THREADS"] = str(opts.nband)
            # avoids numexpr error, probably don't want more than 10 vthreads for ne anyway
            import numexpr as ne
            max_cores = ne.detect_number_of_cores()
            ne_threads = min(max_cores, opts.nband)
            os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)

            if opts.host_address is not None:
                from distributed import Client
                print(f"Initialising distributed client at {opts.host_address}",
                      file=log)
                client = stack.enter_context(Client(opts.host_address))
            else:
                if nthreads_dask * opts.nvthreads > opts.nthreads:
                    print("Warning - you are attempting to use more threads than "
                          "available. This may lead to suboptimal performance.",
                        file=log)
                from dask.distributed import Client, LocalCluster
                print("Initialising client with LocalCluster.", file=log)
                cluster = LocalCluster(processes=True, n_workers=opts.nworkers,
                                    threads_per_worker=opts.nthreads_per_worker,
                                    memory_limit=0)  # str(mem_limit/nworkers)+'GB'
                cluster = stack.enter_context(cluster)
                client = stack.enter_context(Client(cluster))

            client.wait_for_workers(opts.nworkers)
            client.amm.stop()

            return _spotless_dist(**opts)

    else:
        raise NotImplementedError("Only distributed currently implemented")


def _spotless_dist(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import xarray as xr
    import dask
    import dask.array as da
    from distributed import Client, wait, get_client, as_completed
    from pfb.opt.power_method import power_method_dist as power_method
    from pfb.opt.pcg import pcg_dist as pcg
    from pfb.opt.primal_dual import primal_dual_dist as primal_dual
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, dds2fits, dds2fits_mfs
    from pfb.utils.dist import (get_resid_and_stats, accum_wsums,
                                compute_residual, update_results,
                                get_epsb, l1reweight, get_cbeam_area,
                                set_wsum, get_eps, set_l1weight)
    from pfb.operators.hessian import hessian_psf_slice
    from pfb.operators.psi import im2coef_dist as im2coef
    from pfb.operators.psi import coef2im_dist as coef2im
    from pfb.prox.prox_21m import prox_21m
    from pfb.prox.prox_21 import prox_21
    from pfb.wavelets.wavelets import wavelet_setup
    import pywt
    from copy import deepcopy
    from operator import getitem
    from itertools import cycle

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    dds_name = f'{basename}{opts.postfix}.dds.zarr'

    client = get_client()
    names = [w['name'] for w in client.scheduler_info()['workers'].values()]

    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan':-1,
                                          'x':-1,
                                          'y':-1,
                                          'x_psf':-1,
                                          'y_psf':-1,
                                          'yo2':-1,
                                          'b':-1,
                                          'c':-1})

    real_type = dds[0].DIRTY.dtype
    cell_rad = dds[0].cell_rad
    complex_type = np.result_type(real_type, np.complex64)

    nband = len(dds)
    nx, ny = dds[0].nx, dds[0].ny
    nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf
    nx_psf, nyo2_psf = dds[0].PSFHAT.shape
    npad_xl = (nx_psf - nx)//2
    npad_xr = nx_psf - nx - npad_xl
    npad_yl = (ny_psf - ny)//2
    npad_yr = ny_psf - ny - npad_yl
    psf_padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    lastsize = ny + np.sum(psf_padding[-1])

    # dictionary setup
    print("Setting up dictionary", file=log)
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)
    iy, sy, ntot, nmax = wavelet_setup(
                                np.zeros((1, nx, ny), dtype=real_type),
                                bases, opts.nlevels)
    ntot = tuple(ntot)

    psiH = partial(im2coef,
                    bases=bases,
                    ntot=ntot,
                    nmax=nmax,
                    nlevels=opts.nlevels)
    # avoid pickling dumba Dict
    psi = partial(coef2im,
                    bases=bases,
                    ntot=ntot,
                    iy=None,  #dict(iy),
                    sy=None,  #dict(sy),
                    nx=nx,
                    ny=ny)

    print('Scattering data', file=log)
    Afs = {}
    for ds, i in zip(dds, cycle(range(opts.nworkers))):
        Af = client.submit(hessian_psf_slice, ds, nbasis, nmax,
                           opts.nvthreads,
                           opts.sigmainv,
                           workers={names[i]})
        Afs[names[i]] = Af

    # wait expects a list
    wait(list(Afs.values()))

    # assumed constant
    wsum = client.submit(accum_wsums, Afs, workers={names[0]}).result()
    pix_per_beam = client.submit(get_cbeam_area, Afs, wsum, workers={names[0]}).result()

    for wid, A in Afs.items():
        client.submit(set_wsum, A, wsum,
                      pure=False,
                      workers={wid})

    try:
        l1ds = xds_from_zarr(f'{dds_name}::L1WEIGHT', chunks={'b':-1,'c':-1})
        if 'L1WEIGHT' in l1ds:
            l1weightfs = client.submit(set_l1weight, l1ds,
                                      workers={names[0]})
        else:
            raise
    except Exception as e:
        print(f'Did not find l1weights at {dds_name}/L1WEIGHT. '
              'Initialising to unity', file=log)
        l1weightfs = client.submit(np.ones, (nbasis, nmax), dtype=real_type,
                                 workers={names[0]})

    if opts.hessnorm is None:
        print('Getting spectral norm of Hessian approximation', file=log)
        hessnorm = power_method(Afs, nx, ny, nband)
    else:
        hessnorm = opts.hessnorm
    print(f'hessnorm = {hessnorm:.3e}', file=log)

    # future contains mfs residual and stats
    residf = client.submit(get_resid_and_stats, Afs, wsum,
                           workers={names[0]})
    residual_mfs = client.submit(getitem, residf, 0, workers={names[0]})
    rms = client.submit(getitem, residf, 1, workers={names[0]}).result()
    rmax = client.submit(getitem, residf, 2, workers={names[0]}).result()
    print(f"It {0}: max resid = {rmax:.3e}, rms = {rms:.3e}", file=log)
    for k in range(opts.niter):
        print('Solving for update', file=log)
        xfs = {}
        for wid, A in Afs.items():
            xf = client.submit(pcg, A,
                               opts.cg_maxit,
                               opts.cg_minit,
                               opts.cg_tol,
                               opts.sigmainv,
                               workers={wid})
            xfs[wid] = xf

        # wait expects a list
        wait(list(xfs.values()))

        print('Solving for model', file=log)
        modelfs, dualfs = primal_dual(
                            Afs, xfs,
                            psi, psiH,
                            opts.rmsfactor*rms,
                            hessnorm,
                            l1weightfs,
                            nu=len(bases),
                            tol=opts.pd_tol,
                            maxit=opts.pd_maxit,
                            positivity=opts.positivity,
                            gamma=opts.gamma,
                            verbosity=opts.pd_verbose)

        print('Computing residual', file=log)
        residfs = {}
        for wid, A in Afs.items():
            residf = client.submit(compute_residual, A, modelfs[wid],
                                   workers={wid})
            residfs[wid] = residf

        wait(list(residfs.values()))

        # l1reweighting
        if k+1 >= opts.l1reweight_from:
            print('L1 reweighting', file=log)
            l1weightfs = client.submit(l1reweight, modelfs, residfs, l1weightfs,
                                       psiH, wsum, pix_per_beam, workers=names[0])

        # dump results so we can continue from if needs be
        print('Updating results', file=log)
        ddsfs = {}
        modelpfs = {}
        for i, (wid, A) in enumerate(Afs.items()):
            modelpfs[wid] = client.submit(getattr, A, 'model', workers={wid})
            dsf = client.submit(update_results, A, dds[i], modelfs[i], dualfs[i], residfs[i],
                                pure=False,
                                workers={wid})
            ddsfs[wid] = dsf

        dds = []
        for f in as_completed(list(ddsfs.values())):
            dds.append(f.result())
        writes = xds_to_zarr(dds, dds_name,
                             columns=('MODEL','DUAL','RESIDUAL'),
                             rechunk=True)
        l1weight = da.from_array(l1weightfs.result(), chunks=(1, 4096**2))
        dvars = {}
        dvars['L1WEIGHT'] = (('b','c'), l1weight)
        l1ds = xr.Dataset(dvars)
        l1writes = xds_to_zarr(l1ds, f'{dds_name}::L1WEIGHT')
        client.compute(writes, l1writes)

        residf = client.submit(get_resid_and_stats, Afs, wsum,
                               workers={names[0]})
        residual_mfs = client.submit(getitem, residf, 0, workers={names[0]})
        rms = client.submit(getitem, residf, 1, workers={names[0]}).result()
        rmax = client.submit(getitem, residf, 2, workers={names[0]}).result()
        eps_num = []
        eps_den = []
        for wid in Afs.keys():
            fut = client.submit(get_epsb, modelpfs[wid], modelfs[wid], workers={wid})
            eps_num.append(client.submit(getitem, fut, 0, workers={wid}))
            eps_den.append(client.submit(getitem, fut, 1, workers={wid}))

        eps = client.submit(get_eps, eps_num, eps_den, workers={names[0]}).result()
        print(f"It {k+1}: max resid = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)


        if eps < opts.tol:
            break

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

    client.close()

    print("All done here.", file=log)
    return


def Idty(x):
    return x
