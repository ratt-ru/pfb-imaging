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
        if opts.nworkers is None:
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

            return _spotless(**opts)


def _spotless(**kw):
    opts = OmegaConf.create(kw)
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
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.opt.power_method import power_method
    from pfb.opt.pcg import pcg
    from pfb.opt.primal_dual import primal_dual_optimised as primal_dual
    from pfb.operators.hessian import hessian_xds
    from pfb.operators.psf import psf_convolve_cube
    from pfb.operators.psi import im2coef
    from pfb.operators.psi import coef2im
    from copy import copy, deepcopy
    from ducc0.misc import make_noncritical
    from pfb.wavelets.wavelets import wavelet_setup
    from pfb.prox.prox_21m import prox_21m_numba as prox_21
    from pfb.prox.prox2 import prox2
    # from pfb.prox.prox_21 import prox_21
    from pfb.utils.misc import fitcleanbeam

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}_{opts.postfix}.dds.zarr'
    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan':-1,
                                          'x':-1,
                                          'y':-1,
                                          'x_psf':-1,
                                          'y_psf':-1,
                                          'yo2':-1})
    if opts.memory_greedy:
        dds = dask.persist(dds)[0]

    nx_psf, ny_psf = dds[0].x_psf.size, dds[0].y_psf.size
    lastsize = ny_psf

    # stitch dirty/psf in apparent scale
    output_type = dds[0].DIRTY.dtype
    dirty, model, residual, psf, psfhat, beam, wsums, dual = dds2cubes(
                                                               dds,
                                                               opts.nband,
                                                               apparent=False)
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
    nband = opts.nband
    nx = dds[0].x.size
    ny = dds[0].y.size
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


    # image space hessian
    # pre-allocate arrays for doing FFT's
    xout = np.empty(dirty.shape, dtype=dirty.dtype, order='C')
    xout = make_noncritical(xout)
    xpad = np.empty(psf.shape, dtype=dirty.dtype, order='C')
    xpad = make_noncritical(xpad)
    xhat = np.empty(psfhat.shape, dtype=psfhat.dtype)
    xhat = make_noncritical(xhat)
    psf_convolve = partial(psf_convolve_cube, xpad, xhat, xout, psfhat, lastsize,
                       nthreads=opts.nthreads)

    if opts.hessnorm is None:
        print("Finding spectral norm of Hessian approximation", file=log)
        hessnorm, hessbeta = power_method(psf_convolve, (nband, nx, ny),
                                          tol=opts.pm_tol,
                                          maxit=opts.pm_maxit,
                                          verbosity=opts.pm_verbose,
                                          report_freq=opts.pm_report_freq)
    else:
        hessnorm = opts.hessnorm
        print(f"Using provided hessnorm of beta = {hessnorm:.3e}", file=log)

    # # PCG related options
    # cgopts = {}
    # cgopts['tol'] = opts.cg_tol
    # cgopts['maxit'] = opts.cg_maxit
    # cgopts['minit'] = opts.cg_minit
    # cgopts['verbosity'] = opts.cg_verbose
    # cgopts['report_freq'] = opts.cg_report_freq
    # cgopts['backtrack'] = opts.backtrack


    print("Setting up dictionary", file=log)
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)
    iy, sy, ntot, nmax = wavelet_setup(
                                np.zeros((1, nx, ny), dtype=dirty.dtype),
                                bases, opts.nlevels)
    ntot = tuple(ntot)

    psiH = partial(im2coef,
                   bases=bases,
                   ntot=ntot,
                   nmax=nmax,
                   nlevels=opts.nlevels,
                   nthreads=opts.nthreads)
    psi = partial(coef2im,
                  bases=bases,
                  ntot=ntot,
                  iy=iy,
                  sy=sy,
                  nx=nx,
                  ny=ny,
                  nthreads=opts.nthreads)

    # get clean beam area to convert residual units during l1reweighting
    # TODO - could refine this with comparison between dirty and restored
    # if contiuing the deconvolution
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.25, pixsize=1.0)[0]
    pix_per_beam = GaussPar[0]*GaussPar[1]*np.pi/4
    print(f"Number of pixels per beam estimated as {pix_per_beam}",
          file=log)

    # We do the following to set hyper-parameters in an intuitive way
    # i) convert residual units so it is comparable to model
    # ii) project residual into dual domain
    # iii) compute the rms in the space where thresholding happens
    tmp = np.zeros((nband, nbasis, nmax), dtype=dirty.dtype)
    fsel = wsums > 0
    tmp2 = residual.copy()
    tmp2[fsel] *= wsum/wsums[fsel, None, None]
    psiH(tmp2/pix_per_beam, tmp)
    rms_comps = np.std(np.sum(tmp, axis=0),
                       axis=-1)[:, None]  # preserve axes

    # TODO - load from dds if present
    if dual is None:
        dual = np.zeros((nband, nbasis, nmax), dtype=dirty.dtype)
        l1weight = np.ones((nbasis, nmax), dtype=dirty.dtype)
    else:
        if opts.l1reweight_from == 0:
            print('Initialising with L1 reweighted', file=log)
            psiH(model, tmp)
            mcomps = np.sum(tmp, axis=0)
            l1weight = (1 + opts.rmsfactor)/(1 + (np.abs(mcomps)/rms_comps)**2)
            # l1weight[l1weight < 1.0] = 0.0
        else:
            l1weight = np.ones((nbasis, nmax), dtype=dirty.dtype)


    # for generality the prox function only takes the
    # array variable and step size as inputs
    prox21 = partial(prox_21, weight=l1weight)

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    print(f"Iter 0: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    for k in range(opts.niter):
        print('Solving for model', file=log)
        modelp = deepcopy(model)
        data = residual + psf_convolve(model)
        grad21 = lambda x: psf_convolve(x) - data
        model, dual = primal_dual(model,
                                  dual,
                                  opts.rmsfactor*rms,
                                  psi,
                                  psiH,
                                  hessnorm,
                                  prox21,
                                  grad21,
                                  nu=nbasis,
                                  positivity=opts.positivity,
                                  tol=opts.pd_tol,
                                  maxit=opts.pd_maxit,
                                  verbosity=opts.pd_verbose,
                                  report_freq=opts.pd_report_freq,
                                  gamma=opts.gamma)

        save_fits(basename + f'_model_{k+1}.fits', np.mean(model, axis=0), hdr_mfs)

        print("Getting residual", file=log)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        save_fits(basename + f'_residual_{k+1}.fits', residual_mfs, hdr_mfs)

        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        print(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        if k+1 >= opts.l1reweight_from:
            print('Computing L1 weights', file=log)
            # convert residual units so it is comparable to model
            tmp2[fsel] = residual[fsel] * wsum/wsums[fsel, None, None]
            psiH(tmp2/pix_per_beam, tmp)
            rms_comps = np.std(np.sum(tmp, axis=0),
                               axis=-1)[:, None]  # preserve axes
            psiH(model, tmp)
            mcomps = np.sum(tmp, axis=0)
            # the logic here is that weights shoudl remain the same for model
            # components that are rmsfactor times larger than the rms
            # high SNR values should experience relatively small thresholding
            # whereas small values should be strongly thresholded
            l1weight = (1 + opts.rmsfactor)/(1 + (np.abs(mcomps)/rms_comps)**2)
            # l1weight[l1weight < 1.0] = 0.0
            prox = partial(prox_21, weight=l1weight, axis=0)

        print("Updating results", file=log)
        dds_out = []
        for ds in dds:
            b = ds.bandid
            r = da.from_array(residual[b]*wsum)
            m = da.from_array(model[b])
            d = da.from_array(dual[b])
            ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                                  'MODEL': (('x', 'y'), m),
                                  'DUAL': (('c', 'n'), d)})
            dds_out.append(ds_out)
        writes = xds_to_zarr(dds_out, dds_name,
                             columns=('RESIDUAL', 'MODEL', 'DUAL'),
                             rechunk=True)
        dask.compute(writes)

        if eps < opts.tol:
            print(f"Converged after {k+1} iterations.", file=log)
            break
        # if rmax <= threshold:
        #     print("Terminating because final threshold has been reached",
        #           file=log)
        #     break

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

    print("All done here.", file=log)


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
    dds_name = f'{basename}_{opts.postfix}.dds.zarr'

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
    complex_type = np.result_type(dirty.dtype, np.complex64)

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
                                np.zeros((1, nx, ny), dtype=dirty.dtype),
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
        l1weightfs = client.submit(np.ones, (nbasis, nmax), dtype=dirty.dtype,
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
