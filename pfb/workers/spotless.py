# flake8: noqa
import os
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
    Spotless algorithm
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

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(opts, stack, log,
                   scheduler=opts.scheduler,
                   auto_restrict=False)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        if opts.scheduler=='distributed':
            _spotless_dist(**opts)
        else:
            _spotless(**opts)

    print("All done here.", file=log)


def _spotless(ddsi=None, **kw):
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
    from pfb.utils.misc import l1reweight_func
    from pfb.operators.hessian import hessian_xds
    from pfb.operators.psf import psf_convolve_cube
    from pfb.operators.psi import Psi
    from copy import copy, deepcopy
    from ducc0.misc import make_noncritical
    from pfb.prox.prox_21m import prox_21m_numba as prox_21
    # from pfb.prox.prox_21 import prox_21
    from pfb.utils.misc import fitcleanbeam, fit_image_cube


    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}_{opts.suffix}.dds'
    if ddsi is not None:
        dds = []
        for ds in ddsi:
            dds.append(ds.chunk({'row':-1,
                                 'chan':-1,
                                 'x':-1,
                                 'y':-1,
                                 'x_psf':-1,
                                 'y_psf':-1,
                                 'yo2':-1}))
    else:
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
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))
    try:
        assert freq_out.size == opts.nband
    except Exception as e:
        print(f"Number of output frequencies={freq_out.size} "
              f"does not match nband={opts.nband}", file=log)
        raise e
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
    if 'niters' in dds[0].attrs:
        iter0 = dds[0].niters
    else:
        iter0 = 0


    # stitch dirty/psf in apparent scale
    print("Combining slices into cubes", file=log)
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

    # set up vis space Hessian
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['do_wgridding'] = opts.do_wgridding
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads  # nvthreads since dask parallel over band

    # TOD - add beam
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
    # nthreads = nvthreads*nthreads_dask because dask not involved
    psf_convolve = partial(psf_convolve_cube, xpad, xhat, xout, psfhat, lastsize,
                           nthreads=opts.nvthreads*opts.nthreads_dask)

    if opts.hessnorm is None:
        print("Finding spectral norm of Hessian approximation", file=log)
        hessnorm, hessbeta = power_method(psf_convolve, (nband, nx, ny),
                                          tol=opts.pm_tol,
                                          maxit=opts.pm_maxit,
                                          verbosity=opts.pm_verbose,
                                          report_freq=opts.pm_report_freq)
        # inflate slightly for stability
        hessnorm *= 1.05
    else:
        hessnorm = opts.hessnorm
        print(f"Using provided hessnorm of beta = {hessnorm:.3e}", file=log)

    print("Setting up dictionary", file=log)
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)
    psi = Psi(nband, nx, ny, bases, opts.nlevels, opts.nthreads_dask)
    Nxmax = psi.Nxmax
    Nymax = psi.Nymax

    # get clean beam area to convert residual units during l1reweighting
    # TODO - could refine this with comparison between dirty and restored
    print("Fitting PSF", file=log)
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    pix_per_beam = GaussPar[0]*GaussPar[1]*np.pi/4
    print(f"Number of pixels per beam estimated as {pix_per_beam}",
          file=log)

    # We do the following to set hyper-parameters in an intuitive way
    # i) convert residual units so it is comparable to model
    # ii) project residual into dual domain
    # iii) compute the rms in the space where thresholding happens
    psiHoutvar = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=dirty.dtype)
    fsel = wsums > 0
    tmp2 = residual.copy()
    tmp2[fsel] *= wsum/wsums[fsel, None, None]
    psi.dot(tmp2/pix_per_beam, psiHoutvar)
    rms_comps = np.std(np.sum(psiHoutvar, axis=0),
                       axis=(-1,-2))[:, None, None]  # preserve axes

    # a value less than zero turns L1 reweighting off
    # we'll start on convergence or at the iteration
    # indicated by l1reweight_from, whichever comes first
    if opts.l1reweight_from < 0:
        l1reweight_from = np.inf
    else:
        l1reweight_from = opts.l1reweight_from

    if dual is None or dual.shape[1] != nbasis:  # nbasis could change
        dual = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=dirty.dtype)
        l1weight = np.ones((nbasis, Nymax, Nxmax), dtype=dirty.dtype)
        reweighter = None
    else:
        if l1reweight_from == 0:
            print('Initialising with L1 reweighted', file=log)
            reweighter = partial(l1reweight_func,
                                 psi.dot,
                                 psiHoutvar,
                                 opts.rmsfactor,
                                 rms_comps,
                                 alpha=opts.alpha)
            l1weight = reweighter(model)
        else:
            l1weight = np.ones((nbasis, Nymax, Nxmax), dtype=dirty.dtype)
            reweighter = None

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count = 0
    print(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    for k in range(iter0, iter0 + opts.niter):
        print('Solving for model', file=log)
        modelp = deepcopy(model)
        data = residual + psf_convolve(model)
        grad21 = lambda x: psf_convolve(x) - data
        if k == 0:
            rmsfactor = opts.init_factor * opts.rmsfactor
        else:
            rmsfactor = opts.rmsfactor
        model, dual = primal_dual(model,
                                  dual,
                                  rmsfactor*rms,
                                  psi.hdot,
                                  psi.dot,
                                  hessnorm,
                                  prox_21,
                                  l1weight,
                                  reweighter,
                                  grad21,
                                  nu=nbasis,
                                  positivity=opts.positivity,
                                  tol=opts.pd_tol,
                                  maxit=opts.pd_maxit,
                                  verbosity=opts.pd_verbose,
                                  report_freq=opts.pd_report_freq,
                                  gamma=opts.gamma)

        # write component model
        print(f"Writing model at iter {k+1} to "
              f"{basename}_{opts.suffix}_model_{k+1}.mds", file=log)
        try:
            coeffs, Ix, Iy, expr, params, texpr, fexpr = \
                fit_image_cube(time_out, freq_out[fsel], model[None, fsel, :, :],
                               wgt=wsums[None, fsel],
                               nbasisf=int(np.sum(fsel)),
                               method='Legendre')
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
            attrs = {
                'spec': 'genesis',
                'cell_rad_x': cell_rad,
                'cell_rad_y': cell_rad,
                'npix_x': nx,
                'npix_y': ny,
                'texpr': texpr,
                'fexpr': fexpr,
                'center_x': dds[0].x0,
                'center_y': dds[0].y0,
                'ra': dds[0].ra,
                'dec': dds[0].dec,
                'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
                'parametrisation': expr  # already converted to str
            }

            coeff_dataset = xr.Dataset(data_vars=data_vars,
                               coords=coords,
                               attrs=attrs)
            coeff_dataset.to_zarr(f"{basename}_{opts.suffix}_model_{k+1}.mds")
        except Exception as e:
            print(f"Exception {e} raised during model fit .", file=log)

        save_fits(np.mean(model, axis=0),
                  basename + f'_{opts.suffix}_model_{k+1}.fits',
                  hdr_mfs)

        print("Getting residual", file=log)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        save_fits(residual_mfs,
                  basename + f'_{opts.suffix}_residual_{k+1}.fits',
                  hdr_mfs)

        rmsp = rms
        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        # base this on rmax?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        print(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        print("Updating results", file=log)
        dds_out = []
        for ds in dds:
            b = ds.bandid
            r = da.from_array(residual[b]*wsum)
            m = da.from_array(model[b])
            d = da.from_array(dual[b])
            mbest = da.from_array(best_model[b])
            ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                                  'MODEL': (('x', 'y'), m),
                                  'DUAL': (('c', 'i', 'j'), d),
                                  'MODEL_BEST': (('x', 'y'), mbest)})
            ds_out = ds_out.assign_attrs({'parametrisation': 'id',
                                          'niters': k+1,
                                          'best_rms': best_rms,
                                          'best_rmax': best_rmax})
            dds_out.append(ds_out)
        writes = xds_to_zarr(dds_out, dds_name,
                             columns=('RESIDUAL', 'MODEL',
                                      'DUAL', 'MODEL_BEST'),
                             rechunk=True)
        dask.compute(writes)

        if eps < opts.tol:
            # do not converge prematurely
            if k+1 - iter0 >= l1reweight_from:
                # start reweighting
                l1reweight_from = 0
            else:
                print(f"Converged after {k+1} iterations.", file=log)
                break

        if k+1 - iter0 >= l1reweight_from:
            print('Computing L1 weights', file=log)
            # convert residual units so it is comparable to model
            tmp2[fsel] = residual[fsel] * wsum/wsums[fsel, None, None]
            psi.dot(tmp2/pix_per_beam, psiHoutvar)
            rms_comps = np.std(np.sum(psiHoutvar, axis=0),
                               axis=(-1,-2))[:, None, None]  # preserve axes
            # we redefine the reweighter here since the rms has changed
            reweighter = partial(l1reweight_func,
                                 psi.dot,
                                 psiHoutvar,
                                 opts.rmsfactor,
                                 rms_comps,
                                 alpha=opts.alpha)
            l1weight = reweighter(model)

        if rms > rmsp:
            diverge_count += 1
            if diverge_count > opts.diverge_count:
                print("Algorithm is diverging. Terminating.", file=log)
                break

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})

    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', f'{basename}_{opts.suffix}', norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds, 'MODEL', f'{basename}_{opts.suffix}', norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds, 'RESIDUAL', f'{basename}_{opts.suffix}', norm_wsum=True))
        fitsout.append(dds2fits(dds, 'MODEL', f'{basename}_{opts.suffix}', norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    return

def _spotless_dist(**kw):
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
    from pfb.utils.dist import grad_actor

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    # dds_name = f'{basename}_{opts.suffix}.dds'
    dds_name = f'{basename}.dds'
    ds_store = DaskMSStore(dds_name)
    try:
        assert ds_store.exists()
    except Exception as e:
        raise ValueError(f'No dds at {dds_name}')

    client = get_client()
    client.amm.stop()
    names = list(client.scheduler_info()['workers'].keys())

    try:
        assert len(names) == opts.nband
    except Exception as e:
        raise ValueError("You must initialise a worker per imaging band")

    ds_list = ds_store.fs.glob(f'{ds_store.url}/*')

    url_prepend = ds_store.protocol + '://'
    ds_list = list(map(lambda x: url_prepend + x, ds_list))

    mds_name = f'{basename}.mds'
    mdsstore = DaskMSStore(mds_name)

    # create an actor cache dir
    actor_cache_dir = ds_store.url.rstrip('.dds') + '.cache'
    ac_store = DaskMSStore(actor_cache_dir)

    if ac_store.exists() and opts.reset_cache:
        print(f"Removing {ac_store.url}.dds", file=log)
        ac_store.rm(recursive=True)

    optsp_name = ac_store.url + '/opts.pkl'
    # does this use the correct protocol?
    fs = fsspec.filesystem(ac_store.url.split(':', 1)[0])
    if ac_store.exists() and not opts.reset_cache:
        # get opts from previous run
        with fs.open(optsp_name, 'rb') as f:
            optsp = pickle.load(f)

        # check if we need to remake the data products
        verify_attrs = ['robustness', 'epsilon',
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
        f = client.submit(grad_actor,
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

    # we do this in case we want to do MF weighting
    futures = list(map(lambda a: a.get_counts(), actors))
    counts = list(map(lambda f: f.result(), futures))
    if opts.mf_weighting:
        counts = np.sum(counts, axis=0)
    else:
        counts = None  # will use self.counts in this case

    print("Setting imaging weights", file=log)
    futures = list(map(lambda a: a.set_imaging_weights(counts=counts), actors))
    wsums = np.array(list(map(lambda f: f.result(), futures)))
    fsel = wsums > 0
    wsum = np.sum(wsums)
    real_type = wsum.dtype
    complex_type = np.result_type(real_type, np.complex64)
    futures = list(map(lambda a: a.get_image_info(), actors))
    results = list(map(lambda f: f.result(), futures))
    nx, ny, nmax, cell_rad, ra, dec, x0, y0, freq_out, time_out = results[0]
    freq_out = [freq_out]
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

    print(f"Image size set to ({nx}, {ny})", file=log)

    radec = [ra, dec]
    cell_deg = np.rad2deg(cell_rad)
    freq_out = np.array(freq_out)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    time_out = np.array((time_out,))


    # get size of dual domain
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)

    # initialise dual and model (assumed constant in time)
    if mdsstore.exists():
        mds = xr.open_zarr(mdsstore.url)

        # we only want to load these once
        model_coeffs = mds.coefficients.values
        locx = mds.location_x.values
        locy = mds.location_y.values
        params = mds.params.values
        coeffs = mds.coefficients.values
        iter0 = mds.niters

        # TODO - save dual in mds
        model = np.zeros((nband, nx, ny))
        dual = np.zeros((nband, nbasis, nmax))
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
        dual = np.zeros((nband, nbasis, nmax))
        iter0 = 0

    print("Computing image data products", file=log)
    futures = []
    for b in range(nband):
        fut = actors[b].set_image_data_products(model[b], dual[b],
                                                from_cache=from_cache)
        futures.append(fut)

    try:
        results = list(map(lambda f: f.result(), futures))
    except Exception as e:
        print("Error raised - ", e)
        print([r.result() for r in result()])

        import ipdb; ipdb.set_trace()
    psf_mfs = np.sum([r[0] for r in results], axis=0)/wsum
    assert (psf_mfs.max() - 1.0) < opts.epsilon  # make sure wsum is consistent
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    pix_per_beam = GaussPar[0]*GaussPar[1]*np.pi/4
    residual_mfs = np.sum([r[1] for r in results], axis=0)/wsum
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    if iter0 == 0:
        save_fits(residual_mfs,
                  basename + f'_{opts.suffix}_dirty_mfs.fits',
                  hdr_mfs)


    futures = list(map(lambda a: a.set_wsum_and_data(wsum), actors))
    results = list(map(lambda f: f.result(), futures))

    if opts.hessnorm is None:
        print('Getting spectral norm of Hessian approximation', file=log)
        hessnorm = power_method(actors, nx, ny, nband)
        # inflate so we don't have to recompute after each L2 reweight
        hessnorm *= 1.05
    else:
        hessnorm = opts.hessnorm
    print(f'hessnorm = {hessnorm:.3e}', file=log)

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
        print('Solving for model', file=log)
        primal_dual(actors,
                    rms*opts.rmsfactor,
                    hessnorm,
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

        bandids = [r[2] for r in results]
        modelp = model.copy()
        for i, b in enumerate(bandids):
            model[b] = results[i][0]
            dual[b] = results[i][1]

        if np.isnan(model).any():
            import ipdb; ipdb.set_trace()

            raise ValueError('Model is nan')
        # save model and residual
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
                fut = actors[b].set_image_data_products(model[b], dual[b], dof=dof)
                futures.append(fut)

            results = list(map(lambda f: f.result(), futures))
            psf_mfs = np.sum([r[0] for r in results], axis=0)
            wsum = psf_mfs.max()
            psf_mfs /= wsum
            GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
            pix_per_beam = GaussPar[0]*GaussPar[1]*np.pi/4
            residual_mfs = np.sum([r[1] for r in results], axis=0)/wsum

            # don't keep the cubes on the runner
            del results

            futures = list(map(lambda a: a.set_wsum_and_data(wsum), actors))
            results = list(map(lambda f: f.result(), futures))

            # # TODO - how much does hessnorm change after L2 reweight?
            # print('Getting spectral norm of Hessian approximation', file=log)
            # hessnorm = power_method(actors, nx, ny, nband)
            # print(f'hessnorm = {hessnorm:.3e}', file=log)

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
            rms_comps = np.std(residual_mfs)*nband/pix_per_beam
            l1weight = l1reweight_func(actors, opts.rmsfactor, rms_comps,
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

