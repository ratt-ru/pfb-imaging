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
        set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        if opts.scheduler=='distributed':
            return _spotless_dist(**opts)
        else:
            return _spotless(**opts)


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

    if dual is None or dual.shape[1] != nbasis:  # nbasis could change
        dual = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=dirty.dtype)
        l1weight = np.ones((nbasis, Nymax, Nxmax), dtype=dirty.dtype)
        reweighter = None
    else:
        if opts.l1reweight_from == 0:
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


    # for generality the prox function should only take
    # array variable and step size as inputs
    # prox21 = partial(prox_21, weight=l1weight)

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

        if k+1 - iter0 >= opts.l1reweight_from:
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
            print(f"Converged after {k+1} iterations.", file=log)
            break
        # if rmax <= threshold:
        #     print("Terminating because final threshold has been reached",
        #           file=log)
        #     break

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

    print("All done here.", file=log)


# def _spotless_dist(**kw):
#     opts = OmegaConf.create(kw)
#     OmegaConf.set_struct(opts, True)

#     import numpy as np
#     import xarray as xr
#     import dask
#     import dask.array as da
#     from distributed import Client, wait, get_client, as_completed
#     from pfb.opt.power_method import power_method_dist as power_method
#     from pfb.opt.pcg import pcg_dist as pcg
#     from pfb.opt.primal_dual import primal_dual_dist as primal_dual
#     from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
#     from pfb.utils.fits import load_fits, dds2fits, dds2fits_mfs
#     from pfb.utils.dist import (get_resid_and_stats, accum_wsums,
#                                 compute_residual, update_results,
#                                 get_epsb, l1reweight, get_cbeam_area,
#                                 set_wsum, get_eps, set_l1weight)
#     from pfb.operators.hessian import hessian_psf_slice
#     from pfb.operators.psi import im2coef_dist as im2coef
#     from pfb.operators.psi import coef2im_dist as coef2im
#     from pfb.prox.prox_21m import prox_21m
#     from pfb.prox.prox_21 import prox_21
#     from pfb.wavelets.wavelets import wavelet_setup
#     import pywt
#     from copy import deepcopy
#     from operator import getitem
#     from itertools import cycle

#     basename = f'{opts.output_filename}_{opts.product.upper()}'
#     dds_name = f'{basename}_{opts.suffix}.dds'

#     client = get_client()
#     names = [w['name'] for w in client.scheduler_info()['workers'].values()]

#     dds = xds_from_zarr(dds_name, chunks={'row':-1,
#                                           'chan':-1,
#                                           'x':-1,
#                                           'y':-1,
#                                           'x_psf':-1,
#                                           'y_psf':-1,
#                                           'yo2':-1,
#                                           'b':-1,
#                                           'c':-1})

#     real_type = dds[0].DIRTY.dtype
#     cell_rad = dds[0].cell_rad
#     complex_type = np.result_type(dirty.dtype, np.complex64)

#     nband = len(dds)
#     nx, ny = dds[0].nx, dds[0].ny
#     nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf
#     nx_psf, nyo2_psf = dds[0].PSFHAT.shape
#     npad_xl = (nx_psf - nx)//2
#     npad_xr = nx_psf - nx - npad_xl
#     npad_yl = (ny_psf - ny)//2
#     npad_yr = ny_psf - ny - npad_yl
#     psf_padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
#     unpad_x = slice(npad_xl, -npad_xr)
#     unpad_y = slice(npad_yl, -npad_yr)
#     lastsize = ny + np.sum(psf_padding[-1])

#     # dictionary setup
#     print("Setting up dictionary", file=log)
#     bases = tuple(opts.bases.split(','))
#     nbasis = len(bases)
#     iy, sy, ntot, nmax = wavelet_setup(
#                                 np.zeros((1, nx, ny), dtype=dirty.dtype),
#                                 bases, opts.nlevels)
#     ntot = tuple(ntot)

#     psiH = partial(im2coef,
#                     bases=bases,
#                     ntot=ntot,
#                     nmax=nmax,
#                     nlevels=opts.nlevels)
#     # avoid pickling dumba Dict
#     psi = partial(coef2im,
#                     bases=bases,
#                     ntot=ntot,
#                     iy=None,  #dict(iy),
#                     sy=None,  #dict(sy),
#                     nx=nx,
#                     ny=ny)

#     print('Scattering data', file=log)
#     Afs = {}
#     for ds, i in zip(dds, cycle(range(opts.nworkers))):
#         Af = client.submit(hessian_psf_slice, ds, nbasis, nmax,
#                            opts.nvthreads,
#                            opts.sigmainv,
#                            workers={names[i]})
#         Afs[names[i]] = Af

#     # wait expects a list
#     wait(list(Afs.values()))

#     # assumed constant
#     wsum = client.submit(accum_wsums, Afs, workers={names[0]}).result()
#     pix_per_beam = client.submit(get_cbeam_area, Afs, wsum, workers={names[0]}).result()

#     for wid, A in Afs.items():
#         client.submit(set_wsum, A, wsum,
#                       pure=False,
#                       workers={wid})

#     try:
#         l1ds = xds_from_zarr(f'{dds_name}::L1WEIGHT', chunks={'b':-1,'c':-1})
#         if 'L1WEIGHT' in l1ds:
#             l1weightfs = client.submit(set_l1weight, l1ds,
#                                       workers={names[0]})
#         else:
#             raise
#     except Exception as e:
#         print(f'Did not find l1weights at {dds_name}/L1WEIGHT. '
#               'Initialising to unity', file=log)
#         l1weightfs = client.submit(np.ones, (nbasis, nmax), dtype=dirty.dtype,
#                                  workers={names[0]})

#     if opts.hessnorm is None:
#         print('Getting spectral norm of Hessian approximation', file=log)
#         hessnorm = power_method(Afs, nx, ny, nband)
#     else:
#         hessnorm = opts.hessnorm
#     print(f'hessnorm = {hessnorm:.3e}', file=log)

#     # future contains mfs residual and stats
#     residf = client.submit(get_resid_and_stats, Afs, wsum,
#                            workers={names[0]})
#     residual_mfs = client.submit(getitem, residf, 0, workers={names[0]})
#     rms = client.submit(getitem, residf, 1, workers={names[0]}).result()
#     rmax = client.submit(getitem, residf, 2, workers={names[0]}).result()
#     print(f"It {0}: max resid = {rmax:.3e}, rms = {rms:.3e}", file=log)
#     for k in range(opts.niter):
#         print('Solving for update', file=log)
#         xfs = {}
#         for wid, A in Afs.items():
#             xf = client.submit(pcg, A,
#                                opts.cg_maxit,
#                                opts.cg_minit,
#                                opts.cg_tol,
#                                opts.sigmainv,
#                                workers={wid})
#             xfs[wid] = xf

#         # wait expects a list
#         wait(list(xfs.values()))

#         print('Solving for model', file=log)
#         modelfs, dualfs = primal_dual(
#                             Afs, xfs,
#                             psi, psiH,
#                             opts.rmsfactor*rms,
#                             hessnorm,
#                             l1weightfs,
#                             nu=len(bases),
#                             tol=opts.pd_tol,
#                             maxit=opts.pd_maxit,
#                             positivity=opts.positivity,
#                             gamma=opts.gamma,
#                             verbosity=opts.pd_verbose)

#         print('Computing residual', file=log)
#         residfs = {}
#         for wid, A in Afs.items():
#             residf = client.submit(compute_residual, A, modelfs[wid],
#                                    workers={wid})
#             residfs[wid] = residf

#         wait(list(residfs.values()))

#         # l1reweighting
#         if k+1 >= opts.l1reweight_from:
#             print('L1 reweighting', file=log)
#             l1weightfs = client.submit(l1reweight, modelfs, residfs, l1weightfs,
#                                        psiH, wsum, pix_per_beam, workers=names[0])

#         # dump results so we can continue from if needs be
#         print('Updating results', file=log)
#         ddsfs = {}
#         modelpfs = {}
#         for i, (wid, A) in enumerate(Afs.items()):
#             modelpfs[wid] = client.submit(getattr, A, 'model', workers={wid})
#             dsf = client.submit(update_results, A, dds[i], modelfs[i], dualfs[i], residfs[i],
#                                 pure=False,
#                                 workers={wid})
#             ddsfs[wid] = dsf

#         dds = []
#         for f in as_completed(list(ddsfs.values())):
#             dds.append(f.result())
#         writes = xds_to_zarr(dds, dds_name,
#                              columns=('MODEL','DUAL','RESIDUAL'),
#                              rechunk=True)
#         l1weight = da.from_array(l1weightfs.result(), chunks=(1, 4096**2))
#         dvars = {}
#         dvars['L1WEIGHT'] = (('b','c'), l1weight)
#         l1ds = xr.Dataset(dvars)
#         l1writes = xds_to_zarr(l1ds, f'{dds_name}::L1WEIGHT')
#         client.compute(writes, l1writes)

#         residf = client.submit(get_resid_and_stats, Afs, wsum,
#                                workers={names[0]})
#         residual_mfs = client.submit(getitem, residf, 0, workers={names[0]})
#         rms = client.submit(getitem, residf, 1, workers={names[0]}).result()
#         rmax = client.submit(getitem, residf, 2, workers={names[0]}).result()
#         eps_num = []
#         eps_den = []
#         for wid in Afs.keys():
#             fut = client.submit(get_epsb, modelpfs[wid], modelfs[wid], workers={wid})
#             eps_num.append(client.submit(getitem, fut, 0, workers={wid}))
#             eps_den.append(client.submit(getitem, fut, 1, workers={wid}))

#         eps = client.submit(get_eps, eps_num, eps_den, workers={names[0]}).result()
#         print(f"It {k+1}: max resid = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}",
#               file=log)


#         if eps < opts.tol:
#             break

#     # convert to fits files
#     fitsout = []
#     if opts.fits_mfs:
#         fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', basename, norm_wsum=True))
#         fitsout.append(dds2fits_mfs(dds, 'MODEL', basename, norm_wsum=False))

#     if opts.fits_cubes:
#         fitsout.append(dds2fits(dds, 'RESIDUAL', basename, norm_wsum=True))
#         fitsout.append(dds2fits(dds, 'MODEL', basename, norm_wsum=False))

#     if len(fitsout):
#         print("Writing fits", file=log)
#         dask.compute(fitsout)

#     client.close()

#     print("All done here.", file=log)
#     return


# def Idty(x):
#     return x
