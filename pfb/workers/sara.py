# flake8: noqa
import os
import sys
from contextlib import ExitStack
from pfb.workers.main import cli
import click
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
    Deconvolution using SARA regularisation
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
    from ducc0.misc import resize_thread_pool, thread_pool_size
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/sara_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import xds_from_url

    basename = f'{basedir}/{oname}'
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_store = DaskMSStore(f'{basename}_{opts.suffix}.dds')
    dds_list = dds_store.fs.glob(f'{dds_store.url}/*.zarr')

    with ExitStack() as stack:
        ti = time.time()
        _sara(**opts)

        dds = xds_from_url(dds_store.url)

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
            try:
                dds2fits(dds_list,
                     'MPSF',
                     f'{fits_oname}_{opts.suffix}',
                     norm_wsum=False,
                     nthreads=opts.nthreads,
                     do_mfs=opts.fits_mfs,
                     do_cube=opts.fits_cubes)
                print('Done writing MPSF', file=log)
            except Exception as e:
                print(e)

        print(f"All done after {time.time() - ti}s", file=log)


def _sara(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from functools import partial
    import numpy as np
    import xarray as xr
    import numexpr as ne
    from pfb.utils.fits import set_wcs, save_fits, load_fits
    from pfb.utils.naming import xds_from_url, xds_from_list
    from pfb.opt.power_method import power_method
    from pfb.opt.pcg import pcg
    from pfb.opt.primal_dual import primal_dual_optimised as primal_dual
    from pfb.utils.misc import l1reweight_func, taperf
    from pfb.operators.hessian import hess_direct, hessian_psf_cube
    from pfb.operators.psi import Psi
    from pfb.operators.gridder import compute_residual
    from copy import copy, deepcopy
    from ducc0.misc import empty_noncritical
    from pfb.prox.prox_21m import prox_21m_numba as prox_21
    # from pfb.prox.prox_21 import prox_21
    from pfb.utils.misc import fitcleanbeam, fit_image_cube, eval_coeffs_to_slice
    from daskms.fsspec_store import DaskMSStore
    from ducc0.fft import c2c

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds_store = DaskMSStore(dds_name)
    dds_list = dds_store.fs.glob(f'{dds_store.url}/*.zarr')
    dds = xds_from_list(dds_list,
                        drop_vars=['UVW', 'WEIGHT', 'MASK'],
                        nthreads=opts.nthreads)

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
    if time_out.size > 1:
        raise NotImplementedError('Only static models currently supported')

    nband = freq_out.size

    # only need this to get ny_psf
    psf = np.stack([ds.PSF.values for ds in dds], axis=0)
    dds = [ds.drop_vars('PSF') for ds in dds]

    # stitch dirty/psf in apparent scale
    # drop_vars to avoid duplicates in memory
    # and avoid unintentional side effects?
    output_type = dds[0].DIRTY.dtype
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
    psfhat = np.stack([ds.PSFHAT.values for ds in dds], axis=0)
    dds = [ds.drop_vars('PSFHAT') for ds in dds]
    beam = np.stack([ds.BEAM.values for ds in dds], axis=0)
    wsums = np.stack([ds.wsum for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands

    wsum = np.sum(wsums)
    wsums /= wsum
    psf /= wsum
    psfhat /= wsum
    abspsf = np.abs(psfhat)
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)
    model_mfs = np.mean(model[fsel], axis=0)

    # for intermediary results
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    x0 = dds[0].x0
    y0 = dds[0].y0
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    if 'niters' in dds[0].attrs:
        iter0 = dds[0].niters
    else:
        iter0 = 0

    # Allow calling with pd_tol as float
    try:
        ntol = len(opts.pd_tol)
        pd_tol = opts.pd_tol
    except TypeError:
        assert ininstance(opts.pd_tol, float)
        ntol = 1
        pd_tol = [opts.pd_tol]
    niters = opts.niter
    if ntol < niters:
        pd_tolf = pd_tol[-1]
        pd_tol += [pd_tolf]*niters  # no harm in too many

    # image space hessian
    # pre-allocate arrays for doing FFT's
    real_type = 'f8'
    complex_type = 'c16'
    if opts.hess_approx == 'direct':
        npix = np.maximum(nx, ny)
        nyo2 = psfhat.shape[-1]
        taperxy = taperf((nx, ny), np.minimum(int(0.1*npix), 32))
        xhat = empty_noncritical((nband, nx_psf, nyo2),
                                dtype=complex_type)
        xpad = empty_noncritical((nband, nx_psf, ny_psf),
                                dtype=real_type)
        xout = empty_noncritical((nband, nx, ny),
                                dtype=real_type)
        precond = lambda x, mode : hess_direct(
                                    x,
                                    xpad=xpad,
                                    xhat=xhat,
                                    xout=xout,
                                    psfhat=abspsf,
                                    taperxy=taperxy,
                                    lastsize=ny_psf,
                                    nthreads=opts.nthreads,
                                    sigmainvsq=wsums[:, None, None]*opts.epsfactor,
                                    mode=mode)
    elif opts.hess_approx == 'psf':
        raise NotImplementedError
    elif opts.hess_approx == 'wgt':
        raise NotImplementedError

    if opts.hess_norm is None:
        # if the grid worker had been rerun hess_norm won't be in attrs
        if 'hess_norm' in dds[0].attrs:
            hess_norm = dds[0].hess_norm
            print(f"Using previously estimated hess_norm of {hess_norm:.3e}",
                  file=log)
        else:
            print("Finding spectral norm of Hessian approximation", file=log)
            func = lambda x: precond(x, 'forward')
            hess_norm, hessbeta = power_method(func, (nband, nx, ny),
                                            tol=opts.pm_tol,
                                            maxit=opts.pm_maxit,
                                            verbosity=opts.pm_verbose,
                                            report_freq=opts.pm_report_freq)
            # inflate slightly for stability
            hess_norm *= 1.05
    else:
        hess_norm = opts.hess_norm
        print(f"Using provided hess-norm of beta = {hess_norm:.3e}", file=log)

    print("Setting up dictionary", file=log)
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)
    psi = Psi(nband, nx, ny, bases, opts.nlevels, opts.nthreads)
    Nxmax = psi.Nxmax
    Nymax = psi.Nymax

    # number of frequency basis functions
    if opts.nbasisf is None:
        nbasisf = int(np.sum(fsel))
    else:
        nbasisf = opts.nbasisf
    print(f"Using {nbasisf} frequency basis functions", file=log)

    # a value less than zero turns L1 reweighting off
    # we'll start on convergence or at the iteration
    # indicated by l1-reweight-from, whichever comes first
    l1_reweight_from = opts.l1_reweight_from
    l1reweight_active = False
    # we need an array to put the components in for reweighting
    outvar = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=real_type)

    dual = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=residual.dtype)
    if l1_reweight_from == 0:
        print('Initialising with L1 reweighted', file=log)
        if not update.any():
            raise ValueError("Cannot reweight before any updates have been performed")
        # divide by taper so as not to bias rms_comps
        psi.dot(update/taperxy, outvar)
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
    print(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    if opts.skip_model:
        mrange = []
    else:
        mrange = range(iter0, iter0 + opts.niter)
    for k in mrange:
        print('Solving for update', file=log)
        update = precond(residual, 'backward')
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs,
                  fits_oname + f'_{opts.suffix}_update_{k+1}.fits',
                  hdr_mfs)

        modelp = deepcopy(model)
        xtilde = model + opts.gamma * update
        grad21 = lambda x: -precond(xtilde - x, 'forward')/opts.gamma
        if iter0 == 0:
            lam = opts.init_factor * opts.rmsfactor * rms
        else:
            lam = opts.rmsfactor*rms
        print(f'Solving for model with lambda = {lam}', file=log)
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

        # write component model
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

        print(f'Computing residual', file=log)
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            resid = compute_residual(ds_name,
                                     nx, ny,
                                     cell_rad, cell_rad,
                                     ds_name,
                                     model[b],
                                     nthreads=opts.nthreads,
                                     epsilon=opts.epsilon,
                                     do_wgridding=opts.do_wgridding,
                                     double_accum=opts.double_accum)
            residual[b] = resid
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
            ds = ds.assign(**{
                'MODEL': (('x', 'y'), model[b]),
                'MODEL_BEST': (('x', 'y'), best_model[b]),
                'UPDATE': (('x', 'y'), update[b]),
            })
            attrs = {}
            attrs['rms'] = best_rms
            attrs['rmax'] = best_rmax
            attrs['niters'] = k+1
            attrs['hess_norm'] = hess_norm
            ds = ds.assign_attrs(**attrs)
            ds.to_zarr(ds_name, mode='a')


        print(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        if eps < opts.tol:
            # do not converge prematurely
            if l1_reweight_from > 0 and not l1reweight_active:  # only happens once
                # start reweighting
                l1_reweight_from = k+1 - iter0
                l1reweight_active = True
            else:
                print(f"Converged after {k+1} iterations.", file=log)
                break

        if k+1 - iter0 >= l1_reweight_from:
            print('Computing L1 weights', file=log)
            # divide by taper so as not to bias rms_comps
            psi.dot(update/taperxy, outvar)
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

    return
