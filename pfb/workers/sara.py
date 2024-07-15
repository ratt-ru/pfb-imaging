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

# create default parameters from schema
defaults = {}
for key in schema.sara["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.sara["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.sara)
def sara(**kw):
    '''
    Deconvolution using SARA regularisation
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)

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

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/sara_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import xds_from_url

    basename = f'{basedir}/{oname}'
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_store = DaskMSStore(f'{basename}_{opts.suffix}.dds')

    with ExitStack() as stack:
        ti = time.time()
        _sara(**opts)

        dds = xds_from_url(dds_store.url)

        from pfb.utils.fits import dds2fits, dds2fits_mfs

        if opts.fits_mfs or opts.fits:
            print(f"Writing fits files to {fits_oname}_{opts.suffix}",
                  file=log)

        # convert to fits files
        if opts.fits_mfs:
            dds2fits_mfs(dds,
                         'RESIDUAL',
                         f'{fits_oname}_{opts.suffix}',
                         norm_wsum=True)
            dds2fits_mfs(dds,
                         'MODEL',
                         f'{fits_oname}_{opts.suffix}',
                         norm_wsum=False)

        if opts.fits_cubes:
            dds2fits(dds,
                     'RESIDUAL',
                     f'{fits_oname}_{opts.suffix}',
                     norm_wsum=True)
            dds2fits(dds,
                     'MODEL',
                     f'{fits_oname}_{opts.suffix}',
                     norm_wsum=False)

    print(f"All done after {time.time() - ti}s", file=log)


def _sara(ddsi=None, **kw):
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
    from pfb.utils.misc import l1reweight_func
    from pfb.operators.psf import psf_convolve_cube2
    from pfb.operators.psi import Psi
    from pfb.operators.gridder import compute_residual
    from copy import copy, deepcopy
    from ducc0.misc import empty_noncritical
    from pfb.prox.prox_21m import prox_21m_numba as prox_21
    # from pfb.prox.prox_21 import prox_21
    from pfb.utils.misc import fitcleanbeam, fit_image_cube
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
        # are these sorted correctly?
        dds = xds_from_url(dds_store.url)

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

    # stitch dirty/psf in apparent scale
    # drop_vars to avoid duplicates in memory
    output_type = dds[0].DIRTY.dtype
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
        model = np.zeros((nband, nx, ny))
    psf = np.stack([ds.PSF.values for ds in dds], axis=0)
    dds = [ds.drop_vars('PSF') for ds in dds]
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands

    wsum = np.sum(wsums)
    psf /= wsum
    residual /= wsum
    psf_mfs = np.sum(psf, axis=0)
    residual_mfs = np.sum(residual, axis=0)

    # for intermediary results (not currently written)
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

    # image space hessian
    # pre-allocate arrays for doing FFT's
    xhat = empty_noncritical(psf.shape, dtype='c16')
    nxpadl = (nx_psf - nx)//2
    nxpadr = nx_psf - nx - nxpadl
    nypadl = (ny_psf - ny)//2
    nypadr = ny_psf - ny - nypadl
    if nx_psf != nx:
        slicex = slice(nxpadl, -nxpadr)
    else:
        slicex = slice(None)
    if ny_psf != ny:
        slicey = slice(nypadl, -nypadr)
    else:
        slicey = slice(None)
    psf = np.fft.fftshift(psf.astype('c16'), axes=(1,2))
    c2c(psf, axes=(1,2), forward=True, inorm=0, out=psf,
        nthreads=opts.nthreads)
    psf_convolve = partial(psf_convolve_cube2,
                           xhat,
                           psf,
                           slicex,
                           slicey,
                           nthreads=opts.nthreads)

    if opts.hess_norm is None:
        print("Finding spectral norm of Hessian approximation", file=log)
        hess_norm, hessbeta = power_method(psf_convolve, (nband, nx, ny),
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
    psiHoutvar = np.zeros((nband, nbasis, Nymax, Nxmax),
                          dtype=residual.dtype)
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

    # TODO - should we cache this?
    dual = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=residual.dtype)
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
        l1weight = np.ones((nbasis, Nymax, Nxmax), dtype=residual.dtype)
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
                                  hess_norm,
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
        print(f"Writing model to {basename}_{opts.suffix}_model.mds",
              file=log)
        try:
            coeffs, Ix, Iy, expr, params, texpr, fexpr = \
                fit_image_cube(time_out,
                               freq_out[fsel],
                               model[None, fsel, :, :],
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
            resid = compute_residual(ds_name,
                                     nx, ny,
                                     cell_rad, cell_rad,
                                     ds_name,
                                     model[b],
                                     x0=ds.x0, y0=ds.y0,
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
        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        # what to base this on?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()
            for ds_name, ds in zip(dds_list, dds):
                b = int(ds.bandid)
                ds = ds.assign(**{
                    'MODEL_BEST': (('x', 'y'), best_model[b])
                })
                ds = ds.assign_attrs(**{
                    'rms': best_rms,
                    'rmax': best_rmax
                })
                ds.to_zarr(ds_name, mode='a')


        print(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        if eps < opts.tol:
            # do not converge prematurely
            if k+1 - iter0 < l1reweight_from:  # only happens once
                # start reweighting
                l1reweight_from = k+1 - iter0
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

        # keep track of total number of iterations
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            ds = ds.assign_attrs(**{
                'niter': k,
            })

            ds.to_zarr(ds_name, mode='a')

    return

