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
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{str(ldir)}/sara_{timestamp}.log')

    print(f'Logs will be written to {str(ldir)}/spotless_{timestamp}.log', file=log)
    from daskms.experimental.zarr import xds_from_zarr
    from daskms.fsspec_store import DaskMSStore
    import fsspec
    # TODO - there must be a neater way to do this with fsspec
    # basedir = Path(opts.output_filename).resolve().parent
    # basedir.mkdir(parents=True, exist_ok=True)
    # basename = f'{opts.output_filename}_{opts.product.upper()}'
    if '://' in opts.output_filename:
        protocol = opts.output_filename.split('://')[0]
    else:
        protocol = 'file'

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path('/'.join(opts.output_filename.split('/')[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = (opts.output_filename.split('/')[-1] + f'_{opts.product.upper()}'
             + f'_{opts.suffix}')
    basename = f'{basedir}/{oname}'
    opts.output_filename = basename
    dds_name = f'{basename}.dds'
    dds_store = DaskMSStore(dds_name)

    if opts.fits_output_folder is not None:
        # this should be a file system
        fs = fsspec.filesystem('file')
        fbasedir = fs.expand_path(opts.fits_output_folder)[0]
        if not fs.exists(fbasedir):
            fs.makedirs(fbasedir)
        fits_oname = f'{fbasedir}/{oname}'
        opts.fits_output_folder = fbasedir
    else:
        fits_oname = f'{basedir}/{oname}'
        opts.fits_output_folder = basedir

    OmegaConf.set_struct(opts, True)

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

        ti = time.time()
        _sara(**opts)

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})

    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds, 'RESIDUAL',
                                    f'{fits_oname}',
                                    norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds, 'MODEL',
                                    f'{fits_oname}',
                                    norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds, 'RESIDUAL',
                                f'{fits_oname}',
                                norm_wsum=True))
        fitsout.append(dds2fits(dds, 'MODEL',
                                f'{fits_oname}',
                                norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    print(f"All done here after {time.time() - ti}s", file=log)


def _sara(ddsi=None, **kw):
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
    from ducc0.misc import resize_thread_pool, thread_pool_size
    nthreads_tot = opts.nthreads_dask * opts.nvthreads
    resize_thread_pool(nthreads_tot)
    print(f'ducc0 max number of threads set to {thread_pool_size()}', file=log)

    basename = opts.output_filename
    fits_oname = opts.fits_output_folder + basename.split('/')[1]

    dds_name = f'{basename}.dds'
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
        print(f"Writing model to {basename}_model.mds", file=log)
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
            coeff_dataset.to_zarr(f"{basename}_model.mds",
                                  mode='w')
        except Exception as e:
            print(f"Exception {e} raised during model fit .", file=log)

        save_fits(np.mean(model, axis=0),
                  fits_oname + f'_model_{k+1}.fits',
                  hdr_mfs)

        print("Getting residual", file=log)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        save_fits(residual_mfs,
                  fits_oname + f'_residual_{k+1}.fits',
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

    return

