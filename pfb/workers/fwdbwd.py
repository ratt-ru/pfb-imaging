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
log = pyscilog.get_logger('FWDBWD')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.fwdbwd["inputs"].keys():
    defaults[key] = schema.fwdbwd["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.fwdbwd)
def fwdbwd(**kw):
    '''
    Minimises

    (V - R f(x)).H W (V - R f(x)) + sigma_{21} | psi.H x |_{2,1}

    where f: R^N -> R^N is some function, R is the degridding operator
    and psi is an over-complete dictionary of functions. Here sigma_{21}
    is the strength of the regulariser.
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{str(ldir)}/fwdbwd_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/fwdbwd_{timestamp}.log', file=log)

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _fwdbwd(**opts)

def _fwdbwd(ddsi=None, **kw):
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
    from pfb.opt.primal_dual import primal_dual_optimised2 as primal_dual
    from pfb.utils.misc import l1reweight_func, setup_non_linearity
    from pfb.operators.hessian import hessian_xds
    from pfb.operators.psf import psf_convolve_cube
    from pfb.operators.psi import im2coef
    from pfb.operators.psi import coef2im
    from copy import copy, deepcopy
    from ducc0.misc import make_noncritical
    from pfb.wavelets.wavelets import wavelet_setup
    from pfb.prox.prox_21m import prox_21m_numba as prox_21
    # from pfb.prox.prox_21 import prox_21
    from pfb.utils.misc import fitcleanbeam

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}_{opts.postfix}.dds'
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

    # set up vis space Hessian
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['do_wgridding'] = opts.do_wgridding
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads  # nvthreads since dask parallel over band
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
                           nthreads=opts.nvthreads*opts.nthreads_dask)  # nthreads = nvthreads*nthreads_dask because dask not involved

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
                   nthreads=opts.nvthreads*opts.nthreads_dask) # nthreads = nvthreads*nthreads_dask because dask not involved
    psi = partial(coef2im,
                  bases=bases,
                  ntot=ntot,
                  iy=iy,
                  sy=sy,
                  nx=nx,
                  ny=ny,
                  nthreads=opts.nvthreads*opts.nthreads_dask) # nthreads = nvthreads*nthreads_dask because dask not involved

    # get clean beam area to convert residual units during l1reweighting
    # TODO - could refine this with comparison between dirty and restored
    # if contiuing the deconvolution
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    pix_per_beam = GaussPar[0]*GaussPar[1]*np.pi/4
    print(f"Number of pixels per beam estimated as {pix_per_beam}",
          file=log)

    # We do the following to set hyper-parameters in an intuitive way
    # i) convert residual units so it is comparable to model
    # ii) project residual into dual domain
    # iii) compute the rms in the space where thresholding happens
    psiHoutvar = np.zeros((nband, nbasis, nmax), dtype=dirty.dtype)
    fsel = wsums > 0
    tmp2 = residual.copy()
    tmp2[fsel] *= wsum/wsums[fsel, None, None]
    psiH(tmp2/pix_per_beam, psiHoutvar)
    rms_comps = np.std(np.sum(psiHoutvar, axis=0),
                       axis=-1)[:, None]  # preserve axes

    func, finv, dfunc = setup_non_linearity(mode=opts.non_linearity)

    def gradf(residual, x):
        return -2*residual * dfunc(x)

    def hessian_psf(psfo, x0, sigmainv, v):
        '''
        psfo is the convolution operator and x0 is the fixed value of x at which
        we evaluate the operator. v is the vector to be acted on.
        '''
        dx0 = dfunc(x0)
        return 2 * dx0 * psfo(dx0 * v)  + v*sigmainv

    if 'PARAM' in dds[0] and dds[0].non_linearity == opts.non_linearity:
        print("Found matching non-linearity for PARAM in dds", file=log)
        x = [ds.PARAM.data for ds in dds]
        x = da.stack(x).compute()
    elif model.any() and not opts.restart:
        print("Initialsing PARAM from MODEL in dds", file=log)
        # fall back and compute param from model in this case
        x = finv(model)
        # finv is not necessarily exact so we need to recompute residual
        print("Computing residual", file=log)
        model = func(x)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')
        # in this case the dual is also probably not useful
        if dual is not None:
            dual[...] = 0.0
    else:
        print("Initialising PARAM to all zeros", file=log)
        x = np.zeros_like(dirty)
        model = func(x)
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()

    if dual is None:
        dual = np.zeros((nband, nbasis, nmax), dtype=dirty.dtype)
        l1weight = np.ones((nbasis, nmax), dtype=dirty.dtype)
        reweighter = None
    else:
        if opts.l1reweight_from == 0:
            print('Initialising with L1 reweighted', file=log)
            reweighter = partial(l1reweight_func, psiH, psiHoutvar, opts.rmsfactor, rms_comps)
            l1weight = reweighter(x)
            # l1weight[l1weight < 1.0] = 0.0
        else:
            l1weight = np.ones((nbasis, nmax), dtype=dirty.dtype)
            reweighter = None


    # for generality the prox function only takes the
    # array variable and step size as inputs
    # prox21 = partial(prox_21, weight=l1weight)

    hessbeta = None
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    print(f"Iter 0: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    for k in range(opts.niter):
        xp = x.copy()
        j = -gradf(residual, xp)
        print("Finding spectral norm of Hessian approximation", file=log)
        # hessian depends on x and sigmainv so need to do this at every iteration
        sigmainv = np.maximum(np.std(j), opts.sigmainv)
        hesspsf = partial(hessian_psf, psf_convolve, xp, sigmainv)
        hessnorm, hessbeta = power_method(hesspsf, (nband, nx, ny),
                                          b0=hessbeta,
                                          tol=opts.pm_tol,
                                          maxit=opts.pm_maxit,
                                          verbosity=opts.pm_verbose,
                                          report_freq=opts.pm_report_freq)


        print(f"Solving forward step with sigmainv = {sigmainv}", file=log)
        delx = pcg(hesspsf,
                   j,
                   tol=opts.cg_tol,
                   maxit=opts.cg_maxit,
                   minit=opts.cg_minit,
                   verbosity=opts.cg_verbose,
                   report_freq=opts.cg_report_freq,
                   backtrack=opts.backtrack)

        save_fits(np.mean(delx, axis=0),
                  basename + f'_{opts.postfix}_update_{k+1}.fits',
                  hdr_mfs)

        if opts.sigma21 is None:
            sigma21 = opts.rmsfactor*np.std(j)
        else:
            sigma21 = opts.sigma21
        print(f'Solving bacward step with sig21 = {sigma21}', file=log)
        data = xp + opts.gamma * delx
        if opts.non_linearity != 'id':
            bedges = np.histogram_bin_edges(data.ravel(), bins='fd')
            dhist, _ = np.histogram(data.ravel(), bins=bedges)
            dmax = dhist.argmax()
            dmode = (bedges[dmax] + bedges[dmax+1])/2.0
            data -= dmode
            x -= dmode
        grad21 = lambda v: hesspsf(v - data)
        x, dual = primal_dual(x,
                              dual,
                              sigma21,
                              psi,
                              psiH,
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
        if opts.non_linearity != 'id':
            x += dmode
        save_fits(np.mean(x, axis=0),
                  basename + f'_{opts.postfix}_param_{k+1}.fits',
                  hdr_mfs)

        model = func(x)
        save_fits(np.mean(model, axis=0),
                  basename + f'_{opts.postfix}_model_{k+1}.fits',
                  hdr_mfs)

        print("Getting residual", file=log)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        save_fits(residual_mfs,
                  basename + f'_{opts.postfix}_residual_{k+1}.fits',
                  hdr_mfs)

        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(x - xp)/np.linalg.norm(x)

        print(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        if k+1 >= opts.l1reweight_from:
            print('Computing L1 weights', file=log)
            # convert residual units so it is comparable to model
            tmp2[fsel] = residual[fsel] * wsum/wsums[fsel, None, None]
            psiH(tmp2/pix_per_beam, psiHoutvar)
            rms_comps = np.std(np.sum(psiHoutvar, axis=0),
                               axis=-1)[:, None]  # preserve axes
            # we redefine the reweighter here since the rms has changed
            reweighter = partial(l1reweight_func, psiH, psiHoutvar, opts.rmsfactor, rms_comps)
            l1weight = reweighter(x)
            # l1weight[l1weight < 1.0] = 0.0
            # prox21 = partial(prox_21, weight=l1weight, axis=0)

        # import ipdb; ipdb.set_trace()
        print("Updating results", file=log)
        dds_out = []
        for ds in dds:
            b = ds.bandid
            r = da.from_array(residual[b]*wsum)
            m = da.from_array(model[b])
            d = da.from_array(dual[b])
            xb = da.from_array(x[b])
            ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                                  'MODEL': (('x', 'y'), m),
                                  'DUAL': (('c', 'n'), d),
                                  'PARAM': (('x', 'y'), xb)})
            ds_out = ds_out.assign_attrs({'non_linearity': opts.non_linearity})
            dds_out.append(ds_out)
        writes = xds_to_zarr(dds_out, dds_name,
                             columns=('RESIDUAL', 'MODEL', 'DUAL', 'PARAM'),
                             rechunk=True)
        dask.compute(writes)

        if eps < opts.tol:
            print(f"Converged after {k+1} iterations.", file=log)
            break


    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})

    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds, 'MODEL', f'{basename}_{opts.postfix}', norm_wsum=False))
        fitsout.append(dds2fits_mfs(dds, 'PARAM', f'{basename}_{opts.postfix}', norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))
        fitsout.append(dds2fits(dds, 'MODEL', f'{basename}_{opts.postfix}', norm_wsum=False))
        fitsout.append(dds2fits(dds, 'PARAM', f'{basename}_{opts.postfix}', norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    print("All done here.", file=log)
