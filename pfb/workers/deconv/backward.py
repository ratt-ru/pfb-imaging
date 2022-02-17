# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('BACKWARD')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.backward)
def backward(model_name='MODEL',
             mask=None,
             nband=None,
             output_filename=None,
             product='I',
             row_chunk=-1,
             sigmainv=1e-5,
             sigma21=1e-3,
             positivity=True,
             bases='self,db1,db2',
             nlevels=3,
             hessnorm=None,
             niter=5,
             use_psf=True,
             fits_mfs=True,
             fits_cubes=False,
             do_residual=True,
             pd_tol=1e-5,
             pd_maxit=100,
             pd_verbose=1,
             pd_report_freq=25,
             pm_tol=1e-5,
             pm_maxit=50,
             pm_verbose=1,
             pm_report_freq=50,
             host_address=None,
             nworkers=1,
             nthreads_per_worker=1,
             nvthreads=None,
             nthreads=None,
             mem_limit=None,
             scheduler='single-threaded',
             epsilon=1e-7,
             wstack=True,
             double_accum=True):
    '''
    Solves

    argmin_x r(x) + (v - x).H U (v - x) / (2 * gamma)

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default we will use all
    available resources.

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = 1

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    if LocalCluster:
        nvthreads = nthreads//(nworkers*nthreads_per_worker)
    else:
        nvthreads = nthreads//nthreads-per-worker

    where nvthreads refers to the number of threads used to scale vertically
    (eg. the number threads given to each gridder instance).

    '''
    args = OmegaConf.create(locals())
    pyscilog.log_to_file(f'{args.output_filename}_{args.product}.log')

    if args.nworkers is None:
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log, scheduler=args.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _backward(**args)

def _backward(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)

    import numpy as np
    import dask
    import dask.array as da
    import xarray as xr
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.operators.psi import im2coef, coef2im
    from pfb.operators.hessian import hessian_xds
    from pfb.opt.primal_dual import primal_dual
    from pfb.opt.power_method import power_method
    from pfb.prox.prox_21m import prox_21m
    from pfb.prox.prox_21 import prox_21
    from astropy.io import fits
    import pywt

    basename = f'{args.output_filename}_{args.product.upper()}'

    xds_name = f'{basename}.xds.zarr'
    mds_name = f'{basename}.mds.zarr'

    xds = xds_from_zarr(xds_name, chunks={'row':args.row_chunk})
    # daskms bug?
    for i, ds in enumerate(xds):
        xds[i] = ds.chunk({'row':-1})
    # only a single mds (for now)
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny
    for ds in xds:
        assert ds.nx == nx
        assert ds.ny == ny

    if 'UPDATE' in mds:
        update = mds.UPDATE.values
        assert update.shape == (nband, nx, ny)
    else:
        raise ValueError("No update found in model dataset. "
                         "Use forward worker to populate it. ", file=log)

    if args.model_name in mds:
        model = mds.get(args.model_name).values
        assert model.shape == (nband, nx, ny)
        print(f"Initialising model from {args.model_name} in mds", file=log)
    else:
        print('Initialising model to zeros', file=log)
        model = np.zeros((nband, nx, ny), dtype=xds[0].DIRTY.dtype)

    data = model + update

    from pfb.utils.misc import init_mask
    mask = init_mask(args.mask, mds, xds[0].DIRTY.dtype, log)

    # dictionary setup
    print("Setting up dictionary", file=log)
    bases = args.bases.split(',')
    ntots = []
    iys = {}
    sys = {}
    for base in bases:
        if base == 'self':
            y, iy, sy = model[0].ravel(), 0, 0
        else:
            alpha = pywt.wavedecn(model[0], base, mode='zero',
                                  level=args.nlevels)
            y, iy, sy = pywt.ravel_coeffs(alpha)
        iys[base] = iy
        sys[base] = sy
        ntots.append(y.size)

    # get padding info
    nmax = np.asarray(ntots).max()
    padding = []
    nbasis = len(ntots)
    for i in range(nbasis):
        padding.append(slice(0, ntots[i]))


    # initialise dictionary operators
    bases = da.from_array(np.array(bases, dtype=object), chunks=-1)
    ntots = da.from_array(np.array(ntots, dtype=object), chunks=-1)
    padding = da.from_array(np.array(padding, dtype=object), chunks=-1)
    psiH = partial(im2coef, bases=bases, ntot=ntots, nmax=nmax,
                   nlevels=args.nlevels)
    psi = partial(coef2im, bases=bases, padding=padding,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    # we set the alphas used for reweighting using the
    # current clean residuals when available
    alpha = np.ones(nbasis) * 1e-5
    if 'CLEAN_RESIDUAL' in xds[0]:
        cresid = mds.CLEAN_RESIDUAL.values
        resid_comps = psiH(cresid)
        for m in range(nbasis):
            alpha[m] = np.std(resid_comps[:, m])

    wsum = 0.0
    for ds in xds:
        wsum += ds.WSUM.values[0]

    hessopts = {}
    hessopts['cell'] = xds[0].cell_rad
    hessopts['wstack'] = args.wstack
    hessopts['epsilon'] = args.epsilon
    hessopts['double_accum'] = args.double_accum
    hessopts['nthreads'] = args.nvthreads

    if args.use_psf:
        from pfb.operators.psf import psf_convolve_xds

        nx_psf, ny_psf = xds[0].nx_psf, xds[0].ny_psf
        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding[-1])

        psfopts = {}
        psfopts['padding'] = padding
        psfopts['unpad_x'] = unpad_x
        psfopts['unpad_y'] = unpad_y
        psfopts['lastsize'] = lastsize
        psfopts['nthreads'] = args.nvthreads

        hess = partial(psf_convolve_xds, xds=xds, psfopts=psfopts,
                       wsum=wsum, sigmainv=args.sigmainv, mask=mask,
                       compute=True)

    else:
        print("Solving for update using vis space approximation", file=log)
        hess = partial(hessian_xds, xds=xds, hessopts=hessopts,
                        wsum=wsum, sigmainv=args.sigmainv, mask=mask,
                        compute=True)

    if args.hessnorm is None:
        print("Finding spectral norm of Hessian approximation", file=log)
        hessnorm, _ = power_method(hess, (nband, nx, ny), tol=args.pm_tol,
                         maxit=args.pm_maxit, verbosity=args.pm_verbose,
                         report_freq=args.pm_report_freq)
    else:
        hessnorm = args.hessnorm

    if 'DUAL' in mds:
        dual = mds.DUAL.values
        assert dual.shape == (nband, nbasis, nmax)
    else:
        dual = np.zeros((nband, nbasis, nmax), dtype=model.dtype)

    if 'WEIGHT' in mds:
        weight = mds.WEIGHT.values
        assert weight.shape == (nbasis, nmax)
    else:
        weight = np.ones((nbasis, nmax), dtype=model.dtype)

    modelp = model.copy()
    print("Solving for model", file=log)
    for i in range(args.niter):
        # prox = partial(prox_21m, sigma=args.sigma21, weight=weight, axis=0)
        model, dual = primal_dual(hess, data, model, dual, args.sigma21,
                                  psi, psiH, weight, hessnorm, prox_21,
                                  nu=nbasis, positivity=args.positivity,
                                  tol=args.pd_tol, maxit=args.pd_maxit,
                                  verbosity=args.pd_verbose,
                                  report_freq=args.pd_report_freq)

        # reweight
        l2_norm = np.linalg.norm(psiH(model), axis=0)
        for m in range(nbasis):
            # if adapt_sig21:
            #     _, sigmas[m] = expon.fit(l2_norm[m], floc=0.0)
            #     print('basis %i, sigma %f'%sigmas[m], file=log)

            weight[m] = alpha[m]/(alpha[m] + l2_norm[m])

    print("Saving results", file=log)
    mask = np.any(model, axis=0).astype(bool)
    mask = da.from_array(mask, chunks=(-1, -1))
    model = da.from_array(model, chunks=(1, -1, -1), name=False)
    modelp = da.from_array(modelp, chunks=(1, -1, -1), name=False)
    dual = da.from_array(dual, chunks=(1, -1, -1), name=False)
    weight = da.from_array(weight, chunks=(-1, -1), name=False)

    mds = mds.assign(**{
                     'MASK': (('x', 'y'), mask),
                     'MODEL': (('band', 'x', 'y'), model),
                     'MODELP': (('band', 'x', 'y'), modelp),
                     'DUAL': (('band', 'basis', 'coef'), dual),
                     'WEIGHT': (('basis', 'coef'), weight)})

    dask.compute(xds_to_zarr(mds, mds_name, columns='ALL'))

    # # debugging
    # model = da.from_array(update, chunks=(1, -1, -1))

    print("Computing residual", file=log)
    # compute apparent residual per dataset
    from pfb.operators.hessian import hessian
    # Required because of https://github.com/ska-sa/dask-ms/issues/171
    xdsw = xds_from_zarr(xds_name, columns='DIRTY')
    writes = []
    for ds, dsw in zip(xds, xdsw):
        dirty = ds.DIRTY.data
        wgt = ds.WEIGHT.data
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        beam = ds.BEAM.data
        b = ds.bandid
        # we only want to apply the beam once here
        residual = (dirty -
                    hessian(beam * model[b], uvw, wgt, freq, None,
                    hessopts))
        dsw = dsw.assign(**{'RESIDUAL': (('x', 'y'), residual)})
        writes.append(dsw)

    dask.compute(xds_to_zarr(writes, xds_name, columns='RESIDUAL'))

    if args.fits_mfs or args.fits_cubes:
        print("Writing fits files", file=log)
        xds = xds_from_zarr(xds_name)
        residual = np.zeros((nband, nx, ny), dtype=xds[0].DIRTY.dtype)
        wsums = np.zeros(nband)
        for ds in xds:
            b = ds.bandid
            wsums[b] += ds.WSUM.values[0]
            residual[b] += ds.RESIDUAL.values
        wsum = np.sum(wsums)
        residual /= wsum

        # construct a header from xds attrs
        ra = mds.ra
        dec = mds.dec
        radec = [ra, dec]

        cell_rad = mds.cell_rad
        cell_deg = np.rad2deg(cell_rad)

        freq_out = mds.band.values
        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        model_mfs = np.mean(model, axis=0)
        save_fits(f'{basename}_model_mfs.fits', model_mfs, hdr_mfs)
        residual_mfs = np.sum(residual, axis=0)
        save_fits(f'{basename}_residual_mfs.fits', residual_mfs, hdr_mfs)

        if args.fits_cubes:
            # need residual in Jy/beam
            wsums = np.amax(psf, axes=(1,2))
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            fmask = wsums > 0
            residual[fmask] /= wsums[fmask, None, None]
            save_fits(f'{basename}_model.fits', model, hdr)
            save_fits(f'{basename}_residual.fits',
                      residual, hdr)

    print("All done here.", file=log)
