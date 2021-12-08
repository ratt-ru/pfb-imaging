# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('BACKWARD')

@cli.command(context_settings={'show_default': True})
@click.option('-xds', '--xds', type=str, required=True,
              help="Path to xarray dataset containing data products")
@click.option('-mds', '--mds', type=str, required=True,
              help="Path to xarray dataset containing model, dual and "
              "weights from previous iteration.")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-mask', '--mask',
              help="Path to mask.fits.")
@click.option('-pmask', '--point-mask',
              help="Path to point source mask.fits.")
@click.option('-bases', '--bases', default='self',
              help='Wavelet bases to use. Give as str separated by | eg.'
              '-bases self|db1|db2|db3|db4')
@click.option('-nlevels', '--nlevels', default=3,
              help='Number of wavelet decomposition levels')
@click.option('-hessnorm', '--hessnorm', type=float,
              help="Spectral norm of Hessian approximation")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('-sinv', '--sigmainv', type=float, default=1e-3,
              help='Standard deviation of assumed GRF prior used '
              'for preconditioning.')
@click.option('-sig21', '--sigma21', type=float, default=1e-3,
              help='Sparsity threshold level.')
@click.option('-niter', '--niter', type=int, default=10,
              help='Number of reweighting iterations. '
              'Reweighting will take place after every primal dual run.')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('--use-beam/--no-use-beam', default=True)
@click.option('--use-psf/--no-use-psf', default=True)
@click.option('-pdtol', "--pd-tol", type=float, default=1e-5,
              help="Tolerance of conjugate gradient")
@click.option('-pdmaxit', "--pd-maxit", type=int, default=100,
              help="Maximum number of iterations for primal dual")
@click.option('-pdverb', "--pd-verbose", type=int, default=1,
              help="Verbosity of primal dual. "
              "Set to 2 for debugging or zero for silence.")
@click.option('-pdrf', "--pd-report-freq", type=int, default=10,
              help="Report freq for primal dual.")
@click.option('-pmtol', "--pm-tol", type=float, default=1e-5,
              help="Tolerance of power method")
@click.option('-pmmaxit', "--pm-maxit", type=int, default=100,
              help="Maximum number of iterations for power method")
@click.option('-pmverb', "--pm-verbose", type=int, default=1,
              help="Verbosity of power method. "
              "Set to 2 for debugging or zero for silence.")
@click.option('-pmrf', "--pm-report-freq", type=int, default=10,
              help="Report freq for power method.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
def backward(**kw):
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
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')

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
    from astropy.io import fits
    import pywt

    xds = xds_from_zarr(args.xds, chunks={'row':args.row_chunk})
    nband = xds[0].nband
    nx = xds[0].nx
    ny = xds[0].ny
    wsum = 0.0
    for ds in xds:
        wsum += ds.WSUM.values.sum()

    try:
        # always only one mds
        mds = xr.open_zarr(args.mds,
                           chunks={'band': 1, 'x': -1, 'y': -1})
    except Exception as e:
        print(f"You have to pass in a valid model dataset. "
              f"No model found at {args.mds}", file=log)
        raise e

    if 'UPDATE' in mds:
        update = mds.UPDATE.values
        assert update.shape == (nband, nx, ny)
    else:
        raise ValueError("No update found in model dataset. "
                         "Use forward worker to populate it. ", file=log)

    if 'MODEL' in mds:
        model = mds.MODEL.values
        assert model.shape == (nband, nx, ny)
    else:
        model = np.zeros((nband, nx, ny), dtype=xds[0].DIRTY.dtype)

    data = model + update

    if args.mask is not None:
        print("Initialising mask", file=log)
        mask = load_fits(args.mask).squeeze()
        # could be using a model to initialise mask
        if len(mask.shape) == 3:
            mask = np.any(mask, axis=0)
        elif len(mask.shape) == 2:
            mask = np.where(mask, 1.0, 0.0)
        else:
            raise ValueError("Incorrect number of dimensions for mask")
        assert mask.shape == (nx, ny)
        mask = mask[None].astype(model.dtype)
    else:
        mask = np.ones((1, nx, ny), dtype=model.dtype)

    if args.point_mask is not None:
        print("Initialising point source mask", file=log)
        pmask = load_fits(args.point_mask).squeeze()
        # passing model as mask
        if len(pmask.shape) == 3:
            print("Detected third axis on pmask. "
                  "Initialising pmask from model.", file=log)
            pmask = np.any(pmask, axis=0).astype(model.dtype)

        assert pmask.shape == (nx, ny)
    else:
        pmask = np.ones((nx, ny), dtype=model.dtype)


    # dictionary setup
    print("Setting up dictionary", file=log)
    bases = args.bases.split('|')
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
    psiH = partial(im2coef, pmask=pmask, bases=bases, ntot=ntots, nmax=nmax,
                   nlevels=args.nlevels)
    psi = partial(coef2im, pmask=pmask, bases=bases, padding=padding,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    hessopts = {}
    hessopts['cell'] = xds[0].cell_rad
    hessopts['wstack'] = args.wstack
    hessopts['epsilon'] = args.epsilon
    hessopts['double_accum'] = args.double_accum
    hessopts['nthreads'] = args.nvthreads

    if args.use_psf:
        print("Initialising psf", file=log)
        from pfb.operators.psf import psf_convolve_xds
        from ducc0.fft import r2c
        normfact = 1.0

        psf = xds[0].PSF.data
        _, nx_psf, ny_psf = psf.shape

        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding_psf = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding_psf[-1])

        # add psfhat to Dataset
        for i, ds in enumerate(xds):
            psf_pad = iFs(ds.PSF.data.compute(), axes=(1, 2))
            psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                         nthreads=args.nthreads, inorm=0)

            psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)
            ds = ds.assign({'PSFHAT':(('band', 'x_psf', 'y_psfo2'), psfhat)})
            xds[i] = ds

        psfopts = {}
        psfopts['padding'] = padding_psf[1:]
        psfopts['unpad_x'] = unpad_x
        psfopts['unpad_y'] = unpad_y
        psfopts['lastsize'] = lastsize
        psfopts['nthreads'] = args.nvthreads

        hess = partial(psf_convolve_xds, xds=xds, psfopts=psfopts,
                       sigmainv=args.sigmainv, wsum=wsum, mask=mask,
                       compute=True)

    else:
        hess = partial(hessian_xds, xds=xds, hessopts=hessopts,
                       sigmainv=args.sigmainv, wsum=wsum, mask=mask,
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

    # psisq = lambda x: psi(psiH(x))

    # psinorm = power_method(psisq, (nband, nx, ny), tol=args.pm_tol,
    #                        maxit=args.pm_maxit, verbosity=args.pm_verbose,
    #                        report_freq=args.pm_report_freq)

    # quit()

    # import pdb; pdb.set_trace()

    print("Solving for model", file=log)
    for i in range(args.niter):
        # prox = partial(prox_21m, sigma=args.sigma21, weight=weight, axis=0)
        model, dual = primal_dual(hess, data, model, dual, args.sigma21,
                                  psi, psiH, weight, hessnorm, prox_21m,
                                  nu=nbasis,
                                  tol=args.pd_tol, maxit=args.pd_maxit,
                                  verbosity=args.pd_verbose,
                                  report_freq=args.pd_report_freq)

        # # reweight
        # l2_norm = np.linalg.norm(psi.hdot(model), axis=1)
        # for m in range(psi.nbasis):
        #     if adapt_sig21:
        #         _, sigmas[m] = expon.fit(l2_norm[m], floc=0.0)
        #         print('basis %i, sigma %f'%sigmas[m], file=log)

        #     weights21[m] = alpha[m]/(alpha[m] + l2_norm[m]) * sigmas[m]/sig_21


    model = da.from_array(model, chunks=(1, -1, -1), name=False)
    dual = da.from_array(dual, chunks=(1, -1, -1), name=False)
    weight = da.from_array(weight, chunks=(-1, -1), name=False)

    mds = mds.assign(**{
                     #'UPDATE': (('band', 'x', 'y'), update),
                     'MODEL': (('band', 'x', 'y'), model),
                     'DUAL': (('band', 'basis', 'coef'), dual),
                     'WEIGHT': (('basis', 'coef'), weight)})

    mds.to_zarr(args.mds, mode='a')



    # compute apparent residual per dataset
    from pfb.operators.hessian import hessian
    # Required because of https://github.com/ska-sa/dask-ms/issues/171
    xdsw = xds_from_zarr(args.xds, chunks={'band': 1}, columns='DIRTY')
    writes = []
    for ds, dsw in zip(xds, xdsw):
        dirty = ds.DIRTY.data
        wgt = ds.WEIGHT.data
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        fbin_idx = ds.FBIN_IDX.data
        fbin_counts = ds.FBIN_COUNTS.data
        beam = ds.BEAM.data
        band_id = ds.band_id.data
        # we only want to apply the beam once here
        residual = (dirty -
                    hessian(uvw, wgt, freq, beam * model[band_id], None,
                    fbin_idx, fbin_counts, hessopts))
        dsw = dsw.assign(**{'RESIDUAL': (('band', 'x', 'y'), residual)})
        writes.append(xds_to_zarr(dsw, args.xds, columns='RESIDUAL'))

    dask.compute(writes)

    xds = xds_from_zarr(args.xds, chunks={'band': 1})
    residual = []
    wsum = 0
    for ds in xds:
        wsum += ds.WSUM.data.sum()
        residual.append(ds.RESIDUAL.data)
    residual = da.stack(residual).sum(axis=0)/wsum
    residual = residual.compute()

    # construct a header from xds attrs
    ra = mds.ra
    dec = mds.dec
    radec = [ra, dec]

    cell_rad = mds.cell_rad
    cell_deg = np.rad2deg(cell_rad)

    freq_out = mds.band.values
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    # TODO - add wsum info

    print("Saving results", file=log)
    save_fits(args.output_filename + '_model.fits', model, hdr)
    model_mfs = np.mean(model, axis=0)
    save_fits(args.output_filename + '_model_mfs.fits', model_mfs, hdr_mfs)
    save_fits(args.output_filename + '_residual.fits', residual, hdr)
    residual_mfs = np.sum(residual, axis=0)
    save_fits(args.output_filename + '_residual_mfs.fits', residual_mfs, hdr_mfs)

    print("All done here.", file=log)
