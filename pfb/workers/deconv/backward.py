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

# create default parameters from schema
defaults = {}
for key in schema.backward["inputs"].keys():
    defaults[key] = schema.backward["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.backward)
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
    # defaults.update(kw['nworkers'])
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}_{opts.product}{opts.postfix}.log')

    if opts.nworkers is None:
        opts.nworkers = opts.nband

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _backward(**opts)

def _backward(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import dask
    import dask.array as da
    import xarray as xr
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.utils.misc import setup_image_data, init_mask
    from pfb.operators.psi import im2coef, coef2im
    from pfb.operators.hessian import hessian_xds
    from pfb.opt.primal_dual import primal_dual
    from pfb.opt.power_method import power_method
    from pfb.prox.prox_21m import prox_21m
    from pfb.prox.prox_21 import prox_21
    from astropy.io import fits
    import pywt

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}{opts.postfix}.dds.zarr'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'

    dds = xds_from_zarr(dds_name, chunks={'row':opts.row_chunk})
    # only a single mds (for now)
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny
    for ds in dds:
        assert ds.nx == nx
        assert ds.ny == ny

    if 'UPDATE' in mds:
        update = mds.UPDATE.values
        assert update.shape == (nband, nx, ny)
    else:
        raise ValueError("No update found in model dataset. "
                         "Use forward worker to populate it. ", file=log)

    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)
    if opts.model_name in mds:
        model = mds.get(opts.model_name).values
        assert model.shape == (nband, nx, ny)
        print(f"Initialising model from {opts.model_name} in mds", file=log)
    else:
        print('Initialising model to zeros', file=log)
        model = np.zeros((nband, nx, ny), dtype=real_type)

    data = model + update

    from pfb.utils.misc import init_mask
    mask = init_mask(opts.mask, mds, dds[0].DIRTY.dtype, log)

    # dictionary setup
    print("Setting up dictionary", file=log)
    bases = opts.bases.split(',')
    ntots = []
    iys = {}
    sys = {}
    for base in bases:
        if base == 'self':
            y, iy, sy = model[0].ravel(), 0, 0
        else:
            alpha = pywt.wavedecn(model[0], base, mode='zero',
                                  level=opts.nlevels)
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
                   nlevels=opts.nlevels)
    psi = partial(coef2im, bases=bases, padding=padding,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    # residual after forward iteration can be useful for setting
    # hyper-parameters
    residual, wsum, _, psfhat, mean_beam = setup_image_data(dds, opts, 'FORWARD_RESIDUAL', log=log)

    # we set the alphas used for reweighting using the
    # current clean residuals when available
    if opts.alpha is None:
        alpha = np.ones(nbasis) * 1e-5
        if have_resid:
            resid_comps = psiH(cresid)
            for m in range(nbasis):
                alpha[m] = np.std(resid_comps[:, m])
        else:
            print("No residual in dds and alpha was not provided, "
                  "setting alpha to 1e-5.", file=log)
    else:
        alpha = np.ones(nbasis) * opts.alpha

    if opts.sigma21 is None:
        sigma21 = 1e-4
        if have_resid:
            resid_mfs = np.sum(residual, axis=0)
            sigma21 = np.std(resid_mfs)
        else:
            print("No residual in dds and sigma21 was not provided, "
                  "setting sigma21 to 1e-4.", file=log)
    else:
        sigma21 = opts.sigma21

    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads

    if opts.use_psf:
        from pfb.operators.psf import psf_convolve_xds

        nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf
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
        psfopts['nthreads'] = opts.nvthreads

        if opts.mean_ds:
            print("Using mean-ds approximation", file=log)
            from pfb.operators.psf import psf_convolve_cube
            # the PSF is normalised so we don't need to pass wsum
            hess = partial(psf_convolve_cube, psfhat=psfhat,
                           beam=mean_beam * mask[None],
                           psfopts=psfopts, sigmainv=opts.sigmainv,
                           compute=True)
        else:
            from pfb.operators.psf import psf_convolve_xds
            hess = partial(psf_convolve_xds, xds=dds, psfopts=psfopts,
                           wsum=wsum, sigmainv=opts.sigmainv, mask=mask,
                           compute=True)

    else:
        print("Solving for update using vis space approximation", file=log)
        hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                        wsum=wsum, sigmainv=opts.sigmainv, mask=mask,
                        compute=True)

    if opts.hessnorm is None:
        print("Finding spectral norm of Hessian approximation", file=log)
        hessnorm, _ = power_method(hess, (nband, nx, ny), tol=opts.pm_tol,
                         maxit=opts.pm_maxit, verbosity=opts.pm_verbose,
                         report_freq=opts.pm_report_freq)
    else:
        hessnorm = opts.hessnorm

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
    for i in range(opts.niter):
        # prox = partial(prox_21m, sigma=opts.sigma21, weight=weight, axis=0)
        model, dual = primal_dual(hess, data, model, dual, sigma21,
                                  psi, psiH, weight, hessnorm, prox_21,
                                  nu=nbasis, positivity=opts.positivity,
                                  tol=opts.pd_tol, maxit=opts.pd_maxit,
                                  verbosity=opts.pd_verbose,
                                  report_freq=opts.pd_report_freq)

        # reweight
        l2_norm = np.linalg.norm(psiH(model), axis=0)
        for m in range(nbasis):
            # if adapt_sig21:
            #     _, sigmas[m] = expon.fit(l2_norm[m], floc=0.0)
            #     print('basis %i, sigma %f'%sigmas[m], file=log)

            weight[m] = alpha[m]/(alpha[m] + l2_norm[m])

    if opts.niter==0:
        model = data

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

    if opts.do_residual:
        print("Computing residual", file=log)
        # compute apparent residual per dataset
        from pfb.operators.hessian import hessian
        # Required because of https://github.com/ska-sa/dask-ms/issues/171
        ddsw = xds_from_zarr(dds_name, columns='DIRTY')
        writes = []
        for ds, dsw in zip(dds, ddsw):
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

        dask.compute(xds_to_zarr(writes, dds_name, columns='RESIDUAL'))

    if opts.fits_mfs or opts.fits_cubes:
        print("Writing fits files", file=log)
        dds = xds_from_zarr(dds_name)
        residual = np.zeros((nband, nx, ny), dtype=dds[0].DIRTY.dtype)
        wsums = np.zeros(nband)
        for ds in dds:
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

        if opts.fits_cubes:
            # need residual in Jy/beam
            wsums = np.amax(psf, axes=(1,2))
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            fmask = wsums > 0
            residual[fmask] /= wsums[fmask, None, None]
            save_fits(f'{basename}_model.fits', model, hdr)
            save_fits(f'{basename}_residual.fits',
                      residual, hdr)

    print("All done here.", file=log)
