# flake8: noqa
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
    Forward backward steps
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'fwdbwd_{timestamp}.log')

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

def _fwdbwd(**kw):
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
    from pfb.operators.hessian import hessian, hessian_xds
    from pfb.operators.psf import _hessian_reg_psf
    from pfb.opt.pcg import pcg
    from pfb.opt.primal_dual import primal_dual
    from pfb.opt.power_method import power_method
    from pfb.prox.prox_21m import prox_21m
    from pfb.prox.prox_21 import prox_21
    from pfb.wavelets.wavelets import wavelet_setup
    import pywt
    from copy import deepcopy

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

    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)
    if opts.model_name in mds:
        model = mds.get(opts.model_name).values
        assert model.shape == (nband, nx, ny)
        print(f"Initialising model from {opts.model_name} in mds", file=log)
    else:
        print('Initialising model to zeros', file=log)
        model = np.zeros((nband, nx, ny), dtype=real_type)

    mask = init_mask(opts.mask, mds, dds[0].DIRTY.dtype, log)

    # combine images over scan/spw for preconditioner
    dirty = [da.zeros((nx, ny), chunks=(-1, -1),
                         dtype=real_type) for _ in range(nband)]
    wsums = [da.zeros(1, dtype=real_type) for _ in range(nband)]
    nx_psf, ny_psf = dds[0].PSF.shape
    nx_psf, nyo2_psf = dds[0].PSFHAT.shape
    psfhat = [da.zeros((nx_psf, nyo2_psf), chunks=(-1, -1),
                        dtype=complex_type) for _ in range(nband)]
    mean_beam = [da.zeros((nx, ny), chunks=(-1, -1),
                            dtype=real_type) for _ in range(nband)]
    for ds in dds:
        b = ds.bandid
        dirty[b] += ds.DIRTY.data * ds.BEAM.data
        psfhat[b] += ds.PSFHAT.data
        mean_beam[b] += ds.BEAM.data * ds.WSUM.data[0]
        wsums[b] += ds.WSUM.data[0]
    wsums = da.stack(wsums).squeeze()
    wsum = wsums.sum()
    dirty = da.stack(dirty)/wsum
    psfhat = da.stack(psfhat)/wsum
    mean_beam = da.stack(mean_beam)/wsums[:, None, None]
    residual, psfhat, mean_beam, wsum = dask.compute(dirty,
                                                     psfhat,
                                                     mean_beam,
                                                     wsum)

    # dictionary setup
    print("Setting up dictionary", file=log)
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)
    iys, sys, ntot, nmax = wavelet_setup(residual, bases, opts.nlevels)
    ntot = tuple(ntot)
    psiH = partial(im2coef, bases=bases, ntot=ntot, nmax=nmax,
                   nlevels=opts.nlevels)
    psi = partial(coef2im, bases=bases, ntot=ntot,
                  iy=iys, sy=sys, nx=nx, ny=ny)

    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads

    nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf
    npad_xl = (nx_psf - nx)//2
    npad_xr = nx_psf - nx - npad_xl
    npad_yl = (ny_psf - ny)//2
    npad_yr = ny_psf - ny - npad_yl
    psf_padding = ((0,0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    lastsize = ny + np.sum(psf_padding[-1])

    # the PSF is normalised so we don't need to pass wsum
    hess = partial(_hessian_reg_psf, beam=mean_beam * mask[None],
                   psfhat=psfhat, nthreads=opts.nthreads,
                   sigmainv=opts.sigmainv, padding=psf_padding,
                   unpad_x=unpad_x, unpad_y=unpad_y, lastsize=lastsize)

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

    # for saving intermediaries
    ra = mds.ra
    dec = mds.dec
    radec = [ra, dec]
    cell_rad = mds.cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq_out = mds.band.values
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)

    residual_mfs = np.sum(residual, axis=0)
    rms = np.std(residual_mfs)
    rmax = residual_mfs.max()
    print(f"It {0}: max resid = {rmax:.3e}, rms = {rms:.3e}", file=log)
    for i in range(opts.niter):
        print('Getting update', file=log)
        update, fwd_resid = pcg(hess, mask[None] * residual,
                     tol=opts.cg_tol, maxit=opts.cg_maxit,
                     minit=opts.cg_minit, verbosity=opts.cg_verbose,
                     report_freq=opts.cg_report_freq,
                     backtrack=opts.backtrack, return_resid=True)

        save_fits(f'{basename}_update_{i}.fits', update, hdr)
        save_fits(f'{basename}_fwd_resid_{i}.fits', fwd_resid, hdr)

        print('Getting model', file=log)
        modelp = deepcopy(model)
        data = model + update
        model, dual = primal_dual(hess, data, model, dual, opts.sigma21,
                                  psi, psiH, weight, hessnorm, prox_21,
                                  nu=nbasis, positivity=opts.positivity,
                                  tol=opts.pd_tol, maxit=opts.pd_maxit,
                                  verbosity=opts.pd_verbose,
                                  report_freq=opts.pd_report_freq)

        # reweight
        l2_norm = np.linalg.norm(psiH(model), axis=0)
        for m in range(nbasis):
            weight[m] = opts.alpha/(opts.alpha + l2_norm[m])

        model_dask = da.from_array(model, chunks=(1, -1, -1))

        print('Computing residual', file=log)
        # first write it to disk per dataset
        out_ds = []
        for ds in dds:
            dirty = ds.DIRTY.data
            wgt = ds.WEIGHT.data
            uvw = ds.UVW.data
            freq = ds.FREQ.data
            beam = ds.BEAM.data
            vis_mask = ds.MASK.data
            b = ds.bandid
            # we only want to apply the beam once here
            residual = dirty - hessian(beam * model_dask[b], uvw, wgt,
                                       vis_mask, freq, None, hessopts)
            ds = ds.assign(**{'RESIDUAL': (('x', 'y'), residual)})
            out_ds.append(ds)

        writes = xds_to_zarr(out_ds, dds_name, columns='RESIDUAL')
        dask.compute(writes)

        # reconstruct from disk
        dds = xds_from_zarr(dds_name, chunks={'row':opts.row_chunk})
        residual = [da.zeros((nx, ny), chunks=(-1, -1),
                    dtype=real_type) for _ in range(nband)]
        for ds in dds:
            b = ds.bandid
            # we ave to apply the beam here in case it varies with ds
            # the mask will be applied prior to massing into PCG
            residual[b] += ds.RESIDUAL.data * ds.BEAM.data
        residual = da.stack(residual)/wsum
        residual = residual.compute()
        residual_mfs = np.sum(residual, axis=0)
        rms = np.std(residual_mfs)
        rmax = residual_mfs.max()
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)
        print(f"It {i+1}: max resid = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)
        if eps < opts.tol:
            break

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
                     'DUAL': (('band', 'basis', 'coef'), dual),
                     'WEIGHT': (('basis', 'coef'), weight)})

    dask.compute(xds_to_zarr(mds, mds_name, columns='ALL'))

    if opts.fits_mfs or opts.fits_cubes:
        print("Writing fits files", file=log)
        dds = xds_from_zarr(dds_name)

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

        residual = np.zeros((nband, nx, ny), dtype=dds[0].DIRTY.dtype)
        wsums = np.zeros(nband)
        for ds in dds:
            b = ds.bandid
            wsums[b] += ds.WSUM.values[0]
            residual[b] += ds.RESIDUAL.values
        wsum = np.sum(wsums)
        residual_mfs = np.sum(residual, axis=0)/wsum
        save_fits(f'{basename}_residual_mfs.fits', residual_mfs, hdr_mfs)

        if opts.fits_cubes:
            # need residual in Jy/beam
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(f'{basename}_model.fits', model, hdr)
            fmask = wsums > 0
            residual[fmask] /= wsums[fmask, None, None]
            save_fits(f'{basename}_residual.fits',
                    residual, hdr)

    print("All done here.", file=log)
