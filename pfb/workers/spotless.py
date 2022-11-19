# flake8: noqa
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
    defaults[key] = schema.spotless["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.spotless)
def spotless(**kw):
    '''
    Spotless algorithm
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'spotless_{timestamp}.log')

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

        return _spotless(**opts)

def _spotless(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import dask
    import dask.array as da
    from distributed import Client, wait, get_client
    from pfb.opt.power_method import power_method_dist as power_method
    from pfb.opt.pcg import pcg_dist as pcg
    from pfb.opt.primal_dual import primal_dual_dist as primal_dual
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.utils.dist import (get_resid_and_stats, accum_wsums,
                                compute_residual, init_dual_and_model,
                                get_eps, l1reweight, get_cbeam_area)
    from pfb.operators.psf import _hessian_reg_psf_slice
    from pfb.operators.psi import im2coef_dist as im2coef
    from pfb.operators.psi import coef2im_dist as coef2im
    from pfb.prox.prox_21m import prox_21m
    from pfb.prox.prox_21 import prox_21
    from pfb.wavelets.wavelets import wavelet_setup
    import pywt
    from copy import deepcopy
    from operator import getitem
    from pfb.wavelets.wavelets import wavelet_setup

    dds = xds_from_zarr(opts.dds, chunks={'row':-1, 'chan':-1})
    basename = f'{opts.dds.rstrip(".dds.zarr")}'

    client = get_client()
    names = [w['name'] for w in client.scheduler_info()['workers'].values()]

    # dds = client.persist(client)
    ddsf = client.scatter(dds)
    # names={}
    # for ds in ddsf:
    #     b = ds.result().bandid
    #     tmp = client.who_has(ds)
    #     for key in tmp.keys():
    #         names[b] = tmp[key]

    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)

    nband = len(dds)
    nx, ny = dds[0].nx, dds[0].ny
    nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf
    nx_psf, nyo2_psf = dds[0].PSFHAT.shape
    npad_xl = (nx_psf - nx)//2
    npad_xr = nx_psf - nx - npad_xl
    npad_yl = (ny_psf - ny)//2
    npad_yr = ny_psf - ny - npad_yl
    psf_padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    lastsize = ny + np.sum(psf_padding[-1])

    # header for saving intermediaries
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq_out = np.zeros((nband))
    for ds in dds:
        b = ds.bandid
        freq_out[b] = ds.freq_out
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)

    # assumed to stay the same
    wsum = client.submit(accum_wsums, ddsf).result()
    pix_per_beam = client.submit(get_cbeam_area, ddsf, wsum).result()

    # manually persist psfhat and beam on workers
    psfhatf = client.map(lambda ds: ds.PSFHAT.values, ddsf)
    beamf = client.map(lambda ds: ds.BEAM.values, ddsf)

    # this makes for cleaner algorithms but is it a bad pattern?
    Afs = []
    for psfhat, beam in zip(psfhatf, beamf):
        tmp = client.who_has(psfhat)
        Af = client.submit(partial,
                           _hessian_reg_psf_slice,
                           psfhat=psfhat,
                           beam=beam,
                           wsum=wsum,
                           nthreads=opts.nthreads,
                           sigmainv=opts.sigmainv,
                           padding=psf_padding,
                           unpad_x=unpad_x,
                           unpad_y=unpad_y,
                           lastsize=lastsize,
                           workers=list(tmp.values())[0])
        Afs.append(Af)

    # dictionary setup
    print("Setting up dictionary", file=log)
    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)
    iy, sy, ntot, nmax = wavelet_setup(
                                np.zeros((1, nx, ny), dtype=real_type),
                                bases, opts.nlevels)
    ntot = tuple(ntot)
    # we only want to transfer these once
    psiH = partial(im2coef,
                   bases=bases,
                   ntot=ntot,
                   nmax=nmax,
                   nlevels=opts.nlevels)

    psi = partial(coef2im,
                  bases=bases,
                  ntot=ntot,
                  iy=None,  #iy,
                  sy=None,  #sy,
                  nx=nx,
                  ny=ny)

    # compile these before sending them to each worker?
    alphatmp = np.zeros((nbasis, nmax), dtype=real_type)
    xtmp = psi(alphatmp)
    alphatmp = psiH(xtmp)

    del xtmp, alphatmp

    # initialise for backward step
    ddsf = client.map(init_dual_and_model, ddsf,
                      nx=nx,
                      ny=ny,
                      nbasis=nbasis,
                      nmax=nmax,
                      pure=False)

    # TODO - we want to put the l1weights for each basis on the same worker
    # that ratio is computed on
    l1weight = client.submit(np.ones, (nbasis, nmax), workers=[names[0]])

    if opts.hessnorm is None:
        print('Getting spectral norm of Hessian approximation', file=log)
        hessnorm = power_method(Afs, nx, ny, nband).result()
    else:
        hessnorm = opts.hessnorm
    print(f'hessnorm = {hessnorm:.3e}', file=log)

    # future contains mfs residual and stats
    residf = client.submit(get_resid_and_stats, ddsf, wsum)
    residual_mfs = client.submit(getitem, residf, 0)
    rms = client.submit(getitem, residf, 1).result()
    rmax = client.submit(getitem, residf, 2).result()
    print(f"It {0}: max resid = {rmax:.3e}, rms = {rms:.3e}", file=log)
    for i in range(opts.niter):
        print('Getting update', file=log)
        ddsf = client.map(pcg, ddsf, Afs,
                          pure=False,
                          wsum=wsum,
                          sigmainv=opts.sigmainv,
                          tol=opts.cg_tol,
                          maxit=opts.cg_maxit,
                          minit=opts.cg_minit)

        wait(ddsf)
        # save_fits(f'{basename}_update_{i}.fits', update, hdr)
        # save_fits(f'{basename}_fwd_resid_{i}.fits', fwd_resid, hdr)

        print('Getting model', file=log)
        modelp = client.map(lambda ds: ds.MODEL.values, ddsf)
        ddsf = primal_dual(ddsf, Afs,
                           psi, psiH,
                           opts.rmsfactor*rms, #opts.sigma21,
                           hessnorm,
                           wsum,
                           l1weight,
                           tol=opts.pd_tol,
                           maxit=opts.pd_maxit,
                           positivity=opts.positivity,
                           gamma=opts.gamma,
                           verbosity=opts.pd_verbose)

        print('Computing residual', file=log)
        ddsf = client.map(compute_residual, ddsf,
                          pure=False,
                          cell=cell_rad,
                          wstack=opts.wstack,
                          epsilon=opts.epsilon,
                          double_accum=opts.double_accum,
                          nthreads=opts.nvthreads)

        wait(ddsf)

        residf = client.submit(get_resid_and_stats, ddsf, wsum)
        residual_mfs = client.submit(getitem, residf, 0)
        rms = client.submit(getitem, residf, 1).result()
        rmax = client.submit(getitem, residf, 2).result()
        eps = client.submit(get_eps, modelp, ddsf).result()
        print(f"It {i+1}: max resid = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        # l1reweighting
        if i+1 >= opts.l1reweight_from:
            print('L1 reweighting', file=log)
            l1weight = client.submit(l1reweight, ddsf, l1weight,
                                     psiH, wsum, pix_per_beam)
            # import pdb; pdb.set_trace()

        if eps < opts.tol:
            break

    # # future to collection
    # print('Writing results', file=log)
    dds = dask.delayed(Idty)(ddsf).compute()
    # writes = xds_to_zarr(dds,
    #                      '/home/landman/testing/pfb/out/test2.dds.zarr',
    #                      columns=('MODEL','DUAL','UPDATE'))
    # dask.compute(writes)

    if opts.fits_mfs or opts.fits_cubes:
        print("Writing fits files", file=log)

        # construct a header from xds attrs
        ra = dds[0].ra
        dec = dds[0].dec
        radec = [ra, dec]

        cell_rad = dds[0].cell_rad
        cell_deg = np.rad2deg(cell_rad)

        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        residual = np.zeros((nband, nx, ny), dtype=real_type)
        model = np.zeros((nband, nx, ny), dtype=real_type)
        wsums = np.zeros(nband)
        for ds in dds:
            b = ds.bandid
            wsums[b] += ds.WSUM.values[0]
            residual[b] += ds.RESIDUAL.values
            model[b] = ds.MODEL.values
        wsum = np.sum(wsums)
        residual_mfs = np.sum(residual, axis=0)/wsum
        model_mfs = np.mean(model, axis=0)
        save_fits(f'{basename}_residual_mfs.fits', residual_mfs, hdr_mfs)
        save_fits(f'{basename}_model_mfs.fits', model_mfs, hdr_mfs)

        if opts.fits_cubes:
            # need residual in Jy/beam
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(f'{basename}_model.fits', model, hdr)
            fmask = wsums > 0
            residual[fmask] /= wsums[fmask, None, None]
            save_fits(f'{basename}_residual.fits',
                    residual, hdr)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here.", file=log)
    return


def Idty(x):
    return x
