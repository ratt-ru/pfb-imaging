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
    from pfb.utils.dist import get_resid_and_stats, accum_wsums
    # from pfb.operators.psi import im2coef, coef2im
    from pfb.prox.prox_21m import prox_21m
    from pfb.prox.prox_21 import prox_21
    from pfb.wavelets.wavelets import wavelet_setup
    import pywt
    from copy import deepcopy
    from operator import getitem

    dds = xds_from_zarr(opts.dds, chunks={'row':-1, 'chan':-1})
    basename = f'{opts.dds.rstrip(".dds.zarr")}'

    client = get_client()
    names = [w['name'] for w in client.scheduler_info()['workers'].values()]

    dds = client.persist(client)
    ddsf = client.scatter(dds)

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
    psf_padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    lastsize = ny + np.sum(psf_padding[-1])

    # header for saving intermediaries
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq_out = np.array((nband))
    for ds in dds:
        b = ds.bandid
        freq_out[b] = ds.freq_out
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)

    # assumed to stay the same
    wsum = client.submit(accum_wsums, ddsf).result()

    # initialise for backward step
    ddsf = client.map(init_dual_and_model, ddsf, nx=nx, ny=ny, pure=False)
    # TODO - we want to put the l1weights for each basis on the same worker
    # that ratio is computed on
    l1weight = client.submit(np.ones, (1, nx*ny), workers=[names[0]])

    print('Getting spectral norm of Hessian approximation', file=log)
    hessnorm = power_method(ddsf, nx, ny, nband, opts.nvthreads,
                            psf_padding, unpad_x, unpad_y,
                            lastsize, sigmainv, wsum).result()

    # this future contains residual and stats
    residf = get_resid_and_stats(dds, wsum)
    residual_mfs = client.sumbit(getitem, residf, 0)
    rms = client.sumbit(getitem, residf, 1).result()
    rmax = client.sumbit(getitem, residf, 2).result()
    print(f"It {0}: max resid = {rmax:.3e}, rms = {rms:.3e}", file=log)
    for i in range(opts.niter):
        print('Getting update', file=log)
        ddsf = client.map(pcg, ddsf,
                          pure=False,
                          wsum=wsum,
                          tol=opts.cg_tol,
                          maxit=opts.cg_maxit,
                          minit=opts.cg_minit,
                          nthreads=opts.nvthreads,
                          sigmainv=opts.sigmainv,
                          psf_padding=psf_padding,
                          unpad_x=unpad_x,
                          unpad_y=unpad_y,
                          lastsize=lastsize)

        # save_fits(f'{basename}_update_{i}.fits', update, hdr)
        # save_fits(f'{basename}_fwd_resid_{i}.fits', fwd_resid, hdr)

        print('Getting model', file=log)
        ddsf = primal_dual(ddsf,
                           psf_padding, unpad_x, unpad_y, lastsize,
                           opts.sigma21, hessnorm, wsum, l1weight)

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

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here.", file=log)
    return


