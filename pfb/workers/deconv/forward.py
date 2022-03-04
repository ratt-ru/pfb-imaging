# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FORWARD')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.forward["inputs"].keys():
    defaults[key] = schema.forward["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.forward)
def forward(**kw):
    '''
    Forward step aka flux mop.

    Solves

    x = (A.H R.H W R A + sigmainv**2 I)^{-1} ID

    with a suitable approximation to R.H W R (eg. convolution with the PSF,
    BDA'd weights or none). Here A is the combination of mask and an
    average beam pattern.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default the gridder will
    use all available resources.

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
    args = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{args.output_filename}_{args.product}{args.postfix}.log')

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

        return _forward(**args)

def _forward(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)

    import numpy as np
    import xarray as xr
    import dask
    import dask.array as da
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits
    from pfb.operators.psi import im2coef, coef2im
    from pfb.operators.hessian import hessian_xds
    from pfb.opt.pcg import pcg
    from astropy.io import fits
    import pywt

    basename = f'{args.output_filename}_{args.product.upper()}'

    dds_name = f'{basename}{args.postfix}.dds.zarr'
    mds_name = f'{basename}{args.postfix}.mds.zarr'

    dds = xds_from_zarr(dds_name, chunks={'row':args.row_chunk})
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny
    for ds in dds:
        assert ds.nx == nx
        assert ds.ny == ny

    # stitch residuals after beam application
    if args.residual_name in dds[0]:
        rname = args.residual_name
    else:
        rname = 'DIRTY'
    print(f'Using {rname} as residual', file=log)
    output_type = dds[0].DIRTY.dtype
    residual = np.zeros((nband, nx, ny), dtype=output_type)
    wsum = 0
    for ds in dds:
        dirty = ds.get(rname).values
        beam = ds.BEAM.values
        b = ds.bandid
        residual[b] += dirty * beam
        wsum += ds.WSUM.values[0]
    residual /= wsum

    from pfb.utils.misc import init_mask
    mask = init_mask(args.mask, mds, output_type, log)

    try:
        x0 = mds.CLEAN_MODEL.values
    except:
        print("Initialising model to all zeros", file=log)
        x0 = np.zeros((nband, nx, ny), dtype=output_type)

    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = args.wstack
    hessopts['epsilon'] = args.epsilon
    hessopts['double_accum'] = args.double_accum
    hessopts['nthreads'] = args.nvthreads

    if args.use_psf:
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
        psfopts['nthreads'] = args.nvthreads

        hess = partial(psf_convolve_xds, xds=dds, psfopts=psfopts,
                       wsum=wsum, sigmainv=args.sigmainv, mask=mask,
                       compute=True)

    else:
        print("Solving for update using vis space approximation", file=log)
        hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                       wsum=wsum, sigmainv=args.sigmainv, mask=mask,
                       compute=True)

    # # import pdb; pdb.set_trace()
    # x = np.random.randn(nband, nx, ny)  #.astype(np.float32)
    # res = hess(x)
    # dask.visualize(res, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=args.output_filename + '_hess_I_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(res, filename=args.output_filename +
    #                '_hess_I_graph.pdf', optimize_graph=False)

    cgopts = {}
    cgopts['tol'] = args.cg_tol
    cgopts['maxit'] = args.cg_maxit
    cgopts['minit'] = args.cg_minit
    cgopts['verbosity'] = args.cg_verbose
    cgopts['report_freq'] = args.cg_report_freq
    cgopts['backtrack'] = args.backtrack

    print("Solving for update", file=log)
    update = pcg(hess, mask * residual, x0, **cgopts)

    print("Writing update.", file=log)
    update = da.from_array(update, chunks=(1, -1, -1))
    mds = mds.assign(**{'UPDATE': (('band', 'x', 'y'),
                     update)})
    dask.compute(xds_to_zarr(mds, mds_name, columns='UPDATE'))

    if args.do_residual:
        print("Computing residual", file=log)
        from pfb.operators.hessian import hessian
        # Required because of https://github.com/ska-sa/dask-ms/issues/171
        ddsw = xds_from_zarr(dds_name, columns='DIRTY')
        writes = []
        for ds, dsw in zip(dds, ddsw):
            dirty = ds.get(rname).data
            wgt = ds.WEIGHT.data
            uvw = ds.UVW.data
            freq = ds.FREQ.data
            beam = ds.BEAM.data
            b = ds.bandid
            # we only want to apply the beam once here
            residual = (dirty -
                        hessian(beam * update[b], uvw, wgt, freq, None,
                                hessopts))
            dsw = dsw.assign(**{'FORWARD_RESIDUAL': (('x', 'y'),
                                                      residual)})

            writes.append(dsw)

        dask.compute(xds_to_zarr(writes, dds_name, columns='FORWARD_RESIDUAL'))

    if args.fits_mfs or args.fits_cubes:
        print("Writing fits files", file=log)
        # construct a header from xds attrs
        radec = [dds[0].ra, dds[0].dec]
        cell_rad = dds[0].cell_rad
        cell_deg = np.rad2deg(cell_rad)
        freq_out = mds.freq.data
        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        update_mfs = np.mean(update, axis=0)
        save_fits(f'{basename}{args.postfix}_update_mfs.fits', update_mfs, hdr_mfs)

        if args.do_residual:
            dds = xds_from_zarr(dds_name)
            residual = np.zeros((nband, nx, ny), dtype=np.float32)
            wsums = np.zeros(nband)
            for ds in dds:
                b = ds.bandid
                wsums[b] += ds.WSUM.values
                residual[b] += ds.FORWARD_RESIDUAL.values.astype(np.float32)
            wsum = np.sum(wsums)
            residual /= wsum

            residual_mfs = np.sum(residual, axis=0)
            save_fits(f'{basename}{args.postfix}_forward_residual_mfs.fits',
                      residual_mfs, hdr_mfs)

        if args.fits_cubes:
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(f'{basename}{args.postfix}_update.fits', update, hdr)

            if args.do_residual:
                fmask = wsums > 0
                residual[fmask] /= wsums[fmask, None, None]
                save_fits(f'{basename}{args.postfix}_forward_residual.fits',
                          residual, hdr)

    print("All done here.", file=log)
