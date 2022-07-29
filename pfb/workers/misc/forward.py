# flake8: noqa
from contextlib import ExitStack
from pfb.workers.experimental import cli
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
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'forward_{timestamp}.log')

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

        return _forward(**opts)

def _forward(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import xarray as xr
    import dask
    import dask.array as da
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits
    from pfb.utils.misc import setup_image_data, init_mask
    from pfb.operators.hessian import hessian_xds
    from pfb.opt.pcg import pcg
    from astropy.io import fits
    import pywt
    from pfb.operators.hessian import hessian

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}{opts.postfix}.dds.zarr'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'

    dds = xds_from_zarr(dds_name, chunks={'row':opts.row_chunk})
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny
    for ds in dds:
        assert ds.nx == nx
        assert ds.ny == ny

    # stitch residuals after beam application
    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)
    dirty, residual, wsum, _, psfhat, mean_beam = setup_image_data(dds,
                                                                   opts,
                                                                   log=log)
    dirty_mfs = np.sum(dirty, axis=0)
    if residual is None:
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()
    model = mds.MODEL.values
    mask = init_mask(opts.mask, mds, real_type, log)

    try:
        print("Initialising update using CLEAN_MODEL in mds", file=log)
        x0 = mds.CLEAN_MODEL.values
    except:
        print("Initialising update to all zeros", file=log)
        x0 = np.zeros((nband, nx, ny), dtype=real_type)

    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads

    if opts.use_psf:
        print("Solving for update using image space approximation", file=log)
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

    # # import pdb; pdb.set_trace()
    # x = np.random.randn(nband, nx, ny)  #.astype(np.float32)
    # res = hess(x)
    # dask.visualize(res, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=opts.output_filename + '_hess_I_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(res, filename=opts.output_filename +
    #                '_hess_I_graph.pdf', optimize_graph=False)

    cgopts = {}
    cgopts['tol'] = opts.cg_tol
    cgopts['maxit'] = opts.cg_maxit
    cgopts['minit'] = opts.cg_minit
    cgopts['verbosity'] = opts.cg_verbose
    cgopts['report_freq'] = opts.cg_report_freq
    cgopts['backtrack'] = opts.backtrack

    print("Solving for update", file=log)
    update = pcg(hess, mask * residual, x0, **cgopts)

    print("Writing update.", file=log)
    model += update
    update = da.from_array(update, chunks=(1, -1, -1))
    model = da.from_array(model, chunks=(1, -1, -1))
    mds = mds.assign(**{'UPDATE': (('band', 'x', 'y'),
                     update)})

    mds = mds.assign(**{'FORWARD_MODEL': (('band', 'x', 'y'),
                     model)})

    dask.compute(xds_to_zarr(mds, mds_name, columns=('UPDATE', 'FORWARD_MODEL')))

    if opts.do_residual:
        print('Computing final residual', file=log)
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
            residual = dirty - hessian(beam * model[b], uvw, wgt,
                                       vis_mask, freq, None, hessopts)
            ds = ds.assign(**{'FORWARD_RESIDUAL': (('x', 'y'), residual)})
            out_ds.append(ds)

        writes = xds_to_zarr(out_ds, dds_name, columns='FORWARD_RESIDUAL')
        dask.compute(writes)

    if opts.fits_mfs or opts.fits_cubes:
        print("Writing fits files", file=log)
        # construct a header from xds attrs
        radec = [dds[0].ra, dds[0].dec]
        cell_rad = dds[0].cell_rad
        cell_deg = np.rad2deg(cell_rad)
        freq_out = mds.freq.data
        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        update_mfs = np.mean(update, axis=0)
        save_fits(f'{basename}{opts.postfix}_update_mfs.fits', update_mfs, hdr_mfs)

        if opts.do_residual:
            dds = xds_from_zarr(dds_name)
            residual = np.zeros((nband, nx, ny), dtype=np.float32)
            wsums = np.zeros(nband)
            for ds in dds:
                b = ds.bandid
                wsums[b] += ds.WSUM.values
                residual[b] += ds.FORWARD_RESIDUAL.values.astype(np.float32)
            wsum = np.sum(wsums)

            residual_mfs = np.sum(residual, axis=0)/wsum
            save_fits(f'{basename}{opts.postfix}_forward_residual_mfs.fits',
                      residual_mfs, hdr_mfs)

        if opts.fits_cubes:
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(f'{basename}{opts.postfix}_update.fits', update, hdr)

            if opts.do_residual:
                fmask = wsums > 0
                residual[fmask] /= wsums[fmask, None, None]
                save_fits(f'{basename}{opts.postfix}_forward_residual.fits',
                          residual, hdr)

    print("All done here.", file=log)
