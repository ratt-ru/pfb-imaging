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

    dds_name = f'{basename}_{opts.postfix}.dds.zarr'

    dds = xds_from_zarr(dds_name, chunks={'row':opts.row_chunk,
                                          'chan': -1})

    # stitch image space data products
    output_type = dds[0].DIRTY.dtype
    dirty, model, residual, psf, psfhat, beam, wsums, dual = dds2cubes(
                                                               dds,
                                                               opts.nband,
                                                               apparent=False)
    dirty_mfs = np.sum(dirty, axis=0)
    if residual is None:
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()

    mask = init_mask(opts.mask, model, output_type, log)

    # set up vis space Hessian
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads
    hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=opts.sigmainv,
                   mask=mask, dtype=output_type,
                   compute=True, use_beam=False)

    if opts.use_psf:
        print("Solving for update using image space hessian approximation",
              file=log)
        xout = np.empty(dirty.shape, dtype=dirty.dtype, order='C')
        xout = make_noncritical(xout)
        xpad = np.empty(psf.shape, dtype=dirty.dtype, order='C')
        xpad = make_noncritical(xpad)
        xhat = np.empty(psfhat.shape, dtype=psfhat.dtype)
        xhat = make_noncritical(xhat)
        psf_convolve = partial(psf_convolve_cube, xpad, xhat, xout, psfhat, lastsize,
                        nthreads=opts.nthreads)
    else:
        print("Solving for update using vis space hessian approximation",
              file=log)
        psf_convolve = hess

    cgopts = {}
    cgopts['tol'] = opts.cg_tol
    cgopts['maxit'] = opts.cg_maxit
    cgopts['minit'] = opts.cg_minit
    cgopts['verbosity'] = opts.cg_verbose
    cgopts['report_freq'] = opts.cg_report_freq
    cgopts['backtrack'] = opts.backtrack

    print("Solving for update", file=log)
    x0 = np.zeros((nband, nx, ny), dtype=output_type)
    update = pcg(hess, mask * residual, x0,
                 tol=opts.cg_tol,
                 maxit=opts.cg_maxit,
                 minit=opts.cg_minit,
                 verbosity=opts.cg_verbose,
                 report_freq=opts.cg_report_freq,
                 backtrack=opts.backtrack)
    model += opts.gamma * update


    print("Getting residual", file=log)
    convimage = hess(model)
    ne.evaluate('dirty - convimage', out=residual,
                casting='same_kind')
    ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                casting='same_kind')


    print("Updating results", file=log)
    dds_out = []
    for ds in dds:
        b = ds.bandid
        r = da.from_array(residual[b]*wsum)
        m = da.from_array(model[b])
        u = da.from_array(update[b])
        ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                                'MODEL': (('x', 'y'), m),
                                'UPDATE': (('x', 'y'), u)})
        dds_out.append(ds_out)
    writes = xds_to_zarr(dds_out, dds_name,
                            columns=('RESIDUAL', 'MODEL', 'DUAL'),
                            rechunk=True)
    dask.compute(writes)

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})

    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', basename, norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds, 'MODEL', basename, norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds, 'RESIDUAL', basename, norm_wsum=True))
        fitsout.append(dds2fits(dds, 'MODEL', basename, norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    print("All done here.", file=log)
