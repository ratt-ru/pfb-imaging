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
    from pfb.utils.fits import (set_wcs, save_fits, dds2fits,
                                dds2fits_mfs, load_fits)
    from pfb.opt.pcg import pcg
    from astropy.io import fits

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    xds_name = f'{basename}.xds.zarr'
    dds_name = f'{basename}_{opts.postfix}.dds.zarr'

    xds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan': -1})
    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan': -1,
                                          'band': 1,
                                          'x': -1,
                                          'y': -1})

    # stitch residuals after beam application
    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)
    nband = 0
    ntime = 0
    for ds in dds:
        nband = np.maximum(ds.bandid, nband)
        ntime = np.maximum(ds.timeid, ntime)

    nx = ds.x.size
    ny = ds.y.size
    nx_psf = ds.x_psf.size
    ny_psf = ds.y_psf.size
    nyo2 = ds.yo2.size

    dirty = np.zeros((ntime, nband, nx, ny), dtype=real_type)
    psfhat = np.zeros((ntime, nband, nx_psf, nyo2), dtype=real_type)
    x = np.zeros((ntime, nband, nx, ny), dtype=real_type)
    vis = {}
    wgt = {}
    for dsi, dsv in zip(dds, xds):
        t = dsi.timeid
        assert t == dsv.bandid
        b = dsi.bandid
        assert b == dsv.bandid
        vis[f'time{t}band{b}'] = dsv.VIS.values
        wgt[f'time{t}band{b}'] = dsv.WEIGHT.values
        dirty[t, b] = ds.DIRTY.values
        psfhat[t, b] = ds.PSFHAT.values
        if 'MODEL' in dsi:
            x[t, b] = dsi.MODEL.values


    # mask = init_mask(opts.mask, mds, real_type, log)

    # pre-allocate arrays for doing FFT's
    xout = np.empty(dirty.shape, dtype=ID.dtype, order='C')
    xout = make_noncritical(xout)
    xpad = np.empty((ntime, nband, nx_psf, nyo2), dtype=ID.dtype, order='C')
    xpad = make_noncritical(xpad)
    xhat = np.empty(psfhat.shape, dtype=psfhat.dtype)
    xhat = make_noncritical(xhat)
    hess = partial(hess_psf, xpad, xhat, xout, psfhat, lastsize)

    print("Solving for update", file=log)
    update = pcg(hess, residual, x,
                 tol=opts.cg_tol,
                 maxit= opts.cg_maxit,
                 minit=opts.cg_minit,
                 verbosity=opts.cg_verbose,
                 report_freq=opts.cg_report_freq,
                 backtracl=opts.backtrack,
                 return_resid=False)

    print("Writing update.", file=log)
    dds_out = []
    for ds in dds:
        t = ds.timeid
        b = ds.bandid
        ds = ds.assign(**{'MODEL': (('x','y'), da.from_array(update[t, b]))})
        dds_out.append(ds)

    dask.compute(xds_to_zarr(dds_out, dds_name, columns=('MODEL',)))

    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan': -1,
                                          'band': 1,
                                          'x': -1,
                                          'y': -1})

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


def hess_psf(xpad,    # preallocated array to store padded image
             xhat,    # preallocated array to store FTd image
             xout,    # preallocated array to store output image
             psfhat,
             lastsize,
             x,       # input image, not overwritten
             nthreads=1,
             sigmainv=1.0):
    _, nx, ny = x.shape
    xpad[...] = 0.0
    xpad[:, :, 0:nx, 0:ny] = x
    r2c(xpad, axes=(-2, -1), nthreads=nthreads,
        forward=True, inorm=0, out=xhat)
    xhat *= psfhat
    c2r(xhat, axes=(-2, -1), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    xout[...] = xpad[:, :, 0:nx, 0:ny]
    return xout + sigmainv*x
