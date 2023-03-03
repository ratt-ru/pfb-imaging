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
    from pfb.opt.pcg import cg_dct
    from pfb.operators.hessian import hess_vis
    from astropy.io import fits
    from glob import glob

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    # ther eis only one xds
    xds_name = f'{basename}.xds.zarr'
    xds = xds_from_zarr(xds_name, chunks={'row':-1,
                                          'chan': -1})
    xds = dask.persist(xds)[0]

    # get number of times and bands
    real_type = xds[0].WEIGHT.dtype
    complex_type = np.result_type(real_type, np.complex64)
    nband = 0
    ntime = 0
    for ds in xds:
        nband = np.maximum(ds.bandid, nband)
        ntime = np.maximum(ds.timeid, ntime)


    # there can be multiple ddss
    dds_names = glob(f'{basename}_*.dds.zarr')
    nfields = len(dds_names)
    dds = {}
    dirty = {}
    x = {}
    xout = {}
    for name in dds_names:
        dds[name] = {}
        dirty[name] = {}
        x[name] = {}
        xout[name] = {}
        tmpds = xds_from_zarr(name, chunks={'x': -1,
                                            'y': -1,
                                            'band': 1})
        for ds in tmpds:
            t = ds.timeid
            b = ds.bandid
            dds[name][f't{t}b{b}'] = {}
            dds[name][f't{t}b{b}']['nx'] = ds.x.size
            dds[name][f't{t}b{b}']['ny'] = ds.y.size
            dds[name][f't{t}b{b}']['cell'] = ds.cell_rad
            dds[name][f't{t}b{b}']['x0'] = ds.x0
            dds[name][f't{t}b{b}']['y0'] = ds.y0
            dirty[name][f't{t}b{b}'] = ds.DIRTY.values
            if 'MODEL' in ds:
                x[name][f't{t}b{b}'] = ds.MODEL.values
            else:
                x[name][f't{t}b{b}'] = np.zeros(ds.DIRTY.shape, dtype=real_type)


    # pre-allocate arrays for doing FFT's
    hess = partial(hess_vis, xds, dds, xout,
                   sigmainv=opts.sigmainv,
                   wstack=opts.wstack,
                   nthreads=opts.nthreads,
                   epsilon=opts.epsilon)

    print("Solving for update", file=log)
    update, residual = cg_dct(hess, dirty, x,
                              tol=opts.cg_tol,
                              maxit= opts.cg_maxit,
                              verbosity=opts.cg_verbose,
                              report_freq=opts.cg_report_freq)

    for name in dds_names:
        print(f"Writing results for {name}", file=log)
        tmpds = xds_from_zarr(name, chunks={'x': -1,
                                            'y': -1,
                                            'band': 1})
        dds_out = []
        for ds in tmpds:
            t = ds.timeid
            b = ds.bandid
            ds = ds.assign(**{'MODEL': (('x','y'), da.from_array(update[name][f't{t}b{b}'])),
                              'RESIDUAL': (('x','y'), da.from_array(residual[name][f't{t}b{b}']))})
            dds_out.append(ds)

        dask.compute(xds_to_zarr(dds_out, name, columns=('MODEL','RESIDUAL')))

        dds = xds_from_zarr(name, chunks={'row':-1,
                                          'chan': -1,
                                          'band': 1,
                                          'x': -1,
                                          'y': -1})

        outname = name.rstrip('.dds.zarr')

        # convert to fits files
        fitsout = []
        if opts.fits_mfs:
            fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', outname, norm_wsum=True))
            fitsout.append(dds2fits_mfs(dds, 'MODEL', outname, norm_wsum=False))

        if opts.fits_cubes:
            fitsout.append(dds2fits(dds, 'RESIDUAL', outname, norm_wsum=True))
            fitsout.append(dds2fits(dds, 'MODEL', outname, norm_wsum=False))

        if len(fitsout):
            print(f"Writing fits for {name}", file=log)
            dask.compute(fitsout)

    print("All done here.", file=log)
