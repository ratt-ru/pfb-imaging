# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('CLEAN')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.clean["inputs"].keys():
    defaults[key] = schema.clean["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.clean)
def clean(**kw):
    '''
    Single-scale clean.

    The algorithm always acts on the average apparent dirty image and PSF
    provided by xds. This means there is only ever a single dirty image
    and PSF and the algorithm only provides an approximate apparent model
    that is compatible with them. The intrinsic model can be obtained using
    the forward worker.

    Two variants of single scale clean are currently implemented viz.
    Hogbom and Clark.

    Hogbom is the vanilla clean implementation, the full PSF will be
    subtracted at every iteration.

    Clark clean defines an adaptive mask that changes between iterations.
    The mask is defined by all pixels that are above sub-peak-factor * Imax
    where Imax is the current maximum in the MFS residual. A sub-minor cycle
    is performed only within this mask i.e. peak finding and PSF subtraction
    is confined to the mask until the residual within the mask decreases to
    sub-peak-factor * Imax. At the end of the sub-minor cycle an approximate
    residual is computed as

    IR -= PSF.convolve(model)

    with no mask in place. The mask is then recomputed and the sub-minor cycle
    repeated until the residual reaches peak-factor * Imax0 where Imax0 is the
    peak in the residual at the outset of the minor cycle.

    At the end of each minor cycle we recompute the residual using

    IR = R.H W (V - R x) = ID - R.H W R x

    where ID is the dirty image, R and R.H are degridding and gridding
    operators respectively and W are the effective weights i.e. the weights
    after image weighting, applying calibration solutions and taking the
    weighted sum over correlations. This is usually called a major cycle
    but we get away from explicitly loading in the visibilities by writing
    the residual in terms of the dirty image and an application of the
    Hessian.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default the gridder will
    use all available resources.

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
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}_{opts.product}{opts.postfix}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _clean(**opts)

def _clean(**kw):
    opts = OmegaConf.create(kw)
    # always combine over ds during cleaning
    opts['mean_ds'] = True
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import xarray as xr
    import numexpr as ne
    import dask
    import dask.array as da
    from dask.distributed import performance_report
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.deconv.hogbom import hogbom
    from pfb.deconv.clark import clark
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr

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

    nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf

    # stitch dirty/psf in apparent scale
    if opts.residual_name in dds[0]:
        rname = opts.residual_name
    else:
        rname = 'DIRTY'
    print(f'Using {rname} as residual', file=log)
    output_type = dds[0].DIRTY.dtype
    residual, wsum, psf, psfhat, _ = setup_image_data(dds, opts, rname, apparent=True, log=log)
    psf_mfs = np.sum(psf, axis=0)
    dirty_mfs = np.sum(dirty, axis=0)
    assert (psf_mfs.max() - 1.0) < 2*opts.epsilon

    # set up Hessian
    from pfb.operators.hessian import hessian_xds
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads
    # always clean in apparent scale
    # we do not want to use the mask here
    hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=0, mask=np.ones_like(dirty_mfs),
                   compute=True, use_beam=False)

    # to set up psf convolve when using Clark
    if opts.use_clark:
        from pfb.operators.psf import psf_convolve_cube
        from pfb.operators.fft import fft_cube
        from ducc0.fft import r2c
        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding[-1])

        # precompute so we don't need to repeat on each hessian call
        psfopts = {}
        psfopts['padding'] = padding
        psfopts['unpad_x'] = unpad_x
        psfopts['unpad_y'] = unpad_y
        psfopts['lastsize'] = lastsize
        psfopts['nthreads'] = opts.nvthreads
        psfo = partial(psf_convolve_cube, psfhat=psfhat, beam=None, psfopts=psfopts)

    rms = np.std(dirty_mfs)
    rmax = np.abs(dirty_mfs).max()

    print("Iter %i: peak residual = %f, rms = %f" % (0, rmax, rms), file=log)

    residual = dirty.copy()
    residual_mfs = dirty_mfs.copy()
    model = np.zeros_like(residual)
    for k in range(opts.nmiter):
        if opts.use_clark:
            print("Running Clark", file=log)
            x = clark(residual, psf, psfo=psfo,
                      gamma=opts.gamma,
                      pf=opts.peak_factor,
                      maxit=opts.clark_maxit,
                      subpf=opts.sub_peak_factor,
                      submaxit=opts.sub_maxit,
                      verbosity=opts.verbose,
                      report_freq=opts.report_freq)
        else:
            print("Running Hogbom", file=log)
            x = hogbom(residual, psf,
                       gamma=opts.gamma,
                       pf=opts.peak_factor,
                       maxit=opts.hogbom_maxit,
                       verbosity=opts.verbose,
                       report_freq=opts.report_freq)

        model += x

        print("Getting residual", file=log)
        convimage = hess(x)
        ne.evaluate('residual - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        # save_fits(opts.output_filename + f'_residual_mfs{k}.fits',
        #           residual_mfs, hdr_mfs)
        # save_fits(opts.output_filename + f'_model_mfs{k}.fits',
        #           np.mean(model, axis=0), hdr_mfs)
        # save_fits(opts.output_filename + f'_convim_mfs{k}.fits',
        #           np.sum(convimage, axis=0), hdr_mfs)

        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()

        print("Iter %i: peak residual = %f, rms = %f" % (
                k+1, rmax, rms), file=log)

        if opts.threshold is not None:
            if rmax <= opts.threshold:
                print("Terminating because final threshold has been reached",
                      file=log)
                break

    print("Saving results", file=log)
    if opts.update_mask:
        mask = np.any(model, axis=0)
        # from scipy import ndimage
        # mask = ndimage.binary_dilation(mask)
        # mask = ndimage.binary_closing(mask)
        # mask = ndimage.binary_erosion(mask)
        if 'MASK' in mds:
            mask = np.logical_or(mask, mds.MASK.values)
        mds = mds.assign(**{
                'MASK': (('x', 'y'), da.from_array(mask, chunks=(-1, -1)))
        })


    model = da.from_array(model, chunks=(1, -1, -1))
    mds = mds.assign(**{
            'CLEAN_MODEL': (('band', 'x', 'y'), model)
    })

    dask.compute(xds_to_zarr(mds, mds_name, columns='ALL'))

    if opts.do_residual:
        print("Computing residual", file=log)
        from pfb.operators.hessian import hessian
        # Required because of https://github.com/ska-sa/dask-ms/issues/171
        ddsw = xds_from_zarr(dds_name, chunks={'band': 1}, columns='DIRTY')
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
                        hessian(model[b], uvw, wgt, freq, None,
                        hessopts))
            dsw = dsw.assign(**{'CLEAN_RESIDUAL': (('x', 'y'), residual)})
            writes.append(dsw)

        dask.compute(xds_to_zarr(writes, dds_name, columns='CLEAN_RESIDUAL'))

    if opts.fits_mfs or opts.fits_cubes:
        print("Writing fits files", file=log)
        # construct a header from xds attrs
        ra = dds[0].ra
        dec = dds[0].dec
        radec = [ra, dec]
        cell_rad = dds[0].cell_rad
        cell_deg = np.rad2deg(cell_rad)
        freq_out = mds.freq.data
        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        model_mfs = np.mean(model, axis=0)

        save_fits(f'{basename}_clean_model_mfs.fits', model_mfs, hdr_mfs)

        if opts.do_residual:
            dds = xds_from_zarr(dds_name, chunks={'band': 1})
            residual = [da.zeros((nx, ny), chunks=(-1, -1)) for _ in range(nband)]
            wsums = np.zeros(nband)
            for ds in dds:
                b = ds.bandid
                wsums[b] += ds.WSUM.values[0]
                residual[b] += ds.CLEAN_RESIDUAL.data
            wsum = np.sum(wsums)
            residual = (da.stack(residual)/wsum).compute()

            residual_mfs = np.sum(residual, axis=0)
            save_fits(f'{basename}_clean_residual_mfs.fits',
                      residual_mfs, hdr_mfs)

        if opts.fits_cubes:
            # need residual in Jy/beam
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(f'{basename}_clean_model.fits', model, hdr)

            if opts.do_residual:
                fmask = wsums > 0
                residual[fmask] /= wsums[fmask, None, None]
                save_fits(f'{basename}_clean_residual.fits',
                          residual, hdr)

    print("All done here.", file=log)
