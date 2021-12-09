# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('CLEAN')

@cli.command()
@click.option('-xds', '--xds', required=True,
              help="Path to xarray dataset containing data products")
@click.option('-mds', '--mds', type=str,
              help="Path to xarray dataset containing model products.")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('--use-clark/--no-use-clark', default=True)
@click.option('--update-mask/--no-upadte-mask', default=True)
@click.option('--fits-mfs/--no-fits-mfs', default=True)
@click.option('--no-fits-cubes/--fits-cubes', default=True)
@click.option('-nmiter', '--nmiter', type=int, default=5,
              help="Number of major cycles")
@click.option('-th', '--threshold', type=float,
              help='Stop cleaning when the MFS residual reaches '
              'this threshold.')
@click.option('-gamma', "--gamma", type=float, default=0.05,
              help="Minor loop gain of Hogbom/Clark")
@click.option('-pf', "--peak-factor", type=float, default=0.15,
              help="Peak factor of Hogbom/Clark")
@click.option('-spf', "--sub-peak-factor", type=float, default=0.75,
              help="Peak factor in sub-minor loop of Clark")
@click.option('-maxit', "--maxit", type=int, default=50,
              help="Maximum number of iterations for Hogbom/Clark")
@click.option('-smaxit', "--sub-maxit", type=int, default=5000,
              help="Maximum number of sub-minor iterations for Clark")
@click.option('-verb', "--verbose", type=int, default=0,
              help="Verbosity of Hogbom/Clark. Set to 2 for debugging or "
              "zero for silence.")
@click.option('-rf', "--report-freq", type=int, default=10,
              help="Report freq for Hogbom/Clark.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use per worker")
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
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
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')

    if args.nworkers is None:
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(args, stack, log, scheduler=args.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _clean(**args)

def _clean(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)

    import numpy as np
    import xarray as xr
    import numexpr as ne
    import dask
    import dask.array as da
    from dask.distributed import performance_report
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.deconv.hogbom import hogbom
    from pfb.deconv.clark import clark
    from daskms.experimental.zarr import xds_from_zarr

    xds = xds_from_zarr(args.xds, chunks={'row':args.row_chunk})
    nband = xds[0].nband
    nx = xds[0].nx
    ny = xds[0].ny
    nx_psf = xds[0].nx_psf
    ny_psf = xds[0].ny_psf

    # stitch dirty/psf in apparent scale
    dirty = np.zeros((nband, nx, ny), dtype=args.output_type)
    psf = np.zeros((nband, nx_psf, ny_psf), dtype=args.output_type)
    wsum = 0
    for ds in xds:
        if 'RESIDUAL' in ds:
            d = ds.RESIDUAL.values
        else:
            d = ds.DIRTY.values
        band_id = ds.band_id.values
        dirty[band_id] += d
        psf[band_id] += ds.PSF.values
        wsum += ds.WSUM.values.sum()
    dirty /= wsum
    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)
    dirty_mfs = np.sum(dirty, axis=0)
    assert (psf_mfs.max() - 1.0) < 2*args.epsilon

    if args.mds is not None:
        mds_name = args.mds
        # only one mds (for now)
        try:
            mds = xr.open_zarr(mds_name,
                               chunks={'band': 1, 'x': -1, 'y': -1})
        except Exception as e:
            print(f'{args.mds} not found or invalid', file=log)
            raise e
    else:
        mds_name = args.output_filename + '.mds.zarr'
        print(f"Model dataset not passed in. Initialising as {mds_name}.",
              file=log)
        # may not exist yet
        mds = xr.Dataset()

    # set up Hessian
    from pfb.operators.hessian import hessian_xds
    hessopts = {}
    hessopts['cell'] = xds[0].cell_rad
    hessopts['wstack'] = args.wstack
    hessopts['epsilon'] = args.epsilon
    hessopts['double_accum'] = args.double_accum
    hessopts['nthreads'] = args.nvthreads
    # always clean in apparent scale
    hess = partial(hessian_xds, xds=xds, hessopts=hessopts,
                   wsum=wsum, sigmainv=0, mask=np.ones_like(dirty_mfs),
                   compute=True, use_beam=False)

    # to set up psf convolve when using Clark
    if args.use_clark:
        from pfb.operators.psf import psf_convolve
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
        psf_pad = iFs(psf, axes=(1, 2))
        psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                     nthreads=args.nvthreads, inorm=0)

        psfhat = da.from_array(psfhat, chunks=(1, -1, -1))
        psfopts = {}
        psfopts['padding'] = padding[1:]
        psfopts['unpad_x'] = unpad_x
        psfopts['unpad_y'] = unpad_y
        psfopts['lastsize'] = lastsize
        psfopts['nthreads'] = args.nvthreads
        psfo = partial(psf_convolve, psfhat=psfhat, beam=None, psfopts=psfopts)

    rms = np.std(dirty_mfs)
    rmax = np.abs(dirty_mfs).max()

    print("Iter %i: peak residual = %f, rms = %f" % (0, rmax, rms), file=log)

    residual = dirty.copy()
    residual_mfs = dirty_mfs.copy()
    model = np.zeros_like(residual)
    for k in range(args.nmiter):
        if args.use_clark:
            print("Running Clark", file=log)
            x = clark(residual, psf, psfo=psfo,
                      gamma=args.gamma,
                      pf=args.peak_factor,
                      maxit=args.maxit,
                      subpf=args.sub_peak_factor,
                      submaxit=args.sub_maxit,
                      verbosity=args.verbose,
                      report_freq=args.report_freq)
        else:
            print("Running Hogbom", file=log)
            x = hogbom(residual, psf,
                       gamma=args.gamma,
                       pf=args.peak_factor,
                       maxit=args.maxit,
                       verbosity=args.verbose,
                       report_freq=args.report_freq)

        model += x
        print("Getting residual", file=log)
        convimage = hess(x)
        ne.evaluate('residual - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        # save_fits(args.output_filename + f'_residual_mfs{k}.fits',
        #           residual_mfs, hdr_mfs)
        # save_fits(args.output_filename + f'_model_mfs{k}.fits',
        #           np.mean(model, axis=0), hdr_mfs)
        # save_fits(args.output_filename + f'_convim_mfs{k}.fits',
        #           np.sum(convimage, axis=0), hdr_mfs)

        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()

        print("Iter %i: peak residual = %f, rms = %f" % (
                k+1, rmax, rms), file=log)

        if args.threshold is not None:
            if rmax <= args.threshold:
                print("Terminating because final threshold has been reached",
                      file=log)
                break

    print("Saving results", file=log)
    if args.update_mask:
        try:
            mask = mds.MASK.values
        except:
            mask = np.zeros((nx, ny))
        mask = np.logical_or(mask, np.any(model, axis=0)).astype(args.output_type)
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask)
        mds = mds.assign(**{
                'MASK': (('x', 'y'), da.from_array(mask, chunks=(-1, -1)))
        })


    mds = mds.assign(**{
            'CLEAN_MODEL': (('band', 'x', 'y'), model),
            'CLEAN_RESIDUAL': (('band', 'x', 'y'), residual)
    })

    mds.to_zarr(mds_name, mode='a')

    if args.fits_mfs or not args.no_fits_cubes:
        print("Writing fits files", file=log)
        # construct a header from xds attrs
        ra = xds[0].ra
        dec = xds[0].dec
        radec = [ra, dec]

        cell_rad = xds[0].cell_rad
        cell_deg = np.rad2deg(cell_rad)

        freq_out = np.unique(np.concatenate([ds.band.values for ds in xds]))
        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        model_mfs = np.mean(model, axis=0)

        save_fits(args.output_filename + '_model_mfs.fits', model_mfs, hdr_mfs)
        save_fits(args.output_filename + '_residual_mfs.fits', residual_mfs, hdr_mfs)

        if not args.no_fits_cubes:
            # need residual in Jy/beam
            wsums = np.amax(psf, axes=(1,2))
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(args.output_filename + '_residual.fits',
                      residual/wsums[:, None, None], hdr)
            save_fits(args.output_filename + '_model.fits', model, hdr)

    print("All done here.", file=log)
