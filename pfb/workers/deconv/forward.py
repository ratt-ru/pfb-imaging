# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FORWARD')

@cli.command(context_settings={'show_default': True})
@click.option('-xds', '--xds', type=str, required=True,
              help="Path to xarray dataset containing data products.")
@click.option('-mds', '--mds', type=str,
              help="Path to xarray dataset containing model products.")
@click.option('-rname', '--residual-name', default='RESIDUAL',
              help='Name of residual to use in xds')
@click.option('-mask', '--mask', default=None,
              help="Either path to mask.fits or set to mds to use "
              "the mask contained in the mds.")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('-sinv', '--sigmainv', type=float, default=1.0,
              help='Standard deviation of assumed GRF prior.'
              'Set it to rms/nband if uncertain')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('--use-psf/--no-use-psf', default=True)
@click.option('--fits-mfs/--no-fits-mfs', default=True)
@click.option('--no-fits-cubes/--fits-cubes', default=True)
@click.option('--do-residual/--no-do-residual', default=True)
@click.option('-cgtol', "--cg-tol", type=float, default=1e-5,
              help="Tolerance of conjugate gradient")
@click.option('-cgminit', "--cg-minit", type=int, default=10,
              help="Minimum number of iterations for conjugate gradient")
@click.option('-cgmaxit', "--cg-maxit", type=int, default=100,
              help="Maximum number of iterations for conjugate gradient")
@click.option('-cgverb', "--cg-verbose", type=int, default=0,
              help="Verbosity of conjugate gradient. "
              "Set to 2 for debugging or zero for silence.")
@click.option('-cgrf', "--cg-report-freq", type=int, default=10,
              help="Report freq for conjugate gradient.")
@click.option('--backtrack/--no-backtrack', default=True,
              help="Backtracking during cg iterations.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
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
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')

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

    xds = xds_from_zarr(args.xds, chunks={'row':args.row_chunk})
    nband = xds[0].nband
    nx = xds[0].nx
    ny = xds[0].ny

    # stitch residuals after beam application
    if args.residual_name in xds[0]:
        rname = args.residual_name
    else:
        rname = 'DIRTY'
    print(f'Using {rname} as residual', file=log)
    residual = np.zeros((nband, nx, ny), dtype=args.output_type)
    wsum = 0
    for ds in xds:
        d = ds.get(rname).values
        b = ds.BEAM.values
        band_id = ds.band_id.values
        residual[band_id] += d * b
        wsum += ds.WSUM.values.sum()
    residual /= wsum

    if args.mds is not None:
        mds_name = args.mds
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
        mds = mds.assign_attrs({'nband': nband})
        mds = mds.assign_attrs({'nx': nx})
        mds = mds.assign_attrs({'ny': ny})


    from pfb.utils.misc import init_mask
    mask = init_mask(args.mask, mds, args.output_type, log)

    try:
        x0 = mds.CLEAN_MODEL.values
    except:
        print("Initialising model to all zeros", file=log)
        x0 = np.zeros((nband, nx, ny), dtype=args.output_type)

    hessopts = {}
    hessopts['cell'] = xds[0].cell_rad
    hessopts['wstack'] = args.wstack
    hessopts['epsilon'] = args.epsilon
    hessopts['double_accum'] = args.double_accum
    hessopts['nthreads'] = args.nvthreads

    if args.use_psf:
        print("Initialising psf", file=log)
        from pfb.operators.psf import psf_convolve_xds
        from ducc0.fft import r2c
        normfact = 1.0

        psf = xds[0].PSF.data
        _, nx_psf, ny_psf = psf.shape

        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding[-1])

        # add psfhat to Dataset
        for i, ds in enumerate(xds):
            if 'PSFHAT' not in ds:
                psf_pad = iFs(ds.PSF.data.compute(), axes=(1, 2))
                psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                            nthreads=args.nthreads, inorm=0)

                psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)
                ds = ds.assign({'PSFHAT':(('band', 'x_psf', 'y_psfo2'), psfhat)})
                xds[i] = ds

        # LB - this rechunking of the data is really very annoying.
        # Why is it necessary again?
        # # update dataset on disk
        # xds_to_zarr(xds, args.xds, columns=['PSFHAT']).compute()

        psfopts = {}
        psfopts['padding'] = padding[1:]
        psfopts['unpad_x'] = unpad_x
        psfopts['unpad_y'] = unpad_y
        psfopts['lastsize'] = lastsize
        psfopts['nthreads'] = args.nvthreads

        hess = partial(psf_convolve_xds, xds=xds, psfopts=psfopts,
                       sigmainv=args.sigmainv, wsum=wsum, mask=mask,
                       compute=True)

    else:
        print("Solving for update using vis space approximation", file=log)
        hess = partial(hessian_xds, xds=xds, hessopts=hessopts,
                       sigmainv=args.sigmainv, wsum=wsum, mask=mask,
                       compute=True)

    # # import pdb; pdb.set_trace()
    # x = np.random.randn(nband, nbasis, nmax).astype(np.float32)
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

    ra = xds[0].ra
    dec = xds[0].dec
    radec = [ra, dec]

    cell_rad = xds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    freq_out = np.unique(np.concatenate([ds.band.values for ds in xds], axis=0))

    if 'ra' not in mds.attrs:
        mds = mds.assign_attrs({'ra': ra})

    if 'dec' not in mds.attrs:
        mds = mds.assign_attrs({'dec': dec})

    if 'cell_rad' not in mds.attrs:
        mds = mds.assign_attrs({'cell_rad': cell_rad})

    if 'nband' not in mds.attrs:
        mds = mds.assign_attrs({'nband': nband})

    if 'nx' not in mds.attrs:
        mds = mds.assign_attrs({'nx': nx})

    if 'ny' not in mds.attrs:
        mds = mds.assign_attrs({'ny': ny})

    if 'band' not in mds.coords:
        mds = mds.assign_coords({'band': da.from_array(freq_out, chunks=1)})

    mds.to_zarr(mds_name, mode='a')

    if args.do_residual:
        print("Computing residual", file=log)
        from pfb.operators.hessian import hessian
        # Required because of https://github.com/ska-sa/dask-ms/issues/171
        xdsw = xds_from_zarr(args.xds, chunks={'band': 1}, columns='DIRTY')
        writes = []
        for ds, dsw in zip(xds, xdsw):
            dirty = ds.get(rname).data
            wgt = ds.WEIGHT.data
            uvw = ds.UVW.data
            freq = ds.FREQ.data
            fbin_idx = ds.FBIN_IDX.data
            fbin_counts = ds.FBIN_COUNTS.data
            beam = ds.BEAM.data
            band_id = ds.band_id.data
            # we only want to apply the beam once here
            residual = (dirty -
                        hessian(uvw, wgt, freq, beam * update[band_id], None,
                        fbin_idx, fbin_counts, hessopts))
            dsw = dsw.assign(**{'FORWARD_RESIDUAL': (('band', 'x', 'y'), residual)})
            writes.append(xds_to_zarr(dsw, args.xds, columns='RESIDUAL'))

        dask.compute(writes)

    if args.fits_mfs or not args.no_fits_cubes:
        print("Writing fits files", file=log)
        # construct a header from xds attrs
        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        update_mfs = np.mean(update, axis=0)
        save_fits(args.output_filename + '_update_mfs.fits', update_mfs, hdr_mfs)

        if args.do_residual:
            xds = xds_from_zarr(args.xds, chunks={'band': 1})
            residual = np.zeros((nband, nx, ny), dtype=args.output_type)
            wsums = np.zeros(nband)
            for ds in xds:
                band_id = ds.band_id.values
                wsums[band_id] += ds.WSUM.values
                residual[band_id] += ds.FORWARD_RESIDUAL.values
            wsum = np.sum(wsums)
            residual /= wsum

            residual_mfs = np.sum(residual, axis=0)
            save_fits(args.output_filename + '_forward_residual_mfs.fits',
                      residual_mfs, hdr_mfs)

        if not args.no_fits_cubes:
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(args.output_filename + '_update.fits', update, hdr)

            if args.do_residual:
                fmask = wsums > 0
                residual[fmask] /= wsums[fmask, None, None]
                save_fits(args.output_filename + '_forward_residual.fits',
                          residual, hdr)

    print("All done here.", file=log)
