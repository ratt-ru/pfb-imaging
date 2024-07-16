# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FLUXMOP')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.fluxmop["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.fluxmop["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.fluxmop)
def fluxmop(**kw):
    '''
    Forward step aka flux mop.

    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)

    from pfb.utils.naming import set_output_names
    opts, basedir, oname = set_output_names(opts)

    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        if opts.nworkers > 1:
            ntpw = nthreads//opts.nworkers
            opts.nthreads = ntpw//2
            ncpu = ntpw//2
        else:
            opts.nthreads = nthreads//2
            ncpu = ncpu//2

    OmegaConf.set_struct(opts, True)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool, thread_pool_size
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/fluxmop_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import xds_from_url
    from pfb.utils.fits import dds2fits, dds2fits_mfs

    basename = f'{basedir}/{oname}'
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_store = DaskMSStore(f'{basename}_{opts.suffix}.dds')

    with ExitStack() as stack:
        from distributed import wait
        from pfb import set_client
        if opts.nworkers > 1:
            client = set_client(opts.nworkers, stack, log)
            from distributed import as_completed
        else:
            print("Faking client", file=log)
            from pfb.utils.dist import fake_client
            client = fake_client()
            names = [0]
            as_completed = lambda x: x

        ti = time.time()
        _fluxmop(**opts)

        dds_list = dds_store.fs.glob(f'{dds_store.url}/*.zarr')

        # convert to fits files
        if opts.fits_mfs or opts.fits_cubes:
            print(f"Writing fits files to {fits_oname}_{opts.suffix}",
                  file=log)
            futures = []
            fut = client.submit(
                    dds2fits,
                    dds_list,
                    'RESIDUAL',
                    f'{fits_oname}_{opts.suffix}',
                    norm_wsum=True,
                    nthreads=opts.nthreads,
                    do_mfs=opts.fits_mfs,
                    do_cube=opts.fits_cubes)
            futures.append(fut)
            fut = client.submit(
                    dds2fits,
                    dds_list,
                    'MODEL',
                    f'{fits_oname}_{opts.suffix}',
                    norm_wsum=False,
                    nthreads=opts.nthreads,
                    do_mfs=opts.fits_mfs,
                    do_cube=opts.fits_cubes)
            futures.append(fut)
            fut = client.submit(
                    dds2fits,
                    dds_list,
                    'UPDATE',
                    f'{fits_oname}_{opts.suffix}',
                    norm_wsum=False,
                    nthreads=opts.nthreads,
                    do_mfs=opts.fits_mfs,
                    do_cube=opts.fits_cubes)
            futures.append(fut)
            fut = client.submit(
                    dds2fits,
                    dds_list,
                    'X0',
                    f'{fits_oname}_{opts.suffix}',
                    norm_wsum=False,
                    nthreads=opts.nthreads,
                    do_mfs=opts.fits_mfs,
                    do_cube=opts.fits_cubes)
            futures.append(fut)

            for fut in as_completed(futures):
                column = fut.result()
                print(f'Done writing {column}', file=log)

    print(f"All done after {time.time() - ti}s", file=log)

def _fluxmop(ddsi=None, **kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from itertools import cycle
    import numpy as np
    import xarray as xr
    from pfb.utils.fits import load_fits, set_wcs, save_fits
    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import xds_from_url, xds_from_list
    from pfb.utils.misc import init_mask, dds2cubes
    from pfb.operators.hessian import hessian_xds, hessian_psf_cube
    from pfb.operators.psf import psf_convolve_cube
    from pfb.opt.pcg import pcg_dds
    from ducc0.misc import resize_thread_pool, thread_pool_size
    from ducc0.fft import c2c
    iFs = np.fft.ifftshift
    Fs = np.fft.fftshift

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds_store = DaskMSStore(dds_name)
    dds_list = dds_store.fs.glob(f'{dds_store.url}/*.zarr')
    if ddsi is not None:
        dds = []
        for ds in ddsi:
            dds.append(ds.chunk({'row':-1,
                                 'chan':-1,
                                 'x':-1,
                                 'y':-1,
                                 'x_psf':-1,
                                 'y_psf':-1,
                                 'yo2':-1}))
    else:
        # are these sorted correctly?
        dds = xds_from_url(dds_store.url)

    nx, ny = dds[0].x.size, dds[0].y.size
    nx_psf, ny_psf = dds[0].x_psf.size, dds[0].y_psf.size
    lastsize = ny_psf
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))

    nband = freq_out.size
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    if opts.residual_name in dds[0]:
        residual = np.stack([getattr(ds, opts.residual_name).values for ds in dds],
                            axis=0)
    else:
        print("Using dirty image as residual", file=log)
        residual = np.stack([ds.DIRTY.values for ds in dds], axis=0)
    if opts.model_name in dds[0]:
        model = np.stack([getattr(ds, opts.model_name).values for ds in dds],
                         axis=0)
    else:
        model = np.zeros((nband, nx, ny))
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands
    wsum = np.sum(wsums)
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if opts.mask is not None:
        if opts.mask=='model':
            mask = np.any(model > opts.min_model, axis=0)
            assert mask.shape == (nx, ny)
            mask = mask.astype(residual.dtype)
            print('Using model > 0 to create mask', file=log)
        else:
            mask = load_fits(opts.mask, dtype=residual.dtype).squeeze()
            assert mask.shape == (nx, ny)
            if opts.or_mask_with_model:
                print("Combining model with input mask", file=log)
                mask = np.logical_or(mask>0, model_mfs>0).astype(residual.dtype)


            mask = mask.astype(residual.dtype)
            print('Using provided fits mask', file=log)
            if opts.zero_model_outside_mask and not opts.or_mask_with_model:
                model[:, mask<1] = 0
                print("Recomputing residual since asked to zero model", file=log)
                convimage = hess(model)
                ne.evaluate('dirty - convimage', out=residual,
                            casting='same_kind')
                ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                            casting='same_kind')
                save_fits(np.mean(model[fsel], axis=0),
                  basename + f'_{opts.suffix}_model_mfs_zeroed.fits',
                  hdr_mfs)
                save_fits(residual_mfs,
                  basename + f'_{opts.suffix}_residual_mfs_zeroed.fits',
                  hdr_mfs)


    else:
        mask = np.ones((nx, ny), dtype=residual.dtype)
        print('Caution - No mask is being applied', file=log)

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    print(f"Initial peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)

    print("Solving for update", file=log)
    try:
        from distributed import get_client
        client = get_client()
        names = list(client.scheduler_info()['workers'].keys())
        from distributed import as_completed
    except:
        from pfb.utils.dist import fake_client
        client = fake_client()
        names = [0]
        as_completed = lambda x: x
    futures = []
    for wname, ds, ds_name in zip(cycle(names), dds, dds_list):
        fut = client.submit(
            pcg_dds,
            ds_name,
            opts.sigmainvsq,
            opts.sigma,
            use_psf=opts.use_psf,
            residual_name=opts.residual_name,
            model_name=opts.model_name,
            mask=mask,
            do_wgridding=opts.do_wgridding,
            epsilon=opts.epsilon,
            double_accum=opts.double_accum,
            nthreads=opts.nthreads,
            tol=opts.cg_tol,
            maxit=opts.cg_maxit,
            verbosity=opts.cg_verbose,
            report_freq=opts.cg_report_freq,
            workers=wname
        )
        futures.append(fut)

    nds = len(futures)
    n_launched = 1
    for fut in as_completed(futures):
        print(f"\rProcessed: {n_launched}/{nds}", end='', flush=True)
        r, b = fut.result()
        residual[b] = r
        n_launched += 1

    print("\n")  # after progressbar above

    residual_mfs = np.sum(residual/wsum, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    print(f"Final peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)

    return
