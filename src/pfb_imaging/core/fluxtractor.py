# flake8: noqa
from contextlib import ExitStack
import click
from omegaconf import OmegaConf
from pfb_imaging.utils import logging as pfb_logging
from scabha.schema_utils import clickify_parameters
from pfb_imaging.parser.schemas import schema
import ray

log = pfb_logging.get_logger('FLUXTRACTOR')


@click.command(context_settings={'show_default': True})
@clickify_parameters(schema.fluxtractor)
@click.pass_context
def fluxtractor(ctx, **kw):
    '''
    Forward step aka flux mop.
    '''
    opts = OmegaConf.create(kw)

    from pfb_imaging.utils.naming import set_output_names
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

    from pfb_imaging import set_envs
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/fluxtractor_{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    pfb_logging.log_options_dict(log, opts)

    from pfb_imaging.utils.naming import xds_from_url

    basename = opts.output_filename
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_name =f'{basename}_{opts.suffix}.dds'

    # these are passed through to child Ray processes
    renv = {"env_vars": ctx.obj["env_vars"]}
    if opts.nworkers==1:
        renv["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"

    ray.init(num_cpus=opts.nworkers,
             logging_level='INFO',
             ignore_reinit_error=True,
             runtime_env=renv)

    ti = time.time()
    _fluxtractor(**opts)

    _, dds_list = xds_from_url(dds_name)

    # convert to fits files
    if opts.fits_mfs or opts.fits_cubes:
        from pfb_imaging.utils.fits import rdds2fits
        log.info(f"Writing fits files to {fits_oname}_{opts.suffix}")
        tasks = []
        task = rdds2fits.remote(
                dds_list,
                'RESIDUAL_MOPPED',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=True,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes)
        tasks.append(task)

        task = rdds2fits.remote(
                dds_list,
                'MODEL_MOPPED',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=False,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes)
        tasks.append(task)

        task = rdds2fits(
                dds_list,
                'UPDATE',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=False,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes)
        tasks.append(task)
        
        task = rdds2fits(
                dds_list,
                'X0',
                f'{fits_oname}_{opts.suffix}',
                norm_wsum=False,
                nthreads=opts.nthreads,
                do_mfs=opts.fits_mfs,
                do_cube=opts.fits_cubes)
        tasks.append(fut)

        remaining_tasks = tasks.copy()
        while remaining_tasks:
            # Wait for at least 1 task to complete
            ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

            # Process the completed task
            for task in ready:
                try:
                    column = ray.get(task)
                    log.info(f'Done writing {column}')
                
                # LB - why is this except here?
                except Exception as e:
                    continue

        log.info(f"All done after {time.time() - ti}s")
        ray.shutdown()

def _fluxtractor(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from itertools import cycle
    import numpy as np
    import xarray as xr
    from pfb_imaging.utils.fits import load_fits, set_wcs
    from daskms.fsspec_store import DaskMSStore
    from pfb_imaging.utils.naming import xds_from_url
    from pfb_imaging.utils.modelspec import fit_image_cube
    from pfb_imaging.opt.pcg import pcg_dds

    basename = opts.output_filename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds_store = DaskMSStore(dds_name)
    dds, dds_list = xds_from_url(dds_store.url)

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
        log.info("Using dirty image as residual")
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
            log.info('Using model > 0 to create mask')
        else:
            mask = load_fits(opts.mask, dtype=residual.dtype).squeeze()
            assert mask.shape == (nx, ny)
            if opts.or_mask_with_model:
                log.info("Combining model with input mask")
                mask = np.logical_or(mask>0, model_mfs>0).astype(residual.dtype)


            mask = mask.astype(residual.dtype)
            log.info('Using provided fits mask')
    else:
        mask = np.ones((nx, ny), dtype=residual.dtype)
        log.info('Caution - No mask is being applied')

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    log.info(f"Initial peak residual = {rmax:.3e}, rms = {rms:.3e}")

    log.info("Solving for update")
    tasks = []
    for ds_name in dds_list:
        task = pcg_dds.remote(
            ds_name,
            opts.eta,
            use_psf=opts.use_psf,
            residual_name=opts.residual_name,
            model_name=opts.model_name,
            mask=mask,
            do_wgridding=opts.do_wgridding,
            epsilon=opts.epsilon,
            double_accum=opts.double_accum,
            nthreads=opts.nthreads,
            zero_model_outside_mask=opts.zero_model_outside_mask,
            tol=opts.cg_tol,
            maxit=opts.cg_maxit,
            verbosity=opts.cg_verbose,
            report_freq=opts.cg_report_freq
        )
        tasks.append(task)

    nds = len(tasks)
    n_launched = 1
    remaining_tasks = tasks.copy()
    while remaining_tasks:
        # Wait for at least 1 task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        r, b = ray.get(ready)
        residual[b] = r
        n_launched += 1

        print(f"\rProcessed: {n_launched}/{nds}", end='\n', flush=True)

    residual_mfs = np.sum(residual/wsum, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    log.info(f"Final peak residual = {rmax:.3e}, rms = {rms:.3e}")

    log.info(f"Writing model to {basename}_{opts.suffix}_model.mds")

    try:
        coeffs, Ix, Iy, expr, params, texpr, fexpr = \
            fit_image_cube(time_out,
                           freq_out[fsel],
                           model[None, fsel, :, :],
                           wgt=wsums[None, fsel],
                           nbasisf=fsel.size,
                           method='Legendre',
                           sigmasq=1e-10)
        # save interpolated dataset
        data_vars = {
            'coefficients': (('par', 'comps'), coeffs),
        }
        coords = {
            'location_x': (('x',), Ix),
            'location_y': (('y',), Iy),
            # 'shape_x':,
            'params': (('par',), params),  # already converted to list
            'times': (('t',), time_out),  # to allow rendering to original grid
            'freqs': (('f',), freq_out)
        }
        mattrs = {
            'spec': 'genesis',
            'cell_rad_x': cell_rad,
            'cell_rad_y': cell_rad,
            'npix_x': nx,
            'npix_y': ny,
            'texpr': texpr,
            'fexpr': fexpr,
            'center_x': dds[0].x0,
            'center_y': dds[0].y0,
            'flip_u': dds[0].flip_u,
            'flip_v': dds[0].flip_v,
            'flip_w': dds[0].flip_w,
            'ra': dds[0].ra,
            'dec': dds[0].dec,
            'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
            'parametrisation': expr  # already converted to str
        }
        for key, val in opts.items():
            mattrs[key] = val

        coeff_dataset = xr.Dataset(data_vars=data_vars,
                            coords=coords,
                            attrs=mattrs)
        coeff_dataset.to_zarr(f"{basename}_{opts.suffix}_model_mopped.mds",
                              mode='w')
    except Exception as e:
            log.info(f"Exception {e} raised during model fit .")

    return
