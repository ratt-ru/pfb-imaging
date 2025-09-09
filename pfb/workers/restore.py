# flake8: noqa
from contextlib import ExitStack
import click
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

log = pfb_logging.get_logger('RESTORE')


@click.command(context_settings={'show_default': True})
@clickify_parameters(schema.restore)
def restore(**kw):
    '''
    Create fits image cubes from data products (eg. restored images).
    '''
    opts = OmegaConf.create(kw)

    from pfb.utils.naming import set_output_names
    opts, basedir, oname = set_output_names(opts)

    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        opts.nthreads = nthreads//2
        ncpu = ncpu//2

    OmegaConf.set_struct(opts, True)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool, thread_pool_size
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{opts.log_directory}/restore_{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    pfb_logging.log_options_dict(log, opts)

    with ExitStack() as stack:
        if opts.nworkers > 1:
            from pfb import set_client
            client = set_client(opts.nworkers, log, stack=stack, client_log_level=opts.log_level)
        else:
            log.info("Faking client")
            from pfb.utils.dist import fake_client
            client = fake_client()
            names = [0]
        ti = time.time()
        _restore(**opts)

    log.info(f"All done after {time.time() - ti}s")

def _restore(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import numpy as np
    from pfb.utils.naming import xds_from_url, xds_from_list
    from pfb.utils.fits import dds2fits
    from pfb.utils.misc import Gaussian2D, fitcleanbeam, convolve2gaussres
    from ducc0.fft import c2c
    from pfb.utils.restoration import restore_image
    from itertools import cycle
    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import get_opts
    from distributed import wait
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

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds, dds_list = xds_from_url(dds_name)
    dds_store = DaskMSStore(dds_name)
    if '://' in dds_store.url:
        protocol = dds_store.url.split('://')[0]
    else:
        protocol = 'file'

    # get MFS PSF PARS
    try:
        psfpars_mfs = get_opts(dds_store.url,
                            protocol,
                            name='psfparsn_mfs.pkl')
    except:
        log.error_and_raise("Could not load MFS PSF pamaters. "
                            "Run grid worker with psf=true to remake.",
                            RuntimeError)

    if opts.drop_bands is not None:
        ddso = []
        ddso_list = []
        for ds, dsl in zip(dds, dds_list):
            b = int(ds.bandid)
            if b not in opts.drop_bands:
                ddso.append(ds)
                ddso_list.append(dsl)
        dds = ddso
        dds_list = ddso_list

    timeids = np.unique(np.array([int(ds.timeid) for ds in dds]))
    freqs = [ds.freq_out for ds in dds]
    freqs = np.unique(freqs)
    nband = freqs.size
    ntime = timeids.size
    ncorr = dds[0].corr.size
    cell_asec = dds[0].cell_rad * 180/np.pi * 3600
    log.info(f'Number of output times = {ntime}')
    log.info(f'Number of output bands = {nband}')


    if opts.gausspar is None:
        gaussparf = None
        gaussparf_mfs = None
        log.info("Using native resolution")
    elif opts.gausspar == [0,0,0]:
        # This is just to figure out what resolution to convolve to
        dds = xds_from_list(dds_list, nthreads=opts.nthreads,
                            drop_all_but=['PSFPARSN'])
        emaj = 0.0
        emin = 0.0
        pas = []
        for ds in dds:
            gausspars = ds.PSFPARSN.values
            emaj = np.maximum(emaj, gausspars[:, 0].max())
            emin = np.maximum(emin, gausspars[:, 1].max())
            pas.append(np.mean(gausspars[:, 2]))
        pa = np.mean(np.array(pas))
        log.info(f"Using lowest resolution of ({emaj*cell_asec:.3e} asec, "
              f"{emin*cell_asec:.3e} asec, {pa:.3e} degrees) for restored images")
        # these are n pixel units
        gaussparf_mfs = [emaj, emin, pa]
        gaussparf = gaussparf_mfs * ncorr
    else:
        gaussparf_mfs = list(opts.gausspar)
        emaj = gaussparf_mfs[0]
        emin = gaussparf_mfs[1]
        pa = gaussparf_mfs[2]
        if 'i' in opts.outputs.lower():
            log.info(f"Using specified resolution of ({emaj:.3e} asec, {emin:.3e} asec, "
                f"{pa:.3e} degrees) for restored image")
        # convert to pixel units
        for i in range(2):
            gaussparf_mfs[i] /= cell_asec
        gaussparf = [gaussparf_mfs]*ncorr

    # create restored images
    futrestore = []
    for ds_name in dds_list:
        fut = client.submit(restore_image,
                            ds_name,
                            opts.model_name,
                            opts.residual_name,
                            gaussparf=gaussparf,
                            nthreads=opts.nthreads)
        futrestore.append(fut)

    futures = []
    if 'd' in opts.outputs.lower():
        fut = client.submit(
                        dds2fits,
                        dds_list,
                        'DIRTY',
                        f'{fits_oname}_{opts.suffix}',
                        norm_wsum=True,
                        nthreads=opts.nthreads,
                        do_mfs='d' in opts.outputs,
                        do_cube='D' in opts.outputs,
                        psfpars_mfs=psfpars_mfs)
        futures.append(fut)

    if 'm' in opts.outputs.lower():
        fut = client.submit(
                        dds2fits,
                        dds_list,
                        opts.model_name,
                        f'{fits_oname}_{opts.suffix}',
                        norm_wsum=False,
                        nthreads=opts.nthreads,
                        do_mfs='m' in opts.outputs,
                        do_cube='M' in opts.outputs,
                        psfpars_mfs=psfpars_mfs)
        futures.append(fut)

    if 'r' in opts.outputs.lower():
        fut = client.submit(
                        dds2fits,
                        dds_list,
                        opts.residual_name,
                        f'{fits_oname}_{opts.suffix}',
                        norm_wsum=True,
                        nthreads=opts.nthreads,
                        do_mfs='r' in opts.outputs,
                        do_cube='R' in opts.outputs,
                        psfpars_mfs=psfpars_mfs)
        futures.append(fut)

    if opts.nworkers > 1:
        wait(futrestore)

    if 'i' in opts.outputs.lower():
        fut = client.submit(
                        dds2fits,
                        dds_list,
                        'IMAGE',
                        f'{fits_oname}_{opts.suffix}',
                        norm_wsum=False,
                        nthreads=opts.nthreads,
                        do_mfs='i' in opts.outputs,
                        do_cube='I' in opts.outputs,
                        psfpars_mfs=psfpars_mfs)
        futures.append(fut)

    # TODO(LB) - we may want to add these outputs back in, at least the useful ones
    # if 'f' in opts.outputs:
    #     rhat_mfs = c2c(residual_mfs, forward=True,
    #                    nthreads=opts.nthreads, inorm=0)
    #     rhat_mfs = np.fft.fftshift(rhat_mfs)
    #     save_fits(np.abs(rhat_mfs),
    #               f'{fits_oname}_{opts.suffix}.abs_fft_residual_mfs.fits',
    #               hdr_mfs,
    #               overwrite=opts.overwrite)
    #     save_fits(np.angle(rhat_mfs),
    #               f'{fits_oname}_{opts.suffix}.phase_fft_residual_mfs.fits',
    #               hdr_mfs,
    #               overwrite=opts.overwrite)

    # if 'F' in opts.outputs:
    #     rhat = c2c(residual, axes=(1,2), forward=True,
    #                nthreads=opts.nthreads, inorm=0)
    #     rhat = np.fft.fftshift(rhat, axes=(1,2))
    #     save_fits(np.abs(rhat),
    #               f'{fits_oname}_{opts.suffix}.abs_fft_residual.fits',
    #               hdr,
    #               overwrite=opts.overwrite)
    #     save_fits(np.angle(rhat),
    #               f'{fits_oname}_{opts.suffix}.phase_fft_residual.fits',
    #               hdr,
    #               overwrite=opts.overwrite)

    # if 'c' in opts.outputs:
    #     if GaussPar is None:
    #         raise ValueError("Clean beam in output but no PSF in dds")
    #     cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)
    #     save_fits(cpsf_mfs,
    #               f'{fits_oname}_{opts.suffix}.cpsf_mfs.fits',
    #               hdr_mfs,
    #               overwrite=opts.overwrite)

    # if 'C' in opts.outputs:
    #     if GaussPars is None:
    #         raise ValueError("Clean beam in output but no PSF in dds")
    #     cpsf = np.zeros(residual.shape, dtype=output_type)
    #     for v in range(opts.nband):
    #         gpar = GaussPars[v]
    #         if not np.isnan(gpar).any():
    #             cpsf[v] = Gaussian2D(xx, yy, gpar, normalise=False)
    #     save_fits(cpsf,
    #               f'{fits_oname}_{opts.suffix}.cpsf.fits',
    #               hdr,
    #               overwrite=opts.overwrite)

    if opts.nworkers > 1:
        wait(futures)

    return
