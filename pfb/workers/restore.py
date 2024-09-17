# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('RESTORE')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
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
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    with ExitStack() as stack:
        if opts.nworkers > 1:
            from pfb import set_client
            client = set_client(opts.nworkers, log, stack=stack, client_log_level=opts.log_level)
        else:
            print("Faking client", file=log)
            from pfb.utils.dist import fake_client
            client = fake_client()
            names = [0]
        ti = time.time()
        _restore(**opts)

    print(f"All done after {time.time() - ti}s", file=log)

def _restore(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import numpy as np
    from pfb.utils.naming import xds_from_url, xds_from_list
    from pfb.utils.fits import save_fits, add_beampars, set_wcs, dds2fits
    from pfb.utils.misc import Gaussian2D, fitcleanbeam, convolve2gaussres
    from ducc0.fft import c2c
    from pfb.utils.restoration import restore_cube
    from itertools import cycle
    from daskms.fsspec_store import DaskMSStore
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
    dds_store = DaskMSStore(dds_name)
    dds, dds_list = xds_from_url(dds_store.url)

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

    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))
    nband = freq_out.size
    ntime = time_out.size
    if ntime > 1:
        raise NotImplementedError('Multiple output times not yet supported')
    print(f'Number of output bands = {nband}', file=log)

    # get available gausspars
    # we need to compute MFS gausspars on the runner
    wsums = np.array([ds.wsum for ds in dds])
    wsum = np.sum(wsums)
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    cell_asec = cell_deg*3600
    if 'gaussparn' in dds[0].attrs:
        gaussparn = [ds.gaussparn for ds in dds]
        psf_mfs = np.sum([ds.PSF.values for ds in dds], axis=0)/wsum
        gaussparn_mfs = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    else:
        gaussparn = None
        gaussparn_mfs = None

    if 'gaussparu' in dds[0].attrs:
        gaussparu = [ds.gaussparu for ds in dds]
        upsf_mfs = np.sum([ds.UPSF.values * ds.wsum for ds in dds], axis=0)/wsum
        gaussparu_mfs = fitcleanbeam(upsf_mfs[None], level=0.5, pixsize=1.0)[0]
    else:
        gaussparu = None
        gaussparu_mfs = None

    if 'gaussparm' in dds[0].attrs:
        gaussparm = [ds.gaussparm for ds in dds]
        mpsf_mfs = np.sum([ds.MPSF.values * ds.wsum for ds in dds], axis=0)/wsum
        gaussparm_mfs = fitcleanbeam(mpsf_mfs[None], level=0.5, pixsize=1.0)[0]
    else:
        gaussparm = None
        gaussparm_mfs = None

    if opts.gausspar is None:
        gaussparf = gaussparn
        gaussparf_mfs = gaussparn_mfs
        gaussparfu = gaussparu
        gaussparfu_mfs = gaussparu_mfs
        print("Using native resolution", file=log)
    elif opts.gausspar == [0,0,0]:
        gaussparf = [gaussparn[0]]*nband
        emaj = gaussparf[0][0] * cell_asec
        emin = gaussparf[0][1] * cell_asec
        pa = gaussparf[0][2]
        gaussparf_mfs = gaussparn[0]
        print(f"Using lowest resolution of ({emaj:.3e} asec, {emin:.3e} asec, "
              f"{pa:.3e} degrees) for restored images", file=log)

        if gaussparu is not None:
            # inflate and cicularise
            emaj = gaussparu[0][0]
            emin = gaussparu[0][1]
            emaj = np.maximum(emaj, emin)
            print(f"Lowest uniform resolution is {emaj*cell_asec:.3e} asec",
                  file=log)
            emaj *= opts.inflate_factor
            gaussparfu = gaussparu[0]
            gaussparfu[0] = emaj
            gaussparfu[1] = emaj
            gaussparfu[2] = 0.0
            print(f"Setting uniformly blurred image resolution to {emaj*cell_asec:.3e} asec",
                  file=log)
            gaussparfu_mfs = gaussparfu
            gaussparfu = [gaussparfu]*nband

    else:
        gaussparf = list(opts.gausspar)
        emaj = gaussparf[0]
        emin = gaussparf[1]
        pa = gaussparf[2]
        if 'i' in opts.outputs.lower():
            print(f"Using specified resolution of ({emaj:.3e} asec, {emin:.3e} asec, "
                f"{pa:.3e} degrees) for restored image", file=log)
        gaussparfu = [opts.gausspar[0]*opts.inflate_factor,
                      opts.gausspar[1]*opts.inflate_factor,
                      opts.gausspar[2]]  # don't inflat PA
        emaj = gaussparfu[0]
        emin = gaussparfu[1]
        pa = gaussparfu[2]
        if 'u' in opts.outputs.lower():
            print(f"Using inflated resolution of ({emaj:.3e} asec, {emin:.3e} asec, "
                f"{pa:.3e} degrees) for uniformly blurred image", file=log)
        # convert to pixel units
        for i in range(2):
            gaussparf[i] /= cell_asec
            gaussparfu[i] /= cell_asec
        gaussparf_mfs = gaussparf
        gaussparfu_mfs = gaussparfu
        gaussparf = [gaussparf]*nband
        gaussparfu = [gaussparfu]*nband

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
                        do_cube='D' in opts.outputs)
        futures.append(fut)

    if 'm' in opts.outputs.lower():
        fut = client.submit(
                        dds2fits,
                        dds_list,
                        'MODEL',
                        f'{fits_oname}_{opts.suffix}',
                        norm_wsum=False,
                        nthreads=opts.nthreads,
                        do_mfs='m' in opts.outputs,
                        do_cube='M' in opts.outputs)
        futures.append(fut)

    if 'r' in opts.outputs.lower():
        fut = client.submit(
                        dds2fits,
                        dds_list,
                        'RESIDUAL',
                        f'{fits_oname}_{opts.suffix}',
                        norm_wsum=True,
                        nthreads=opts.nthreads,
                        do_mfs='r' in opts.outputs,
                        do_cube='R' in opts.outputs)
        futures.append(fut)

    if 'i' in opts.outputs.lower():
        if gaussparn is None:
            raise ValueError('Could not make restored image since Gausspars not in dds')
        fut = client.submit(
                        restore_cube,
                        dds_list,
                        f'{fits_oname}_{opts.suffix}' + '_image',
                        'MODEL',
                        'RESIDUAL',
                        gaussparn,
                        gaussparn_mfs,
                        gaussparf,
                        gaussparf_mfs,
                        gaussparm=gaussparm,
                        gaussparm_mfs=gaussparm_mfs,
                        nthreads=opts.nthreads,
                        unit='Jy/beam',
                        output_dtype='f4')
        futures.append(fut)

    if 'u' in opts.outputs.lower():
        if gaussparu is None:
            raise ValueError('Could not make uniformly blurred image since Gausspars not in dds')
        fut = client.submit(
                        restore_cube,
                        dds_list,
                        f'{fits_oname}_{opts.suffix}' + '_uimage',
                        'MODEL',
                        'UPDATE',
                        gaussparu,
                        gaussparu_mfs,
                        gaussparfu,
                        gaussparfu_mfs,
                        gaussparm=gaussparm,
                        gaussparm_mfs=gaussparm_mfs,
                        nthreads=opts.nthreads,
                        unit='Jy/pixel',
                        output_dtype='f4')
        futures.append(fut)

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
