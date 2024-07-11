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

# create default parameters from schema
defaults = {}
for key in schema.restore["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.restore["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.restore)
def restore(**kw):
    '''
    Create fits image cubes from data products (eg. restored images).
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)

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

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{opts.log_directory}/restore_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    with ExitStack() as stack:
        if opts.nworkers > 1:
            from pfb import set_client
            set_client(opts.nworkers, stack, log)
            client = get_client()
        else:
            print("Faking client", file=log)
            from pfb.utils.dist import fake_client
            client = fake_client()
            names = [0]
        ti = time.time()
        return _restore(**opts)

    print(f"All done after {time.time() - ti}s", file=log)

def _restore(ddsi=None, **kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import numpy as np
    from pfb.utils.naming import xds_from_url, xds_from_list
    from pfb.utils.fits import (save_fits, add_beampars, set_wcs,
                                dds2fits, dds2fits_mfs)
    from pfb.utils.misc import Gaussian2D, fitcleanbeam, convolve2gaussres, dds2cubes
    from ducc0.fft import c2c
    from pfb.utils.restoration import restore_ds
    from itertools import cycle
    from daskms.fsspec_store import DaskMSStore
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
        dds = xds_from_list(dds_list)

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

    # init fits headers
    radec = (dds[0].ra, dds[0].dec)
    ref_freq = np.mean(freq_out)
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    cell_asec = cell_deg * 3600
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)

    if opts.overwrite:
        print("Warning! Potentially overwriting output images",
              file=log)

    # stitch dirty/psf in apparent scale
    # drop_vars to avoid duplicates in memory
    output_type = dds[0].DIRTY.dtype
    if 'r' in opts.outputs.lower() and opts.residual_name not in dds[0]:
        raise ValueError(f'{opts.residual_name} not found in dds. '
                         'Unable to produce requested residual.')
    if 'm' in opts.outputs.lower() and opts.model_name not in dds[0]:
        raise ValueError(f'{opts.model_name} not found in dds. '
                         'Unable to produce requested model.')
    if 'i' in opts.outputs and 'PSF' not in dds[0]:
        if opts.gausspar is None:
            raise ValueError('PSF not found in dds. '
                             'Unable to produce restored image.')
        else:
            print("Warning - unable to homogenise residual since no PSF "
                  "found in dds.", file=log)
        # psf = np.stack([ds.PSF.values for ds in dds], axis=0)
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    wsum = np.sum(wsums)
    fmask = wsums > 0  # keep track of empty bands
    if (~fmask).all():
        raise ValueError("All data seem to be flagged")

    if opts.gausspar is None:
        gaussparf = None
        print("Using native resolution", file=log)
    elif opts.gausspar == [0,0,0]:
        bid0 = np.array([int(ds.bandid) for ds in dds]).min()
        # TODO - there could be more than one ds per band
        ds0 = [ds for ds in dds if int(ds.bandid) == bid0][0]
        gaussparf = fitcleanbeam(ds0.PSF.values[None], level=0.5,
                                 pixsize=1)[0]
        emaj = gaussparf[0] * cell_asec
        emin = gaussparf[1] * cell_asec
        pa = gaussparf[2]
        print(f"Using lowest resolution of ({emaj:.3e} asec, {emin:.3e} asec, "
              f"{pa:.3e} degrees)", file=log)

        gaussparu = restore_ds(dds_list[0],
                               opts.outputs,
                               residual_name=opts.residual_name,
                               model_name=opts.model_name,
                               nthreads=opts.nthreads,
                               gaussparf=gaussparf)

    else:
        gaussparf = opts.gausspar
        emaj = gaussparf[0]
        emin = gaussparf[1]
        pa = gaussparf[2]
        print(f"Using specified resolution of ({emaj:.3e} asec, {emin:.3e} asec, "
              f"{pa:.3e} degrees)", file=log)

    import ipdb; ipdb.set_trace()

    # compute per band images on workers
    futures = []
    for wname, ds, ds_name in zip(cycle(names), dds, dds_list):
        fut = client.submit(restore_ds,
                            ds_name,
                            opts.outputs,
                            residual_name=opts.residual_name,
                            model_name=opts.model_name,
                            nthreads=opts.nthreads,
                            gaussparf=opts.gausspar)
        futures.append(fut)

    # mfs images on runner
    psf_mfs = np.sum([ds.PSF.values for ds in dds], axis=0)/wsum
    model_mfs = np.sum([getattr(ds, opts.model_name).values for ds in dds],
                       axis=0)/np.sum(fmask)
    residual_mfs = np.sum([getattr(ds, opts.residual_name).values for ds in dds],
                          axis=0)/wsum
    if not model_mfs.any():
        print("Warning - model is empty", file=log)

    # lm in pixel coordinates
    lpsf = -(nx//2) + np.arange(nx)
    mpsf = -(ny//2) + np.arange(ny)
    xx, yy = np.meshgrid(lpsf, mpsf, indexing='ij')

    # fit restoring psf (pixel units)
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)
    hdr_mfs = add_beampars(hdr_mfs, GaussPar, unit2deg=cell_deg)



    if 'm' in opts.outputs:
        save_fits(model_mfs,
                  f'{basename}_{opts.suffix}.model_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'M' in opts.outputs:
        save_fits(model,
                  f'{basename}_{opts.suffix}.model.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'r' in opts.outputs:
        save_fits(residual_mfs,
                  f'{basename}_{opts.suffix}.residual_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'R' in opts.outputs:
        save_fits(residual,
                  f'{basename}_{opts.suffix}.residual.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'f' in opts.outputs:
        rhat_mfs = c2c(residual_mfs, forward=True,
                       nthreads=opts.nthreads, inorm=0)
        rhat_mfs = np.fft.fftshift(rhat_mfs)
        save_fits(np.abs(rhat_mfs),
                  f'{basename}_{opts.suffix}.abs_fft_residual_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)
        save_fits(np.angle(rhat_mfs),
                  f'{basename}_{opts.suffix}.phase_fft_residual_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'F' in opts.outputs:
        rhat = c2c(residual, axes=(1,2), forward=True,
                   nthreads=opts.nthreads, inorm=0)
        rhat = np.fft.fftshift(rhat, axes=(1,2))
        save_fits(np.abs(rhat),
                  f'{basename}_{opts.suffix}.abs_fft_residual.fits',
                  hdr,
                  overwrite=opts.overwrite)
        save_fits(np.angle(rhat),
                  f'{basename}_{opts.suffix}.phase_fft_residual.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'd' in opts.outputs:
        dirty_mfs = np.sum(dirty, axis=0)
        save_fits(dirty_mfs,
                  f'{basename}_{opts.suffix}.dirty_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'D' in opts.outputs:
        dirty[fmask] /= wsums[fmask, None, None]/wsum
        save_fits(dirty,
                  f'{basename}_{opts.suffix}.dirty.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'i' in opts.outputs and psf is not None:
        image_mfs = convolve2gaussres(model_mfs[None], xx, yy,
                                      GaussPar[0], opts.nthreads,
                                      norm_kernel=False)[0]  # peak of kernel set to unity
        image_mfs += residual_mfs
        save_fits(image_mfs,
                  f'{basename}_{opts.suffix}.image_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'I' in opts.outputs and psf is not None:
        image = np.zeros_like(model)
        for b in range(nband):
            image[b:b+1] = convolve2gaussres(model[b:b+1], xx, yy,
                                            GaussPars[b], opts.nthreads,
                                            norm_kernel=False)  # peak of kernel set to unity
            image[b] += residual[b]
        save_fits(image,
                  f'{basename}_{opts.suffix}.image.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'c' in opts.outputs:
        if GaussPar is None:
            raise ValueError("Clean beam in output but no PSF in dds")
        cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)
        save_fits(cpsf_mfs,
                  f'{basename}_{opts.suffix}.cpsf_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'C' in opts.outputs:
        if GaussPars is None:
            raise ValueError("Clean beam in output but no PSF in dds")
        cpsf = np.zeros(residual.shape, dtype=output_type)
        for v in range(opts.nband):
            gpar = GaussPars[v]
            if not np.isnan(gpar).any():
                cpsf[v] = Gaussian2D(xx, yy, gpar, normalise=False)
        save_fits(cpsf,
                  f'{basename}_{opts.suffix}.cpsf.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here", file=log)
