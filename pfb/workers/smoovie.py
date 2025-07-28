from pfb.workers.main import cli
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
pfb_logging.init('pfb')
log = pfb_logging.get_logger('SMOOVIE')
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.smoovie)
def smoovie(**kw):
    '''
    Smooth high cadence imaging results
    '''
    opts = OmegaConf.create(kw)

    from pfb.utils.naming import set_output_names
    opts, basedir, oname = set_output_names(opts)

    import psutil
    ncpu = psutil.cpu_count(logical=False)
    # to prevent flickering 
    opts.nthreads = 1
    remprod = opts.product.upper().strip('IQUV')
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported",
                            NotImplementedError)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/smoovie_{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    log.info('Input Options:')
    for key in opts.keys():
        log.info('     %25s = %s' % (key, opts[key]))

    from pfb import set_envs
    set_envs(opts.nthreads, ncpu)

    # import dask
    # dask.config.set(**{'array.slicing.split_large_chunks': False})
    # from pfb import set_client
    # client = set_client(opts.nworkers, log, client_log_level=opts.log_level)

    ti = time.time()
    _smoovie(**opts)

    log.info(f"All done after {time.time() - ti}s")

    try:
        from distributed import get_client
        client = get_client()
        client.close()
    except Exception as e:
        pass


def _smoovie(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import xarray as xr
    import numpy as np
    from pfb.utils.naming import xds_from_url
    import matplotlib.pyplot as plt
    from streamjoy import stream, wrap_matplotlib
    from daskms.fsspec_store import DaskMSStore

    basename = opts.output_filename
    if opts.scratch_dir is not None:
        scratch_dir = opts.scratch_dir
    else:
        scratch_dir = basename.rsplit('/', 1)[0]

    # xds contains vis products, no imaging weights applied
    fds_name = f'{basename}.fds' if opts.fds is None else opts.fds
    fds_store = DaskMSStore(fds_name)
    try:
        assert fds_store.exists()
    except Exception as e:
        log.error_and_raise(f"There must be a dataset at {fds_store.url}",
                            ValueError)

    log.info(f"Lazy loading fds from {fds_store.url}")
    fds, fds_list = xds_from_url(fds_store.url)

    # TODO - scan selection

    # get input times and frequencies
    freqs_in = []
    times_in = []
    for ds in fds:
        freqs_in.append(ds.FREQ.values)
        times_in.append(ds.TIME.values)

    freqs_in = np.unique(np.array(freqs_in))
    times_in = np.unique(np.array(times_in))

    @wrap_matplotlib()
    def plot_frame(ds):
        wsum = ds.wsum
        utc = ds.utc
        scan = ds.scanid
        fnum = ds.ffrac
        band = ds.bandid
        resid = ds.cube.values[0, 0, :, :]
        # nstokes = resid.shape[0]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        # for i, ax in enumerate(axs):
        rms = ds.rms[0, 0]
        im1 = ax.imshow(resid,
                    vmin=-opts.min_frac*rms,
                    vmax=opts.max_frac*rms,
                    cmap=opts.cmap)

        plt.xticks([]), plt.yticks([])
        ax.annotate(
            f'{basename}_band{band:04d}_scan{scan:04d}_{opts.product[0]}' + '\n' + fnum + '\n' + utc,
            xy=(0.0, 0.0),
            xytext=(0.05, 0.05),
            xycoords='axes fraction',
            textcoords='axes fraction',
            ha='left', va='bottom',
            fontsize=15,
            color=opts.text_colour)
        return fig

    outfmt = opts.out_format.lower()
    idfy = f'fps{opts.fps}'
    if opts.animate_axis == 'time':
        # bin freq axis and make movie for each bin
        fds_dict = {}
        for ds in fds:
            b = ds.bandid
            fds_dict.setdefault(b, [])
            fds_dict[b].append(ds)

        for b, dslist in fds_dict.items():
            
            log.info(f"Writing movie to {basename}_band{b}_{idfy}.{outfmt}")
            rmss = [ds.rms for ds in dslist]
            medrms = np.median(rmss)
            nframe = len(dslist)
            for i, ds in enumerate(dslist):
                ds.attrs['rms'] = medrms
                ds.attrs['ffrac'] = f'{i}/{nframe}'

            if outfmt == 'gif':
                outim = stream(
                        dslist,
                        renderer=plot_frame,
                        intro_title=f"{basename}-Band{b:04d}",
                        optimize=opts.optimize,
                        processes=True,
                        threads_per_worker=1,
                        fps=opts.fps,
                        max_frames=-1,
                        uri=f'{basename}_band{b}_{idfy}.gif',
                        scratch_dir=f'{scratch_dir}/streamjoy_scratch',
                        # client=client
                    )
            elif outfmt == 'mp4':
                outim = stream(
                        dslist,
                        renderer=plot_frame,
                        intro_title=f"{basename}-Band{b:04d}",
                        write_kwargs={'crf':opts.crf},
                        processes=True,
                        threads_per_worker=1,
                        fps=opts.fps,
                        max_frames=-1,
                        uri=f'{basename}_band{b}_{idfy}.mp4',
                        scratch_dir=f'{scratch_dir}/streamjoy_scratch',
                        # client=client
                    )
            else:
                log.error_and_raise(f"Unsupported format {opts.out_format}", ValueError)

    else:
        log.error_and_raise(f"Can't animate axis {opts.animate_axis}",
                            NotImplementedError)
