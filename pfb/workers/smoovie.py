import os
from pathlib import Path
from contextlib import ExitStack
from pfb.workers.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('SMOOVIE')
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
    if opts.product.upper() not in ["I","Q", "U", "V"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/smoovie_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from pfb import set_client
    client = set_client(opts.nworkers, log, client_log_level=opts.log_level)

    ti = time.time()
    _smoovie(**opts)

    print(f"All done after {time.time() - ti}s", file=log)

    try:
        client.close()
    except Exception as e:
        raise e


def _smoovie(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import xarray as xr
    import numpy as np
    from pfb.utils.naming import xds_from_url
    from distributed import get_client
    import matplotlib.pyplot as plt
    from streamjoy import stream, wrap_matplotlib
    from daskms.fsspec_store import DaskMSStore

    try:
        client = get_client()
    except:
        client = None

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
        raise ValueError(f"There must be a dataset at {fds_store.url}")

    print(f"Lazy loading fds from {fds_store.url}", file=log)
    fds, fds_list = xds_from_url(fds_store.url)

    # TODO - scan selection

    # get input times and frequencies
    freqs_in = []
    times_in = []
    for ds in fds:
        freqs_in.append(ds.freq_out)
        times_in.append(ds.time_out)

    freqs_in = np.unique(np.array(freqs_in))
    times_in = np.unique(np.array(times_in))

    ntimes_in = times_in.size
    nfreqs_in = freqs_in.size

    @wrap_matplotlib()
    def plot_frame(ds):
        # with worker_client() as client:
        #     ds = client.compute(frame, sync=True)
        # frame is a list containing
        # 0 - out image
        # 1 - median rms
        # 2 - utc
        # 3 - scan number
        # 4 - frame fraction
        # 5 - band id
        wsum = ds.wsum
        utc = ds.utc
        scan = ds.scanid
        fnum = ds.ffrac
        band = ds.bandid
        resid = ds.RESIDUAL.values/wsum
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        im1 = ax.imshow(resid,
                    vmin=-opts.min_frac*ds.rms,
                    vmax=opts.max_frac*ds.rms,
                    cmap=opts.cmap)

        plt.xticks([]), plt.yticks([])
        ax.annotate(
            f'{basename}_band{band:04d}_scan{scan:04d}' + '\n' + fnum + '\n' + utc,
            xy=(0.0, 0.0),
            xytext=(0.05, 0.05),
            xycoords='axes fraction',
            textcoords='axes fraction',
            ha='left', va='bottom',
            fontsize=15,
            color=opts.text_colour)
        return fig


    if opts.animate_axis == 'time':
        # bin freq axis and make movie for each bin
        fds_dict = {}
        for ds in fds:
            b = ds.bandid
            fds_dict.setdefault(b, [])
            fds_dict[b].append(ds)

        for b, dslist in fds_dict.items():
            rmss = [ds.rms for ds in dslist]
            medrms = np.median(rmss)
            nframe = len(dslist)
            for i, ds in enumerate(dslist):
                ds.attrs['rms'] = medrms
                ds.attrs['ffrac'] = f'{i}/{nframe}'

            idfy = f'fps{opts.fps}'
            if opts.out_format.lower() == 'gif':
                outim = stream(
                        dslist,
                        renderer=plot_frame,
                        intro_title=f"{basename}-Band{b:04d}",
                        optimize=opts.optimize,
                        threads_per_worker=1,
                        fps=opts.fps,
                        max_frames=-1,
                        uri=f'{basename}_band{b}_{idfy}.gif',
                        scratch_dir=f'{scratch_dir}/streamjoy_scratch',
                        client=client
                    )
            elif opts.out_format.lower() == 'mp4':
                outim = stream(
                        dslist,
                        renderer=plot_frame,
                        intro_title=f"{basename}-Band{b:04d}",
                        write_kwargs={'crf':opts.crf},
                        threads_per_worker=1,
                        fps=opts.fps,
                        max_frames=-1,
                        uri=f'{basename}_band{b}_{idfy}.mp4',
                        scratch_dir=f'{scratch_dir}/streamjoy_scratch',
                        client=client
                    )
            else:
                raise ValueError(f"Unsupported format {opts.out_format}")


            # outim.fps = opts.fps
            # outim.write(f'{basename}_band{b}.gif')

    else:
        raise NotImplementedError(f"Can't animate axis {opts.animate_axis}")


import dask
import numpy as np
from casacore.quanta import quantity
from datetime import datetime
from distributed import worker_client
# from pfb.utils.fits import save_fits
def sum_blocks(fds, animate='time'):
    with worker_client() as client:
        fds = client.compute(fds, sync=True)
    # fds = dask.compute(fds)[0]
    outim = np.zeros((fds[0].x.size, fds[0].y.size))
    wsum = 0.0
    cout = 0.0
    for ds in fds:
        outim += ds.RESIDUAL.values * ds.wsum
        cout += getattr(ds, f'{animate}_out') * ds.wsum
        wsum += ds.wsum

    if wsum:
        outim /= wsum
        cout /= wsum
    if animate=='time':
        unix_time = quantity(f'{cout}s').to_unix_time()
        utc = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
    rms = np.std(outim)

    return [outim, rms, utc, fds[0].scanid]
