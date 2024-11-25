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
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        opts.nthreads = nthreads//2
        ncpu = ncpu//2

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
    from pfb.utils.fits import dds2fits
    from pfb.utils.naming import xds_from_url
    from distributed import get_client, as_completed
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from streamjoy import stream, wrap_matplotlib
    from distributed.diagnostics.progressbar import progress
    from daskms.fsspec_store import DaskMSStore

    try:
        client = get_client()
        names = list(client.scheduler_info()['workers'].keys())
    except:
        from pfb.utils.dist import fake_client
        client = fake_client()
        names = [0]
        as_completed = lambda x: x

    basename = opts.output_filename

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

    # filter freq
    if opts.freq_range is not None:
        fmin, fmax = opts.freq_range.split(':')
        for ds, dsl in zip(fds, fds_list):
            if ds.freq_out < fmin or ds.freq_out > fmax:
                    fds.pop(ds)
                    fds_list.pop(dsl)

    # filter time
    if opts.time_range is not None:
        tmin, tmax = opts.time_range.split(':')
        for ds, dsl in zip(fds, fds_list):
            if ds.time_out < tmin or ds.time_out > tmax:
                    fds.pop(ds)
                    fds_list.pop(dsl)

    # get output times and frequencies
    b0 = np.inf
    t0 = np.inf
    freqs_out = []
    times_out = []
    for ds in fds:
        freqs_out.append(ds.freq_out)
        times_out.append(ds.time_out)
        if ds.bandid < b0:
            b0 = ds.bandid
        if ds.timeid < t0:
            t0 = ds.timeid

    freqs_out = np.unique(np.array(freqs_out))
    times_out = np.unique(np.array(times_out))
    ntimes_out = times_out.size
    nfreqs_out = freqs_out.size

    # get scan numbers
    scan_bins = {}
    for ds in fds:
        scan_bins.setdefault(ds.scanid, 0)
        scan_bins[ds.scanid] += 1

    # adjust tid and bid after selection
    # this should make tid and bid unique
    scan_numbers = list(scan_bins.keys())
    bin_counts = np.array(list(scan_bins.values()))
    bin_cumcounts = np.cumsum(np.concatenate(([0],bin_counts)))
    for i, ds in enumerate(fds):
        tid = ds.timeid
        bid = ds.bandid
        # index of scan number
        sid = scan_numbers.index(ds.scanid)
        fds[i] = ds.assign_attrs(**{
            'timeid': tid + bin_cumcounts[sid] - t0,
            'bandid': bid + bin_cumcounts[sid] - b0
        })

    # Time and freq selection
    print(f"Time and freq selection takes ({nfreqs_in},{ntimes_in}) cube to ({nfreqs_out},{ntimes_out})", file=log)
    fbin = nfreqs_out if opts.freq_bin in (-1, None, 0) else opts.freq_bin
    nf = int(np.ceil(nfreqs_out/fbin))

    # time and freq binning
    if opts.respect_scan_boundaries:  # and len(scan_numbers) > 1:
        max_bin = np.max(bin_counts)
        if opts.time_bin in (-1, None, 0) or opts.time_bin > max_bin:
            # one bin per scan
            tbins = bin_counts
            nt = bin_counts.size
        else:
            nts = list(map(lambda x: int(np.ceil(x/opts.time_bin)), bin_counts))
            nt = np.sum(nts)
            tbins = np.zeros(nt)
            nlow = 0
            ntot = np.sum(bin_counts)
            for t in range(nt):
                nhigh = np.minimum(nlow + opts.time_bin, ntot)
                # check if scan would be straddled
                idxl = np.where(nlow < bin_cumcounts)[0][0]
                idxh = np.where(nhigh <= bin_cumcounts)[0][0]
                if idxl == idxh and t < nt-1:
                    # not straddled
                    tbins[t] = opts.time_bin
                    nlow += opts.time_bin
                else:
                    # straddled
                    tbins[t] = bin_cumcounts[idxl] - nlow
                    nlow = bin_cumcounts[idxl]

    else:
        max_bin = ntimes_out if opts.time_bin in (-1, None, 0) else opts.time_bin
        if opts.time_bin > ntimes_out:
            max_bin = ntimes_out
        nt = int(np.ceil(ntimes_out/max_bin))
        tbins = np.array([max_bin]*nt)
        if ntimes_out % max_bin:
            tbins[-1] = ntimes_out - (nt-1)*max_bin
    tidx = np.cumsum(np.concatenate(([0], tbins))).astype(int)
    tbins = tbins.astype(int)
    print(f"Time and freq binning takes ({nfreqs_out},{ntimes_out}) cube to ({nf},{nt})", file=log)


    @wrap_matplotlib()
    def plot_frame(frame):
        # frame is a list containing
        # 0 - out image
        # 1 - median rms
        # 2 - utc
        # 3 - scan number
        # 4 - frame fraction
        # 5 - band id
        im = frame[0]
        rms = frame[1]
        utc = frame[2]
        scan = frame[3]
        fnum = frame[4]
        band = frame[5]
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        im1 = ax.imshow(im,
                    vmin=-opts.min_frac*rms,
                    vmax=opts.max_frac*rms,
                    cmap=opts.cmap)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im1, cax=cax, orientation='vertical')

        plt.xticks([]), plt.yticks([])
        ax.annotate(
            f'{basename}_band{band:04d}_scan{scan:04d}' + '\n' + fnum + '\n' + utc,
            xy=(0.0, 0.0),
            xytext=(0.05, 0.05),
            xycoords='axes fraction',
            textcoords='axes fraction',
            ha='left', va='bottom',
            fontsize=20,
            color=opts.text_colour)
        return fig


    if opts.animate_axis == 'time':
        # bin freq axis and make ovie for each bin
        fbins = np.arange(fbin/2, nfreqs_out + fbin/2, fbin)
        fdso = {}
        for ds in fds:
            tmp = np.abs(fbins - ds.bandid)
            b = np.where(tmp == tmp.min())[0][-1]
            fdso.setdefault(b, [])
            fdso[b].append(ds)

        for b, dlist in fdso.items():
            futures = []
            for t in range(nt):
                nlow = tidx[t]
                nhigh = nlow + tbins[t]
                fut = client.submit(sum_blocks, dlist[nlow:nhigh],
                                    priority=-t)
                futures.append(fut)

            # this should preserve order
            progress(futures)
            results = client.gather(futures)

            # drop empty frames
            nframes = len(results)
            for i, res in enumerate(results):
                if not res[1]:
                    results.pop(i)
            nframeso = len(results)
            print(f"Dropped {nframes - nframeso} empty frames in band {b}", file=log)
            # get median rms
            medrms = np.median([res[1] for res in results])

            # replace rms with medrms and add metadata
            for i, res in enumerate(results):
                res[1] = medrms
                res.append(f'{str(i)}/{nt}')  # frame fraction
                res.append(b)  # band number

            # results should have
            # 0 - out image
            # 1 - median rms
            # 2 - utc
            # 3 - scan number
            # 4 - frame fraction
            # 5 - band id

            # TODO:
            # - progressbar
            # - investigate writing frames to disk as xarray dataset and passing instead of frames
            idfy = f'fps{opts.fps}_tbin{opts.time_bin}_fbin{opts.freq_bin}'
            if opts.out_format.lower() == 'gif':
                outim = stream(
                        results,
                        renderer=plot_frame,
                        intro_title=f"{basename}-Band{b:04d}",
                        optimize=opts.optimize,
                        threads_per_worker=1,
                        fps=opts.fps,
                        max_frames=-1,
                        uri=f'{basename}_band{b}_{idfy}.gif'
                    )
            elif opts.out_format.lower() == 'mp4':
                outim = stream(
                        results,
                        renderer=plot_frame,
                        intro_title=f"{basename}-Band{b:04d}",
                        write_kwargs={'crf':opts.crf},
                        threads_per_worker=1,
                        fps=opts.fps,
                        max_frames=-1,
                        uri=f'{basename}_band{b}_{idfy}.mp4'
                    )
            else:
                raise ValueError(f"Unsupported format {opts.out_format}")


            # outim.fps = opts.fps
            # outim.write(f'{basename}_band{b}_fps{opts.fps}_tbin'
            #             f'{opts.time_bin}_fbin{opts.freq_bin}.gif')


    elif opts.animate_axis == 'freq':
        tbins = np.arange(0, ntimes_out, opts.time_bin)
        tbins = np.append(tbins, ntimes_out)
        fdso = {}
        for ds in fds:
            tmp = np.abs(tbins - ds.timeid)
            t = np.where(tmp == tmp.min())[0][0]
            fdso.setdefault(t, [])
            fdso[t].append(ds)
    else:
        raise ValueError(f"Can't animate axis {opts.animate_axis}")


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
