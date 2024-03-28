import os
from pathlib import Path
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('SMOOVIE')
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.smoovie["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.smoovie["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.smoovie)
def smoovie(**kw):
    '''
    Produce image data products
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{ldir}/smoovie_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/smoovie_{timestamp}.log', file=log)
    from daskms.fsspec_store import DaskMSStore
    fdsstore = DaskMSStore(opts.fds.rstrip('/'))
    try:
        assert fdsstore.exists()
    except:
        raise ValueError(f"No fds at {opts.fds}")
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    with ExitStack() as stack:
        os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nthreads)
        os.environ["MKL_NUM_THREADS"] = str(opts.nthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nthreads)
        os.environ["NUMBA_NUM_THREADS"] = str(opts.nthreads)
        import numexpr as ne
        max_cores = ne.detect_number_of_cores()
        ne_threads = min(max_cores, opts.nthreads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)
        import dask
        dask.config.set(**{'array.slicing.split_large_chunks': False})

        # set up client
        host_address = opts.host_address or os.environ.get("DASK_SCHEDULER_ADDRESS")
        if host_address is not None:
            from distributed import Client
            print("Initialising distributed client.", file=log)
            client = stack.enter_context(Client(host_address))
        else:
            from dask.distributed import Client, LocalCluster
            print("Initialising client with LocalCluster.", file=log)
            with dask.config.set({"distributed.scheduler.worker-saturation":  1.1}):
                cluster = LocalCluster(processes=True,
                                       n_workers=opts.nworkers,
                                       threads_per_worker=opts.nthreads,
                                       memory_limit=0,  # str(mem_limit/nworkers)+'GB'
                                       asynchronous=False)
                cluster = stack.enter_context(cluster)
                client = stack.enter_context(Client(cluster))

        client.wait_for_workers(opts.nworkers)

        import xarray as xr
        import numpy as np
        from pfb.utils.fits import dds2fits, dds2fits_mfs
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        from streamjoy import stream, wrap_matplotlib

        @wrap_matplotlib()
        def plot_frame(frame):
            im = frame[0]
            rms = frame[1]
            utc = frame[2]
            fnum = frame[3]
            band = frame[4]
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
            ax.imshow(im,
                      vmin=-opts.min_frac*rms,
                      vmax=opts.max_frac*rms,
                      cmap=opts.cmap)
            plt.xticks([]), plt.yticks([])
            ax.annotate(
                f'{opts.outname}_band{b:04d}' + '\n' + fnum + '\n' + utc,
                xy=(0.0, 0.0),
                xytext=(0.05, 0.05),
                xycoords='axes fraction',
                textcoords='axes fraction',
                ha='left', va='bottom',
                fontsize=20,
                color=opts.text_colour)
            return fig

        # returns sorted list
        fdslist = fdsstore.fs.glob(f'{opts.fds}/*')

        # lazy load fds
        fds = list(map(xr.open_zarr, fdslist))

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
            for ds in fds:
                if ds.freq_out < fmin or ds.freq_out > fmax:
                     fds.pop(ds)

        # filter time
        if opts.time_range is not None:
            tmin, tmax = opts.time_range.split(':')
            for ds in fds:
                if ds.time_out < tmin or ds.time_out > tmax:
                     fds.pop(ds)

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

        # Time and freq selection
        print(f"Time and freq selection takes ({nfreqs_in},{ntimes_in}) cube to ({nfreqs_out},{ntimes_out})", file=log)
        nf = int(np.ceil(nfreqs_out/opts.freq_bin))
        nt = int(np.ceil(ntimes_out/opts.time_bin))
        print(f"Time and freq binning takes ({nfreqs_out},{ntimes_out}) cube to ({nf},{nt})", file=log)

        # reset band and time id after selection
        for i, ds in enumerate(fds):
            tid = ds.timeid
            bid = ds.bandid
            fds[i] = ds.assign_attrs(**{
                'timeid': tid - t0,
                'bandid': bid - b0
            })


        # bin freq or time
        if opts.animate_axis == 'time':
            fbins = np.arange(0, nfreqs_out, opts.freq_bin)
            fbins = np.append(fbins, nfreqs_out)
            fdso = {}
            for ds in fds:
                tmp = np.abs(fbins - ds.bandid)
                b = np.where(tmp == tmp.min())[0][0]
                fdso.setdefault(b, [])
                fdso[b].append(ds)


            for b, dlist in fdso.items():
                futures = []
                for t in range(nt):
                    nlow = t*opts.time_bin
                    nhigh = np.minimum(nlow + opts.time_bin, ntimes_out)
                    fut = client.submit(sum_blocks, dlist[nlow:nhigh])
                    futures.append(fut)

                # this should preserve order
                results = client.gather(futures)
                medrms = np.median([res[1] for res in results])

                # replace rms with medrms and add metadata
                for i, res in enumerate(results):
                    res[1] = medrms
                    res.append(f'{str(i)}/{nt}')  # frame fraction
                    res.append(b)  # band number

                # TODO - submit streams in parallel
                outim = stream(
                    results,
                    renderer=plot_frame,
                    intro_title=f"{opts.outname}-Band{b:04d}",
                    optimize=True,
                    threads_per_worker=1
                    # uri=f'{opts.outname}.band{b:0:4d}.mp4'
                )

                outim.fps = opts.fps
                outim.write(f'{opts.outname}.band{b:04d}.gif')


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




        # # convert to fits files
        # if opts.fits_mfs or opts.fits_cubes:
        #     print("Writing fits", file=log)
        #     tj = time.time()

        # fitsout = []
        # if opts.fits_mfs:
        #     fitsout.append(dds2fits_mfs(fds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))

        # if opts.fits_cubes:
        #     fitsout.append(dds2fits(fds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))

        # if len(fitsout):
        #     with compute_context(opts.scheduler, f'{ldir}/fastim_fits_{timestamp}'):
        #         dask.compute(fitsout)
        #     print(f"Fits writing took {time.time() - tj}s", file=log)

        # if opts.movie_mfs or opts.movie_cubes:
        #     print("Filming", file=log)
        #     fontPath = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
        #     sans30  =  ImageFont.truetype ( fontPath, 30 )
        #     tj = time.time()

        # if opts.movie_mfs:
        #     frames = []
        #     for t in range(len(times_out)):
        #         res = np.zeros(fds[0].RESIDUAL.shape)
        #         rmss = np.zeros(len(times_out))
        #         wsum = 0.0
        #         for ds in fds:
        #             # bands share same time axis so accumulate
        #             if ds.timeid != t:
        #                 continue
        #             else:
        #                 res += ds.RESIDUAL.values
        #                 wsum += ds.WSUM.values
        #                 utc = ds.utc

        #         # min to zero
        #         res -= res.min()
        #         # max to 255
        #         res *= 255/res.max()
        #         res = res.astype('uint8')
        #         nn = Image.fromarray(res)
        #         prog = str(t).zfill(len(str(nframes)))+' / '+str(nframes)
        #         draw = ImageDraw.Draw(nn)
        #         draw.text((0.03*nx,0.90*ny),'Frame : '+prog,fill=('white'),font=sans30)
        #         draw.text((0.03*nx,0.93*ny),'Time  : '+utc,fill=('white'),font=sans30)
        #         # draw.text((0.03*nx,0.96*ny),'Image : '+ff,fill=('white'),font=sans30)
        #         frames.append(nn)
        #     frames[0].save(f'{basename}_{opts.postfix}_animated_mfs.gif',
        #                    save_all=True,
        #                    append_images=frames[1:],
        #                    duration=35,
        #                    loop=1)

        # if opts.movie_cubes:
        #     for b in range(len(freqs_out)):
        #         frames = []
        #         for ds in fds:
        #             if ds.bandid != b:
        #                 continue
        #             res = ds.RESIDUAL.values
        #             # min to zero
        #             res -= res.min()
        #             # max to 255
        #             res *= 255/res.max()
        #             res = res.astype('uint8')
        #             nn = Image.fromarray(res)
        #             tt = ds.utc
        #             t = ds.timeid
        #             prog = str(t).zfill(len(str(nframes)))+' / '+str(nframes)
        #             draw = ImageDraw.Draw(nn)
        #             draw = ImageDraw.Draw(nn)
        #             draw.text((0.03*nx,0.90*ny),'Frame : '+prog,fill=('white'),font=sans30)
        #             draw.text((0.03*nx,0.93*ny),'Time  : '+utc,fill=('white'),font=sans30)
        #             frames.append(nn)
        #         frames[0].save(f'{basename}_{opts.postfix}_animated_band{b:04d}.gif',
        #                        save_all=True,
        #                        append_images=frames[1:],
        #                        duration=35,
        #                        loop=1)


        # if opts.movie_mfs or opts.movie_cubes:
        #     print(f"Filming took {time.time() - tj}s", file=log)


        client.close()

        print("All done here.", file=log)


import dask
import numpy as np
from casacore.quanta import quantity
from datetime import datetime
# from pfb.utils.fits import save_fits
def sum_blocks(fds, animate='time'):
    fds = dask.compute(fds)[0]
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

    return [outim, rms, utc]
