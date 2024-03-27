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
    fdslist = fdsstore.fs.glob(opts.fds.rstrip('/'))
    try:
        assert len(fdslist) == 1
    except:
        raise ValueError(f"No fds at {opts.fds}")
    opts.fds = fdslist[0]
    OmegaConf.set_struct(opts, True)
    basename = f'{opts.output_filename}'

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)
        import dask
        import numpy as np  # has to be after set client
        from pfb.utils.misc import compute_context
        from pfb.utils.fits import dds2fits, dds2fits_mfs
        from PIL import Image, ImageDraw, ImageFont

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)


        fds = xds_from_zarr(f'{basename}_{opts.postfix}.fds',
                            chunks={'x': -1, 'y': -1})
        # TODO!!
        fds = dask.persist(fds)[0]


        # need to redo if not making the fds
        freqs_out = []
        times_out = []
        for ds in fds:
            freqs_out.append(ds.freq_out)
            times_out.append(ds.time_out)

        freqs_out = np.unique(np.array(freqs_out))
        times_out = np.unique(np.array(times_out))
        nframes = times_out.size
        nx, ny = fds[0].RESIDUAL.shape

        # convert to fits files
        if opts.fits_mfs or opts.fits_cubes:
            print("Writing fits", file=log)
            tj = time.time()

        fitsout = []
        if opts.fits_mfs:
            fitsout.append(dds2fits_mfs(fds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))

        if opts.fits_cubes:
            fitsout.append(dds2fits(fds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))

        if len(fitsout):
            with compute_context(opts.scheduler, f'{ldir}/fastim_fits_{timestamp}'):
                dask.compute(fitsout)
            print(f"Fits writing took {time.time() - tj}s", file=log)

        if opts.movie_mfs or opts.movie_cubes:
            print("Filming", file=log)
            fontPath = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
            sans30  =  ImageFont.truetype ( fontPath, 30 )
            tj = time.time()

        if opts.movie_mfs:
            frames = []
            for t in range(len(times_out)):
                res = np.zeros(fds[0].RESIDUAL.shape)
                rmss = np.zeros(len(times_out))
                wsum = 0.0
                for ds in fds:
                    # bands share same time axis so accumulate
                    if ds.timeid != t:
                        continue
                    else:
                        res += ds.RESIDUAL.values
                        wsum += ds.WSUM.values
                        utc = ds.utc

                # min to zero
                res -= res.min()
                # max to 255
                res *= 255/res.max()
                res = res.astype('uint8')
                nn = Image.fromarray(res)
                prog = str(t).zfill(len(str(nframes)))+' / '+str(nframes)
                draw = ImageDraw.Draw(nn)
                draw.text((0.03*nx,0.90*ny),'Frame : '+prog,fill=('white'),font=sans30)
                draw.text((0.03*nx,0.93*ny),'Time  : '+utc,fill=('white'),font=sans30)
                # draw.text((0.03*nx,0.96*ny),'Image : '+ff,fill=('white'),font=sans30)
                frames.append(nn)
            frames[0].save(f'{basename}_{opts.postfix}_animated_mfs.gif',
                           save_all=True,
                           append_images=frames[1:],
                           duration=35,
                           loop=1)

        if opts.movie_cubes:
            for b in range(len(freqs_out)):
                frames = []
                for ds in fds:
                    if ds.bandid != b:
                        continue
                    res = ds.RESIDUAL.values
                    # min to zero
                    res -= res.min()
                    # max to 255
                    res *= 255/res.max()
                    res = res.astype('uint8')
                    nn = Image.fromarray(res)
                    tt = ds.utc
                    t = ds.timeid
                    prog = str(t).zfill(len(str(nframes)))+' / '+str(nframes)
                    draw = ImageDraw.Draw(nn)
                    draw = ImageDraw.Draw(nn)
                    draw.text((0.03*nx,0.90*ny),'Frame : '+prog,fill=('white'),font=sans30)
                    draw.text((0.03*nx,0.93*ny),'Time  : '+utc,fill=('white'),font=sans30)
                    frames.append(nn)
                frames[0].save(f'{basename}_{opts.postfix}_animated_band{b:04d}.gif',
                               save_all=True,
                               append_images=frames[1:],
                               duration=35,
                               loop=1)


        if opts.movie_mfs or opts.movie_cubes:
            print(f"Filming took {time.time() - tj}s", file=log)


        if opts.scheduler=='distributed':
            from distributed import get_client
            client = get_client()
            client.close()

        print("All done here.", file=log)
