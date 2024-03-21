# flake8: noqa
import os
from pathlib import Path
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FASTIM')
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.fastim["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.fastim["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.fastim)
def fastim(**kw):
    '''
    Produce image data products
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{ldir}/init_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/init_{timestamp}.log', file=log)
    from daskms.fsspec_store import DaskMSStore
    msnames = []
    for ms in opts.ms:
        msstore = DaskMSStore(ms.rstrip('/'))
        mslist = msstore.fs.glob(ms.rstrip('/'))
        try:
            assert len(mslist) > 0
            msnames.append(*list(map(msstore.fs.unstrip_protocol, mslist)))
        except:
            raise ValueError(f"No MS at {ms}")
    opts.ms = msnames
    if opts.gain_table is not None:
        gainnames = []
        for gt in opts.gain_table:
            gainstore = DaskMSStore(gt.rstrip('/'))
            gtlist = gainstore.fs.glob(gt.rstrip('/'))
            try:
                assert len(gtlist) > 0
                gainnames.append(*list(map(gainstore.fs.unstrip_protocol, gtlist)))
            except Exception as e:
                raise ValueError(f"No gain table  at {gt}")
        opts.gain_table = gainnames
    if opts.product.upper() not in ["I","Q", "U", "V"]:
            # "XX", "YX", "XY", "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")
    if opts.transfer_model_from is not None:
        modelstore = DaskMSStore(opts.transfer_model_from.rstrip('/'))
        opts.transfer_model_from = modelstore.url
    OmegaConf.set_struct(opts, True)
    basename = f'{opts.output_filename}_{opts.product.upper()}'

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)
        import dask
        from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
        import numpy as np  # has to be after set client
        from pfb.utils.misc import compute_context
        from pfb.utils.fits import dds2fits, dds2fits_mfs
        from PIL import Image, ImageDraw, ImageFont

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)


        if opts.make_fds:
            print("Constructing graph", file=log)
            ti = time.time()
            out_datasets = _fastim(**opts)

            # assign time and band ids
            freqs_out = []
            times_out = []
            for ds in out_datasets:
                freqs_out.append(ds.freq_out)
                times_out.append(ds.time_out)

            freqs_out = np.unique(np.array(freqs_out))
            times_out = np.unique(np.array(times_out))

            for i, ds in enumerate(out_datasets):
                bandid = np.where(freqs_out == ds.freq_out)[0][0]
                timeid = np.where(times_out == ds.time_out)[0][0]
                ds = ds.assign_attrs(**{
                    'bandid': bandid,
                    'timeid': timeid
                })
                out_datasets[i] = ds

            print(f"Graph construction took {time.time() - ti}s", file=log)
            print("Compute starting", file=log)
            ti = time.time()
            if len(out_datasets):
                writes = xds_to_zarr(out_datasets,
                                    f'{basename}_{opts.postfix}.fds',
                                    columns='ALL',
                                    rechunk=True)
            else:
                raise ValueError('No datasets found to write. '
                                'Data completely flagged maybe?')

            # dask.visualize(writes, color="order", cmap="autumn",
            #                node_attr={"penwidth": "4"},
            #                filename=basename + '_writes_I_ordered_graph.pdf',
            #                optimize_graph=False)
            # dask.visualize(writes, filename=basename +
            #                '_writes_I_graph.pdf', optimize_graph=False)
            # import ipdb; ipdb.set_trace()

            with compute_context(opts.scheduler, f'{ldir}/fastim_{timestamp}'):
                dask.compute(writes, optimize_graph=False)


            print(f"Compute took {time.time() - ti}s", file=log)

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

def _fastim(**kw):
    opts = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if (not isinstance(opts.ms, list) and not
        isinstance(opts.ms, ListConfig)):
        opts.ms = [opts.ms]
    if opts.gain_table is not None:
        if (not isinstance(opts.gain_table, list) and not
            isinstance(opts.gain_table,ListConfig)):
            opts.gain_table = [opts.gain_table]
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
    from daskms.fsspec_store import DaskMSStore
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.stokes2im import single_stokes_image
    import xarray as xr

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    fdsstore = DaskMSStore(f'{basename}_{opts.postfix}.fds')
    if fdsstore.exists():
        if opts.overwrite:
            print(f"Overwriting {basename}_{opts.postfix}.fds", file=log)
            fdsstore.rm(recursive=True)
        else:
            raise ValueError(f"{basename}_{opts.postfix}.fds exists. "
                             "Set overwrite to overwrite it. ")

    if opts.gain_table is not None:
        tmpf = lambda x: x.rstrip('/') + f'::{opts.gain_term}'
        gain_names = list(map(tmpf, opts.gain_table))
    else:
        gain_names = None

    if opts.freq_range is not None:
        fmin, fmax = opts.freq_range.strip(' ').split(':')
        if len(fmin) > 0:
            freq_min = float(fmin)
        else:
            freq_min = -np.inf
        if len(fmax) > 0:
            freq_max = float(fmax)
        else:
            freq_max = np.inf
    else:
        freq_min = -np.inf
        freq_max = np.inf

    print('Constructing mapping', file=log)
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gain_chunks, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               gain_names,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_image,
                               freq_min=freq_min,
                               freq_max=freq_max)

    max_freq = 0
    for ms in opts.ms:
        for idt in freqs[ms].keys():
            freq = freqs[ms][idt]
            max_freq = np.maximum(max_freq, freq.max())

    # cell size
    cell_N = 1.0 / (2 * uv_max * max_freq / lightspeed)

    if opts.cell_size is not None:
        cell_size = opts.cell_size
        cell_rad = cell_size * np.pi / 60 / 60 / 180
        if cell_N / cell_rad < 1:
            raise ValueError("Requested cell size too large. "
                             "Super resolution factor = ", cell_N / cell_rad)
        print(f"Super resolution factor = {cell_N/cell_rad}", file=log)
    else:
        cell_rad = cell_N / opts.super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi
        print(f"Cell size set to {cell_size} arcseconds", file=log)

    if opts.nx is None:
        fov = opts.field_of_view * 3600
        npix = int(fov / cell_size)
        npix = good_size(npix)
        while npix % 2:
            npix += 1
            npix = good_size(npix)
        nx = npix
        ny = npix
    else:
        nx = opts.nx
        ny = opts.ny if opts.ny is not None else nx
        cell_deg = np.rad2deg(cell_rad)
        fovx = nx*cell_deg
        fovy = ny*cell_deg
        print(f"Field of view is ({fovx:.3e},{fovy:.3e}) degrees")

    print(f"Image size = (nx={nx}, ny={ny})", file=log)

    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    # crude arithmetic
    dc = opts.data_column.replace(" ", "")
    if "+" in dc:
        dc1, dc2 = dc.split("+")
    elif "-" in dc:
        dc1, dc2 = dc.split("-")
    else:
        dc1 = dc
        dc2 = None

    # assumes measurement sets have the same columns
    columns = (dc1,
               opts.flag_column,
               'UVW', 'ANTENNA1',
               'ANTENNA2', 'TIME', 'INTERVAL', 'FLAG_ROW')
    schema = {}
    schema[opts.flag_column] = {'dims': ('chan', 'corr')}
    schema[dc1] = {'dims': ('chan', 'corr')}
    if dc2 is not None:
        columns += (dc2,)
        schema[dc2] = {'dims': ('chan', 'corr')}

    # only WEIGHT column gets special treatment
    # any other column must have channel axis
    if opts.sigma_column is not None:
        print(f"Initialising weights from {opts.sigma_column} column", file=log)
        columns += (opts.sigma_column,)
        schema[opts.sigma_column] = {'dims': ('chan', 'corr')}
    elif opts.weight_column is not None:
        print(f"Using weights from {opts.weight_column} column", file=log)
        columns += (opts.weight_column,)
        # hack for https://github.com/ratt-ru/dask-ms/issues/268
        if opts.weight_column != 'WEIGHT':
            schema[opts.weight_column] = {'dims': ('chan', 'corr')}
    else:
        print(f"No weights provided, using unity weights", file=log)

    if opts.transfer_model_from is not None:
        try:
            mds = xds_from_zarr(opts.transfer_model_from,
                                chunks={'params':-1, 'comps':-1})[0]
            # this should be fairly small but should
            # it rather be read in the dask call?
            mds = dask.persist(mds)[0]
        except Exception as e:
            import ipdb; ipdb.set_trace()
            raise ValueError(f"No dataset found at {opts.transfer_model_from}")
    else:
        mds = None

    out_datasets = []
    for ims, ms in enumerate(opts.ms):
        xds = xds_from_ms(ms, chunks=ms_chunks[ms], columns=columns,
                          table_schema=schema, group_cols=group_by)

        if opts.gain_table is not None:
            gds = xds_from_zarr(gain_names[ims],
                                chunks=gain_chunks[ms])

        for ids, ds in enumerate(xds):
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            # TODO - cleaner syntax
            if opts.fields is not None:
                if fid not in opts.fields:
                    continue
            if opts.ddids is not None:
                if ddid not in opts.ddids:
                    continue
            if opts.scans is not None:
                if scanid not in opts.scans:
                    continue


            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"
            nrow = ds.sizes['row']
            ncorr = ds.sizes['corr']

            idx = (freqs[ms][idt]>=freq_min) & (freqs[ms][idt]<=freq_max)
            if not idx.any():
                continue

            for ti, (tlow, tcounts) in enumerate(zip(time_mapping[ms][idt]['start_indices'],
                                           time_mapping[ms][idt]['counts'])):

                It = slice(tlow, tlow + tcounts)
                ridx = row_mapping[ms][idt]['start_indices'][It]
                rcnts = row_mapping[ms][idt]['counts'][It]
                # select all rows for output dataset
                Irow = slice(ridx[0], ridx[-1] + rcnts[-1])

                for flow, fcounts in zip(freq_mapping[ms][idt]['start_indices'],
                                         freq_mapping[ms][idt]['counts']):
                    Inu = slice(flow, flow + fcounts)

                    subds = ds[{'row': Irow, 'chan': Inu}]
                    subds = subds.chunk({'row':-1, 'chan': -1})
                    if opts.gain_table is not None:
                        # Only DI gains currently supported
                        subgds = gds[ids][{'gain_time': It, 'gain_freq': Inu}]
                        subgds = subgds.chunk({'gain_time': -1, 'gain_freq': -1})
                        jones = subgds.gains.data
                    else:
                        jones = None

                    if opts.product.upper() in ["I", "Q", "U", "V"]:
                        out_ds = single_stokes_image(
                            ds=subds,
                            mds=mds,
                            jones=jones,
                            opts=opts,
                            nx=nx,
                            ny=ny,
                            freq=freqs[ms][idt][Inu],
                            chan_width=chan_widths[ms][idt][Inu],
                            utime=utimes[ms][idt][It],
                            tbin_idx=ridx,
                            tbin_counts=rcnts,
                            cell_rad=cell_rad,
                            radec=radecs[ms][idt],
                            antpos=antpos[ms],
                            poltype=poltype[ms])
                    # elif opts.product.upper() in ["XX", "YX", "XY", "YY",
                    #                               "RR", "RL", "LR", "LL"]:
                    #     out_ds = single_corr(
                    #         ds=subds,
                    #         jones=jones,
                    #         opts=opts,
                    #         freq=freqs[ms][idt][Inu],
                    #         chan_width=chan_widths[ms][idt][Inu],
                    #         utimes=utimes[ms][idt][It],
                    #         tbin_idx=ridx,
                    #         tbin_counts=rcnts,
                    #         cell_rad=cell_rad,
                    #         radec=radecs[ms][idt])
                    else:
                        raise NotImplementedError(f"Product {args.product} not "
                                                "supported yet")
                    # if all data in a dataset is flagged we return None and
                    # ignore this chunk of data
                    if out_ds is not None:
                        out_datasets.append(out_ds)

    return out_datasets
