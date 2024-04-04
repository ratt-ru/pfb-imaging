# flake8: noqa
import os
import sys
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
    pyscilog.log_to_file(f'{ldir}/fastim_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/fastim_{timestamp}.log', file=log)
    if opts.product.upper() not in ["I","Q", "U", "V"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")
    from daskms.fsspec_store import DaskMSStore
    msstore = DaskMSStore(opts.ms.rstrip('/'))
    mslist = msstore.fs.glob(opts.ms.rstrip('/'))
    try:
        assert len(mslist) == 1
    except:
        raise ValueError(f"There must be a single MS corresponding to {opts.ms}")
    opts.ms = mslist[0]
    if opts.gain_table is not None:
        gainstore = DaskMSStore(opts.gain_table.rstrip('/'))
        gtlist = gainstore.fs.glob(opts.gain_table.rstrip('/'))
        try:
            assert len(gtlist) == 1
        except Exception as e:
            raise ValueError(f"There must be a single gain table corresponding to {opts.gain_table}")
        opts.gain_table = gtlist[0]
    if opts.transfer_model_from is not None:
        tmf = opts.transfer_model_from.rstrip('/')
        modelstore = DaskMSStore(tmf)
        modellist = modelstore.fs.glob(tmf)
        try:
            assert len(modellist) == 1
        except Exception as e:
            raise ValueError(f"There must be a single model corresponding to {opts.gain_table}")
        opts.transfer_model_from = modellist[0]
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    with ExitStack() as stack:
        os.environ["OMP_NUM_THREADS"] = str(opts.nvthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nvthreads)
        os.environ["MKL_NUM_THREADS"] = str(opts.nvthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nvthreads)
        paths = sys.path
        ppath = [paths[i] for i in range(len(paths)) if 'pfb/bin' in paths[i]]
        if len(ppath):
            ldpath = ppath[0].replace('bin', 'lib')
            ldcurrent = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ["LD_LIBRARY_PATH"] = f'{ldpath}:{ldcurrent}'
            # TODO - should we fall over in else?
        os.environ["NUMBA_NUM_THREADS"] = str(opts.nvthreads)

        import numexpr as ne
        max_cores = ne.detect_number_of_cores()
        ne_threads = min(max_cores, opts.nvthreads)
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
            cluster = LocalCluster(processes=True,
                                    n_workers=opts.nworkers,
                                    threads_per_worker=opts.nthreads_dask,
                                    memory_limit=0,  # str(mem_limit/nworkers)+'GB'
                                    asynchronous=False)
            cluster = stack.enter_context(cluster)
            client = stack.enter_context(Client(cluster, direct_to_workers=False))

        client.wait_for_workers(opts.nworkers)
        client.amm.stop()

        ti = time.time()
        _fastim(**opts)
        client.close()

        print("All done here.", file=log)

def _fastim(**kw):
    opts = OmegaConf.create(kw)
    from omegaconf import ListConfig
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from dask.graph_manipulation import clone
    from distributed import get_client, wait, as_completed
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.fsspec_store import DaskMSStore
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.stokes2im import single_stokes_image
    import xarray as xr

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    fdsstore = DaskMSStore(f'{basename}.fds')
    if fdsstore.exists():
        if opts.overwrite:
            print(f"Overwriting {basename}.fds", file=log)
            fdsstore.rm(recursive=True)
        else:
            raise ValueError(f"{basename}.fds exists. "
                             "Set overwrite to overwrite it. ")

    if opts.gain_table is not None:
        gain_name = "::".join(opts.gain_table.rstrip('/').rsplit("/", 1))
    else:
        gain_name = None

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


    # generate some futures to initialise as_completed
    # Are these round robin'd?
    client = get_client()
    # futures = client.map(lambda x: x, np.arange(opts.nworkers*opts.nthreads_dask*2))

    print('Constructing mapping', file=log)
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gain_chunks, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               gain_name,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_image,
                               freq_min=freq_min,
                               freq_max=freq_max)

    max_freq = 0
    ms = opts.ms
    # for ms in opts.ms:
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

    # crude column arithmetic
    dc = opts.data_column.replace(" ", "")
    if "+" in dc:
        dc1, dc2 = dc.split("+")
        operator="+"
    elif "-" in dc:
        dc1, dc2 = dc.split("-")
        operator="+"
    else:
        dc1 = dc
        dc2 = None
        operator=None

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
        mdsstore = DaskMSStore(opts.transfer_model_from)
        try:
            mdsstore.exists()
            # mds = mdsstore.url
            mds = xr.open_zarr(mdsstore.url)
            # this should be fairly small but should
            # it rather be read in the dask call?
            # mds = client.persist(mds)
            foo = client.scatter(mds, broadcast=True)
            wait(foo)
        except Exception as e:
            import ipdb; ipdb.set_trace()
            raise ValueError(f"No dataset found at {opts.transfer_model_from}")
    else:
        mds = None

    # ascomp = as_completed(futures)
    futures= []
    xds = xds_from_ms(ms,
                      chunks=ms_chunks[ms],
                      columns=columns,
                      table_schema=schema,
                      group_cols=group_by)

    if opts.gain_table is not None:
        gds = xds_from_zarr(gain_name,
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

        titr = enumerate(zip(time_mapping[ms][idt]['start_indices'],
                            time_mapping[ms][idt]['counts']))
        for ti, (tlow, tcounts) in titr:

            It = slice(tlow, tlow + tcounts)
            ridx = row_mapping[ms][idt]['start_indices'][It]
            rcnts = row_mapping[ms][idt]['counts'][It]
            # select all rows for output dataset
            Irow = slice(ridx[0], ridx[-1] + rcnts[-1])

            fitr = enumerate(zip(freq_mapping[ms][idt]['start_indices'],
                                freq_mapping[ms][idt]['counts']))
            for fi, (flow, fcounts) in fitr:
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

                data2 = None if dc2 is None else getattr(subds, dc2).data
                sc = opts.sigma_column
                sigma = None if sc is None else getattr(subds, sc).data
                wc = opts.weight_column
                weight = None if wc is None else getattr(subds, wc).data
                # # poll until a worker has capacity
                # while not ascomp.has_ready():
                #     pass
                # # get and pop ready future
                # fut = ascomp.next()
                # # get worker that had it
                # wid = list(client.who_has(fut).values())[0][0]

                future = client.submit(single_stokes_image,
                        data=getattr(subds, dc1).data,
                        data2=data2,
                        operator=operator,
                        ant1=clone(subds.ANTENNA1.data),
                        ant2=clone(subds.ANTENNA2.data),
                        uvw=clone(subds.UVW.data),
                        frow=clone(subds.FLAG_ROW.data),
                        flag=subds.FLAG.data,
                        sigma=sigma,
                        weight=weight,
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
                        poltype=poltype[ms],
                        fieldid=subds.FIELD_ID,
                        ddid=subds.DATA_DESC_ID,
                        scanid=subds.SCAN_NUMBER,
                        fds_store=fdsstore.url,
                        bandid=fi,
                        timeid=ti)
                        # workers=wid)  # submit to the same worker

                # add current future to ascomp
                # ascomp.add(future)
                futures.append(future)
                if len(futures) == opts.nworkers*opts.nthreads_dask:
                    wait(futures)
                    futures = []

    # while not ascomp.is_empty():
    #     # pop them as they finish
    #     if ascomp.has_ready():
    #         fut = ascomp.next()
    #         print(fut)

    # import ipdb; ipdb.set_trace()
    wait(futures)
    return
