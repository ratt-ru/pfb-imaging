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
log = pyscilog.get_logger('HCI')
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.hci["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.hci["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.hci)
def hci(**kw):
    '''
    Produce high cadence residual images.
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{ldir}/hci_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/fastim_{timestamp}.log', file=log)
    if opts.product.upper() not in ["I","Q", "U", "V"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")
    from daskms.fsspec_store import DaskMSStore
    msstore = DaskMSStore(opts.ms.rstrip('/'))
    mslist = msstore.fs.glob(opts.ms.rstrip('/'))
    try:
        assert len(mslist) == 1
    except:
        raise ValueError(f"There must be a single MS corresponding "
                         f"to {opts.ms}")
    opts.ms = mslist[0]
    if opts.gain_table is not None:
        gainstore = DaskMSStore(opts.gain_table.rstrip('/'))
        gtlist = gainstore.fs.glob(opts.gain_table.rstrip('/'))
        try:
            assert len(gtlist) == 1
        except Exception as e:
            raise ValueError(f"There must be a single gain table "
                             f"corresponding to {opts.gain_table}")
        opts.gain_table = gtlist[0]
    if opts.transfer_model_from is not None:
        tmf = opts.transfer_model_from.rstrip('/')
        modelstore = DaskMSStore(tmf)
        try:
            assert modelstore.exists()
        except Exception as e:
            raise ValueError(f"There must be a single model corresponding "
                             f"to {opts.transfer_model_from}")
        opts.transfer_model_from = modelstore.url
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    with ExitStack() as stack:
        os.environ["OMP_NUM_THREADS"] = str(opts.nthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nthreads)
        os.environ["MKL_NUM_THREADS"] = str(opts.nthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nthreads)
        paths = sys.path
        ppath = [paths[i] for i in range(len(paths)) if 'pfb/bin' in paths[i]]
        if len(ppath):
            ldpath = ppath[0].replace('bin', 'lib')
            ldcurrent = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ["LD_LIBRARY_PATH"] = f'{ldpath}:{ldcurrent}'
            # TODO - should we fall over in else?
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
            cluster = LocalCluster(processes=True,
                                    n_workers=opts.nworkers,
                                    threads_per_worker=1,
                                    memory_limit=0,
                                    asynchronous=False)
            cluster = stack.enter_context(cluster)
            client = stack.enter_context(Client(cluster,
                                                direct_to_workers=False))

        client.wait_for_workers(opts.nworkers)
        client.amm.stop()

        ti = time.time()
        _hci(**opts)

    print(f"All done after {time.time() - ti}s", file=log)

def _hci(**kw):
    opts = OmegaConf.create(kw)
    from omegaconf import ListConfig, open_dict
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from dask.graph_manipulation import clone
    from distributed import get_client, wait, as_completed, Semaphore
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_from_zarr
    import fsspec
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

    fs = fsspec.filesystem(fdsstore.url.split(':', 1)[0])
    fs.makedirs(fdsstore.url, exist_ok=True)

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



    client = get_client()

    print('Constructing mapping', file=log)
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gain_chunks, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               gain_name,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_degrid_image,
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
        print(f"Field of view is ({fovx:.3e},{fovy:.3e}) degrees",
              file=log)

    print(f"Image size = (nx={nx}, ny={ny})", file=log)

    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    # crude column arithmetic
    dc = opts.data_column.replace(" ", "")
    if "+" in dc:
        dc1, dc2 = dc.split("+")
        operator="+"
    elif "-" in dc:
        dc1, dc2 = dc.split("-")
        operator="-"
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
        try:
            mds = xr.open_zarr(opts.transfer_model_from)
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

    xds = xds_from_ms(ms,
                      chunks=ms_chunks[ms],
                      columns=columns,
                      table_schema=schema,
                      group_cols=group_by)

    if opts.concat_time:
        xds = xr.concat(xds, dime='row')

    if opts.gain_table is not None:
        gds = xds_from_zarr(gain_name,
                            chunks=gain_chunks[ms])
        if opts.concat_time:
            gds = xr.concat(gds, 'gain_time')

    # import ipdb; ipdb.set_trace()



    # a flat list to use with as_completed
    datasets = []

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

            # # TODO - cpdi to cpgi mapping
            # # assumes cpdi is integer multiple of cpgi
            nbandi = freq_mapping[ms][idt]['start_indices'].size
            nfreqs = np.sum(freq_mapping[ms][idt]['counts'])
            if opts.channels_per_grid_image in (0, -1, None):
                cpgi = nfreqs
            else:
                cpgi = opts.channels_per_grid_image
            if opts.channels_per_degrid_image in (0, -1, None):
                cpdi = nfreqs
            else:
                cpdi = opts.channels_per_degrid_image
            fbins_per_band = int(cpgi / cpdi)
            nband = int(np.ceil(nbandi/fbins_per_band))

            for fi in range(nband):
                idx0 = fi * fbins_per_band
                idxf = np.minimum((fi + 1) * fbins_per_band, nfreqs)
                fidx = freq_mapping[ms][idt]['start_indices'][idx0:idxf]
                fcnts = freq_mapping[ms][idt]['counts'][idx0:idxf]
                # need to slice out all data going onto same grid
                Inu = slice(fidx.min(), fidx.min() + np.sum(fcnts))

                subds = ds[{'row': Irow, 'chan': Inu}]
                subds = subds.chunk({'row':-1, 'chan': -1})

                if opts.gain_table is not None:
                    # Only DI gains currently supported
                    subgds = gds[ids][{'gain_time': It, 'gain_freq': Inu}]
                    subgds = subgds.chunk({'gain_time': -1, 'gain_freq': -1})
                    jones = subgds.gains.data
                else:
                    jones = None

                datasets.append([subds,
                                 jones,
                                 freqs[ms][idt][Inu],
                                 utimes[ms][idt][It],
                                 ridx, rcnts,
                                 fidx, fcnts,
                                 radecs[ms][idt],
                                 fi, ti])

    futures = []
    associated_workers = {}
    idle_workers = set(client.scheduler_info()['workers'].keys())
    n_launched = 0

    while idle_workers:   # Seed each worker with a task.

        (subds, jones, freqsi, utimesi, ridx, rcnts, fidx, fcnts,
         radeci, fi, ti) = datasets[n_launched]
        data2 = None if dc2 is None else getattr(subds, dc2).data
        sc = opts.sigma_column
        sigma = None if sc is None else getattr(subds, sc).data
        wc = opts.weight_column
        weight = None if wc is None else getattr(subds, wc).data

        worker = idle_workers.pop()
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
                        freq=freqsi,
                        utime=utimesi,
                        tbin_idx=ridx,
                        tbin_counts=rcnts,
                        fbin_idx=fidx,
                        fbin_counts=fcnts,
                        cell_rad=cell_rad,
                        radec=radeci,
                        antpos=antpos[ms],
                        poltype=poltype[ms],
                        fieldid=subds.FIELD_ID,
                        ddid=subds.DATA_DESC_ID,
                        scanid=subds.SCAN_NUMBER,
                        fds_store=fdsstore,
                        bandid=fi,
                        timeid=ti,
                        wid=worker,
                        pure=False,
                        workers=worker)

        futures.append(future)
        associated_workers[future] = worker
        n_launched += 1

    ac_iter = as_completed(futures)
    nds = len(datasets)
    for completed_future in ac_iter:

        if n_launched == nds:  # Stop once all jobs have been launched.
            continue

        (subds, jones, freqsi, utimesi, ridx, rcnts, fidx, fcnts,
        radeci, fi, ti) = datasets[n_launched]
        data2 = None if dc2 is None else getattr(subds, dc2).data
        sc = opts.sigma_column
        sigma = None if sc is None else getattr(subds, sc).data
        wc = opts.weight_column
        weight = None if wc is None else getattr(subds, wc).data

        worker = associated_workers.pop(completed_future)

        if completed_future.result() != 1:
            import ipdb; ipdb.set_trace()
            raise ValueError("Something went wrong in submit!")

        # future = client.submit(f, xdsl[n_launched], worker, workers=worker)
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
                        freq=freqsi,
                        utime=utimesi,
                        tbin_idx=ridx,
                        tbin_counts=rcnts,
                        fbin_idx=fidx,
                        fbin_counts=fcnts,
                        cell_rad=cell_rad,
                        radec=radeci,
                        antpos=antpos[ms],
                        poltype=poltype[ms],
                        fieldid=subds.FIELD_ID,
                        ddid=subds.DATA_DESC_ID,
                        scanid=subds.SCAN_NUMBER,
                        fds_store=fdsstore,
                        bandid=fi,
                        timeid=ti,
                        wid=worker,
                        pure=False,
                        workers=worker)

        ac_iter.add(future)
        associated_workers[future] = worker
        n_launched += 1

        if opts.progressbar:
            print(f"\rProcessing: {n_launched}/{nds}", end='', flush=True)

    wait(futures)

    return
