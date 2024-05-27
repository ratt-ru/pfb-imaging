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
log = pyscilog.get_logger('IMIT')
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.imit["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.imit["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.imit)
def imit(**kw):
    '''
    Initialise Stokes data products for imaging
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{ldir}/imit_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/fastim_{timestamp}.log', file=log)
    if opts.product.upper() not in ["I","Q", "U", "V"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")
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
    OmegaConf.set_struct(opts, True)

    if opts.product.upper() not in ["I","Q", "U", "V"]:
            # "XX", "YX", "XY", "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")

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
            client = stack.enter_context(Client(cluster,
                                                direct_to_workers=False))

        client.wait_for_workers(opts.nworkers)
        dashboard_url = client.dashboard_link
        print(f"Dask Dashboard URL at {dashboard_url}", file=log)
        client.amm.stop()

        ti = time.time()
        _imit(**opts)

    print(f"All done after {time.time() - ti}s", file=log)

def _imit(**kw):
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
    from daskms.fsspec_store import DaskMSStore
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.stokes2vis import single_stokes_dist
    import xarray as xr
    from uuid import uuid4
    import fsspec

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    xds_store = DaskMSStore(f'{basename}.xds')
    if xds_store.exists():
        if opts.overwrite:
            print(f"Overwriting {basename}.xds", file=log)
            xds_store.rm(recursive=True)
        else:
            raise ValueError(f"{basename}.xds exists. "
                             "Set overwrite to overwrite it. ")

    fs = fsspec.filesystem(xds_store.protocol)
    fs.makedirs(xds_store.url, exist_ok=True)

    print(f"Data products will be stored in {xds_store.url}", file=log)

    if opts.gain_table is not None:
        tmpf = lambda x: '::'.join(x.rsplit('/', 1))
        gain_names = list(map(tmpf, opts.gain_table))
    else:
        gain_names = None

    if opts.freq_range is not None and len(opts.freq_range):
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
                               gain_names,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_image,
                               freq_min=freq_min,
                               freq_max=freq_max)

    # we need this to set cell sizes consistently per band
    max_freq = 0
    for ms in opts.ms:
        for idt in freqs[ms].keys():
            freq = freqs[ms][idt]
            max_freq = np.maximum(max_freq, freq.max())

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


    # a flat list to use with as_completed
    datasets = []

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

                    datasets.append([subds,
                                    jones,
                                    freqs[ms][idt][Inu],
                                    chan_widths[ms][idt][Inu],
                                    utimes[ms][idt][It],
                                    ridx, rcnts,
                                    radecs[ms][idt],
                                    fi, ti, ims])

    # debug call outside of client
    nds = len(datasets)
    for i in range(nds):
        (subds, jones, freqsi, chan_widthi, utimesi, ridx, rcnts,
            radeci, fi, ti, ims) = datasets[i]

        out_store, time_out, freq_out = single_stokes_dist(
            dc1=dc1,
            dc2=dc2,
            operator=operator,
            ds=subds,
            jones=jones,
            opts=opts,
            freq=freqsi,
            chan_width=chan_widthi,
            utime=utimesi,
            tbin_idx=ridx,
            tbin_counts=rcnts,
            radec=radeci,
            antpos=antpos[ms],
            poltype=poltype[ms],
            fieldid=subds.FIELD_ID,
            ddid=subds.DATA_DESC_ID,
            scanid=subds.SCAN_NUMBER,
            xds_store=xds_store.url,
            bandid=fi,
            timeid=ti,
            msid=ims,
            wid=worker,
            max_freq=max_freq,
            uv_max=uv_max
        )

        print(f"Success at {i}!", file=log)

    quit()

    futures = []
    associated_workers = {}
    idle_workers = set(client.scheduler_info()['workers'].keys())
    n_launched = 0
    nds = len(datasets)
    while idle_workers:   # Seed each worker with a task.

        (subds, jones, freqsi, chan_widthi, utimesi, ridx, rcnts,
         radeci, fi, ti, ims) = datasets[n_launched]

        worker = idle_workers.pop()
        future = client.submit(single_stokes_dist,
                        dc1=dc1,
                        dc2=dc2,
                        operator=operator,
                        ds=subds,
                        jones=jones,
                        opts=opts,
                        freq=freqsi,
                        chan_width=chan_widthi,
                        utime=utimesi,
                        tbin_idx=ridx,
                        tbin_counts=rcnts,
                        radec=radeci,
                        antpos=antpos[ms],
                        poltype=poltype[ms],
                        fieldid=subds.FIELD_ID,
                        ddid=subds.DATA_DESC_ID,
                        scanid=subds.SCAN_NUMBER,
                        xds_store=xds_store.url,
                        bandid=fi,
                        timeid=ti,
                        msid=ims,
                        wid=worker,
                        max_freq=max_freq,
                        uv_max=uv_max,
                        pure=False,
                        workers=worker,
                        key='image-'+uuid4().hex)

        futures.append(future)
        associated_workers[future] = worker
        n_launched += 1

        if opts.progressbar:
            print(f"\rProcessing: {n_launched}/{nds}", end='', flush=True)

    ac_iter = as_completed(futures)
    for completed_future in ac_iter:

        if n_launched == nds:  # Stop once all jobs have been launched.
            continue

        (subds, jones, freqsi, chan_widthi, utimesi, ridx, rcnts,
        radeci, fi, ti, ims) = datasets[n_launched]
        data2 = None if dc2 is None else getattr(subds, dc2).data
        sc = opts.sigma_column
        sigma = None if sc is None else getattr(subds, sc).data
        wc = opts.weight_column
        weight = None if wc is None else getattr(subds, wc).data

        res = completed_future.result()
        if isinstance(res[2], int):
            print(res)

            quit()

        worker = associated_workers.pop(completed_future)

        future = client.submit(single_stokes_dist,
                        dc1=dc1,
                        dc2=dc2,
                        operator=operator,
                        ds=subds,
                        jones=jones,
                        opts=opts,
                        freq=freqsi,
                        chan_width=chan_widthi,
                        utime=utimesi,
                        tbin_idx=ridx,
                        tbin_counts=rcnts,
                        radec=radeci,
                        antpos=antpos[ms],
                        poltype=poltype[ms],
                        fieldid=subds.FIELD_ID,
                        ddid=subds.DATA_DESC_ID,
                        scanid=subds.SCAN_NUMBER,
                        xds_store=xds_store.url,
                        bandid=fi,
                        timeid=ti,
                        msid=ims,
                        wid=worker,
                        max_freq=max_freq,
                        uv_max=uv_max,
                        pure=False,
                        workers=worker,
                        key='image-'+uuid4().hex)

        futures.append(future)
        ac_iter.add(future)
        associated_workers[future] = worker
        n_launched += 1

        if opts.progressbar:
            print(f"\rProcessing: {n_launched}/{nds}", end='', flush=True)

    wait(futures)

    times_out = []
    freqs_out = []
    for f in futures:
        result = f.result()
        times_out.append(result[1])
        freqs_out.append(result[2])

    times_out = np.unique(times_out)
    freqs_out = np.unique(freqs_out)

    nband = freqs_out.size
    ntime = times_out.size

    print("\n")  # after progressbar above
    print(f"Freq and time selection resulted in {nband} output bands and "
          f"{ntime} output times", file=log)

    return
