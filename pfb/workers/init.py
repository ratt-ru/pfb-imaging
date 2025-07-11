# flake8: noqa
from pfb.workers.main import cli
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
pfb_logging.init('pfb')
log = pfb_logging.get_logger('INIT')
import time

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.init)
def init(**kw):
    '''
    Initialise Stokes data products for imaging
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

    remprod = opts.product.upper().strip('IQUV')
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported",
                            NotImplementedError)

    from daskms.fsspec_store import DaskMSStore
    msnames = []
    for ms in opts.ms:
        msstore = DaskMSStore(ms.rstrip('/'))
        mslist = msstore.fs.glob(ms.rstrip('/'))
        try:
            assert len(mslist) > 0
            msnames.append(*list(map(msstore.fs.unstrip_protocol, mslist)))
        except:
            log.error_and_raise(f"No MS at {ms}", ValueError)
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
                log.error_and_raise(f"No gain table  at {gt}", ValueError)
        opts.gain_table = gainnames

    OmegaConf.set_struct(opts, True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/init_{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    # TODO - prettier config printing
    log.info('Input Options:')
    for key in opts.keys():
        log.info('     %25s = %s' % (key, opts[key]))

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    # with ExitStack() as stack:
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from pfb import set_client
    client = set_client(opts.nworkers, log, client_log_level=opts.log_level)

    ti = time.time()
    _init(**opts)

    log.info(f"All done after {time.time() - ti}s")

    try:
        client.close()
    except Exception as e:
        pass

def _init(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    from distributed import get_client, as_completed
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms.experimental.zarr import xds_from_zarr
    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.stokes2vis import single_stokes
    from uuid import uuid4
    import fsspec

    basename = f'{opts.output_filename}'

    xds_store = DaskMSStore(f'{basename}.xds')
    if xds_store.exists():
        if opts.overwrite:
            log.info(f"Overwriting {basename}.xds")
            xds_store.rm(recursive=True)
        else:
            log.error_and_raise(f"{basename}.xds exists. "
                                "Set overwrite to overwrite it. ",
                                RuntimeError)

    fs = fsspec.filesystem(xds_store.protocol)
    fs.makedirs(xds_store.url, exist_ok=True)

    log.info(f"Data products will be stored in {xds_store.url}")

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
    worker_keys = client.scheduler_info()['workers'].keys()

    log.info('Constructing mapping')
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gains, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               gain_names,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_image,
                               freq_min=freq_min,
                               freq_max=freq_max,
                               FIELD_IDs=opts.fields,
                               DDIDs=opts.ddids,
                               SCANs=opts.scans)

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
               'UVW', 'ANTENNA1',
               'ANTENNA2', 'TIME', 'INTERVAL', 'FLAG_ROW')
    schema = {}
    if opts.flag_column != 'None':
        columns += (opts.flag_column,)
        schema[opts.flag_column] = {'dims': ('chan', 'corr')}
    schema[dc1] = {'dims': ('chan', 'corr')}
    if dc2 is not None:
        columns += (dc2,)
        schema[dc2] = {'dims': ('chan', 'corr')}

    # only WEIGHT column gets special treatment
    # any other column must have channel axis
    if opts.sigma_column is not None:
        log.info(f"Initialising weights from {opts.sigma_column} column")
        columns += (opts.sigma_column,)
        schema[opts.sigma_column] = {'dims': ('chan', 'corr')}
    elif opts.weight_column is not None:
        log.info(f"Using weights from {opts.weight_column} column")
        columns += (opts.weight_column,)
        # hack for https://github.com/ratt-ru/dask-ms/issues/268
        if opts.weight_column != 'WEIGHT':
            schema[opts.weight_column] = {'dims': ('chan', 'corr')}
    else:
        log.info(f"No weights provided, using unity weights")

    # distinct freq groups
    sgroup = 0
    freq_groups = []
    freq_sgroups = []
    for ms in opts.ms:
        for idt, freq in freqs[ms].items():
            ilo = idt.find('DDID') + 4
            ihi = idt.rfind('_')
            ddid = int(idt[ilo:ihi])
            if (opts.ddids is not None) and (ddid not in opts.ddids):
                continue
            if not len(freq_groups):
                freq_groups.append(freq)
                freq_sgroups.append(sgroup)
                sgroup += freq_mapping[ms][idt]['counts'].size
            else:
                in_group = False
                for fs in freq_groups:
                    if freq.size == fs.size and np.all(freq == fs):
                        in_group = True
                        break
                if not in_group:
                    freq_groups.append(freq)
                    freq_sgroups.append(sgroup)
                    sgroup += freq_mapping[ms][idt]['counts'].size

    # band mapping
    msddid2bid = {}
    for ms in opts.ms:
        msddid2bid[ms] = {}
        for idt, freq in freqs[ms].items():
            # find group where it matches
            for sgroup, fs in zip(freq_sgroups, freq_groups):
                if freq.size == fs.size and np.all(freq == fs):
                    msddid2bid[ms][idt] = sgroup

    # a flat list to use with as_completed
    datasets = []
    for ims, ms in enumerate(opts.ms):
        xds = xds_from_ms(ms, chunks=ms_chunks[ms], columns=columns,
                          table_schema=schema, group_cols=group_by)

        for ds in xds:
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            if (opts.fields is not None) and (fid not in opts.fields):
                continue
            if (opts.ddids is not None) and (ddid not in opts.ddids):
                continue
            if (opts.scans is not None) and (scanid not in opts.scans):
                continue

            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

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
                b0 = msddid2bid[ms][idt]
                for fi, (flow, fcounts) in fitr:
                    Inu = slice(flow, flow + fcounts)
                    subds = ds[{'row': Irow, 'chan': Inu}]
                    if gains[ms][idt] is not None:
                        subgds = gains[ms][idt][{'gain_time': It, 'gain_freq': Inu}]
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
                                    b0 + fi, ti, ims, ms, flow, flow+fcounts])

    futures = []
    associated_workers = {}
    idle_workers = set(client.scheduler_info()['workers'].keys())
    n_launched = 0
    nds = len(datasets)
    while idle_workers and len(datasets):   # Seed each worker with a task.
        # pop so len(datasets) -> 0
        (subds, jones, freqsi, chan_widthi, utimesi, ridx, rcnts,
         radeci, fi, ti, ims, ms, chan_low, chan_high) = datasets.pop(0)

        worker = idle_workers.pop()
        future = client.submit(single_stokes,
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
                        chan_low=chan_low,
                        chan_high=chan_high,
                        radec=radeci,
                        antpos=antpos[ms],
                        poltype=poltype[ms],
                        xds_store=xds_store.url,
                        bandid=fi,
                        timeid=ti,
                        msid=ims,
                        wid=worker,
                        pure=False,
                        workers=worker,
                        key='image-'+uuid4().hex)

        futures.append(future)
        associated_workers[future] = worker
        n_launched += 1

        if opts.progressbar:
            print(f"\rProcessing: {n_launched}/{nds}", end='', flush=True)

    times_out = []
    freqs_out = []
    ac_iter = as_completed(futures)
    for completed_future in ac_iter:
        if isinstance(completed_future.result(), BaseException):
            e = completed_future.result()
            import traceback
            log.error_and_raise(f"Traceback:\n{traceback.format_exc()}", e)

        result = completed_future.result()
        if result is not None:
            times_out.append(result[0])
            freqs_out.append(result[1])

        worker = associated_workers.pop(completed_future)
        # we need this here to release memory for some reason
        client.cancel(completed_future)

        # pop so len(datasets) -> 0
        if len(datasets):
            (subds, jones, freqsi, chan_widthi, utimesi, ridx, rcnts,
            radeci, fi, ti, ims, ms, chan_low, chan_high) = datasets.pop(0)

            future = client.submit(single_stokes,
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
                            chan_low=chan_low,
                            chan_high=chan_high,
                            radec=radeci,
                            antpos=antpos[ms],
                            poltype=poltype[ms],
                            xds_store=xds_store.url,
                            bandid=fi,
                            timeid=ti,
                            msid=ims,
                            wid=worker,
                            pure=False,
                            workers=worker,
                            key='image-'+uuid4().hex)


            ac_iter.add(future)
            associated_workers[future] = worker
            n_launched += 1

        if opts.memory_reporting:
            worker_info = client.scheduler_info()['workers']
            log.info(f'Total memory {worker} MB = ',
                  worker_info[worker]['metrics']['memory']/1e6)

        if opts.progressbar:
            print(f"\rProcessing: {n_launched}/{nds}", end='', flush=True)

        # this should not be necessary but just in case
        if ac_iter.is_empty():
            break

    times_out = np.unique(times_out)
    freqs_out = np.unique(freqs_out)

    nband = freqs_out.size
    ntime = times_out.size

    print("\n")  # after progressbar above
    log.info(f"Freq and time selection resulted in {nband} output bands and "
          f"{ntime} output times")

    return
