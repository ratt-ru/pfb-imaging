# flake8: noqa
from pfb.workers.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('HCI')
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.hci)
def hci(**kw):
    '''
    Produce high cadence residual images.
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
        raise NotImplementedError(f"Product {remprod} not yet supported")


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

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/hci_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

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
    _hci(**opts)

    print(f"All done after {time.time() - ti}s", file=log)

    client.close()

def _hci(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from distributed import get_client, as_completed
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_from_zarr
    import fsspec
    from daskms.fsspec_store import DaskMSStore
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.stokes2im import stokes_image
    import xarray as xr
    from glob import glob

    basename = f'{opts.output_filename}'

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
        tmpf = lambda x: '::'.join(x.rsplit('/', 1))
        gain_names = list(map(tmpf, opts.gain_table))
        gain_name = gain_names[0]
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



    client = get_client()

    # only a single MS for now
    ms = opts.ms[0]
    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    # write model to tmp ds
    if opts.transfer_model_from is not None:
        raise NotImplementedError('Use the degrid app to populate a model column instead')

    print('Constructing mapping', file=log)
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

    max_freq = 0

    # for ms in opts.ms:
    for idt in freqs[ms].keys():
        freq = freqs[ms][idt]
        mask  = (freq <= freq_max) & (freq >= freq_min)
        max_freq = np.maximum(max_freq, freq[mask].max())

    # cell size
    cell_N = 1.0 / (2 * uv_max * max_freq / lightspeed)

    if opts.cell_size is not None:
        cell_size = opts.cell_size
        cell_rad = cell_size * np.pi / 60 / 60 / 180
        if cell_N / cell_rad < 1:
            print(f"Requested cell size of {cell_size} arcseconds could be sub-Nyquist.", file=log)
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

    if opts.model_column is not None:
        columns += (opts.model_column,)
        schema[opts.model_column] = {'dims': ('chan', 'corr')}

    # import ipdb; ipdb.set_trace()
    xds = xds_from_ms(ms,
                    #   chunks=ms_chunks[ms],
                      columns=columns,
                      table_schema=schema,
                      group_cols=group_by)

    # a flat list to use with as_completed
    datasets = []
    for ids, ds in enumerate(xds):
        fid = ds.FIELD_ID
        ddid = ds.DATA_DESC_ID
        scanid = ds.SCAN_NUMBER
        # TODO - cleaner syntax
        if opts.fields is not None:
            if fid not in list(map(int, opts.fields)):
                continue
        if opts.ddids is not None:
            if ddid not in list(map(int, opts.ddids)):
                continue
        if opts.scans is not None:
            if scanid not in list(map(int, opts.scans)):
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
                subds = subds.chunk({'row':-1, 'chan': -1})
                if gains[ms][idt] is not None:
                    subgds = gains[ms][idt][{'gain_time': It, 'gain_freq': Inu}]
                    jones = subgds.gains.data
                else:
                    jones = None

                datasets.append([subds,
                                jones,
                                freqs[ms][idt][Inu],
                                utimes[ms][idt][It],
                                ridx, rcnts,
                                radecs[ms][idt],
                                b0 + fi, ti, ms])

    nds = len(datasets)
    futures = []
    associated_workers = {}
    idle_workers = set(client.scheduler_info()['workers'].keys())
    n_launched = 0
    while idle_workers and len(datasets):   # Seed each worker with a task.
        # pop so len(datasets) -> 0
        (subds, jones, freqsi, utimesi, ridx, rcnts,
        radeci, fi, ti, ms) = datasets.pop(0)

        worker = idle_workers.pop()
        future = client.submit(stokes_image,
                        dc1=dc1,
                        dc2=dc2,
                        operator=operator,
                        ds=subds,
                        jones=jones,
                        opts=opts,
                        nx=nx,
                        ny=ny,
                        freq=freqsi,
                        utime=utimesi,
                        tbin_idx=ridx,
                        tbin_counts=rcnts,
                        cell_rad=cell_rad,
                        radec=radeci,
                        antpos=antpos[ms],
                        poltype=poltype[ms],
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
    for completed_future in ac_iter:
        if isinstance(completed_future.result(), BaseException):
            print(completed_future.result())
            raise RuntimeError('Something went wrong')

        worker = associated_workers.pop(completed_future)
        # need this to release memory for some reason
        client.cancel(completed_future)

        # pop so len(datasets) -> 0
        if len(datasets):
            (subds, jones, freqsi, utimesi, ridx, rcnts,
            radeci, fi, ti, ms) = datasets.pop(0)

            future = client.submit(stokes_image,
                            dc1=dc1,
                            dc2=dc2,
                            operator=operator,
                            ds=subds,
                            jones=jones,
                            opts=opts,
                            nx=nx,
                            ny=ny,
                            freq=freqsi,
                            utime=utimesi,
                            tbin_idx=ridx,
                            tbin_counts=rcnts,
                            cell_rad=cell_rad,
                            radec=radeci,
                            antpos=antpos[ms],
                            poltype=poltype[ms],
                            fds_store=fdsstore,
                            bandid=fi,
                            timeid=ti,
                            wid=worker,
                            pure=False,
                            workers=worker)

            ac_iter.add(future)
            associated_workers[future] = worker
            n_launched += 1

        if opts.memory_reporting:
            worker_info = client.scheduler_info()['workers']
            print(f'Total memory {worker} MB = ',
                worker_info[worker]['metrics']['memory']/1e6, file=log)

        if opts.progressbar:
            print(f"\rProcessing: {n_launched}/{nds}", end='', flush=True)

        # this should not be necessary but just in case
        if ac_iter.is_empty():
            break
    print("\n")  # after progressbar above

    return
