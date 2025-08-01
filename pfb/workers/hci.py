# flake8: noqa
from pfb.workers.main import cli
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
pfb_logging.init('pfb')
log = pfb_logging.get_logger('HCI')
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema
import ray


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
            log.error_and_raise(f"No MS at {ms}",
                                ValueError)
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
                log.error_and_raise(f"No gain table  at {gt}",
                                    ValueError)
        opts.gain_table = gainnames

    OmegaConf.set_struct(opts, True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/hci_{timestamp}.log'
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

    ray.init(num_cpus=opts.nworkers, 
             logging_level='INFO', 
             ignore_reinit_error=True,
             local_mode=opts.nworkers==1)
    
    ti = time.time()
    _hci(**opts)

    log.info(f"All done after {time.time() - ti}s")

    ray.shutdown()

def _hci(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    from distributed import get_client, as_completed
    from daskms import xds_from_storage_ms as xds_from_ms
    import fsspec
    from daskms.fsspec_store import DaskMSStore
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.stokes2im import safe_stokes_image
    import xarray as xr
    from glob import glob

    basename = f'{opts.output_filename}'

    fds_store = DaskMSStore(f'{basename}.fds')
    if fds_store.exists():
        if opts.overwrite:
            log.info(f"Overwriting {basename}.fds")
            fds_store.rm(recursive=True)
        else:
            log.error_and_raise(f"{basename}.fds exists. "
                                "Set overwrite to overwrite it. ",
                                RuntimeError)

    fs = fsspec.filesystem(fds_store.url.split(':', 1)[0])
    fs.makedirs(fds_store.url, exist_ok=True)

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

    # only a single MS for now
    ms = opts.ms[0]
    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    # write model to tmp ds
    if opts.transfer_model_from is not None:
        log.error_and_raise('Use the degrid app to populate a model column instead',
                            NotImplementedError)

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
            log.info(f"Requested cell size of {cell_size} arcseconds could be sub-Nyquist.")
        log.info(f"Super resolution factor = {cell_N/cell_rad}")
    else:
        cell_rad = cell_N / opts.super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi
        log.info(f"Cell size set to {cell_size} arcseconds")

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
        log.info(f"Field of view is ({fovx:.3e},{fovy:.3e}) degrees")

    log.info(f"Image size = (nx={nx}, ny={ny})")

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

    if opts.model_column is not None:
        columns += (opts.model_column,)
        schema[opts.model_column] = {'dims': ('chan', 'corr')}

    tasks = []
    for ims, ms in enumerate(opts.ms):
        xds = xds_from_ms(ms,
                        columns=columns,
                        table_schema=schema,
                        group_cols=group_by)

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
                    subds = subds.chunk({'row':-1, 'chan': -1})
                    if gains[ms][idt] is not None:
                        subgds = gains[ms][idt][{'gain_time': It, 'gain_freq': Inu}]
                        jones = subgds.gains.data
                    else:
                        jones = None

                    fut = safe_stokes_image.remote(
                            dc1=dc1,
                            dc2=dc2,
                            operator=operator,
                            ds=subds,
                            jones=jones,
                            opts=opts,
                            nx=nx,
                            ny=ny,
                            freq=freqs[ms][idt][Inu],
                            utime=utimes[ms][idt][It],
                            tbin_idx=ridx,
                            tbin_counts=rcnts,
                            cell_rad=cell_rad,
                            radec=radecs[ms][idt],
                            antpos=antpos[ms],
                            poltype=poltype[ms],
                            fds_store=fds_store,
                            bandid=b0+fi,
                            timeid=ti,
                            msid=ims,
                    )
                    tasks.append(fut)

    nds = len(tasks)
    ncomplete = 0
    remaining_tasks = tasks.copy()
    while remaining_tasks:
        # Wait for at least 1 task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        
        # Process the completed task
        for task in ready:
            try:
                result = ray.get(task)
            except Exception as e:
                import ipdb; ipdb.set_trace()
                for future in remaining_tasks:
                    ray.cancel(future)
                import traceback
                log.error_and_raise(f"Traceback:\n{traceback.format_exc()}",
                                    result)
            ncomplete += 1
            print(f"Completed: {ncomplete} / {nds}", end='\n', flush=True)

    print("\n")  
    return
