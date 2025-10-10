# flake8: noqa
import click
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
import time
from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema
import ray
import fsspec

log = pfb_logging.get_logger('HCI')


@click.command(context_settings={'show_default': True})
@clickify_parameters(schema.hci)
def hci(**kw):
    '''
    Produce high cadence residual images.
    '''
    opts = OmegaConf.create(kw)

    dir_out = opts.dir_out

    if '://' in dir_out:
        protocol = dir_out.split('://')[0]
        prefix = f'{protocol}://'
    else:
        protocol = 'file'
        prefix = ''

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path('/'.join(dir_out.split('/')[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = dir_out.split('/')[-1]

    opts.dir_out = f'{prefix}{basedir}/{oname}'

    # this should be a file system
    opts.log_directory = basedir

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
            msnames += list(map(msstore.fs.unstrip_protocol, mslist))
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
                gainnames += list(map(gainstore.fs.unstrip_protocol, gtlist))
            except Exception as e:
                log.error_and_raise(f"No gain table  at {gt}",
                                    ValueError)
        opts.gain_table = gainnames

    OmegaConf.set_struct(opts, True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/hci_{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    pfb_logging.log_options_dict(log, opts)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    if opts.object_store_memory is not None:
        mem_limit = opts.object_store_memory * 1e9  # convert GB -> B
    else:
        mem_limit = None

    ray.init(num_cpus=opts.nworkers,
             logging_level='INFO',
             ignore_reinit_error=True,
             object_store_memory=mem_limit,
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
    from daskms import xds_from_storage_ms as xds_from_ms
    import fsspec
    from daskms.fsspec_store import DaskMSStore
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.stokes2im import batch_stokes_image
    import xarray as xr
    import yaml
    from tempfile import TemporaryDirectory
    from zarr import ProcessSynchronizer

    basename = f'{opts.dir_out}'

    if opts.stack and opts.output_format == 'fits':
        raise RuntimeError("Can't stack in fits mode")

    fds_store = DaskMSStore(basename)
    if fds_store.exists():
        if opts.overwrite:
            log.info(f"Overwriting {basename}")
            fds_store.rm(recursive=True)
        else:
            log.error_and_raise(f"{basename} exists. "
                                "Set overwrite to overwrite it. ",
                                RuntimeError)

    fs = fsspec.filesystem(fds_store.url.split(':', 1)[0])
    fs.makedirs(fds_store.url, exist_ok=True)

    if opts.gain_table is not None:
        tmpf = lambda x: '::'.join(x.rsplit('/', 1))
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

    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    # write model to tmp ds
    if opts.transfer_model_from is not None:
        log.error_and_raise('Use the degrid app to populate a model column instead',
                            NotImplementedError)
        
    ipc = opts.integrations_per_chunk
    if ipc % opts.integrations_per_image:
        log.warning("Warning - integrations-per-image does not divide integrations-per-chunk evenly")

    log.info('Constructing mapping')
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gains, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               gain_names,
                               ipi=ipc,
                               cpi=opts.channels_per_image,
                               freq_min=freq_min,
                               freq_max=freq_max,
                               FIELD_IDs=opts.fields,
                               DDIDs=opts.ddids,
                               SCANs=opts.scans)

    max_freq = 0

    # get the full (time, freq) domain
    all_times = []
    all_freqs = []
    for ms in opts.ms:
        for idt in freqs[ms].keys():
            freq = freqs[ms][idt]
            all_freqs.extend(freq)
            all_times.extend(utimes[ms][idt])
            mask  = (freq <= freq_max) & (freq >= freq_min)
            max_freq = np.maximum(max_freq, freq[mask].max())

    all_freqs = np.unique(all_freqs)
    all_times = np.unique(all_times)

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
        if nx%2 or ny%2:
            raise NotImplementedError('Only even number of pixels currently supported')
        fovx = nx*cell_deg
        fovy = ny*cell_deg
        log.info(f"Field of view is ({fovx:.3e},{fovy:.3e}) degrees")

    cell_deg = np.rad2deg(cell_rad)

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

    # construct examplar dataset if asked to stack
    if opts.stack and opts.output_format != 'zarr':
        raise ValueError('Can only stack zarr outputs not fits')
    attrs, ntasks = make_dummy_dataset(opts, fds_store.url, utimes, freqs, radecs,
                                    time_mapping, freq_mapping,
                                    freq_min, freq_max, nx, ny, cell_deg, ipc)

    if opts.inject_transients is not None:
        # we need to do this here because we don't have access
        # to the full domain inside the call to stokes2im
        log.info("Computing transient spectra")
        from pfb.utils.transients import generate_transient_spectra
        with open(opts.inject_transients, 'r') as f:
            config = yaml.safe_load(f)
        transients = config['transients']
        coords = {
            "TIME": (("TIME",), all_times),
            "FREQ": (("FREQ",), all_freqs),
        }
        names = []
        ras = []
        decs = []
        transient_ds = xr.Dataset(coords=coords)
        for transient in transients:
            tprofile, fprofile = generate_transient_spectra(all_times, all_freqs, transient)
            name = transient["name"]
            transient_ds[f'{name}_time_profile'] = (("TIME",), tprofile)
            transient_ds[f'{name}_freq_profile'] = (("FREQ",), fprofile)
            names.append(name)
            ras.append(transient["position"]["ra"])
            decs.append(transient["position"]["dec"])
        transient_ds = transient_ds.assign_attrs({"names": names, "ras": ras, "decs": decs})
        transient_baseoname = opts.inject_transients.removesuffix('yaml')
        transient_ds.to_zarr(f"{transient_baseoname}zarr", mode='a')
        log.info("Spectra computed")

    if opts.model_column is not None:
        columns += (opts.model_column,)
        schema[opts.model_column] = {'dims': ('chan', 'corr')}

    tasks = []
    channel_width = {}
    ncompleted = 0
    # if opts.temp-dir is None dir will be /tmp
    with TemporaryDirectory(prefix='hci-', dir=opts.temp_dir) as tempdir:
        synchronizer = ProcessSynchronizer(tempdir)
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
                        
                        fut = batch_stokes_image.remote(
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
                                rbin_idx=ridx,
                                rbin_counts=rcnts,
                                cell_rad=cell_rad,
                                radec=radecs[ms][idt],
                                antpos=antpos[ms],
                                poltype=poltype[ms],
                                fds_store=fds_store,
                                bandid=b0+fi,
                                timeid=ti,
                                msid=ims,
                                attrs=attrs,
                                integrations_per_image=opts.integrations_per_image,
                                synchronizer=synchronizer
                        )
                        tasks.append(fut)
                        channel_width[b0+fi] = freqs[ms][idt][Inu].max() - freqs[ms][idt][Inu].min()

                        # limit the number of chunks that are processed simultaneously
                        if len(tasks) > opts.max_simul_chunks:
                            ncompleted += 1
                            ready, tasks = ray.wait(tasks, num_returns=1)
                            timeid, bandid = ray.get(ready)[0]
                            print(f"Processed {ncompleted}/{ntasks}", end='\n', flush=True)

        remaining_tasks = tasks.copy()
        while remaining_tasks:
            ncompleted += 1
            ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
            timeid, bandid = ray.get(ready)[0]
            print(f"Processed {ncompleted}/{ntasks}", end='\n', flush=True)

    if opts.stack:
        log.info("Computing means")
        import dask
        import dask.array as da
        from concurrent.futures import ThreadPoolExecutor
        cwidths = []
        for _, val in channel_width.items():
            cwidths.append(val)
        # reduction over FREQ and TIME so use max chunk sizes
        ds = xr.open_zarr(fds_store.url, chunks={'FREQ': -1, 'TIME':-1})
        cube = ds.cube.data
        wsums = ds.weight.data
        # all variables have been normalised by wsum so we first
        # undo the normalisation (wsum=0 where there is no data)
        weighted_cube = cube * wsums[:, :, :, None, None]
        taxis = ds.cube.get_axis_num('TIME')
        faxis = ds.cube.get_axis_num('FREQ')
        wsum = da.sum(wsums, axis=taxis)
        # we need this for the where clause in da.divide, should be cheap
        wsumc = wsum.compute()[:, :, None, None]
        weighted_sum = da.sum(weighted_cube, axis=taxis)
        weighted_mean = da.divide(weighted_sum, wsum[:, :, None, None], where=wsumc>0)
        if opts.psf_out:
            psfsq = (ds.psf.data*wsums[:, :, :, None, None])**2
            weighted_psfsq_sum = da.sum(psfsq, axis=(faxis, taxis))
            wsumsq = da.sum(wsums**2, axis=(faxis, taxis))
            wsumsqc = wsumsq.compute()[:, None, None]
            weighted_psfsq_mean = da.divide(weighted_psfsq_sum, wsumsq[:, None, None], where=wsumsqc>0)
            if opts.psf_relative_size == 1:
                xpsf = "X"
                ypsf = "Y"
            else:
                xpsf = "X_PSF"
                ypsf = "Y_PSF"
            ds['psf2'] = (('STOKES', ypsf, xpsf),  weighted_psfsq_mean)
        # only write new variables
        drop_vars = [key for key in ds.data_vars.keys() if key != 'psf2']
        ds = ds.drop_vars(drop_vars)
        ds['cube_mean'] = (('STOKES', 'FREQ', 'Y', 'X'),  weighted_mean)
        ds['channel_width'] = (('FREQ',), da.from_array(cwidths, chunks=1))
        with dask.config.set(pool=ThreadPoolExecutor(8)):
            ds.to_zarr(fds_store.url, mode='r+')
        log.info("Reduction complete")
    return

def make_dummy_dataset(opts, cds_url, utimes, freqs, radecs, time_mapping, freq_mapping,
                       freq_min, freq_max, nx, ny, cell_deg, time_chunk,
                       spatial_chunk=128):
    import numpy as np
    import dask.array as da
    import xarray as xr
    from daskms import xds_from_storage_ms as xds_from_ms
    from ducc0.fft import good_size
    out_ra = []
    out_dec = []
    out_times = []
    out_freqs = []
    ntasks = 0
    for ms in opts.ms:
        xds = xds_from_ms(ms,
                    columns='TIME',
                    group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])
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

            out_ra.append(radecs[ms][idt][0])
            out_dec.append(radecs[ms][idt][1])

            titr = enumerate(zip(time_mapping[ms][idt]['start_indices'],
                                time_mapping[ms][idt]['counts']))
            for ti, (tlow, tcounts) in titr:
                # It = slice(tlow, tlow + tcounts)
                fitr = enumerate(zip(freq_mapping[ms][idt]['start_indices'],
                                     freq_mapping[ms][idt]['counts']))
                for fi, (flow, fcounts) in fitr:
                    Inu = slice(flow, flow + fcounts)
                    out_freqs.append(np.mean(freqs[ms][idt][Inu]))
                    ntasks += 1

                    # divide tlow:tlow+tcounts into ipi intervals
                    for t0 in range(tlow, tlow+tcounts, opts.integrations_per_image):
                        tmax = np.minimum(tlow+tcounts, t0 + opts.integrations_per_image)
                        It = slice(t0, tmax)
                        out_times.append(np.mean(utimes[ms][idt][It]))
    
    # if not stacking we only run this to get the number of tasks that will be submitted
    if not opts.stack:
        return None, ntasks

    # spatial coordinates
    if opts.phase_dir is None:
        out_ra_deg = np.rad2deg(np.unique(out_ra))
        out_dec_deg = np.rad2deg(np.unique(out_dec))
        if out_ra_deg.size > 1 or out_dec_deg.size > 1:
            raise ValueError('phase-dir must be specified when stacking multiple fields')
    else:
        from pfb.utils.astrometry import format_coords
        from astropy import units
        from astropy.coordinates import SkyCoord
        ra_str, dec_str = opts.phase_dir.split(',')
        coord = SkyCoord(ra_str, dec_str, frame='fk5', unit=(units.hourangle, units.deg))
        out_ra_deg = np.array([coord.ra.value])
        out_dec_deg = np.array([coord.dec.value])

    # remove duplicates
    out_times = np.unique(out_times)
    out_freqs = np.unique(out_freqs)

    n_stokes = len(opts.product)
    n_times = out_times.size
    n_freqs = out_freqs.size

    cube_dims = (n_stokes, n_freqs, n_times, ny, nx)
    cube_chunks = (n_stokes, 1, time_chunk, spatial_chunk, spatial_chunk)

    mean_dims = (n_stokes, n_freqs, ny, nx)
    mean_chunks = (n_stokes, 1, spatial_chunk, spatial_chunk)

    rms_dims = (n_stokes, n_freqs, n_times)
    rms_chunks = (n_stokes, 1, time_chunk)

    ra_dim = 'RA--SIN'
    dec_dim = 'DEC--SIN'
    if out_freqs.size > 1:
        delta_freq = np.diff(out_freqs).min()
    else:
        delta_freq = 1
    if out_times.size > 1:
        delta_time = np.diff(out_times).min()
    else:
        delta_time = 1

    # construct reference header

    from astropy.wcs import WCS
    from astropy.io import fits
    w = WCS(naxis=5)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'INTEGRATION', 'FREQ', 'STOKES']
    w.wcs.cdelt = [-cell_deg, cell_deg, delta_time, delta_freq, 1]
    w.wcs.cunit = ['deg', 'deg', 's', 'Hz', '']
    w.wcs.crval = [out_ra_deg[0], out_dec_deg[0], out_times[0], out_freqs[0], 1]
    w.wcs.crpix = [1 + nx//2, 1 + ny//2, 1, 1, 1]
    w.wcs.equinox = 2000.0
    wcs_hdr = w.to_header()

    # the order does seem to matter here,
    # especially when using with StreamingHDU
    hdr = fits.Header()
    hdr['SIMPLE'] = (True, 'conforms to FITS standard')
    hdr['BITPIX'] = (-32, 'array data type')
    hdr['NAXIS'] = (5, 'number of array dimensions')
    data_shape = (nx, ny, n_times, n_freqs, n_stokes)
    for i, size in enumerate(data_shape, 1):
        hdr[f'NAXIS{i}'] = (size, f'length of data axis {i}')

    hdr['EXTEND'] = True
    hdr['BSCALE'] = 1.0
    hdr['BZERO'] = 0.0
    hdr['BUNIT'] = 'Jy/beam'
    hdr['EQUINOX'] = 2000.0
    hdr['BTYPE'] = 'Intensity'
    hdr.update(wcs_hdr)
    hdr['TIMESCAL'] = delta_time
    if opts.obs_label is not None:
        hdr['OBSLABEL'] = opts.obs_label

    # if we don't pass these into stokes2im they get overwritten
    attrs={
            "fits_header": list(dict(hdr).items()),
            "radec_dims": (ra_dim, dec_dim),
            "fits_dims": (
                ("X", ra_dim),
                ("Y", dec_dim),
                ("TIME", "TIME"),
                ("FREQ", "FREQ"),
                ("STOKES", "STOKES")
            )
        }

    # TODO - why do we sometimes need to round here?
    out_ras = out_ra_deg + np.arange(nx//2, -(nx//2), -1) * cell_deg
    out_decs = out_dec_deg + np.arange(-(ny//2), ny//2) * cell_deg
    out_ras = np.round(out_ras, decimals=12)
    out_decs = np.round(out_decs, decimals=12)


    dummy_ds = xr.Dataset(
        data_vars={
            "cube": (
                ("STOKES", "FREQ", "TIME", "Y", "X"),
                da.empty(cube_dims, chunks=cube_chunks, dtype=np.float32)
            ),
            "cube_mean":(
                ("STOKES", "FREQ", "Y", "X"),
                da.empty(mean_dims, chunks=mean_chunks, dtype=np.float32)
            ),
            "rms": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32)
            ),
            "weight": (
                ("STOKES", "FREQ", "TIME"),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32)
            ),
            "nonzero": (
                ("STOKES", "FREQ", "TIME",),
                da.empty(rms_dims, chunks=rms_chunks, dtype=np.bool_)
            ),
            "psf_maj": (
                    ("STOKES", "FREQ", "TIME",),
                    da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32)
            ),
            "psf_min": (
                    ("STOKES", "FREQ", "TIME",),
                    da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32)
            ),
            "psf_pa": (
                    ("STOKES", "FREQ", "TIME",),
                    da.empty(rms_dims, chunks=rms_chunks, dtype=np.float32)
            ),
            "channel_width": (
                    ("FREQ",),
                    da.empty((n_freqs), chunks=(1), dtype=np.float32)
            ),
        },
        coords={
            "TIME": (("TIME",), out_times),
            "STOKES": (("STOKES",), list(sorted(opts.product))),
            "FREQ": (("FREQ",), out_freqs),
            "X": (("X",), out_ras),
            "Y": (("Y",), out_decs)
        }
        ,
        # Store dicts as tuples as zarr doesn't seem to maintain dict order.
        attrs=attrs
    )

    if opts.beam_model is not None:
        dummy_ds['beam_weight'] = (("STOKES", "FREQ", "TIME", "Y", "X"),
                da.empty(cube_dims, chunks=cube_chunks, dtype=np.float32))

    if opts.psf_out:
        nx_psf = good_size(int(opts.psf_relative_size * nx))
        while nx_psf%2:
            nx_psf = good_size(nx_psf+1)
        ny_psf = good_size(int(opts.psf_relative_size * ny))
        while ny_psf%2:
            ny_psf = good_size(ny_psf+1)
        psf_dims = (n_stokes, n_freqs, n_times, ny_psf, nx_psf)
        if opts.psf_relative_size == 1:
            xpsf = "X"
            ypsf = "Y"
        else:
            xpsf = "X_PSF"
            ypsf = "Y_PSF"
            dummy_ds = dummy_ds.assign_coords(
                {'Y_PSF': ((ypsf,), out_dec_deg + np.arange(-(ny_psf//2), ny_psf//2) * cell_deg),
                'X_PSF': ((xpsf,), out_ra_deg + np.arange(nx_psf//2, -(nx_psf//2), -1) * cell_deg)}
            )
        dummy_ds['psf'] = (("STOKES", "FREQ", "TIME", ypsf, xpsf),
                da.empty(psf_dims, chunks=cube_chunks, dtype=np.float32))
        dummy_ds['psf2'] = (("STOKES", ypsf, xpsf),
                da.empty((n_stokes, ny_psf, nx_psf),
                         chunks=(1, spatial_chunk, spatial_chunk), dtype=np.float32))

    # Write scaffold and metadata to disk.
    dummy_ds.to_zarr(cds_url, mode="w", compute=False)
    return attrs, ntasks
