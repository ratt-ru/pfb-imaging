# flake8: noqa
from pfb.workers.main import cli
import time
from omegaconf import OmegaConf
from pfb.utils import logging as pfb_logging
pfb_logging.init('pfb')
log = pfb_logging.get_logger('DEGRID')


from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.degrid)
def degrid(**kw):
    '''
    Predict model visibilities to measurement sets.
    The default behaviour is to read the frequency mapping from the dds and
    degrid one image per band.
    If channels-per-image is provided, the model is evaluated from the mds.
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

    basename = opts.output_filename

    if opts.mds is None:
        mds_store = DaskMSStore(f'{basename}_{opts.suffix}_model.mds')
    else:
        mds_store = DaskMSStore(opts.mds)
        try:
            assert mds_store.exists()
        except Exception as e:
            log.error_and_raise(f"No mds at {opts.mds}", ValueError)
    opts.mds = mds_store.url

    dds_store = DaskMSStore(f'{basename}_{opts.suffix}.dds')
    if opts.channels_per_image is None and not mds_store.exists():
        try:
            assert dds_store.exists()
        except Exception as e:
            log.error_and_raise(f"There must be a dds at {dds_store.url}. "
                                "Specify mds and channels-per-image to degrid from mds.",
                                ValueError)
    opts.dds = dds_store.url
    OmegaConf.set_struct(opts, True)

    if opts.product.upper() not in ["I"]:
                                    # , "Q", "U", "V", "XX", "YX", "XY",
                                    # "YY", "RR", "RL", "LR", "LL"]:
        log.error_and_raise(f"Product {opts.product} not yet supported",
                            NotImplementedError)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/degrid_{timestamp}.log'
    pfb_logging.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    # TODO - prettier config printing
    log.info('Input Options:')
    for key in opts.keys():
        log.info('     %25s = %s' % (key, opts[key]))

    # we need the collections
    from pfb import set_client
    client = set_client(opts.nworkers, log, None, client_log_level=opts.log_level)

    ti = time.time()
    _degrid(**opts)

    log.info(f"All done after {time.time() - ti}s.")

    try:
        client.close()
    except Exception as e:
        pass

def _degrid(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from dask.distributed import wait, get_client
    from dask.graph_manipulation import clone
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_to_storage_table as xds_to_table
    from daskms.fsspec_store import DaskMSStore
    import dask.array as da
    from pfb.operators.gridder import comps2vis
    from pfb.utils.fits import set_wcs
    from regions import Regions
    from pfb.utils.naming import xds_from_url
    import xarray as xr
    import sympy as sm
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)

    client = get_client()

    dds_store = DaskMSStore(opts.dds)
    mds_store = DaskMSStore(opts.mds)

    if opts.channels_per_image is None:
        if dds_store.exists():
            dds, dds_list = xds_from_url(dds_store.url)
            cpi = 0
            for ds in dds:
                cpi = np.maximum(ds.chan.size, cpi)
        else:
            log.error_and_raise("You must supply channels per image in the "
                                "absence of a dds", ValueError)
    else:
        cpi = opts.channels_per_image

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


    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    log.info('Constructing mapping')
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gains, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               None,
                               ipi=opts.integrations_per_image,
                               cpi=cpi,
                               freq_min=freq_min,
                               freq_max=freq_max)
                            #    FIELD_IDs=opts.fields,
                            #    DDIDs=opts.ddids,
                            #    SCANs=opts.scans)

    mds = xr.open_zarr(opts.mds)
    foo = client.scatter(mds, broadcast=True)
    wait(foo)

    # grid spec
    cell_rad = mds.cell_rad_x
    cell_deg = np.rad2deg(cell_rad)
    nx = mds.npix_x
    ny = mds.npix_y
    x0 = mds.center_x
    y0 = mds.center_y
    radec = (mds.ra, mds.dec)

    # model func
    params = sm.symbols(('t','f'))
    params += sm.symbols(tuple(mds.params.values))
    symexpr = parse_expr(mds.parametrisation)
    modelf = lambdify(params, symexpr)
    texpr = parse_expr(mds.texpr)
    tfunc = lambdify(params[0], texpr)
    fexpr = parse_expr(mds.fexpr)
    ffunc = lambdify(params[1], fexpr)

    # load region file if given
    masks = []
    if opts.region_file is not None:
        rfile = Regions.read(opts.region_file)  # should detect format
        # get wcs for model
        wcs = set_wcs(np.rad2deg(mds.cell_rad_x),
                      np.rad2deg(mds.cell_rad_y),
                      mds.npix_x,
                      mds.npix_y,
                      (mds.ra, mds.dec),
                      mds.freqs.values,
                      header=False)
        wcs = wcs.dropaxis(-1)
        wcs = wcs.dropaxis(-1)

        mask = np.zeros((nx, ny), dtype=np.float64)
        # get a mask for each region
        for region in rfile:
            pixel_region = region.to_pixel(wcs)
            # why the transpose?
            region_mask = pixel_region.to_mask().to_image((ny, nx))
            region_mask = region_mask.T
            mask += region_mask
            masks.append(region_mask)
        if (mask > 1).any():
            log.error_and_raise("Overlapping regions are not supported",
                                ValueError)
        remainder = 1 - mask
        # place DI component first
        masks = [remainder] + masks
    else:
        masks = [np.ones((nx, ny), dtype=np.float64)]

    
    writes = []
    for ms in opts.ms:
        xds = xds_from_ms(ms,
                          chunks=ms_chunks[ms],
                          group_cols=group_by)
        

        for i, mask in enumerate(masks):
            out_data = []
            columns = []
            for k, ds in enumerate(xds):
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

                if i == 0:
                    column_name = opts.model_column
                else:
                    column_name = f'{opts.model_column}{i}'
                columns.append(column_name)

                # time <-> row mapping
                utime = da.from_array(utimes[ms][idt],
                                      chunks=opts.integrations_per_image)
                tidx = da.from_array(time_mapping[ms][idt]['start_indices'],
                                     chunks=1)
                tcnts = da.from_array(time_mapping[ms][idt]['counts'],
                                      chunks=1)

                ridx = da.from_array(row_mapping[ms][idt]['start_indices'],
                                     chunks=opts.integrations_per_image)
                rcnts = da.from_array(row_mapping[ms][idt]['counts'],
                                      chunks=opts.integrations_per_image)

                # freq <-> band mapping (entire freq axis)
                freq = da.from_array(freqs[ms][idt],
                                     chunks=ms_chunks[ms][k]['chan'])
                fcnts = np.array(ms_chunks[ms][k]['chan'])
                fidx = np.concatenate((np.array([0]), np.cumsum(fcnts)))[0:-1]

                fidx = da.from_array(fidx,
                                     chunks=1)
                fcnts = da.from_array(fcnts,
                                      chunks=1)

                # number of chunks need to match in mapping and coord
                ntime_out = len(tidx.chunks[0])
                assert len(utime.chunks[0]) == ntime_out
                nfreq_out = len(fidx.chunks[0])
                assert len(freq.chunks[0]) == nfreq_out
                # and they need to match the number of row chunks
                uvw = clone(ds.UVW.data)
                assert len(uvw.chunks[0]) == len(tidx.chunks[0])

                ncorr = ds.corr.size

                vis = comps2vis(uvw,
                                utime,
                                freq,
                                ridx, rcnts,
                                tidx, tcnts,
                                fidx, fcnts,
                                mask,
                                mds,
                                modelf,
                                tfunc,
                                ffunc,
                                nthreads=opts.nthreads,
                                epsilon=opts.epsilon,
                                do_wgridding=opts.do_wgridding,
                                freq_min=freq_min,
                                freq_max=freq_max,
                                ncorr_out=ncorr,
                                product=opts.product,
                                poltype=poltype)

                # convert to single precision to write to MS
                vis = vis.astype(np.complex64)

                if opts.accumulate:
                    vis += getattr(ds, column_name).data

                out_ds = ds.assign(**{column_name:
                                    (("row", "chan", "corr"), vis)})
                out_data.append(out_ds)

            writes.append(xds_to_table(
                                    out_data, ms,
                                    columns=columns,
                                    rechunk=True))

    # optimize_graph can make things much worse
    log.info("Computing model visibilities")
    dask.compute(writes)  #, optimize_graph=False)

