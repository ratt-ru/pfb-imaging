# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
import time
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('DEGRID')


from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.degrid)
def degrid(**kw):
    '''
    Predict model visibilities to measurement sets.
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
            raise ValueError(f"No MS at {ms}")
    if len(opts.ms) > 1:
        raise ValueError(f"There must be a single MS at {opts.ms}")
    opts.ms = msnames
    modelstore = DaskMSStore(opts.mds.rstrip('/'))
    try:
        assert modelstore.exists()
    except Exception as e:
        raise ValueError(f"There must be a model at  "
                         f"to {opts.mds}")
    opts.mds = modelstore.url

    OmegaConf.set_struct(opts, True)

    if opts.product.upper() not in ["I"]:
                                    # , "Q", "U", "V", "XX", "YX", "XY",
                                    # "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/degrid_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    # we need the collections
    from pfb import set_client
    client = set_client(opts.nworkers, log, None, client_log_level=opts.log_level)

    ti = time.time()
    _degrid(**opts)

    print(f"All done after {time.time() - ti}s.", file=log)

    try:
        client.close()
    except Exception as e:
        raise e

def _degrid(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from dask.distributed import performance_report, wait, get_client
    from dask.graph_manipulation import clone
    from daskms.experimental.zarr import xds_from_zarr
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms import xds_to_storage_table as xds_to_table
    import dask.array as da
    from africanus.constants import c as lightspeed
    from africanus.gridding.wgridder.dask import model as im2vis
    from pfb.operators.gridder import comps2vis, _comps2vis_impl
    from pfb.utils.fits import load_fits, data_from_header, set_wcs
    from regions import Regions
    from astropy.io import fits
    from pfb.utils.misc import compute_context
    import xarray as xr
    import sympy as sm
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr
    from ducc0.misc import resize_thread_pool
    resize_thread_pool(opts.nthreads)

    client = get_client()
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

    print('Constructing mapping', file=log)
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gain_chunks, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               None,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_image)

    # load region file if given
    masks = []
    if opts.region_file is not None:
        # import ipdb; ipdb.set_trace()
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
            raise ValueError("Overlapping regions are not supported")
        remainder = 1 - mask
        # place DI component first
        masks = [remainder] + masks
    else:
        masks = [np.ones((nx, ny), dtype=np.float64)]

    # utime = utimes['file:///home/landman/testing/pfb/MS/point_gauss_nb.MS_p0']['FIELD0_DDID0_SCAN0']
    # freq = freqs['file:///home/landman/testing/pfb/MS/point_gauss_nb.MS_p0']['FIELD0_DDID0_SCAN0']
    # model = np.zeros((nx, ny), dtype=np.float64)
    # tout = tfunc(np.mean(utime))
    # fout = ffunc(np.mean(freq))
    # image = np.zeros((nx, ny), dtype=np.float64)
    # Ix = mds.location_x.values
    # Iy = mds.location_y.values
    # comps = mds.coefficients.values
    # model[Ix, Iy] = modelf(tout, fout, *comps[:, :])  # too magical?
    # import matplotlib.pyplot as plt
    # for mask in masks:
    #     plt.figure(1)
    #     plt.imshow(mask)
    #     plt.colorbar()
    #     plt.figure(2)
    #     plt.imshow(model)
    #     plt.colorbar()
    #     plt.show()
    # import ipdb; ipdb.set_trace()

    print("Computing model visibilities", file=log)
    writes = []
    for ms in opts.ms:
        xds = xds_from_ms(ms,
                          chunks=ms_chunks[ms],
                          group_cols=group_by)

        for i, mask in enumerate(masks):
            out_data = []
            columns = []
            for ds in xds:
                if i == 0:
                    column_name = opts.model_column
                else:
                    column_name = f'{opts.model_column}{i}'
                columns.append(column_name)

                fid = ds.FIELD_ID
                ddid = ds.DATA_DESC_ID
                scanid = ds.SCAN_NUMBER
                idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

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

                # freq <-> band mapping
                freq = da.from_array(freqs[ms][idt],
                                    chunks=opts.channels_per_image)
                fidx = da.from_array(freq_mapping[ms][idt]['start_indices'],
                                    chunks=1)
                fcnts = da.from_array(freq_mapping[ms][idt]['counts'],
                                    chunks=1)

                # number of chunks need to math in mapping and coord
                ntime_out = len(tidx.chunks[0])
                assert len(utime.chunks[0]) == ntime_out
                nfreq_out = len(fidx.chunks[0])
                assert len(freq.chunks[0]) == nfreq_out
                # and they need to match the number of row chunks
                uvw = clone(ds.UVW.data)
                assert len(uvw.chunks[0]) == len(tidx.chunks[0])
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
                                freq_max=freq_max)

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
    dask.compute(writes)  #, optimize_graph=False)

