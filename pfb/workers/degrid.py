# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('DEGRID')


from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.degrid["inputs"].keys():
    defaults[key] = schema.degrid["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.degrid)
def degrid(**kw):
    '''
    Predict model visibilities to measurement sets.
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'degrid_{timestamp}.log')

    from daskms.fsspec_store import DaskMSStore
    msstore = DaskMSStore(opts.ms.rstrip('/'))
    ms = msstore.fs.glob(opts.ms.rstrip('/'))
    try:
        assert len(ms) > 0
        opts.ms = list(map(msstore.fs.unstrip_protocol, ms))
    except:
        raise ValueError(f"No MS at {opts.ms}")

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    if opts.product.upper() not in ["I"]:
                                    # , "Q", "U", "V", "XX", "YX", "XY",
                                    # "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _degrid(**opts)

def _degrid(**kw):
    opts = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(opts.ms, list) and not isinstance(opts.ms, ListConfig) :
        opts.ms = [opts.ms]
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from dask.distributed import performance_report
    from dask.graph_manipulation import clone
    from daskms.experimental.zarr import xds_from_zarr
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms import xds_to_storage_table as xds_to_table
    from daskms.optimisation import inlined_array
    import dask.array as da
    from africanus.constants import c as lightspeed
    from africanus.gridding.wgridder.dask import model as im2vis
    from pfb.operators.gridder import comps2vis
    from pfb.utils.fits import load_fits, data_from_header
    from pfb.utils.misc import restore_corrs, model_from_comps
    from astropy.io import fits
    from pfb.utils.misc import compute_context

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    dds_name = f'{basename}_{opts.postfix}.dds.zarr'
    dds = xds_from_zarr(dds_name,
                        chunks={'x':-1,
                                'y':-1})
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    # get cube info
    nband = 0
    ntime = 0
    mfreqs = []
    mtimes = []
    for ds in dds:
        nband = np.maximum(ds.bandid+1, nband)
        ntime = np.maximum(ds.timeid+1, ntime)
        mfreqs.append(ds.freq_out)
        mtimes.append(ds.time_out)
    mfreqs = np.unique(np.array(mfreqs))
    mtimes = np.unique(np.array(mtimes))
    assert nband == opts.nband
    assert mfreqs.size == nband
    assert mtimes.size == ntime
    assert len(dds) == nband*ntime

    # stack cube
    nx = dds[0].x.size
    ny = dds[0].y.size
    x0 = dds[0].x0
    y0 = dds[0].y0

    # TODO - do this as a dask graph
    # model = [da.zeros((nx, ny)) for _ in range(opts.nband)]
    # wsums = [da.zeros(1) for _ in range(opts.nband)]
    model = np.zeros((ntime, nband, nx, ny), dtype=np.float64)
    wsums = np.zeros((ntime, nband), dtype=np.float64)
    mask = np.zeros((nx, ny), dtype=bool)
    for ds in dds:
        b = ds.bandid
        t = ds.timeid
        model[t, b] = getattr(ds, opts.model_name).values
        wsums[t, b] += ds.WSUM.values[0]

    # model = da.stack(model)
    # wsums = da.stack(wsums).squeeze()
    # model, wsums = dask.compute(model, wsums)
    if not np.any(model):
        raise ValueError('Model is empty')
    radec = (dds[0].ra, dds[0].dec)

    ref_freq = mfreqs[0]

    if opts.nband_out is None:
        nband_out = nband
    else:
        nband_out = opts.nband_out

    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    # freqs, fbin_idx, fbin_counts, band_mapping, freq_out, \
    #     utimes, tbin_idx, tbin_counts, time_mapping, \
    #     ms_chunks, gain_chunks, radecs, \
    #     chan_widths, uv_max, antpos, poltype = \
    #         construct_mappings(opts.ms, None,
    #                            nband_out,
    #                            opts.integrations_per_image)

    print('Constructing mapping', file=log)
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gain_chunks, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               None,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_image)

    # interpolate model
    mask = np.any(model, axis=(0,1))
    Ix, Iy = np.where(mask)
    ncomps = Ix.size

    # components excluding zeros
    beta = model[:, :, Ix, Iy]
    if nband_out != nband:
        order = opts.spectral_poly_order
        print(f"Fitting integrated polynomial of order {order}", file=log)
        if order > nband:
            raise ValueError("spectral-poly-order can't be larger than nband")

        # we are given frequencies at bin centers, convert to bin edges
        delta_freq = mfreqs[1] - mfreqs[0]
        wlow = (mfreqs - delta_freq/2.0)/ref_freq
        whigh = (mfreqs + delta_freq/2.0)/ref_freq
        wdiff = whigh - wlow

        # set design matrix for each component
        Xfit = np.zeros([mfreqs.size, order])
        for i in range(1, order+1):
            Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

        comps = np.zeros((ntime, order, Ix.size))
        from pfb.utils.misc import _model_from_comps
        from pfb.utils.fits import set_wcs
        from pfb.utils.fits import save_fits
        freq_fitted = True
        for t in range(ntime):
            dirty_comps = Xfit.T.dot(wsums[t, :, None]*beta[t])

            hess_comps = Xfit.T.dot(wsums[t, :, None]*Xfit)

            comps[t] = np.linalg.solve(hess_comps, dirty_comps)

            m = _model_from_comps(comps[t], freq_out, mask,
                                np.arange(freq_out.size),
                                ref_freq, freq_fitted)
            mmfs = np.mean(m, axis=0)

            hdr = set_wcs(cell_deg, cell_deg,
                        nx, ny, radec, freq_out)
            hdr_mfs = set_wcs(cell_deg, cell_deg,
                            nx, ny, radec, np.mean(freq_out))

            save_fits(f'{basename}_fitted_model_time{t:04d}.fits', m, hdr)
            save_fits(f'{basename}_fitted_model_time{t:04d}_mfs.fits', mmfs, hdr_mfs)

    else:
        print("Not fitting frequency axis", file=log)
        comps = beta
        freq_fitted = False


    print("Computing model visibilities", file=log)
    mask = da.from_array(mask, chunks=(nx, ny), name=False)
    # comps = da.from_array(comps,
    #                       chunks=(-1, ncomps), name=False)
    # freq_out = da.from_array(freq_out, chunks=-1, name=False)
    writes = []
    for ms in opts.ms:
        xds = xds_from_ms(ms,
                          chunks=ms_chunks[ms],
                          group_cols=group_by)

        out_data = []
        for ds in xds:
            # TODO - rephase if fields don't match
            # radec = radecs[ms][idt]
            # if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
            #     continue
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

            # currently no interpolation in time
            tmap = time_mapping[ms][idt]
            tidx = da.from_array(tbin_idx[ms][idt][tmap['low']], chunks=1)
            tcnts = da.from_array(np.array(ms_chunks[ms][0]['row']), chunks=1)

            # always
            freq = da.from_array(freqs[ms][idt],
                                 chunks=tuple(fbin_counts[ms][idt]))
            fidx = da.from_array(freq_mapping[ms][idt]['start_indices'], chunks=1)
            fcnts = da.from_array(freq_mapping[ms][idt]['counts'], chunks=1)
            uvw = ds.UVW.data

            # we need to do this here because the design matrix is a function of SPW
            freqo = freq_out[band_mapping[ms][idt]]
            if freq_fitted:
                w = (freqo / ref_freq).reshape(freqo.size, 1)
                Xdes = np.tile(w, order) ** np.arange(0, order)  # simple polynomial
            else:
                Xdes = np.eye(freqo.size)
            Xdes = da.from_array(Xdes, chunks=(1, -1))

            vis = comps2vis(uvw,
                            freq,
                            comps,
                            Xdes,
                            mask,
                            tidx,
                            tcnts,
                            fidx,
                            fcnts,
                            cell_rad, cell_rad,
                            x0=x0, y0=y0,
                            nthreads=opts.nvthreads,
                            epsilon=opts.epsilon,
                            wstack=opts.wstack)

            # convert to single precision to write to MS
            vis = vis.astype(np.complex64)

            if opts.accumulate:
                vis += getattr(ds, opts.model_column).data

            vis = inlined_array(vis, [uvw])

            out_ds = ds.assign(**{opts.model_column:
                                 (("row", "chan", "corr"), vis)})
            out_data.append(out_ds)

        writes.append(xds_to_table(out_data, ms,
                                   columns=[opts.model_column],
                                   rechunk=True))

    # dask.visualize(writes, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=opts.output_filename + '_degrid_writes_I_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(writes, filename=opts.output_filename +
    #                '_degrid_writes_I_graph.pdf', optimize_graph=False)

    with compute_context(opts.scheduler, opts.output_filename+'_degrid'):
        dask.compute(writes, optimize_graph=False)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here.", file=log)

