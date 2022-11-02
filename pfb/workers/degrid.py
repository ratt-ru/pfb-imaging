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
    from pfb.utils.fits import load_fits, data_from_header
    from pfb.utils.misc import restore_corrs, model_from_comps
    from astropy.io import fits
    from pfb.utils.misc import compute_context

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'

    mds = xds_from_zarr(mds_name)[0]
    cell_rad = mds.cell_rad
    cell_deg = np.rad2deg(cell_rad)
    mfreqs = mds.freq.data
    model = getattr(mds, opts.model_name).data
    wsums = mds.WSUM.data
    radec = (mds.ra, mds.dec)

    model, mfreqs, wsums = dask.compute(model, mfreqs, wsums)

    if not np.any(model):
        raise ValueError('Model is empty')

    ref_freq = mfreqs[0]
    nband, nx, ny = model.shape

    if opts.nband_out is None:
        nband_out = nband
    else:
        nband_out = opts.nband_out

    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    freqs, fbin_idx, fbin_counts, band_mapping, freq_out, \
        utimes, tbin_idx, tbin_counts, time_mapping, \
        ms_chunks, gain_chunks, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms, None,
                               opts.nband_out,
                               opts.utimes_per_chunk)

    # interpolate model
    mask = np.any(model, axis=0)
    Ix, Iy = np.where(mask)
    ncomps = Ix.size

    # components excluding zeros
    beta = model[:, Ix, Iy]
    if opts.spectral_poly_order:
        order = opts.spectral_poly_order
        print(f"Fitting integrated polynomial of order {order}", file=log)
        if order > mfreqs.size:
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

        dirty_comps = Xfit.T.dot(wsums[:, None]*beta)

        hess_comps = Xfit.T.dot(wsums[:, None]*Xfit)

        comps = np.linalg.solve(hess_comps, dirty_comps)
        freq_fitted = True


        print("Saving fitted models", file=log)
        from pfb.utils.misc import _model_from_comps
        from pfb.utils.fits import set_wcs
        from pfb.utils.fits import save_fits

        m = _model_from_comps(comps, freq_out, mask,
                              np.arange(freq_out.size),
                              ref_freq, freq_fitted)
        mmfs = np.mean(m, axis=0)

        hdr = set_wcs(cell_deg, cell_deg,
                      nx, ny, radec, freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg,
                          nx, ny, radec, np.mean(freq_out))

        save_fits(f'{basename}_fitted_model.fits', m, hdr)
        save_fits(f'{basename}_fitted_model_mfs.fits', mmfs, hdr_mfs)

    else:
        print("Not fitting frequency axis", file=log)
        comps = beta
        freq_fitted = False

    print("Computing model visibilities", file=log)
    mask = da.from_array(mask, chunks=(nx, ny), name=False)
    comps = da.from_array(comps,
                          chunks=(-1, ncomps), name=False)
    freq_out = da.from_array(freq_out, chunks=-1, name=False)
    writes = []
    for ms in opts.ms:
        # DATA used to get required type since MODEL_DATA may not exist yet
        xds = xds_from_ms(ms,
                          chunks=ms_chunks[ms],
                          columns=('UVW','DATA'),
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

            bmap = da.from_array(band_mapping[ms][idt], chunks=1)
            model = model_from_comps(comps, freq_out, mask,
                                     bmap,
                                     ref_freq, freq_fitted)

            model = inlined_array(model, [comps, freq_out, mask])

            # always
            freq = da.from_array(freqs[ms][idt],
                                 chunks=tuple(fbin_counts[ms][idt]))
            fidx = da.from_array(fbin_idx[ms][idt], chunks=1)
            fcnts = da.from_array(fbin_counts[ms][idt], chunks=1)
            uvw = ds.UVW.data

            vis = im2vis(uvw,
                         freq,
                         model,
                         fidx,
                         fcnts,
                         cell_rad,
                         nthreads=opts.nvthreads,
                         epsilon=opts.epsilon,
                         do_wstacking=opts.wstack)


            vis = inlined_array(vis, [uvw])
            vis = vis.astype(ds.DATA.dtype)
            model_vis = restore_corrs(vis, ds.corr.size)

            out_ds = ds.assign(**{opts.model_column:
                                 (("row", "chan", "corr"), model_vis)})
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
        dask.compute(writes, optimize_graph=True)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here.", file=log)

