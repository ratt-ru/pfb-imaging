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
    Currently only predicts from .fits files which can optionally be interpolated
    along the frequency axis.

    By default the real-type argument specifies the type of the
    output unless --output-dtype is explicitly set.

    The row chunk size is determined automatically from mem_limit
    unless it is specified explicitly.

    The chan chunk size is determined automatically from chan <-> band
    mapping unless it is specified explicitly.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default the gridder will
    use all available resources.

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = 1

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    if LocalCluster:
        nvthreads = nthreads//(nworkers*nthreads_per_worker)
    else:
        nvthreads = nthreads//nthreads-per-worker

    where nvthreads refers to the number of threads used to scale vertically
    (eg. the number threads given to each gridder instance).

    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'degrid_{timestamp}.log')

    from daskms.fsspec_store import DaskMSStore
    msstore = DaskMSStore.from_url_and_kw(opts.ms.rstrip('/'), {})
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

    if opts.product.upper() not in ["I", "Q", "U", "V", "XX", "YX", "XY", "YY", "RR", "RL", "LR", "LL"]:
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
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.distributed import performance_report
    from dask.graph_manipulation import clone
    from daskms.experimental.zarr import xds_from_zarr
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms import xds_to_storage_table as xds_to_table
    import dask.array as da
    from africanus.constants import c as lightspeed
    from africanus.gridding.wgridder.dask import model as im2vis
    from pfb.utils.fits import load_fits, data_from_header
    from pfb.utils.misc import restore_corrs, model_from_comps
    from astropy.io import fits

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'
    mds = xds_from_zarr(mds_name)[0]
    cell_rad = mds.cell_rad
    mfreqs = mds.freq.values
    ref_freq = mfreqs[0]
    model = mds.MODEL.values
    nband, nx, ny = model.shape
    wsums = mds.WSUM.values

    if opts.nband_out is None:
        nband_out = nband
    else:
        nband_out = opts.nband_out

    # TODO - optional grouping. We need to construct an identifier between
    # dataset and field/spw/scan identifiers
    group_by = []
    if opts.group_by_field:
        group_by.append('FIELD_ID')
    else:
        raise NotImplementedError("Grouping by field is currently mandatory")

    if opts.group_by_ddid:
        group_by.append('DATA_DESC_ID')
    else:
        raise NotImplementedError("Grouping by DDID is currently mandatory")

    if opts.group_by_scan:
        group_by.append('SCAN_NUMBER')
    else:
        raise NotImplementedError("Grouping by scan is currently mandatory")

    # chan <-> band mapping
    freqs, fbin_idx, fbin_counts, freq_out, band_mapping, chan_chunks = \
        chan_to_band_mapping(opts.ms, nband=nband_out, group_by=group_by)

    # if mstype == 'zarr':
    #     if opts.model_column in xds[0].keys():
    #         model_chunks = getattr(xds[0], opts.model_column).data.chunks
    #     else:
    #         model_chunks = xds[0].DATA.data.chunks
    #         print('Chunking model same as data', file=log)

    ms_chunks = {}
    ncorr = None
    for ms in opts.ms:
        xds = xds_from_ms(ms, group_cols=group_by)
        ms_chunks[ms] = []  # daskms expects a list per ds

        for ds in xds:
            idt = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"

            if ncorr is None:
                ncorr = ds.dims['corr']
            else:
                try:
                    assert ncorr == ds.dims['corr']
                except Exception as e:
                    raise ValueError("All data sets must have the same number of correlations")

            if opts.row_chunk in [0, -1, None]:
                rchunks = ds.dims['row']
            else:
                rchunks = opts.row_chunk

            ms_chunks[ms].append({'row': rchunks,
                                  'chan': chan_chunks[ms][idt]})

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
    else:
        print("Not fitting frequency axis", file=log)
        comps = beta
        freq_fitted = False

    # this is a hack to get on disk chunks when MODEL_DATA does not exits
    on_disk_chunks = {}
    for ms in opts.ms:
        xds = xds_from_ms(ms)
        rc = xds[0].chunks['row'][0]
        fc = xds[0].chunks['chan'][0]
        cc = xds[0].chunks['corr'][0]
        on_disk_chunks[ms] = {0:rc, 1:fc, 2: cc}


    print("Computing model visibilities", file=log)
    mask = da.from_array(mask, chunks=(nx, ny), name=False)
    comps = da.from_array(comps,
                          chunks=(-1, ncomps), name=False)
    freq_out = da.from_array(freq_out, chunks=-1, name=False)
    writes = []
    radec = None  # assumes we are only imaging field 0 of first MS
    for ms in opts.ms:
        # DATA used to get required type since MODEL_DATA may not exist yet
        xds = xds_from_ms(ms, chunks=ms_chunks[ms], columns=('UVW','DATA'),
                          group_cols=group_by)

        # subtables
        ddids = xds_from_table(ms + "::DATA_DESCRIPTION")
        fields = xds_from_table(ms + "::FIELD")
        spws = xds_from_table(ms + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ms + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]

        out_data = []
        for ds in xds:
            field = fields[ds.FIELD_ID]
            radec = field.PHASE_DIR.data.squeeze()

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            # TODO - rephase if fields don't match, requires PB model
            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            idt = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"

            model = model_from_comps(comps, freq_out, mask,
                                     band_mapping[ms][idt],
                                     ref_freq, freq_fitted)

            uvw = ds.UVW.data
            vis_I = im2vis(uvw,
                           freqs[ms][idt],
                           model,
                           fbin_idx[ms][idt],
                           fbin_counts[ms][idt],
                           cell_rad,
                           nthreads=opts.nvthreads,
                           epsilon=opts.epsilon,
                           do_wstacking=opts.wstack)

            # vis_Q = im2vis(uvw,
            #              freqs[ms][idt],
            #              model[1],
            #              fbin_idx[ms][idt],
            #              fbin_counts[ms][idt],
            #              cell_rad,
            #              nthreads=ngridder_threads,
            #              epsilon=opts.epsilon,
            #              do_wstacking=opts.wstack)

            vis_I = vis_I.astype(ds.DATA.dtype)
            model_vis = restore_corrs(vis_I, ncorr)

            # In case MODEL_DATA does not exist we need to chunk it like DATA
            # if not model_exists[ms]:  # we rechunk
            model_vis = model_vis.rechunk(on_disk_chunks[ms])

            out_ds = ds.assign(**{opts.model_column: (("row", "chan", "corr"), model_vis)})
            out_data.append(out_ds)

        writes.append(xds_to_table(out_data, ms, columns=[opts.model_column]))

    # dask.visualize(*writes, filename=opts.output_filename + '_predict_graph.pdf',
    #                optimize_graph=False, collapse_outputs=True)

    # if not opts.mock:
    #     with performance_report(filename=opts.output_filename + '_predict_per.html'):
    #         dask.compute(writes, optimize_graph=False)

    dask.compute(writes)
    # from pfb.utils.misc import compute_context
    # with compute_context(opts.scheduler, opts.output_filename):
    #     dask.compute(writes,
    #                  optimize_graph=False,
    #                  scheduler=opts.scheduler)

    print("All done here.", file=log)

