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
    import xarray as xr
    import sympy as sm
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    mds_name = f'{basename}_{opts.postfix}.coeffs.zarr'
    mds = xds_from_zarr(mds_name)[0]
    cell_rad = mds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    # stack cube
    nx = mds.x.size
    ny = mds.y.size
    x0 = mds.x0
    y0 = mds.y0

    if not np.any(model):
        raise ValueError('Model is empty')
    radec = (mds[0].ra, mds[0].dec)

    ref_freq = mds.ref_freq
    ref_time = mds.ref_times

    params = sm.symbols('t,f')
    params += sm.symbols(tuple(mds.params.values))
    symexpr = parse_expr(mds.parametrisation)
    # signature (t, f, *params)
    modelf = lambdify(params, symexpr)

    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    print('Constructing mapping', file=log)
    row_mapping, freq_mapping, time_mapping, \
        freqs, utimes, ms_chunks, gain_chunks, radecs, \
        chan_widths, uv_max, antpos, poltype = \
            construct_mappings(opts.ms,
                               None,
                               ipi=opts.integrations_per_image,
                               cpi=opts.channels_per_image)


    print("Computing model visibilities", file=log)
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

            # needed for interpolation in time
            utime = da.from_array(utimes[ms][idt],
                                  chunks=opts.integrations_per_image)
            tidx = da.from_array(time_mapping[ms][idt]['start_indices'],
                                 chunks=1)
            tcnts = da.from_array(time_mapping[ms][idt]['counts'],
                                  chunks=1)

            # needed for interpolation in freq
            freq = da.from_array(freqs[ms][idt],
                                 chunks=opts.channels_per_image)
            fidx = da.from_array(freq_mapping[ms][idt]['start_indices'],
                                 chunks=1)
            fcnts = da.from_array(freq_mapping[ms][idt]['counts'],
                                  chunks=1)

            # number of chunks need to math in mapping and coord
            ntime_out = len(tidx.chunks[0])
            assert len(utimes.chunks[0]) == ntime_out
            nfreq_out = len(fidx.chunks[0])
            assert len(freq.chunks[0]) == nfreq_out
            # and they need to match the number of row chunks
            uvw = ds.UVW.data
            assert len(uvw.chunks[0]) == len(tidx.chunks[0])

            # construct design matrix for this SPW
            cpi = opts.channels_per_image
            if nfreq_out == freq.size:
                freq_out = freq
            else:
                freq_out = np.zeros(nfreq_out)
                for b in range(nfreq_out):
                    nf = np.minimum(freq.size, (i+1)*cpi)
                    freq_out[b] = np.mean(freq[i*cpi:nf])

            if freq_fitted:
                w = (freq_out / ref_freq).reshape(nfreq_out, 1)
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

