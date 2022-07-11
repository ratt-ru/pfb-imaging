# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('INIT')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.init["inputs"].keys():
    defaults[key] = schema.init["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.init)
def init(**kw):
    '''
    Create a dirty image, psf and weights from a list of measurement
    sets. MFS images are written out in units of Jy/beam.
    By default only the MFS images are converted to fits files.
    Set the --fits-cubes flag to also produce fits cubes.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. All available resources
    are used by default.

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
    pyscilog.log_to_file(f'{opts.output_filename}_{opts.product}.log')
    from glob import glob
    ms = glob(opts.ms)
    try:
        assert len(ms) > 0
        opts.ms = ms
    except:
        raise ValueError(f"No MS at {opts.ms}")

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    if opts.gain_table is not None:
        gt = glob(opts.gain_table)
        try:
            assert len(gt) > 0
            opts.gain_table = gt
        except Exception as e:
            raise ValueError(f"No gain table  at {opts.gain_table}")

    if opts.product.upper() not in ["I", "Q", "U", "V", "XX", "YX", "XY",
                                    "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _init(**opts)

def _init(**kw):
    opts = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(opts.ms, list) and not isinstance(opts.ms, ListConfig):
        opts.ms = [opts.ms]
    if not isinstance(opts.gain_table, list) and not isinstance(opts.gain_table,
                                                                ListConfig):
        opts.gain_table = [opts.gain_table]
    OmegaConf.set_struct(opts, True)

    import os
    from pathlib import Path
    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from dask.graph_manipulation import clone
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.fits import set_wcs, save_fits
    from pfb.utils.stokes import single_stokes
    from pfb.utils.correlations import single_corr
    from pfb.utils.misc import compute_context, chunkify_rows
    import xarray as xr

    basename = f'{opts.output_filename}_{opts.product}'

    # TODO - optional grouping.
    # We need to construct an identifier between
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
    nband = opts.nband
    freqs, fbin_idx, fbin_counts, freq_out, band_mapping, chan_chunks = \
        chan_to_band_mapping(opts.ms, nband=opts.nband, group_by=group_by)

    # gridder memory budget (TODO)
    max_chan_chunk = 0
    max_freq = 0
    for ms in opts.ms:
        for spw in freqs[ms]:
            counts = fbin_counts[ms][spw].compute()
            freq = freqs[ms][spw].compute()
            max_chan_chunk = np.maximum(max_chan_chunk, counts.max())
            max_freq = np.maximum(max_freq, freq.max())

    # we need the Nyquist limit for setting up the beam interpolation
    # get max uv coords over all datasets
    uvw = []
    u_max = 0.0
    v_max = 0.0
    for ms in opts.ms:
        xds = xds_from_ms(ms, columns='UVW', group_cols=group_by)
        for ds in xds:
            uvw = ds.UVW.data
            u_max = da.maximum(u_max, abs(uvw[:, 0]).max())
            v_max = da.maximum(v_max, abs(uvw[:, 1]).max())
            uv_max = da.maximum(u_max, v_max)

    uv_max = uv_max.compute()
    # approx max cell size
    cell_rad = 1.0 / (2 * uv_max * max_freq / lightspeed)

    # assumes measurement sets have the same columns
    columns = (opts.data_column,
               opts.flag_column,
               'UVW', 'ANTENNA1',
               'ANTENNA2', 'TIME', 'INTERVAL')
    schema = {}
    schema[opts.data_column] = {'dims': ('chan', 'corr')}
    schema[opts.flag_column] = {'dims': ('chan', 'corr')}

    # only WEIGHT column gets special treatment
    # any other column must have channel axis
    if opts.sigma_column is not None:
        print(f"Initialising weights from {opts.sigma_column} column", file=log)
        columns += (opts.sigma_column,)
        schema[opts.sigma_column] = {'dims': ('chan', 'corr')}
    elif opts.weight_column is not None:
        print(f"Using weights from {opts.weight_column} column", file=log)
        columns += (opts.weight_column,)
        if opts.weight_column == 'WEIGHT':
            schema[opts.weight_column] = {'dims': ('corr')}
        else:
            schema[opts.weight_column] = {'dims': ('chan', 'corr')}
    else:
        print(f"No weights provided, using unity weights", file=log)

    # flag row
    if 'FLAG_ROW' in xds[0]:
        columns += ('FLAG_ROW',)

    ms_chunks = {}
    gain_chunks = {}
    tbin_idx = {}
    tbin_counts = {}
    ncorr = None
    for ims, ms in enumerate(opts.ms):
        xds = xds_from_ms(ms, group_cols=group_by, columns=('TIME', 'FLAG'))
        ms_chunks[ms] = []  # daskms expects a list per ds
        gain_chunks[ms] = []
        tbin_idx[ms] = {}
        tbin_counts[ms] = {}
        if opts.gain_table[ims] is not None:
            try:
                G = xds_from_zarr(opts.gain_table[ims].rstrip('/') +
                                  f'::{opts.gain_term}')
            except:
                G = xds_from_zarr(opts.gain_table[ims].rstrip('/') +
                                  f'/{opts.gain_term}')

        for ids, ds in enumerate(xds):
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

            if ncorr is None:
                ncorr = ds.dims['corr']
            else:
                try:
                    assert ncorr == ds.dims['corr']
                except Exception as e:
                    raise ValueError("All data sets must have the same "
                                     "number of correlations")
            time = ds.TIME.values
            if opts.utimes_per_chunk in [0, -1, None]:
                utpc = np.unique(time).size
            else:
                utpc = opts.utimes_per_chunk

            rchunks, tidx, tcounts = chunkify_rows(time,
                                                   utimes_per_chunk=utpc,
                                                   daskify_idx=True)

            tbin_idx[ms][idt] = tidx
            tbin_counts[ms][idt] = tcounts

            ms_chunks[ms].append({'row': rchunks,
                                  'chan': chan_chunks[ms][idt]})

            if opts.gain_table[ims] is not None:
                gain = G[ids]  # TODO - how to make sure they are aligned?
                tmp_dict = {}
                for name, val in zip(gain.GAIN_AXES, gain.GAIN_SPEC):
                    if name == 'gain_t':
                        ntimes = gain.gain_t.size
                        nchunksm1 = ntimes//utpc
                        rem = ntimes - nchunksm1*utpc
                        tmp_dict[name] = (utpc,)*nchunksm1
                        if rem:
                            tmp_dict[name] += (rem,)
                    elif name == 'gain_f':
                        tmp_dict[name] = chan_chunks[ms][idt]
                    elif name == 'dir':
                        if len(val) > 1:
                            raise ValueError("DD gains not supported yet")
                        if val[0] > 1:
                            raise ValueError("DD gains not supported yet")
                        tmp_dict[name] = val
                    else:
                        tmp_dict[name] = val

                gain_chunks[ms].append(tmp_dict)

    out_datasets = []
    radec = None  # assumes we are only imaging field 0 of first MS
    for ims, ms in enumerate(opts.ms):
        xds = xds_from_ms(ms, chunks=ms_chunks[ms], columns=columns,
                          table_schema=schema, group_cols=group_by)

        if opts.gain_table[ims] is not None:
            try:
                G = xds_from_zarr(opts.gain_table[ims].rstrip('/') +
                                  f'::{opts.gain_term}',
                                  chunks=gain_chunks[ms])
            except:
                G = xds_from_zarr(opts.gain_table[ims].rstrip('/') +
                                  f'/{opts.gain_term}',
                                  chunks=gain_chunks[ms])
        # subtables
        ddids = xds_from_table(ms + "::DATA_DESCRIPTION")
        fields = xds_from_table(ms + "::FIELD")
        spws = xds_from_table(ms + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ms + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        # spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]

        # import pdb; pdb.set_trace()
        # corr_type = set(tuple(pols[0].CORR_TYPE.data.squeeze()))
        pol_type='linear'

        # if corr_type.issubset(set([9, 10, 11, 12])):
        #     pol_type = 'linear'
        # elif corr_type.issubset(set([5, 6, 7, 8])):
        #     pol_type = 'circular'
        # else:
        #     raise ValueError(f"Cannot determine polarisation type "
        #                      f"from correlations {pols[0].CORR_TYPE.data}")

        for ids, ds in enumerate(xds):
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"
            nrow = ds.dims['row']
            nchan = ds.dims['chan']
            ncorr = ds.dims['corr']

            field = fields[fid]

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                # TODO - phase shift visibilities
                continue

            spw = spws[ddid]
            chan_width = spw.CHAN_WIDTH.data.squeeze()
            chan_width = chan_width.rechunk(freqs[ms][idt].chunks)

            universal_opts = {
                'tbin_idx':tbin_idx[ms][idt],
                'tbin_counts':tbin_counts[ms][idt],
                'cell_rad':cell_rad,
                'radec':radec
            }

            for b, band_id in enumerate(band_mapping[ms][idt].compute()):
                f0 = fbin_idx[ms][idt][b].compute()
                ff = f0 + fbin_counts[ms][idt][b].compute()
                Inu = slice(f0, ff)

                subds = ds[{'chan': Inu}]
                if opts.gain_table[ims] is not None:
                    # Only DI gains currently supported
                    jones = G[ids][{'gain_f': Inu}].gains.data
                else:
                    jones = None

                if opts.product.upper() in ["I", "Q", "U", "V"]:
                    out_ds = single_stokes(ds=subds,
                                           jones=jones,
                                           opts=opts,
                                           freq=freqs[ms][idt][Inu],
                                           freq_out=freq_out[band_id],
                                           chan_width=chan_width[Inu],
                                           bandid=band_id,
                                           **universal_opts)
                elif opts.product.upper() in ["XX", "YX", "XY", "YY", "RR",
                                              "RL", "LR", "LL"]:
                    out_ds = single_corr(ds=subds,
                                         jones=jones,
                                         opts=opts,
                                         freq=freqs[ms][idt][Inu],
                                         freq_out=freq_out[band_id],
                                         chan_width=chan_width[Inu],
                                         bandid=band_id,
                                         **universal_opts)
                else:
                    raise NotImplementedError(f"Product {args.product} not "
                                              "supported yet")
                # if all data in a dataset is flagged we return None and
                # ignore this chunk of data
                if out_ds is not None:
                    out_datasets.append(out_ds)

    if len(out_datasets):
        writes = xds_to_zarr(out_datasets, f'{basename}.xds.zarr',
                            columns='ALL')
    else:
        raise ValueError('No datasets found to write. '
                         'Data completely flagged maybe?')

    # dask.visualize(writes, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=opts.output_filename + '_writes_I_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(writes, filename=opts.output_filename +
    #                '_writes_I_graph.pdf', optimize_graph=False)

    # import pdb; pdb.set_trace()

    dask.compute(writes, optimize_graph=False)

    print("All done here.", file=log)
