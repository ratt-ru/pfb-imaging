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
    args = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{args.output_filename}_{args.product}.log')
    from glob import glob
    ms = glob(args.ms)
    try:
        assert len(ms) > 0
        args.ms = ms
    except:
        raise ValueError(f"No MS at {args.ms}")

    if args.nworkers is None:
        if args.scheduler=='distributed':
            args.nworkers = args.nband
        else:
            args.nworkers = 1

    if args.gain_table is not None:
        gt = glob(args.gain_table)
        try:
            assert len(gt) > 0
            args.gain_table = gt
        except Exception as e:
            raise ValueError(f"No gain table  at {args.gain_table}")

    if args.product not in ["I", "Q", "U", "V"]:
        raise NotImplementedError(f"Product {args.product} not yet supported")

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log, scheduler=args.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _init(**args)

def _init(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig):
        args.ms = [args.ms]
    if not isinstance(args.gain_table, list) and not isinstance(args.gain_table, ListConfig):
        args.gain_table = [args.gain_table]
    OmegaConf.set_struct(args, True)

    import os
    from pathlib import Path
    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.graph_manipulation import clone
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from africanus.calibration.utils import chunkify_rows
    from ducc0.fft import good_size
    from pfb.utils.fits import set_wcs, save_fits
    from pfb.utils.stokes import single_stokes
    from pfb.utils.misc import compute_context
    import xarray as xr

    # TODO - optional grouping.
    # We need to construct an identifier between
    # dataset and field/spw/scan identifiers
    group_by = []
    if args.group_by_field:
        group_by.append('FIELD_ID')
    else:
        raise NotImplementedError("Grouping by field is currently mandatory")

    if args.group_by_ddid:
        group_by.append('DATA_DESC_ID')
    else:
        raise NotImplementedError("Grouping by DDID is currently mandatory")

    if args.group_by_scan:
        group_by.append('SCAN_NUMBER')
    else:
        raise NotImplementedError("Grouping by scan is currently mandatory")

    # chan <-> band mapping
    nband = args.nband
    freqs, fbin_idx, fbin_counts, freq_out, band_mapping, chan_chunks = \
        chan_to_band_mapping(args.ms, nband=args.nband, group_by=group_by)

    # gridder memory budget (TODO)
    max_chan_chunk = 0
    max_freq = 0
    for ms in args.ms:
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
    for ms in args.ms:
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
    columns = (args.data_column,
               args.flag_column,
               'UVW', 'ANTENNA1',
               'ANTENNA2', 'TIME', 'INTERVAL')
    schema = {}
    schema[args.data_column] = {'dims': ('chan', 'corr')}
    schema[args.flag_column] = {'dims': ('chan', 'corr')}

    # only WEIGHT column gets special treatment
    # any other column must have channel axis
    if args.weight_column is not None:
        columns += (args.weight_column,)
        if args.weight_column == 'WEIGHT':
            schema[args.weight_column] = {'dims': ('corr')}
        else:
            schema[args.weight_column] = {'dims': ('chan', 'corr')}

    # flag row
    if 'FLAG_ROW' in xds[0]:
        columns += ('FLAG_ROW',)

    ms_chunks = {}
    gain_chunks = {}
    tbin_idx = {}
    tbin_counts = {}
    ncorr = None
    for ims, ms in enumerate(args.ms):
        xds = xds_from_ms(ms, group_cols=group_by, columns=('TIME', 'FLAG'))
        ms_chunks[ms] = []  # daskms expects a list per ds
        gain_chunks[ms] = []
        tbin_idx[ms] = {}
        tbin_counts[ms] = {}
        if args.gain_table[ims] is not None:
            G = xds_from_zarr(args.gain_table[ims].rstrip('/') + '::NET')

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
            if args.utimes_per_chunk in [0, -1, None]:
                utpc = np.unique(time).size
            else:
                utpc = args.utimes_per_chunk

            rchunks, tidx, tcounts = chunkify_rows(time,
                                                   utimes_per_chunk=utpc,
                                                   daskify_idx=True)

            tbin_idx[ms][idt] = tidx
            tbin_counts[ms][idt] = tcounts

            ms_chunks[ms].append({'row': rchunks,
                                  'chan': chan_chunks[ms][idt]})

            if args.gain_table[ims] is not None:
                gain = G[ids]  # TODO - how to make sure they are aligned?
                tmp_dict = {}
                for name, val in zip(gain.GAIN_AXES, gain.GAIN_SPEC):
                    if name == 'gain_t':
                        tmp_dict[name] = (utpc,)
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
    for ims, ms in enumerate(args.ms):
        xds = xds_from_ms(ms, chunks=ms_chunks[ms], columns=columns,
                          table_schema=schema, group_cols=group_by)

        if args.gain_table[ims] is not None:
            G = xds_from_zarr(args.gain_table[ims].rstrip('/') + '::NET',
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


        corr_type = set(tuple(pols[0].CORR_TYPE.data.squeeze()))
        if corr_type.issubset(set([9, 10, 11, 12])):
            pol_type = 'linear'
        elif corr_type.issubset(set([5, 6, 7, 8])):
            pol_type = 'circular'
        else:
            raise ValueError(f"Cannot determine polarisation type "
                             f"from correlations {pols[0].CORR_TYPE.data}")

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
                if args.gain_table[ims] is not None:
                    # Only DI gains currently supported
                    jones = G[ids][{'gain_f': Inu}].gains.data
                else:
                    jones = None

                out_ds = single_stokes(ds=subds,
                                       jones=jones,
                                       args=args,
                                       freq=freqs[ms][idt][Inu],
                                       freq_out=freq_out[band_id],
                                       chan_width=chan_width[Inu],
                                       bandid=band_id,
                                       **universal_opts)
                out_datasets.append(out_ds)

    writes = xds_to_zarr(out_datasets, args.output_filename +
                         f'_{args.product.upper()}.xds.zarr',
                         columns='ALL')

    # dask.visualize(writes, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=args.output_filename + '_writes_I_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(writes, filename=args.output_filename +
    #                '_writes_I_graph.pdf', optimize_graph=False)

    with compute_context(args.scheduler, args.output_filename):
        dask.compute(writes,
                     optimize_graph=False,
                     scheduler=args.scheduler)

    print("All done here.", file=log)
