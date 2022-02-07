# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('GRID')


@cli.command(context_settings={'show_default': True})
@click.option('-ms', '--ms', required=True,
              help='Path to measurement set.')
@click.option('-dc', '--data-column', default='DATA',
              help="Data or residual column to image."
              "Must be the same across MSs")
@click.option('-wc', '--weight-column', default=None,
              help="Column containing natural weights."
              "Must be the same across MSs")
@click.option('-iwc', '--imaging-weight-column',
              help="Column containing imaging weights. "
              "Must be the same across MSs")
@click.option('-fc', '--flag-column', default='FLAG',
              help="Column containing data flags."
              "Must be the same across MSs")
@click.option('-gt', '--gain-table',
              help="Path to Quartical gain table containing NET gains."
              "There must be a table for each MS and glob(ms) and glob(gt) "
              "should match up.")
@click.option('-p', '--product', default='I',
              help='Currently supports I, Q, U, and V. '
              'Only single Stokes products currently supported.')
@click.option('-utpc', '--utimes-per-chunk', type=int, default=-1,
              help="Number of unique times in a chunk.")
@click.option('-rochunk', '--row-out-chunk', type=int, default=10000,
              help="Size of row chunks for output weights and uvw")
@click.option('-eps', '--epsilon', type=float, default=1e-7,
              help='Gridder accuracy')
@click.option('-precision', '--precision', default='double',
              help='Either single or double')
@click.option('--group-by-field/--no-group-by-field', default=True)
@click.option('--group-by-ddid/--no-group-by-ddid', default=True)
@click.option('--group-by-scan/--no-group-by-scan', default=True)
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('--fits-mfs/--no-fits-mfs', default=True)
@click.option('--no-fits-cubes/--fits-cubes', default=True)
@click.option('--psf/--no-psf', default=True)
@click.option('--dirty/--no-dirty', default=True)
@click.option('--weights/--no-weights', default=True)
@click.option('--bda-weights/--no-bda-weights', default=False)
@click.option('--do-beam/--no-do-beam', default=False)
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-fov', '--field-of-view', type=float,
              help="Field of view in degrees")
@click.option('-srf', '--super-resolution-factor', type=float,
              help="Will over-sample Nyquist by this factor at max frequency")
@click.option('-psfo', '--psf-oversize', type=float, default=2.0,
              help='Size of the PSF relative to the dirty image.')
@click.option('-cs', '--cell-size', type=float,
              help='Cell size in arcseconds')
@click.option('-nx', '--nx', type=int,
              help="Number of x pixels")
@click.option('-ny', '--ny', type=int,
              help="Number of x pixels")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling "
                   "(eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=int,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. "
              "Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="distributed or single-threaded (for debugging)")
def grid(**kw):
    '''
    Create a dirty image, psf and weights from a list of measurement
    sets. Image cubes are not normalised by wsum as this destroyes
    information. MFS images are written out in units of Jy/beam.
    By default only the MFS images are converted to fits files.
    Set the --fits-cubes flag to also produce fits cubes.

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
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
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

        return _grid(**args)

def _grid(**kw):
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
    from pfb.utils.misc import stitch_images
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

    # assumes measurement sets have the same columns
    xds = xds_from_ms(args.ms[0])
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

    # imaging weights
    if args.imaging_weight_column is not None:
        columns += (args.imaging_weight_column,)
        schema[args.imaging_weight_column] = {'dims': ('chan', 'corr')}

    # get max uv coords over all datasets
    uvw = []
    u_max = 0.0
    v_max = 0.0
    for ms in args.ms:
        xds = xds_from_ms(ms, columns=('UVW'), chunks={'row': -1},
                          group_cols=group_by)

        for ds in xds:
            uvw = ds.UVW.data
            u_max = da.maximum(u_max, abs(uvw[:, 0]).max())
            v_max = da.maximum(v_max, abs(uvw[:, 1]).max())
            uv_max = da.maximum(u_max, v_max)

    uv_max = uv_max.compute()
    del uvw

    # image size
    cell_N = 1.0 / (2 * uv_max * max_freq / lightspeed)

    if args.cell_size is not None:
        cell_size = args.cell_size
        cell_rad = cell_size * np.pi / 60 / 60 / 180
        if cell_N / cell_rad < 1:
            raise ValueError("Requested cell size too small. "
                             "Super resolution factor = ", cell_N / cell_rad)
        print(f"Super resolution factor = {cell_N/cell_rad}", file=log)
    else:
        cell_rad = cell_N / args.super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi
        print(f"Cell size set to {cell_size} arcseconds", file=log)

    if args.nx is None:
        fov = args.field_of_view * 3600
        npix = int(fov / cell_size)
        npix = good_size(npix)
        if npix % 2:
            npix += 1
        nx = npix
        ny = npix
    else:
        nx = args.nx
        ny = args.ny if args.ny is not None else nx

    print(f"Image size set to ({nband}, {nx}, {ny})", file=log)

    nx_psf = good_size(int(args.psf_oversize * nx))
    if nx_psf % 2:
        nx_psf += 1

    ny_psf = good_size(int(args.psf_oversize * ny))
    if ny_psf % 2:
        ny_psf += 1

    print(f"PSF size set to ({nband}, {nx_psf}, {ny_psf})", file=log)

    ms_chunks = {}
    gain_chunks = {}
    tbin_idx = {}
    tbin_counts = {}
    ncorr = None
    for ims, ms in enumerate(args.ms):
        xds = xds_from_ms(ms, group_cols=group_by)
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
                'nx':nx,
                'ny':ny,
                'nx_psf':nx_psf,
                'ny_psf':ny_psf,
                'cell_rad':cell_rad,
                'radec':radec
            }

            nband = fbin_idx[ms][idt].size
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

    print("Initialising model", file=log)
    # TODO - allow non-zero input model
    attrs = {'nband': nband,
             'nx': nx,
             'ny': ny,
             'ra': radec[0],
             'dec': radec[1],
             'cell_rad': cell_rad}
    coords = {'freq': freq_out}
    real_type = np.float64 if args.precision=='double' else np.float32
    model = da.zeros((nband, nx, ny), chunks=(1, -1, -1), dtype=real_type)
    data_vars = {'MODEL': (('band', 'x', 'y'), model),
                 'MASK': (('x', 'y'), np.zeros((nx, ny), dtype=bool))}
    mds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    mds_name = f'{args.output_filename}_{args.product.upper()}.mds.zarr'
    dask.compute(xds_to_zarr([mds], mds_name,columns='ALL'))

    # convert to fits files
    if args.fits_mfs or not args.no_fits_cubes:
        if args.dirty:
            print("Saving dirty as fits", file=log)
            dirty = np.zeros((nband, nx, ny), dtype=np.float32)
            wsums = np.zeros(nband, dtype=np.float32)

            hdr = set_wcs(cell_size / 3600, cell_size / 3600,
                          nx, ny, radec, freq_out)
            hdr_mfs = set_wcs(cell_size / 3600, cell_size / 3600,
                              nx, ny, radec, np.mean(freq_out))

            xds = xds_from_zarr(args.output_filename +
                                f'_{args.product.upper()}.xds.zarr')

            for ds in xds:
                b = ds.bandid
                dirty[b] += ds.DIRTY.values
                wsums[b] += ds.WSUM.values

            for b, w in enumerate(wsums):
                hdr[f'WSUM{b}'] = w
            wsum = np.sum(wsums)
            hdr_mfs[f'WSUM'] = wsum

            dirty_mfs = np.sum(dirty, axis=0, keepdims=True)/wsum

            if args.fits_mfs:
                save_fits(args.output_filename +
                          f'_{args.product.upper()}_dirty_mfs.fits',
                          dirty_mfs, hdr_mfs, dtype=np.float32)

            if not args.no_fits_cubes:
                fmask = wsums > 0
                dirty[fmask] /= wsums[fmask, None, None]
                save_fits(args.output_filename +
                          f'_{args.product.upper()}_dirty.fits', dirty, hdr,
                        dtype=np.float32)

        if args.psf:
            print("Saving PSF as fits", file=log)
            psf = np.zeros((nband, nx_psf, ny_psf), dtype=np.float32)
            wsums = np.zeros(nband, dtype=np.float32)

            hdr_psf = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf,
                              ny_psf, radec, freq_out)
            hdr_psf_mfs = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf,
                                  ny_psf, radec, np.mean(freq_out))

            xds = xds_from_zarr(args.output_filename +
                                f'_{args.product.upper()}.xds.zarr')

            # TODO - add logic to select which spw and scan to reduce over
            for ds in xds:
                b = ds.bandid
                psf[b] += ds.PSF.values
                wsums[b] += ds.WSUM.values

            for b, w in enumerate(wsums):
                hdr_psf[f'WSUM{b}'] = w
            wsum = np.sum(wsums)
            hdr_psf_mfs[f'WSUM'] = wsum

            psf_mfs = np.sum(psf, axis=0, keepdims=True)/wsum

            if args.fits_mfs:
                save_fits(args.output_filename +
                          f'_{args.product.upper()}_psf_mfs.fits', psf_mfs,
                          hdr_psf_mfs, dtype=np.float32)

            if not args.no_fits_cubes:
                fmask = wsums > 0
                psf[fmask] /= wsums[fmask, None, None]
                save_fits(args.output_filename +
                          f'_{args.product.upper()}_psf.fits', psf, hdr_psf,
                        dtype=np.float32)

    print("All done here.", file=log)
