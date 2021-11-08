# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('INIT')


@cli.command()
@click.option('-ms', '--ms', required=True,
              help='Path to measurement set.')
@click.option('-dc', '--data-column', default='DATA',
              help="Data or residual column to image."
              "Must be the same across MSs")
@click.option('-wc', '--weight-column', default='WEIGHT_SPECTRUM',
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
@click.option('-p', '--products', default='I',
              help='Stokes products separated by |.'
              'Currently supports I, Q, U, and V eg. '
              'I|Q will produce I and Q products.')
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-rochunk', '--row-out-chunk', type=int, default=10000,
              help="Size of row chunks for output weights and uvw")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
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
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=int,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
def init(**kw):
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
            raise ValueError(f"No gain table at {args.gain_table}")

    # Stokes products are separated by '|' because click doesn't
    # like arbitrary number of inputs
    args.products = args.products.upper().split('|')

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
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig) :
        args.ms = [args.ms]
    OmegaConf.set_struct(args, True)

    import os
    from pathlib import Path
    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.graph_manipulation import clone
    from dask.distributed import performance_report
    from dask.diagnostics import ProgressBar
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_to_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.misc import stitch_images, plan_row_chunk
    from pfb.utils.fits import set_wcs, save_fits
    from pfb.utils.stokes import single_stokes

    # TODO - optional grouping. We need to construct an identifier between
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
    ms = args.ms
    nband = args.nband
    freqs, fbin_idx, fbin_counts, freq_out, band_mapping, chan_chunks = chan_to_band_mapping(
        ms, nband=args.nband, group_by=group_by)

    # gridder memory budget (TODO)
    max_chan_chunk = 0
    max_freq = 0
    for ims in ms:
        for spw in freqs[ims]:
            counts = fbin_counts[ims][spw].compute()
            freq = freqs[ims][spw].compute()
            max_chan_chunk = np.maximum(max_chan_chunk, counts.max())
            max_freq = np.maximum(max_freq, freq.max())

    # assumes measurement sets have the same columns,
    # number of correlations etc.
    xds = xds_from_ms(ms[0])
    ncorr = xds[0].dims['corr']
    nrow = xds[0].dims['row']

    columns = (args.data_column,
               args.weight_column,
               args.flag_column,
               'UVW', 'ANTENNA1',
               'ANTENNA2', 'TIME')
    schema = {}
    schema[args.data_column] = {'dims': ('chan', 'corr')}\
    # TODO - this won't work with WEIGHT column
    schema[args.weight_column] = {'dims': ('chan', 'corr')}
    schema[args.flag_column] = {'dims': ('chan', 'corr')}

    # flag row
    if 'FLAG_ROW' in xds[0]:
        columns += ('FLAG_ROW',)

    # imaging weights
    if args.imaging_weight_column is not None:
        columns += (args.imaging_weight_column,)
        schema[args.imaging_weight_column] = {'dims': ('chan', 'corr')}

    # # Mueller term (complex valued)
    # if args.mueller_column is not None:
    #     columns += (args.mueller_column,)
    #     schema[args.mueller_column] = {'dims': ('chan', 'corr')}

    # get max uv coords over all datasets
    uvw = []
    u_max = 0.0
    v_max = 0.0
    for ims in ms:
        xds = xds_from_ms(ims, columns=('UVW'), chunks={'row': -1})

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
        if npix % 2:
            npix += 1
        nx = npix
        ny = npix
    else:
        nx = args.nx
        ny = args.ny if args.ny is not None else nx

    print(f"Image size set to ({nband}, {nx}, {ny})", file=log)

    nx_psf = int(args.psf_oversize * nx)
    if nx_psf % 2:
        nx_psf += 1

    ny_psf = int(args.psf_oversize * ny)
    if ny_psf % 2:
        ny_psf += 1

    print(f"PSF size set to ({nband}, {nx_psf}, {ny_psf})", file=log)

    row_chunks = {}
    ncorr = None
    for ims in ms:
        xds = xds_from_ms(ims, group_cols=group_by)
        row_chunks[ims] = {}

        for ds in xds:
            idt = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"
            spw = ds.DATA_DESC_ID

            if ncorr is None:
                ncorr = ds.dims['corr']
            else:
                try:
                    assert ncorr == ds.dims['corr']
                except Exception as e:
                    raise ValueError("All data sets must have the same number of correlations")

            if args.row_chunk in [0, -1, None]:
                row_chunks[ims][idt] = (ds.dims['row'],)
            else:
                row_chunks[ims][idt] = (args.row_chunk,)

    chunks = {}
    for ims in ms:
        chunks[ims] = []  # xds_from_ms expects a list per ds
        for idt in freqs[ims]:
            chunks[ims].append({'row': row_chunks[ims][idt],
                                'chan': chan_chunks[ims][idt]})

    dirties = {}
    psfs = {}
    out_datasets = {}
    radec = None  # assumes we are only imaging field 0 of first MS
    for ims in ms:
        xds = xds_from_ms(ims, chunks=chunks[ims], columns=columns,
                          table_schema=schema, group_cols=group_by)

        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD")
        spws = xds_from_table(ims + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ims + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]


        corr_type = set(tuple(pols[0].CORR_TYPE.data.squeeze()))
        if corr_type.issubset(set([9, 10, 11, 12])):
            pol_type = 'linear'
        elif corr_type.issubset(set([5, 6, 7, 8])):
            pol_type = 'circular'
        else:
            raise ValueError(f"Cannot determine polarisation type "
                             f"from correlations {pols[0].CORR_TYPE.data}")

        for ds in xds:
            idt = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"
            field = fields[ds.FIELD_ID]

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            # TODO - need to use spw table
            spw = ds.DATA_DESC_ID

            # MS may contain auto-correlations
            if 'FLAG_ROW' in ds:
                frow = ds.FLAG_ROW.data | (ds.ANTENNA1.data == ds.ANTENNA2.data)
            else:
                frow = (ds.ANTENNA1.data == ds.ANTENNA2.data)

            data = getattr(ds, args.data_column).data

            if args.weight_column is not None:
                weight = getattr(ds, args.weight_column).data
            else:
                weight = None

            if args.imaging_weight_column is not None:
                imaging_weight = getattr(ds, args.imaging_weight_column).data
            else:
                imaging_weight = None

            # if args.mueller_column is not None:
            #     mueller = getattr(ds, args.mueller_column).data.astype(data.dtype)
            # else:
            #     mueller = None

            if args.flag_column is not None:
                flag = getattr(ds, args.flag_column).data
            else:
                flag = None

            uvw = ds.UVW.data
            frow = frow

            universal_opts = {
                'data':data,
                'weight':weight,
                'imaging_weight':imaging_weight,
                'mueller':None,
                'flag':flag,
                'frow':frow,
                'uvw':uvw,
                'time':ds.TIME.data,
                'fid':ds.FIELD_ID,
                'ddid':ds.DATA_DESC_ID,
                'row_out_chunk':args.row_out_chunk,
                'nthreads':args.nvthreads,
                'epsilon':args.epsilon,
                'wstack':args.wstack,
                'double_accum':args.double_accum,
                'freq':freqs[ims][idt],
                'fbin_idx':fbin_idx[ims][idt],
                'fbin_counts':fbin_counts[ims][idt],
                'band_mapping':band_mapping[ims][idt],
                'freq_out':freq_out,
                'nx':nx,
                'ny':ny,
                'nx_psf':nx_psf,
                'ny_psf':ny_psf,
                'cell_rad':cell_rad,
                'radec':radec,
                'do_dirty':args.dirty,
                'do_psf':args.psf,
                'do_weights':args.weights
            }

            if 'I' in args.products:
                # I always has the same pattern
                idx0 = 0
                idxf = -1
                sign = 1.0  # sign to use in sum
                csign = 1.0  # used to negate complex vals
                out_ds_I = single_stokes(idx0=idx0,
                                         idxf=idxf,
                                         sign=sign,
                                         csign=csign,
                                         **universal_opts)

                out_datasets.setdefault('I', [])
                out_datasets['I'].append(out_ds_I)

            if 'Q' in args.products:
                if pol_type.lower() == 'linear':
                    idx0 = 0
                    idxf = -1
                    sign = -1.0
                    csign = 1.0
                elif pol_type.lower() == 'circular':
                    idx0 = 1
                    idxf = 2
                    sign = 1.0
                    csign = 1.0
                out_ds_Q = single_stokes(idx0=idx0,
                                         idxf=idxf,
                                         sign=sign,
                                         csign=csign,
                                         **universal_opts)
                out_datasets.setdefault('Q', [])
                out_datasets['Q'].append(out_ds_Q)

            if 'U' in args.products:
                if pol_type.lower() == 'linear':
                    idx0 = 1
                    idxf = 2
                    sign = 1.0
                    csign = 1
                elif pol_type.lower() == 'circular':
                    idx0 = 1
                    idxf = 2
                    sign = -1
                    csign = 1.0j
                out_ds_U = single_stokes(idx0=idx0,
                                         idxf=idxf,
                                         sign=sign,
                                         csign=csign,
                                         **universal_opts)
                out_datasets.setdefault('U', [])
                out_datasets['U'].append(out_ds_U)

            if 'V' in args.products:
                if pol_type.lower() == 'linear':
                    idx0 = 1
                    idxf = 2
                    sign = -1.0
                    csign = 1.0j
                elif pol_type.lower() == 'circular':
                    idx0 = 0
                    idxf = -1
                    sign = -1.0
                    csign = 1.0
                out_ds_V = single_stokes(idx0=idx0,
                                         idxf=idxf,
                                         sign=sign,
                                         csign=csign,
                                         **universal_opts)
                out_datasets.setdefault('V', [])
                out_datasets['V'].append(out_ds_V)

    writes = {}
    wlist = []  # for visualisation
    for p in args.products:
        if os.path.isdir(args.output_filename + f'_{p}.zarr'):
            print(f"Removing existing {args.output_filename}_{p}.zarr folder",
                  file=log)
            os.system(f"rm -r {args.output_filename}_{p}.zarr")
        writes[p] = xds_to_zarr(out_datasets[p], args.output_filename +
                                f'_{p}.zarr', columns='ALL')
        wlist.append(writes[p])


    dask.visualize(*wlist, color="order", cmap="autumn",
                   node_attr={"penwidth": "4"},
                   filename=args.output_filename + '_writes_ordered_graph.pdf',
                   optimize_graph=False)
    dask.visualize(*wlist, filename=args.output_filename +
                   '_writes_graph.pdf', optimize_graph=False)
    dask.visualize(writes['I'], color="order", cmap="autumn",
                   node_attr={"penwidth": "4"},
                   filename=args.output_filename + '_writes_I_ordered_graph.pdf',
                   optimize_graph=False)
    dask.visualize(writes['I'], filename=args.output_filename +
                   '_writes_I_graph.pdf', optimize_graph=False)

    from pfb.utils.misc import compute_context

    with compute_context(args.scheduler, args.output_filename):
        dask.compute(writes,
                     optimize_graph=False,
                     scheduler=args.scheduler)

    # convert to fits files
    if args.fits_mfs or not args.no_fits_cubes:
        from daskms.experimental.zarr import xds_from_zarr

        if args.dirty:
            print("Saving dirty as fits", file=log)
            dirty = np.zeros((len(args.products), nband, nx, ny), dtype=args.output_type)

            hdr = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec, freq_out)
            hdr_mfs = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec,
                            np.mean(freq_out))

            for i, p in enumerate(sorted(args.products)):
                xds = xds_from_zarr(args.output_filename + f'_{p}.zarr')

                # TODO - add logic to select which spw and scan to reduce over
                dirties = []
                wsums = np.zeros(nband)
                for ds in xds:
                    dirties.append(ds.DIRTY.values)
                    wsums += ds.WSUM.values

                dirty[i] = stitch_images(dirties, nband, band_mapping)

                for b, w in enumerate(wsums):
                    hdr[f'WSUM{p}{b}'] = w
                wsum = np.sum(wsums)
                hdr_mfs[f'WSUM{p}'] = wsum

                dirty_mfs = np.sum(dirty, axis=1, keepdims=True)/wsum

            if args.fits_mfs:
                save_fits(args.output_filename + '_dirty_mfs.fits', dirty_mfs, hdr_mfs,
                        dtype=args.output_type)

            if not args.no_fits_cubes:
                save_fits(args.output_filename + '_dirty.fits', dirty, hdr,
                        dtype=args.output_type)

        if args.psf:
            print("Saving PSF as fits", file=log)
            psf = np.zeros((len(args.products), nband, nx_psf, ny_psf),
                           dtype=args.output_type)

            hdr_psf = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf, ny_psf, radec, freq_out)
            hdr_psf_mfs = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf, ny_psf, radec,
                                np.mean(freq_out))

            for i, p in enumerate(sorted(args.products)):
                xds = xds_from_zarr(args.output_filename + f'_{p}.zarr')

                # TODO - add logic to select which spw and scan to reduce over
                psfs = []
                wsums = np.zeros(nband)
                for ds in xds:
                    psfs.append(ds.PSF.values)
                    wsums += ds.WSUM.values

                psf[i] = stitch_images(psfs, nband, band_mapping)

                for b, w in enumerate(wsums):
                    hdr_psf[f'WSUM{p}{b}'] = w
                wsum = np.sum(wsums)
                hdr_psf_mfs[f'WSUM{p}'] = wsum

                psf_mfs = np.sum(psf, axis=1, keepdims=True)/wsum

            if args.fits_mfs:
                save_fits(args.output_filename + '_psf_mfs.fits', psf_mfs, hdr_psf_mfs,
                        dtype=args.output_type)

            if not args.no_fits_cubes:
                save_fits(args.output_filename + '_psf.fits', psf, hdr_psf,
                        dtype=args.output_type)



    print("All done here.", file=log)
