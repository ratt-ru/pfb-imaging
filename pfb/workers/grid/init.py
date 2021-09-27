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
@click.option('-dc', '--data-column',
              help="Data or residual column to image."
              "Must be the same across MSs")
@click.option('-wc', '--weight-column',
              help="Column containing natural weights."
              "Must be the same across MSs")
@click.option('-iwc', '--imaging-weight-column',
              help="Column containing imaging weights. "
              "Must be the same across MSs")
@click.option('-fc', '--flag-column', default='FLAG',
              help="Column containing data flags."
              "Must be the same across MSs")
@click.option('-muc', '--mueller-column',
              help="Column containing Mueller terms."
              "Must be the same across MSs")
@click.option('-p', '--products', default='I',
              help='Imaging products. Currently supports I')
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-rochunk', '--row-out-chunk', type=int, default=10000,
              help="Size of row chunks for output weights and uvw")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
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
def init(**kw):
    '''
    Create a dirty image, psf and weight table from a list of measurement
    sets. Image cubes are not normalised by wsum as this destroyes
    information. MFS images are written out in units of Jy/beam.

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
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

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
    import numpy as np
    from pfb.utils.misc import chan_to_band_mapping
    import dask
    from dask.graph_manipulation import clone
    # from dask.distributed import performance_report
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_to_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from africanus.gridding.wgridder.dask import dirty as vis2im
    from ducc0.fft import good_size
    from pfb.utils.misc import stitch_images, plan_row_chunk
    from pfb.utils.fits import set_wcs, save_fits

    # chan <-> band mapping
    ms = args.ms
    nband = args.nband
    freqs, fbin_idx, fbin_counts, freq_out, band_mapping, chan_chunks = chan_to_band_mapping(
        ms, nband=args.nband)

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

    # Mueller term (complex valued)
    if args.mueller_column is not None:
        columns += (args.mueller_column,)
        schema[args.mueller_column] = {'dims': ('chan', 'corr')}

    # get max uv coords over all fields
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

    if args.row_chunk in [0, -1, None]:
        row_chunk = nrow
    else:
        row_chunk = args.row_chunk

    print(f"nrows = {nrow}, row chunks set to {row_chunk} for a total of "
          f"{int(np.ceil(nrow / row_chunk))} chunks", file=log)

    chunks = {}
    for ims in ms:
        chunks[ims] = []  # xds_from_ms expects a list per ds
        for spw in freqs[ims]:
            chunks[ims].append({'row': row_chunk,
                                'chan': chan_chunks[ims][spw]['chan']})

    dirties = []
    psfs = []
    out_datasets = []
    radec = None  # assumes we are only imaging field 0 of first MS
    for ims in ms:
        xds = xds_from_ms(ims, chunks=chunks[ims], columns=columns)

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

        for ds in xds:
            field = fields[ds.FIELD_ID]

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            # TODO - need to use spw table
            spw = ds.DATA_DESC_ID

            # uvw = clone(ds.UVW.data)

            if args.products.lower() == 'i':
                dirty, psf, out_ds = _I(ds, args,
                                        freqs[ims][spw],
                                        fbin_idx[ims][spw],
                                        fbin_counts[ims][spw],
                                        nx, ny, nx_psf, ny_psf, cell_rad)
                dirties.append(dirty)
                psfs.append(psf)
                out_datasets.append(out_ds)
            else:
                raise NotImplementedError("Sorry, not yet")

    # remove dataset if it exists
    if os.path.isdir(args.output_filename + '.zarr'):
        print(f"Removing existing {args.output_filename}.zarr folder", file=log)
        os.system(f"rm -r {args.output_filename}.zarr")

    writes = xds_to_zarr(out_datasets, args.output_filename + '.zarr', columns='ALL')

    result = dask.compute(dirties, psfs, writes, optimize_graph=False)

    dirties = result[0]

    dirty = stitch_images(dirties, nband, band_mapping)

    psfs = result[1]

    psf = stitch_images(psfs, nband, band_mapping)
    wsums = np.amax(psf, axis=(1, 2))
    psf_mfs = np.sum(psf, axis=0)
    wsum = psf_mfs.max()
    psf_mfs /= wsum

    dirty_mfs = np.sum(dirty, axis=0)/wsum

    # save dirty
    hdr = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec, freq_out)
    for i, w in enumerate(wsums):
        hdr[f'WSUM{i}'] = w
    save_fits(args.output_filename + '_dirty.fits', dirty, hdr,
              dtype=args.output_type)
    hdr_mfs = set_wcs(cell_size / 3600, cell_size / 3600, nx, ny, radec,
                      freq_out[0])
    save_fits(args.output_filename + '_dirty_mfs.fits', dirty_mfs, hdr_mfs,
              dtype=args.output_type)

    # save psf
    hdr = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf, ny_psf,
                  radec, freq_out)
    for i, w in enumerate(wsums):
        hdr[f'WSUM{i}'] = w
    save_fits(args.output_filename + '_psf.fits', psf, hdr,
              dtype=args.output_type)
    hdr_mfs = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf, ny_psf,
                      radec, freq_out[0])
    save_fits(args.output_filename + '_psf_mfs.fits', psf_mfs, hdr_mfs,
              dtype=args.output_type)

    print("All done here.", file=log)


def _I(ds, args, freq, fbin_idx, fbin_counts, nx, ny,
       nx_psf, ny_psf, cell_rad):
    import numpy as np
    import dask.array as da
    from xarray import Dataset
    from africanus.gridding.wgridder.dask import dirty as vis2im
    data = getattr(ds, args.data_column).data
    dataxx = data[:, :, 0]
    datayy = data[:, :, -1]

    data_type = getattr(ds, args.data_column).data.dtype
    data_shape = getattr(ds, args.data_column).data.shape
    data_chunks = getattr(ds, args.data_column).data.chunks

    weights = getattr(ds, args.weight_column).data

    if args.imaging_weight_column is not None:
        imaging_weights = getattr(
            ds, args.imaging_weight_column).data

        weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
        weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
    else:
        weightsxx = weights[:, :, 0]
        weightsyy = weights[:, :, -1]

    # adjoint of mueller term
    if args.mueller_column is not None:
        mueller = getattr(ds, args.mueller_column).data
        data = (dataxx *  mueller[:, :, 0].conj() * weightsxx +
                datayy * mueller[:, :, -1].conj() * weightsyy)
        weightsxx *= da.absolute(mueller[:, :, 0])**2
        weightsyy *= da.absolute(mueller[:, :, -1])**2

    else:
        data = weightsxx * dataxx + weightsyy * datayy

    weights = weightsxx + weightsyy
    # TODO - turn off this stupid warning
    data = da.where(weights, data / weights, 0.0j)
    real_type = data.real.dtype

    # MS may contain auto-correlations
    if 'FLAG_ROW' in ds:
        frow = ds.FLAG_ROW.data | (ds.ANTENNA1.data == ds.ANTENNA2.data)
    else:
        frow = (ds.ANTENNA1.data == ds.ANTENNA2.data)

    # only keep data where both corrs are unflagged
    flag = getattr(ds, args.flag_column).data
    flagxx = flag[:, :, 0]
    flagyy = flag[:, :, -1]
    # ducc0 uses uint8 mask not flag
    mask = ~ da.logical_or((flagxx | flagyy), frow[:, None])
    uvw = ds.UVW.data
    dirty = vis2im(uvw,
                   freq,
                   data,
                   fbin_idx,
                   fbin_counts,
                   nx,
                   ny,
                   cell_rad,
                   weights=weights,
                   flag=mask.astype(np.uint8),
                   nthreads=args.nvthreads,
                   epsilon=args.epsilon,
                   do_wstacking=args.wstack,
                   double_accum=args.double_accum)

    psf = vis2im(uvw,
                 freq,
                 weights.astype(data_type),
                 fbin_idx,
                 fbin_counts,
                 nx_psf,
                 ny_psf,
                 cell_rad,
                 flag=mask.astype(np.uint8),
                 nthreads=args.nvthreads,
                 epsilon=args.epsilon,
                 do_wstacking=args.wstack,
                 double_accum=args.double_accum)

    weights = da.where(mask, weights, 0.0)

    data_vars = {
                'FIELD_ID':(('row',), da.full_like(ds.TIME.data,
                            ds.FIELD_ID, chunks=args.row_out_chunk)),
                'DATA_DESC_ID':(('row',), da.full_like(ds.TIME.data,
                            ds.DATA_DESC_ID, chunks=args.row_out_chunk)),
                'WEIGHT':(('row', 'chan'), weights.rechunk({0:args.row_out_chunk})),  # why no 'f4'?
                'UVW':(('row', 'uvw'), uvw.rechunk({0:args.row_out_chunk}))
            }

    coords = {
        'chan': (('chan',), freq)
    }

    out_ds = Dataset(data_vars, coords)

    return dirty, psf, out_ds

def _Q(ds, args, uvw, freq, fbin_idx, fbin_counts, nx, ny, cell_rad):
    data = getattr(ds, args.data_column).data
    dataxx = data[:, :, 0]
    datayy = data[:, :, -1]

    weights = getattr(ds, args.weight_column).data
    if len(weights.shape) < 3:
        weights = da.broadcast_to(
            weights[:, None, :], data.shape, chunks=data.chunks)

    if args.imaging_weight_column is not None:
        imaging_weights = getattr(
            ds, args.imaging_weight_column).data
        if len(imaging_weights.shape) < 3:
            imaging_weights = da.broadcast_to(
                imaging_weights[:, None, :], data.shape, chunks=data.chunks)

        weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
        weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
    else:
        weightsxx = weights[:, :, 0]
        weightsyy = weights[:, :, -1]

    # apply adjoint of mueller term.
    if args.mueller_column is not None:
        mueller = getattr(ds, args.mueller_column).data
        data = (dataxx *  mueller[:, :, 0].conj() * weightsxx -
                datayy * mueller[:, :, -1].conj() * weightsyy)
        weightsxx *= da.absolute(mueller[:, :, 0])**2
        weightsyy *= da.absolute(mueller[:, :, -1])**2

    else:
        data = weightsxx * dataxx - weightsyy * datayy

    # weighted sum corr to Stokes I
    weights = weightsxx + weightsyy
    # TODO - turn off this stupid warning
    data = da.where(weights, data / weights, 0.0j)

    # MS may contain auto-correlations
    if 'FLAG_ROW' in ds:
        frow = ds.FLAG_ROW.data | (ds.ANTENNA1.data == ds.ANTENNA2.data)
    else:
        frow = (ds.ANTENNA1.data == ds.ANTENNA2.data)

    # only keep data where both corrs are unflagged
    flag = getattr(ds, args.flag_column).data
    flagxx = flag[:, :, 0]
    flagyy = flag[:, :, -1]
    # ducc0 uses uint8 mask not flag
    mask = ~ da.logical_or((flagxx | flagyy), frow[:, None])

    dirty = vis2im(uvw,
                   freq,
                   data,
                   fbin_idx,
                   fbin_counts,
                   nx,
                   ny,
                   cell_rad,
                   weights=weights,
                   flag=mask.astype(np.uint8),
                   nthreads=args.nvthreads,
                   epsilon=args.epsilon,
                   do_wstacking=args.wstack,
                   double_accum=args.double_accum)

    return dirty

def _IQ(ds, args, uvw, freq, fbin_idx, fbin_counts, nx, ny, cell_rad):
    data = getattr(ds, args.data_column).data
    dataxx = data[:, :, 0]
    datayy = data[:, :, -1]

    weights = getattr(ds, args.weight_column).data
    if len(weights.shape) < 3:
        weights = da.broadcast_to(
            weights[:, None, :], data.shape, chunks=data.chunks)

    if args.imaging_weight_column is not None:
        imaging_weights = getattr(
            ds, args.imaging_weight_column).data
        if len(imaging_weights.shape) < 3:
            imaging_weights = da.broadcast_to(
                imaging_weights[:, None, :], data.shape, chunks=data.chunks)

        weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
        weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
    else:
        weightsxx = weights[:, :, 0]
        weightsyy = weights[:, :, -1]

    # apply adjoint of mueller term.
    if args.mueller_column is not None:
        mueller = getattr(ds, args.mueller_column).data
        dataxx *= mueller[:, :, 0].conj() * weightsxx
        datayy *= mueller[:, :, -1].conj() * weightsyy
        weightsxx *= da.absolute(mueller[:, :, 0])**2
        weightsyy *= da.absolute(mueller[:, :, -1])**2

    else:
        dataxx *= weightsxx
        datayy *= weightsyy

    # MS may contain auto-correlations
    if 'FLAG_ROW' in ds:
        frow = ds.FLAG_ROW.data | (ds.ANTENNA1.data == ds.ANTENNA2.data)
    else:
        frow = (ds.ANTENNA1.data == ds.ANTENNA2.data)

    # only keep data where both corrs are unflagged
    flag = getattr(ds, args.flag_column).data
    flagxx = flag[:, :, 0]
    flagyy = flag[:, :, -1]
    # ducc0 uses uint8 mask not flag
    mask = (~ da.logical_or((flagxx | flagyy), frow[:, None])).astype(np.uint8)

    dirtyxx = vis2im(uvw,
                     freq,
                     dataxx,
                     fbin_idx,
                     fbin_counts,
                     nx,
                     ny,
                     cell_rad,
                     weights=weightsxx,
                     flag=mask,
                     nthreads=args.nvthreads,
                     epsilon=args.epsilon,
                     do_wstacking=args.wstack,
                     double_accum=args.double_accum)

    dirtyyy = vis2im(uvw,
                     freq,
                     datayy,
                     fbin_idx,
                     fbin_counts,
                     nx,
                     ny,
                     cell_rad,
                     weights=weightsyy,
                     flag=mask,
                     nthreads=args.nvthreads,
                     epsilon=args.epsilon,
                     do_wstacking=args.wstack,
                     double_accum=args.double_accum)

    return dirty