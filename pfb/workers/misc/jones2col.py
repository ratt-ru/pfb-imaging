# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('J2COL')


@cli.command()
@click.option('-ms', '--ms', required=True,
              help='Path to measurement set.')
@click.option('-mc', '--mueller-column', default='CORRECTED_DATA',
              help="Column to write Mueller term to.")
@click.option('-gt', '--gain-table', required=True,
              help="QuartiCal gain table at the same resolution as MS")
@click.option('-acol', '--acol',
              help='Will apply gains to this column if supplied')
@click.option('-c2', '--compareto',
              help="Will compare corrupted vis to this column if provided. "
              "Mainly useful for testing.")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int, default=1,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int, default=1,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=int,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
def jones2col(**kw):
    '''
    Write product of diagonal Jones matrices to 'Mueller' column
    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
    from glob import glob
    ms = glob(args.ms)
    try:
        assert len(ms) == 1
        args.ms = ms
    except:
        raise ValueError(f"There must be exactly one MS at {args.ms}")

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _jones2col(**args)

def _jones2col(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig):
        args.ms = [args.ms]
    OmegaConf.set_struct(args, True)

    import numpy as np
    from daskms.experimental.zarr import xds_from_zarr
    from daskms import xds_from_ms, xds_to_table
    import dask.array as da
    import dask
    from africanus.calibration.utils import chunkify_rows
    from africanus.calibration.utils.dask import corrupt_vis

    # get net gains
    G = xds_from_zarr(args.gain_table + '::NET')

    # chunking info
    t_chunks = G[0].t_chunk.data
    if len(t_chunks) > 1:
        t_chunks = G[0].t_chunk.data[1:-1] - G[0].t_chunk.data[0:-2]
        assert (t_chunks == t_chunks[0]).all()
        utpc = t_chunks[0]
    else:
        utpc = t_chunks[0]
    times = xds_from_ms(args.ms[0], columns=['TIME'])[0].get('TIME').data.compute()
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(
                                        times,
                                        utimes_per_chunk=utpc)
    tbin_idx = da.from_array(tbin_idx, chunks=1)
    tbin_counts = da.from_array(tbin_counts, chunks=1)

    f_chunks = G[0].f_chunk.data
    if len(f_chunks) > 1:
        f_chunks = G[0].f_chunk.data[1:-1] - G[0].f_chunk.data[0:-2]
        assert (f_chunks == f_chunks[0]).all()
        chan_chunks = f_chunks[0]
    else:
        if f_chunks[0]:
            chan_chunks = f_chunks[0]
        else:
            chan_chunks = -1

    columns = ('DATA', 'FLAG', 'FLAG_ROW', 'ANTENNA1', 'ANTENNA2')
    if args.acol is not None:
        columns += (args.acol,)

    # open MS
    xds = xds_from_ms(args.ms[0], chunks={'row': row_chunks, 'chan': chan_chunks},
                      columns=columns,
                      group_cols=('FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'))

    # Current hack probably only works for single field and DDID
    try:
        assert len(xds) == len(G)
    except Exception as e:
        raise ValueError("Number of datasets in gains do not "
                            "match those in MS")

    # assuming scans are aligned
    out_data = []
    for g, ds in zip(G, xds):
        try:
            assert g.SCAN_NUMBER == ds.SCAN_NUMBER
        except Exception as e:
            raise ValueError("Scans not aligned")

        nrow = ds.dims['row']
        nchan = ds.dims['chan']
        ncorr = ds.dims['corr']

        # need to swap axes for africanus
        jones = da.swapaxes(g.gains.data, 1, 2)
        flag = ds.FLAG.data
        frow = ds.FLAG_ROW.data
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data

        frow = (frow | (ant1 == ant2))
        flag = (flag[:, :, 0] | flag[:, :, -1])
        flag = da.logical_or(flag, frow[:, None])

        if args.acol is not None:
            acol = ds.get(args.acol).data.reshape(nrow, nchan, 1, ncorr)
        else:
            acol = da.ones((nrow, nchan, 1, ncorr),
                           chunks=(row_chunks, chan_chunks, 1, -1),
                           dtype=jones.dtype)

        cvis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, acol)

        # compare where unflagged
        if args.compareto is not None:
            flag = flag.compute()
            vis = ds.get(args.compareto).values[~flag]
            print("Max abs difference = ", np.abs(cvis.compute()[~flag] - vis).max())
            quit()

        out_ds = ds.assign(**{args.mueller_column: (("row", "chan", "corr"), cvis)})
        out_data.append(out_ds)

    writes = xds_to_table(out_data, args.ms[0], columns=[args.mueller_column])
    dask.compute(writes)







