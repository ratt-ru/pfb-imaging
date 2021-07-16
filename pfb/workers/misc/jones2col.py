# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('DIRTY')


@cli.command()
@click.option('-ms', '--ms', required=True,
              help='Path to measurement set.')
@click.option('-mc', '--mueller-column', default='CORRECTED_DATA',
              help="Column to write Mueller term to.")
@click.option('-gt', '--gain-table', required=True,
              help="QuartiCal gain table at the same resolution as MS")
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

        return _jones2col(**args)

def _jones2col(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig):
        args.ms = [args.ms]
    OmegaConf.set_struct(args, True)

    from daskms.experimental.zarr import xds_from_zarr
    from daskms import xds_from_ms
    import dask.array as da
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
    times = xds_from_ms(args.ms, columns=['TIME'])[0].get('TIME').data.compute()
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(times, utimes_per_chunk=utpc)

    f_chunks = G[0].f_chunk.data
    if len(f_chunks) > 1:
        f_chunks = G[0].f_chunk.data[1:-1] - G[0].f_chunk.data[0:-2]
        assert (f_chunks == f_chunks[0]).all()
        chan_chunks = f_chunks[0]
    else:
        chan_chunks = f_chunks[0]

    # open MS
    xds = xds_from_ms(args.ms, chunks={'row': row_chunks, 'chan': chan_chunks}
                      columns=('FLAG', 'ANTENNA1', 'ANTENNA2'),
                      group_cols=('FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'))

    # Current hack probably only works for single field and DDID
    try:
        assert len(xds) == len(G)
    except Exception as e:
        raise ValueError("Number of datasets in gains do not "
                            "match those in MS")

    # assuming scans are aligned
    for g, ds in zip(G, xds):
        try:
            assert g.SCAN_NUMBER == ds.SCAN_NUMBER
        except Exception as e:
            raise ValueError("Scans not aligned")

        # need to swap axes for africanus
        gain = da.swapaxes(g.gains.data, (1, 2))


        flag = ds.get('FLAG')
        unitvis = da.ones(flag.shape, chunks=flag.chunks, dtype=gain.dtype)

        cvis = corrupt_vis()


