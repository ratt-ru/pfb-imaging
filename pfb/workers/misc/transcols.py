# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('TRANSCOLS')


@cli.command()
@click.option('-ms1', '--ms1', required=True,
              help='Path to measurement set containing cols to transfer.')
@click.option('-ms2', '--ms2', required=True,
              help='Path to target measurement.')
@click.option('-cols', '--columns', required=True, type=str,
              help='Comma seperated list of columns to transfer')
@click.option('-rc', '--row-chunk', type=int, default=10000)
@click.option('-cc', '--chan-chunk', type=int, default=32)
@click.option('-o', '--output-filename', type=str,
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
def transcols(**kw):
    '''
    Write product of diagonal Jones matrices to 'Mueller' column
    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
    from glob import glob
    ms1 = glob(args.ms1)
    try:
        assert len(ms1) == 1
        args.ms1 = ms1
    except:
        raise ValueError(f"There must be exactly one MS at {args.ms1}")

    ms2 = glob(args.ms2)
    try:
        assert len(ms2) == 1
        args.ms2 = ms2
    except:
        raise ValueError(f"There must be exactly one MS at {args.ms2}")

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _transcols(**args)

def _transcols(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms1, list) and not isinstance(args.ms1, ListConfig):
        args.ms1 = [args.ms1]
    if not isinstance(args.ms2, list) and not isinstance(args.ms2, ListConfig):
        args.ms2 = [args.ms2]

    OmegaConf.set_struct(args, True)

    from daskms import xds_from_ms, xds_to_table
    import dask

    columns = tuple(args.columns.split(','))

    xds1 = xds_from_ms(args.ms1[0],
                       chunks={'row': args.row_chunk,
                               'chan': args.chan_chunk},
                       columns=columns)
    xds2 = xds_from_ms(args.ms2[0],
                       chunks={'row': args.row_chunk,
                               'chan': args.chan_chunk})

    out_data = []
    for ds1, ds2 in zip(xds1, xds2):
        for column in columns:
            data = ds1.get(column).data
            out_ds = ds2.assign(**{column: (("row", "chan", "corr"), data)})
            out_data.append(out_ds)

    writes = xds_to_table(out_data, args.ms2[0], columns=columns)
    dask.compute(writes)
