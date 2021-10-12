# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('NOISE')


@cli.command()
@click.option('-ms', '--ms', required=True,
              help='Path to measurement set.')
@click.option('-ocol', '--output-column', default='NOISE',
              help='Column to write noise realisation to')
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
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
def sim_noise(**kw):
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
        args = set_client(args, stack, log, scheduler=args.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _sim_noise(**args)

def _sim_noise(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig):
        args.ms = [args.ms]
    OmegaConf.set_struct(args, True)

    import numpy as np
    from daskms import xds_from_ms, xds_to_table
    import dask.array as da
    import dask

    xds = xds_from_ms(args.ms[0], columns=('FLAG', 'WEIGHT_SPECTRUM'),
                      chunks={'row':-1}, group_cols=('FIELD_ID',
                      'DATA_DESC_ID', 'SCAN_NUMBER'))
    out_data = []
    for ds in xds:
        w = ds.get('WEIGHT_SPECTRUM')
        f = ds.get('FLAG')

        nrow, nchan, ncorr = f.shape

        nr = da.random.standard_normal(size=(nrow, nchan, ncorr),
                                       chunks=(-1, -1, -1))
        ni = da.random.standard_normal(size=(nrow, nchan, ncorr),
                                       chunks=(-1, -1, -1))
        n = da.where(w > 0, (nr + 1.0j*ni)/da.sqrt(w), 0.0j)

        out_ds = ds.assign(**{args.output_column: (("row", "chan", "corr"), n)})
        out_data.append(out_ds)

    writes = xds_to_table(out_data, args.ms[0], columns=[args.output_column])
    dask.compute(writes)

    print("All done here", file=log)
