from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FLEDGES')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.fledges["inputs"].keys():
    defaults[key] = schema.fledges["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.fledges)
def fledges(**kw):
    '''
    Routine to interpolate gains defined on a grid onto a measurement set
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _fledges(**opts)

def _fledges(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import dask
    import dask.array as da
    from daskms import xds_from_ms, xds_to_table

    xds = xds_from_ms(opts.ms, columns='FLAG', chunks={'row':opts.row_chunk},
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])
    writes = []
    for ds in xds:
        flag = ds.FLAG.data
        flag = da.blockwise(set_flags, 'rfc',
                            flag, 'rfc',
                            opts.chan_lower, None,
                            opts.chan_upper, None,
                            dtype=bool)

        ds = ds.assign(**{'FLAG': (('row','chan','corr'), flag)})

        writes.append(xds_to_table(ds, opts.ms, columns="FLAG"))

    dask.compute(writes)




def set_flags(flag, chan_lower, chan_upper):
    flag[:, 0:chan_lower, :] = True
    flag[:, -chan_upper:, :] = True
    return flag
