from contextlib import ExitStack
from pfb.workers.experimental import cli
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
    Apply a frequency mask
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'fledges_{timestamp}.log')
    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:


        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _fledges(**opts)

def _fledges(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from multiprocessing.pool import ThreadPool
    import dask
    dask.config.set(pool=ThreadPool(opts.nthreads))
    import dask.array as da
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_to_storage_table as xds_to_table
    import numpy as np
    import re

    xds = xds_from_ms(opts.ms, columns='FLAG', chunks={'row':opts.row_chunk},
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

    # 1419.8:1421.3 <=> 2697:2705
    I = np.zeros(xds[0].chan.size, dtype=bool)
    for idx in opts.franges.split(','):
        m = re.match('(-?\d+)?:(-?\d+)?', idx)
        ilow = int(m.group(1)) if m.group(1) is not None else None
        ihigh = int(m.group(2)) if m.group(2) is not None else None
        I[slice(ilow, ihigh)] = True

    I = da.from_array(I, chunks=-1)
    writes = []
    for ds in xds:
        flag = ds.FLAG.data
        flag = da.blockwise(set_flags, 'rfc',
                            flag, 'rfc',
                            I, 'f',
                            dtype=bool)

        dso = ds.assign(**{'FLAG': (('row','chan','corr'), flag)})

        writes.append(xds_to_table(dso, opts.ms, columns="FLAG", rechunk=True))

    dask.compute(writes)

def set_flags(flag, I):
    flag[:, I, :] = True
    return flag
