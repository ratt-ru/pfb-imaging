from contextlib import ExitStack
from pfb.workers.experimental import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('IFT2QC')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.ift2qc["inputs"].keys():
    defaults[key] = schema.ift2qc["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.ift2qc)
def ift2qc(**kw):
    '''
    Routine to interpolate gains defined on a grid onto a measurement set
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'ift2qc_{timestamp}.log')

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

        return _ift2qc(**opts)

def _ift2qc(**kw):  # '/home/landman/projects/ESO137/subsets/qcal/out/ift_gains/20220322_primary_subset_gains.npz'
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import dask
    import dask.array as da
    from daskms import xds_from_ms, xds_from_table
    from daskms.experimental.zarr import xds_to_zarr
    from pfb.utils.misc import interp_gain_grid, array2qcal_ds

    # get interpolation objects
    ms_name = opts.ms.rstrip('/')
    ant_names = xds_from_table(f'{ms_name}::ANTENNA')[0].NAME.values
    gains = np.load(opts.gains, allow_pickle=True)
    gobj_amp, gobj_phase = interp_gain_grid(gains, ant_names)
    # gobj_amp = da.from_array(gobj_amp, chunks=(-1, -1, -1))
    # gobj_phase = da.from_array(gobj_phase, chunks=(-1, -1, -1))

    # FIXME - in general we can't assume that these properties are the same for each dataset
    ms_freq = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0].CHAN_FREQ.values.squeeze()
    fname = xds_from_table(f'{ms_name}::FIELD')[0].NAME.values[0]
    xds = xds_from_ms(opts.ms, columns='TIME',
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])
    writes = []
    for ds in xds:
        # interp
        time = np.unique(ds.TIME.values)
        fid = ds.FIELD_ID
        ddid = ds.DATA_DESC_ID
        sid = ds.SCAN_NUMBER
        D = array2qcal_ds(gobj_amp, gobj_phase, time, ms_freq, ant_names, fid, ddid, sid, fname)
        writes.append(D)

    if not opts.output_filename.endswith('/'):
        outname = opts.output_filename + '/'
    else:
        outname = opts.output_filename
    dask.compute(xds_to_zarr(writes, f'{outname}gains.qc::NET', columns='ALL'))

