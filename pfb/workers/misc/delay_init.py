from contextlib import ExitStack
from pfb.workers.experimental import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('DELAY_INIT')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.delay_init["inputs"].keys():
    defaults[key] = schema.delay_init["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.delay_init)
def delay_init(**kw):
    '''
    Smooth time variable solution
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'delay_init_{timestamp}.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    return _delay_init(**opts)

def _delay_init(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import dask
    import dask.array as da
    from daskms import xds_from_ms, xds_from_table
    from daskms.experimental.zarr import xds_to_zarr
    from pfb.utils.misc import accum_vis, estimate_delay
    import xarray as xr
    from pathlib import Path

    ms_path = Path(opts.ms).resolve()
    ms_name = str(ms_path)
    group_cols = ['FIELD_ID', 'DATA_DESC_ID']
    if opts.split_scans:
        print("Splitting estimate by scan", file=log)
        group_cols.append('SCAN_NUMBER')
    xds = xds_from_ms(ms_name,
                      group_cols=group_cols,
                      chunks={'row': -1, 'chan': -1, 'corr': 1})
    ant_names = xds_from_table(f'{ms_name}::ANTENNA')[0].NAME.values
    # pad frequency to get sufficient resolution in delay space
    spws = dask.compute(xds_from_table(f'{ms_name}::SPECTRAL_WINDOW'))[0]
    fields = dask.compute(xds_from_table(f'{ms_name}::FIELD'))[0]

    ant1 = xds[0].ANTENNA1.values
    ant2 = xds[0].ANTENNA2.values
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    out_ds = []
    for ds in xds:
        fid = ds.FIELD_ID
        ddid = ds.DATA_DESC_ID
        if opts.split_scans:
            sid = int(ds.SCAN_NUMBER)
        else:
            sid = '?'
        fname = fields[fid].NAME.values[0]

        # only diagonal correlations
        if ds.corr.size > 1:
            ds = ds.sel(corr=[0, -1])
            ncorr = 2
        else:
            ncorr = 1

        if opts.ref_ant == -1:
            ref_ant = nant-1
        else:
            ref_ant = opts.ref_ant
        vis_ant = accum_vis(ds.DATA.data, ds.FLAG.data,
                            ds.ANTENNA1.data, ds.ANTENNA2.data,
                            nant, ref_ant=ref_ant)

        vis_ant.rechunk({1:8})

        freq = spws[ddid].CHAN_FREQ.data[0]
        delays = estimate_delay(vis_ant, freq, opts.min_delay)

        utime = np.unique(ds.TIME.values)
        ntime = utime.size
        nchan = freq.size
        ndir = 1
        gain = da.exp(2.0j * np.pi * freq[:, None, None, None] * delays[None, :, None, :])
        gain = da.tile(gain[None, :, :, :, :], (ntime, 1, 1, 1, 1))
        gain = da.rechunk(gain, (-1, -1, -1, -1, -1))
        gflags = da.zeros((ntime, nchan, nant, ndir), chunks=(-1, -1, -1, -1), dtype=np.int8)
        data_vars = {
            'gains':(('gain_time', 'gain_freq', 'antenna', 'direction', 'correlation'), gain),
            'gain_flags':(('gain_time', 'gain_freq', 'antenna', 'direction'), gflags)
        }
        from collections import namedtuple
        gain_spec_tup = namedtuple('gains_spec_tup', 'tchunk fchunk achunk dchunk cchunk')
        attrs = {
            'DATA_DESC_ID': int(ddid),
            'FIELD_ID': int(fid),
            'FIELD_NAME': fname,
            'GAIN_AXES': ('gain_time', 'gain_freq', 'antenna', 'direction', 'correlation'),
            'GAIN_SPEC': gain_spec_tup(tchunk=(int(ntime),),
                                        fchunk=(int(nchan),),
                                        achunk=(int(nant),),
                                        dchunk=(int(1),),
                                        cchunk=(int(ncorr),)),
            'NAME': 'NET',
            'SCAN_NUMBER': sid,
            'TYPE': 'complex'
        }
        if ncorr==1:
            corrs = np.array(['XX'], dtype=object)
        elif ncorr==2:
            corrs = np.array(['XX', 'YY'], dtype=object)
        coords = {
            'gain_freq': (('gain_freq',), freq),
            'gain_time': (('gain_time',), utime),
            'antenna': (('ant'), ant_names),
            'correlation': (('corr'), corrs),
            'direction': (('dir'), np.array([0], dtype=np.int32)),
            'f_chunk': (('f_chunk'), np.array([0], dtype=np.int32)),
            't_chunk': (('t_chunk'), np.array([0], dtype=np.int32))
        }

        ods = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        ods = ods.chunk({ax: "auto" for ax in ods.GAIN_AXES[:2]})
        out_ds.append(ods)

    # LB - Why is this required? ods.chunk should coerce all arrays to have the same chunking
    out_ds = xr.unify_chunks(*out_ds)
    out_path = Path(f'{opts.gain_dir}::{opts.gain_term}').resolve()
    out_name = str(out_path)
    writes = xds_to_zarr(out_ds, out_path, columns='ALL')
    dask.compute(writes)

