from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('KSMOOTH')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.ksmooth["inputs"].keys():
    defaults[key] = schema.ksmooth["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.ksmooth)
def ksmooth(**kw):
    '''
    Per antenna KGB plotter
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'bsmooth.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    return _ksmooth(**opts)

def _ksmooth(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pathlib import Path
    import matplotlib.pyplot as plt
    import dask.array as da
    import dask

    gain_dir = Path(opts.gain_dir).resolve()

    try:
        K = xds_from_zarr(f'{str(gain_dir)}::K')
        G = xds_from_zarr(f'{str(gain_dir)}::G')
        B = xds_from_zarr(f'{str(gain_dir)}::B')
    except Exception as e:
        K = xds_from_zarr(f'{str(gain_dir)}/K')
        G = xds_from_zarr(f'{str(gain_dir)}/G')
        B = xds_from_zarr(f'{str(gain_dir)}/B')

    import pdb; pdb.set_trace()

    nscan = len(xds)
    ntime, nchan, nant, ndir, ncorr = xds[0].gains.data.shape
    import pdb; pdb.set_trace()
    if nchan > 1:
        raise ValueError("nchan can't be > 1")

    ppath = gain_dir.parent
    for p in range(nant):
        for c in range(ncorr):
            fig, ax = plt.subplots(nrows=3, ncols=2,
                                  figsize=(18, 18))
            fig.suptitle(f'Antenna {p}, corr {c}')

            for s, ds in enumerate(xds):
                time = ds.gain_t.values
                jhj = ds.jhj.values.real[0, :, p, 0, c]
                f = ds.gain_flags.values[0, :, p, 0]
                flag = np.logical_or(jhj==0, f)
                amp = np.abs(xds[s].gains.values[0, :, p, 0, c])
                phase = np.angle(xds[s].gains.values[0, :, p, 0, c])
                amp[flag] = np.nan
                phase[flag] = np.nan

                ax[0].plot(time, amp, 'b', alpha=0.5, linewidth=2)
                ax[1].plot(time, phase, 'b', alpha=0.5, linewidth=2)

            ax[0].set_xlabel('time')
            ax[1].set_xlabel('time')

            fig.tight_layout()
            name = f'{str(ppath)}/Antenna{p}corr{c}{opts.postfix}.png'
            plt.savefig(name, dpi=500, bbox_inches='tight')
            plt.close('all')


