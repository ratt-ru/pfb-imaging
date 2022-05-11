from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('BSMOOTH')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.bsmooth["inputs"].keys():
    defaults[key] = schema.bsmooth["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.bsmooth)
def bsmooth(**kw):
    '''
    Smooth bandpass solution
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'bsmooth.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    return _bsmooth(**opts)

def _bsmooth(**kw):
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
        xds = xds_from_zarr(f'{str(gain_dir)}::{opts.gain_term}')
    except Exception as e:
        xds = xds_from_zarr(f'{str(gain_dir)}/{opts.gain_term}')

    nscan = len(xds)
    ntime, nchan, nant, ndir, ncorr = xds[0].gains.data.shape
    if ntime > 1:
        raise ValueError("Bandpass can't have ntime > 1")

    bamp = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
    bphase = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
    wgt = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
    for ds in xds:
        jhj = np.abs(ds.jhj.values)
        g = ds.gains.values
        f = ds.gain_flags.values
        flag = np.any(jhj == 0, axis=-1)
        flag = np.logical_or(flag, f)[:, :, :, :, None]

        # for c in range(ncorr):
        #     jhj[flag, c] = 0.0
        bamp += np.abs(g)*jhj
        bphase += np.angle(g)*jhj
        wgt += jhj

    # import pdb; pdb.set_trace()
    bamp = np.where(wgt > 0, bamp/wgt, np.nan)
    bphase = np.where(wgt > 0, bphase/wgt, np.nan)
    bpass = bamp * np.exp(1.0j*bphase)
    # bpass = da.from_array(bpass, chunks=(-1, -1, -1, -1, -1))
    # xdsw = []
    # for ds in xds:
    #     xdsw.append(ds.assign(**{'gains': (ds.GAIN_AXES, bpass)}))

    ppath = gain_dir.parent
    # dask.compute(xds_to_zarr(xdsw, f'{str(ppath)}/smoothed.qc::{opts.gain_term}',
    #                          columns='ALL'))

    freq = xds[0].gain_f
    # bpass = bpass.compute()
    for p in range(nant):
        for c in range(ncorr):
            fig, ax = plt.subplots(nrows=1, ncols=2,
                                figsize=(18, 18))
            fig.suptitle(f'Antenna {p}, corr {c}')
            amp = np.abs(bpass[0, :, p, 0, c])
            phase = np.angle(bpass[0, :, p, 0, c])
            phase = np.where(phase < -np.pi, phase + 2*np.pi, phase)
            phase = np.where(phase > np.pi, phase - 2*np.pi, phase)
            #phase[~np.isnan(phase)] = np.unwrap(phase[~np.isnan(phase)])


            for s, ds in enumerate(xds):
                jhj = ds.jhj.values.real[0, :, p, 0, c]
                f = ds.gain_flags.values[0, :, p, 0]
                flag = np.logical_or(jhj==0, f)
                tamp = np.abs(xds[s].gains.values[0, :, p, 0, c])
                tphase = np.angle(xds[s].gains.values[0, :, p, 0, c])
                tphase = np.where(tphase < -np.pi, tphase + 2*np.pi, tphase)
                tphase = np.where(tphase > np.pi, tphase - 2*np.pi, tphase)

                # tphase[~np.isnan(tphase)] = np.unwrap(tphase[~np.isnan(tphase)])
                tamp[flag] = np.nan
                tphase[flag] = np.nan

                ax[0].plot(freq, tamp, 'b', alpha=0.15, linewidth=2)
                ax[1].plot(freq, tphase, 'b', alpha=0.15, linewidth=2)

            ax[0].plot(freq, amp, 'k', linewidth=1)
            ax[0].set_xlabel('freq / [MHz]')

            ax[1].plot(freq, phase, 'k', linewidth=1)
            ax[1].set_xlabel('freq / [MHz]')

        fig.tight_layout()
        name = f'{str(ppath)}/Antenna{p}corr{c}{opts.postfix}.png'
        plt.savefig(name, dpi=500, bbox_inches='tight')
        plt.close('all')


