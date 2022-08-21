from contextlib import ExitStack
from pfb.workers.experimental import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('GSMOOTH')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.gsmooth["inputs"].keys():
    defaults[key] = schema.gsmooth["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.gsmooth)
def gsmooth(**kw):
    '''
    Smooth and plot 1D gain solutions with a median filter
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'tsmooth_{timestamp}.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    return _gsmooth(**opts)

def _gsmooth(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pathlib import Path
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})
    import matplotlib.pyplot as plt
    import dask.array as da
    import dask
    from scipy.ndimage import median_filter
    import xarray as xr

    gain_dir = Path(opts.gain_dir).resolve()

    try:
        xds = xds_from_zarr(f'{str(gain_dir)}::{opts.gain_term}')
    except Exception as e:
        xds = xds_from_zarr(f'{str(gain_dir)}/{opts.gain_term}')

    xds_concat = xr.concat(xds, dim='gain_t').sortby('gain_t')

    ntime, nchan, nant, ndir, ncorr = xds_concat.gains.data.shape
    if nchan > 1:
        raise ValueError("Only time smoothing currently supported")
    if ndir > 1:
        raise ValueError("Only time smoothing currently supported")


    wgt = np.abs(xds_concat.jhj.values)
    g = xds_concat.gains.values
    f = xds_concat.gain_flags.values
    flag = np.any(wgt == 0, axis=-1)
    flag = np.logical_or(flag, f)

    for c in range(ncorr):
        wgt[flag, c] = 0.0

    gamp = np.abs(g)
    gphase = np.angle(g)
    time = xds_concat.gain_t.values

    samp = np.zeros_like(amp)
    sphase = np.zeros_like(phase)
    for p in range(nant):
        for c in range(ncorr):
            idx = np.where(jhj[:, 0, p, 0, c] > 0)[0]
            if idx.size < 2:
                continue
            x = time[idx]
            w = np.sqrt(wgt[idx, 0, p, 0, c])
            amp = gamp[idx, 0, p, 0, c]
            amplin = np.interp(time, x, amp)
            samp[0, :, p, 0, c] = median_filter(amplin, size=opts.filter_size)
            phase = bphase[0, idx, p, 0, c]
            phaselin = np.interp(freq, x, phase)
            sphase[0, :, p, 0, c] = median_filter(phaselin,
                                                  size=opts.filter_size)


    bpass = samp * np.exp(1.0j*sphase)
    bpass = da.from_array(bpass, chunks=(-1, -1, -1, -1, -1))
    flag = da.from_array(flag, chunks=(-1, -1, -1, -1))
    for i, ds in enumerate(xds):
        xds[i] = ds.assign(**{'gains': (ds.GAIN_AXES, bpass),
                              'gain_flags': (ds.GAIN_AXES[0:-1], flag)})

    ppath = gain_dir.parent
    writes = xds_to_zarr(xds, f'{str(ppath)}/smoothed.qc::{opts.gain_term}',
                         columns='ALL')

    bpass = dask.compute(bpass, writes)[0]

    if not opts.do_plots:
        quit()

    # set to NaN's for plotting
    bamp = np.where(wgt > 0, bamp, np.nan)
    bphase = np.where(wgt > 0, bphase, np.nan)

    # samp = np.where(wgt > 0, samp, np.nan)
    # sphase = np.where(wgt > 0, sphase, np.nan)

    freq = xds[0].gain_f
    for p in range(nant):
        for c in range(ncorr):
            fig, ax = plt.subplots(nrows=1, ncols=2,
                                figsize=(18, 18))
            fig.suptitle(f'Antenna {p}, corr {c}', fontsize=24)

            for s, ds in enumerate(xds):
                jhj = ds.jhj.values.real[0, :, p, 0, c]
                f = ds.gain_flags.values[0, :, p, 0]
                flag = np.logical_or(jhj==0, f)
                tamp = np.abs(xds[s].gains.values[0, :, p, 0, c])
                tphase = np.angle(xds[s].gains.values[0, :, p, 0, c])
                tamp[flag] = np.nan
                tphase[flag] = np.nan

                ax[0].plot(freq, tamp, label=f'scan-{s}', alpha=0.5, linewidth=1)
                ax[1].plot(freq, np.rad2deg(tphase), label=f'scan-{s}', alpha=0.5, linewidth=1)

            ax[0].plot(freq, bamp[0, :, p, 0, c], 'k', label='inf', linewidth=1)
            ax[0].plot(freq, samp[0, :, p, 0, c], 'r', label='smooth', linewidth=1)
            ax[0].legend()
            ax[0].set_xlabel('freq / [MHz]')

            ax[1].plot(freq, np.rad2deg(bphase[0, :, p, 0, c]), 'k', label='inf', linewidth=1)
            ax[1].plot(freq, np.rad2deg(sphase[0, :, p, 0, c]), 'r', label='smooth', linewidth=1)
            ax[1].legend()
            ax[1].set_xlabel('freq / [MHz]')

            fig.tight_layout()
            name = f'{str(ppath)}/Antenna{p}corr{c}{opts.postfix}.png'
            plt.savefig(name, dpi=500, bbox_inches='tight')
            plt.close('all')


