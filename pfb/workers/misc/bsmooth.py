from contextlib import ExitStack
from pfb.workers.experimental import cli
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
    Smooth and plot 1D gain solutions with a median filter
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    opts.nband = 1  # hack!!!
    OmegaConf.set_struct(opts, True)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'bsmooth_{timestamp}.log')


    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

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
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})
    import matplotlib.pyplot as plt
    import dask.array as da
    import dask
    from scipy.ndimage import median_filter
    from pfb.utils.regression import kanterp3, kanterp2, kanterp

    gain_dir = Path(opts.gain_dir).resolve()

    try:
        xds = xds_from_zarr(f'{str(gain_dir)}::{opts.gain_term}')
        if not len(xds):
            raise ValueError(f'No data at {str(gain_dir)}::{opts.gain_term}')
    except Exception as e:
        raise(e)

    nscan = len(xds)
    ntime, nchan, nant, ndir, ncorr = xds[0].gains.data.shape
    if ntime > 1:
        raise ValueError("Only freq smoothing currently supported")
    if ndir > 1:
        raise ValueError("Only freq smoothing currently supported")

    if opts.ref_ant == -1:
        ref_ant = nant-1
    else:
        ref_ant = opts.ref_ant

    # we assume the freq range is the same per ds
    freq = xds[0].gain_f.values
    # the smoothing coordinate needs to be normalised to lie between (0, 1)
    fmin = freq.min()
    fmax = freq.max()
    nu = (freq - fmin)/(fmax - fmin)
    nu += 0.01
    nu *= 0.99/nu.max()

    bamp = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
    bphase = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
    wgt = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
    for i, ds in enumerate(xds):
        jhj = np.abs(ds.jhj.values)
        g = ds.gains.values
        f = ds.gain_flags.values
        flag = np.any(jhj == 0, axis=-1)
        flag = np.logical_or(flag, f)# [:, :, :, :, None]

        for c in range(ncorr):
            jhj[flag, c] = 0.0

        amp = np.abs(g)
        jhj = np.where(amp < opts.reject_amp_thresh, jhj, 0)

        phase = np.angle(g) - np.angle(g[:, :, ref_ant])[:, :, None]
        jhj = np.where(np.abs(phase) < np.deg2rad(opts.reject_phase_thresh),
                       jhj, 0)

        # remove slope BEFORE averaging
        if opts.detrend:
            for p in range(nant):
                for c in range(ncorr):
                    idx = np.where(jhj[0, :, p, 0, c] > 0)[0]
                    if idx.size < 2:
                        continue
                    # enforce zero offset and slope
                    w = np.sqrt(jhj[0, idx, p, 0, c])
                    y = phase[0, idx, p, 0, c]
                    f = freq[idx]
                    coeffs = np.polyfit(f, y, 1, w=w)
                    phase[0, idx, p, 0, c] -= np.polyval(coeffs, f)

                    # TODO - move slope into K

        if opts.per_scan:
            print(f"Smoothing scan {i}", file=log)
            for p in range(nant):
                for c in range(ncorr):
                    w = np.sqrt(jhj[0, :, p, 0, c])
                    y = amp[0, :, p, 0, c]
                    idx = w>0
                    amplin = np.interp(freq, freq[idx], y[idx])
                    I = slice(128, -128)
                    ms, Ps = kanterp(nu[I], amplin[I], w[I], niter=10, nu0=2,
                                      sigmaf0=np.sqrt(nchan), sigman0=1)
                    amp[0, I, p, 0, c] = ms[0, :]
                    if p == ref_ant:
                        continue
                    y = phase[0, :, p, 0, c]
                    phaselin = np.interp(freq, freq[idx], y[idx])
                    ms, Ps = kanterp(nu[I], phaselin[I], w[I]/amp[0, I, p, 0, c],
                                      niter=10, nu0=2, sigmaf0=np.sqrt(nchan), sigman0=1)
                    phase[0, I, p, 0, c] = ms[0, :]


            bpass = amp * np.exp(1.0j*phase)
            bpass = da.from_array(bpass, chunks=(-1, -1, -1, -1, -1))
            xds[i] = ds.assign(**{'gains': (ds.GAIN_AXES, bpass)})


        else:
            bamp += amp*jhj
            bphase += phase*jhj
            wgt += jhj

    if not opts.per_scan:
        print("Smoothing over all scans", file=log)
        bamp = np.where(wgt > 0, bamp/wgt, 0)
        bphase = np.where(wgt > 0, bphase/wgt, 0)
        flag = wgt == 0
        # flag has no corr axis
        flag = np.any(flag, axis=-1)

        samp = np.zeros_like(bamp)
        sphase = np.zeros_like(bamp)
        for p in range(nant):
            for c in range(ncorr):
                print(f" p = {p}, c = {c}", file=log)
                idx = ~flag[0, :, p, 0]
                if idx.sum() < 2:
                    continue
                w = wgt[0, :, p, 0, c]
                amp = bamp[0, :, p, 0, c]
                amplin = np.interp(freq, freq[idx], amp[idx])
                I = slice(128, -128)
                ms, Ps = kanterp(nu[I], amplin[I], w[I], niter=10, nu0=2,
                                  sigmaf0=np.sqrt(nchan), sigman0=1)
                samp[0, I, p, 0, c] = ms[0, :]
                if p == ref_ant:
                    continue
                phase = bphase[0, :, p, 0, c]
                phaselin = np.interp(freq, freq[idx], phase[idx])
                ms, Ps = kanterp(nu[I], phaselin[I], w[I]/samp[0, I, p, 0, c],
                                  niter=10, nu0=2, sigmaf0=np.sqrt(nchan), sigman0=1)
                sphase[0, I, p, 0, c] = ms[0, :]


        bpass = samp * np.exp(1.0j*sphase)
        bpass = da.from_array(bpass, chunks=(-1, -1, -1, -1, -1))
        flag = da.from_array(flag, chunks=(-1, -1, -1, -1))
        for i, ds in enumerate(xds):
            xds[i] = ds.assign(**{'gains': (ds.GAIN_AXES, bpass),
                                  'gain_flags': (ds.GAIN_AXES[0:-1],
                                                flag.astype(bool))})


    print(f"Writing smoothed gains to {str(gain_dir)}/"
        f"smoothed.qc::{opts.gain_term}", file=log)
    writes = xds_to_zarr(xds, f'{str(gain_dir)}/smoothed.qc::{opts.gain_term}',
                         columns='ALL')

    bpass = dask.compute(bpass, writes)[0]

    if not opts.do_plots:
        print("Not doing plots", file=log)
        print("All done here", file=log)
        quit()

    # set to NaN's for plotting
    bamp = np.where(wgt > 0, bamp, np.nan)
    bphase = np.where(wgt > 0, bphase, np.nan)

    samp = np.abs(bpass)
    sphase = np.angle(bpass)
    # samp = np.where(wgt > 0, samp, np.nan)
    # sphase = np.where(wgt > 0, sphase, np.nan)

    # load the original data for comparitive plotting
    try:
        xds = xds_from_zarr(f'{str(gain_dir)}::{opts.gain_term}')
    except Exception as e:
        raise(e)



    freq = xds[0].gain_f/1e6
    for p in range(nant):
        for c in range(ncorr):
            print(f" p = {p}, c = {c}")
            fig, ax = plt.subplots(nrows=1, ncols=2,
                                figsize=(18, 18))
            fig.suptitle(f'Antenna {p}, corr {c}', fontsize=24)

            for s, ds in enumerate(xds):
                jhj = ds.jhj.values.real[0, :, p, 0, c]
                f = ds.gain_flags.values[0, :, p, 0]
                flag = np.logical_or(jhj==0, f)
                tamp = np.abs(xds[s].gains.values[0, :, p, 0, c])
                tphase = (np.angle(xds[s].gains.values[0, :, p, 0, c]) -
                          np.angle(xds[s].gains.values[0, :, opts.ref_ant, 0, c]))
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
            name = f'{str(gain_dir)}/{opts.gain_term}_Antenna{p}corr{c}.png'
            plt.savefig(name, dpi=250, bbox_inches='tight')
            plt.close('all')

    print("All done here", file=log)
