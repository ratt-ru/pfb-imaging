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
    from pfb.utils.regression import gpr, kanterp, kanterp3

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


    jhj = np.abs(xds_concat.jhj.values)
    g = xds_concat.gains.values
    f = xds_concat.gain_flags.values
    flag = np.any(jhj == 0, axis=-1)
    flag = np.logical_or(flag, f)

    for c in range(ncorr):
        jhj[flag, c] = 0.0

    if opts.ref_ant == -1:
        ref_ant = nant-1
    else:
        ref_ant = opts.ref_ant

    gamp = np.abs(g)
    gphase = np.angle(g) - np.angle(g[:, :, ref_ant])[:, :, None]
    time = xds_concat.gain_t.values
    # the coordinate needs to be scaled to lie in (0, 1)
    tmin = time.min()
    tmax = time.max()
    t = (time - tmin)/(tmax - tmin)
    t += 0.01
    t *= 0.99/t.max()

    samp = np.zeros_like(gamp)
    sampcov = np.zeros_like(gamp)
    sphase = np.zeros_like(gphase)
    sphasecov = np.zeros_like(gamp)
    for p in range(nant):
        if p == ref_ant:
            continue
        for c in range(ncorr):
            print(f" p = {p}, c = {c}")
            idx = np.where(jhj[:, 0, p, 0, c] > 0)[0]
            if idx.size < 2:
                continue
            w = jhj[:, 0, p, 0, c]
            amp = gamp[:, 0, p, 0, c]
            mus, covs = kanterp3(t, amp, w, opts.niter, nu0=2)
            samp[:, 0, p, 0, c] = mus[0, :]
            sampcov[:, 0, p, 0, c] = covs[0, 0, :]
            # mu, cov = gpr(amp, t, w, t)
            # samp[:, 0, p, 0, c] = mu
            # sampcov[:, 0, p, 0, c] = cov
            phase = gphase[:, 0, p, 0, c]
            # phase = np.unwrap(phase, discont=2*np.pi*0.99)
            wp = w/samp[:, 0, p, 0, c]
            mus, covs = kanterp3(t, phase, wp, opts.niter, nu0=2)
            sphase[:, 0, p, 0, c] = mus[0, :]
            sphasecov[:, 0, p, 0, c] = covs[0, 0, :]
            # mu, cov = gpr(phase, t, wp, t)
            # sphase[:, 0, p, 0, c] = mu
            # sphasecov[:, 0, p, 0, c] = cov


    # gs = samp * np.exp(1.0j*sphase)
    # gs = da.from_array(gs, chunks=(-1, -1, -1, -1, -1))
    # flag = da.from_array(flag, chunks=(-1, -1, -1, -1))
    # for i, ds in enumerate(xds):
    #     xds[i] = ds.assign(**{'gains': (ds.GAIN_AXES, gs),
    #                           'gain_flags': (ds.GAIN_AXES[0:-1], flag)})

    # ppath = gain_dir.parent
    # writes = xds_to_zarr(xds, f'{str(gain_dir)}/smoothed.qc::{opts.gain_term}',
    #                      columns='ALL')

    # gs = dask.compute(gs, writes)[0]

    if not opts.do_plots:
        quit()

    # set to NaN's for plotting
    gamp = np.where(jhj > 0, gamp, np.nan)
    gphase = np.where(jhj > 0, gphase, np.nan)

    # samp = np.where(wgt > 0, samp, np.nan)
    # sphase = np.where(wgt > 0, sphase, np.nan)

    print("Plotting results", file=log)
    for p in range(nant):
        for c in range(ncorr):
            print(f" p = {p}, c = {c}")
            fig, ax = plt.subplots(nrows=1, ncols=2,
                                figsize=(18, 18))
            fig.suptitle(f'Antenna {p}, corr {c}', fontsize=24)

            sigma = 1.0/np.sqrt(jhj[:, 0, p, 0, c])
            amp = gamp[:, 0, p, 0, c]
            ax[0].errorbar(time, amp, sigma, fmt='xr')
            ax[0].errorbar(time, samp[:, 0, p, 0, c], np.sqrt(sampcov[:, 0, p, 0, c]),
                           fmt='ok', label='smooth')
            ax[0].legend()
            ax[0].set_xlabel('time')

            sigmap = sigma/amp
            phase = gphase[:, 0, p, 0, c]
            # phase = np.unwrap(phase, discont=2*np.pi*0.99)
            ax[1].errorbar(time, np.rad2deg(phase), sigmap, fmt='xr')
            phase = sphase[:, 0, p, 0, c]
            # phase = np.unwrap(phase, discont=2*np.pi*0.99)
            ax[1].errorbar(time, np.rad2deg(phase), np.rad2deg(np.sqrt(sphasecov[:, 0, p, 0, c])),
                       fmt='ok', label='smooth')
            ax[1].legend()
            ax[1].set_xlabel('time')

            fig.tight_layout()
            name = f'{str(gain_dir)}/{opts.gain_term}_Antenna{p}corr{c}.png'
            plt.savefig(name, dpi=250, bbox_inches='tight')
            plt.close('all')

    print("All done here.", file=log)


