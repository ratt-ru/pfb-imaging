from contextlib import ExitStack
from pfb.workers.main import cli
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
    pyscilog.log_to_file(f'bsmooth.log')
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
    from ducc0.fft import c2c, good_size
    from africanus.calibration.utils import chunkify_rows
    from pfb.utils.misc import accum_vis
    xds = xds_from_ms(opts.ms, group_cols=['SCAN_NUMBER'],
                      chunks={'row': -1, 'chan': -1, 'corr': 1})
    ant1 = xds[0].ANTENNA1.values
    ant2 = xds[0].ANTENNA2.values
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    spw = xds_from_table(f'{opts.ms}::SPECTRAL_WINDOW')[0]
    freq = spw.CHAN_FREQ.data
    vis_ants = []
    for ds in xds:
        rchunks, tbin_idx, tbin_cnts = chunkify_rows(ds.TIME.values, -1)
        utime = ds.TIME.values[tbin_idx]
        vis_ant = accum_vis(ds.DATA.data, ds.FLAG.data,
                            ds.ANTENNA1.data, ds.ANTENNA2.data,
                            tbin_idx, tbin_cnts)

        vis_ants.append(vis_ant)

    vis_ants = dask.compute(vis_ants)

    # pad frequency to get sufficient resolution in delay space
    delta_freq = 1.0/opts.min_delay
    fmax = freq.min() + 1/delta_freq
    fexcess = fmax - freq.max()
    freq_cell = freq[1]-freq[0]
    if fexcess > 0:
        npad = np.int(np.ceil(fexcess/freq_cell))
        npix = (nchan + npad)
    else:
        npix = nchan
    while npix%2:
        npix = good_size(npix+1)
    npad = npix - nchan
    fft_freq = np.fft.fftfreq(npix, freq_cell)
    fft_freq = np.fft.fftshift(fft_freq)
    delays = np.zeros((nant, ncorr), dtype=np.float64)
    import matplotlib.pyplot as plt
    for vis_ant in vis_ants:
        vis = np.pad(vis_ant, (0, npad, 0),
                     mode='constant')
        pspecs = np.abs(c2c(vis, axes=1, nthreads=opts.nthreads))
        for p in range(nant):
            for c in range(ncorr):
                plt.plot(fft_freq, pspecs)
                plt.show()
                plt.close('all')
                delay_idx = np.argmax(pspecs[p, :, c])
                delay = fft_freq[delay_idx]
                print(delay)
