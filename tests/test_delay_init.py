import numpy as np
from pyrap.tables import table
from africanus.calibration.utils import corrupt_vis, chunkify_rows
from pfb.utils.misc import _accum_vis_impl, _estimate_delay_impl, _estimate_tec_impl
from ducc0.fft import good_size


def test_delay_init():
    ms = table('/home/landman/projects/LOFAR/combined.MS')
    ms = ms.query(query='ANTENNA1!=ANTENNA2')
    time = ms.getcol('TIME')
    utime = np.unique(time)
    tf = utime[20]
    ms = ms.query(query=f'TIME<={tf}')
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    time = ms.getcol('TIME')
    utime = np.unique(time)
    ntime = utime.size
    mvis = ms.getcol('DATA')
    mvis = mvis[:, :, 0:1].astype(np.complex128)
    flag = ms.getcol('FLAG')[:, :, 0:1]
    # mvis[...] = 1.0 + 0.0j
    nrow, _, ncorr = mvis.shape

    spw = table('/home/landman/projects/LOFAR/combined.MS::SPECTRAL_WINDOW')
    freq = spw.getcol('CHAN_FREQ').squeeze()
    nchan = freq.size

    ref_ant = 20
    # delays = 1e-7*np.random.randn(nant, ncorr)
    fctr = 1
    nuinv = fctr/freq
    tec_nyq = 1/(2*(nuinv.max() - nuinv.min()))
    # tecs = 10*tec_nyq*np.random.randn(nant, ncorr)
    # tecs[ref_ant] = 0
    max_tec = 2*np.pi/(nuinv[-2]-nuinv[-1])

    # phase = freq[None, :, None] * delays[:, None, :]  # tecs/freq
    # phase = fctr*tecs[:, None, :] / freq[None, :, None]
    # phase = np.tile(phase[None, :, :, :], (ntime, 1, 1, 1))
    # gain = np.exp(2j * np.pi * phase[:, :, :, None, :])
    # gain = np.exp(1j * phase[:, :, :, None, :])

    # _, tbin_idx, tbin_counts = chunkify_rows(time, -1)


    # try rotating out model phase for complex field
    # vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, gain, mvis[:, :, None, :])
    # flag = np.zeros_like(vis, dtype=bool)
    # import pdb; pdb.set_trace()
    vis_ant = _accum_vis_impl(mvis, flag, ant1, ant2, ref_ant)
    # import pdb; pdb.set_trace()
    tecs_rec = _estimate_tec_impl(vis_ant, freq, tec_nyq, max_tec, fctr)
    import pdb; pdb.set_trace()
    # scale delays to account for ref_ant
    # print(delays[ref_ant])
    tecs -= tecs[ref_ant:ref_ant+1]
    print(np.abs(tecs - tecs_rec).max())


test_delay_init()

