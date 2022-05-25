# import numpy as np
# from pyrap.tables import table
# from africanus.calibration.utils import corrupt_vis
# from africanus.calibration.utils import chunkify_rows
# from pfb.utils.misc import _accum_vis_impl, _estimate_delay_impl
# from ducc0.fft import good_size


# def test_delay_init():
#     ms = table('/home/landman/projects/ESO137/subsets/msdir/ms1_primary_subset.ms')
#     ms = ms.query(query='ANTENNA1!=ANTENNA2')
#     ant1 = ms.getcol('ANTENNA1')
#     ant2 = ms.getcol('ANTENNA2')
#     nant = np.maximum(ant1.max(), ant2.max()) + 1
#     time = ms.getcol('TIME')
#     utime = np.unique(time)
#     ntime = utime.size
#     mvis = ms.getcol('MODEL_DATA')
#     mvis = mvis[:, :, 0:1].astype(np.complex128)
#     mvis[...] = 1.0 + 0.0j
#     nrow, _, ncorr = mvis.shape

#     spw = table('/home/landman/projects/ESO137/subsets/msdir/ms1_primary_subset.ms::SPECTRAL_WINDOW')
#     freq = spw.getcol('CHAN_FREQ').squeeze()
#     nchan = freq.size

#     ref_ant = 20
#     delays = 1e-7*np.random.randn(nant, ncorr)
#     # delays[ref_ant] = 0
#     phase = freq[None, :, None] * delays[:, None, :]
#     phase = np.tile(phase[None, :, :, :], (ntime, 1, 1, 1))
#     gain = np.exp(2j * np.pi * phase[:, :, :, None, :])

#     _, tbin_idx, tbin_counts = chunkify_rows(time, -1)


#     # try rotating out model phase for complex field
#     vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, gain, mvis[:, :, None, :])
#     flag = np.zeros_like(vis, dtype=bool)
#     vis_ant = _accum_vis_impl(vis, flag, ant1, ant2, ref_ant)

#     delays_rec = _estimate_delay_impl(vis_ant, freq, 1e-11)

#     # scale delays to account for ref_ant
#     # print(delays[ref_ant])
#     delays -= delays[ref_ant:ref_ant+1]
#     print(np.abs(delays - delays_rec).max())


# test_delay_init()
# # nchan = 100
# # fmin = 1054535156.25
# # fupper = 1075224609.375
# # nu = np.linspace(fmin, fupper, nchan)
# # fcell = nu[1]-nu[0]

# # max_delay = 1/fcell

# # print(max_delay)

# # delay = max_delay/5

# # min_delay = 1e-8
# # fmax = fmin + 1/min_delay
# # fexcess = fmax - fupper
# # if fexcess > 0:
# #     npad = np.int(np.ceil(fexcess/fcell))
# #     npix = (nchan + npad)
# # else:
# #     npix = nchan

# # lag = np.fft.fftfreq(npix, d=fcell)
# # lag = np.fft.fftshift(lag)

# # s = np.exp(2j*np.pi*nu*delay)

# # shat = np.fft.fft(s, n=npix)
# # shat = np.fft.fftshift(shat)

# # pspec = np.abs(shat)
# # idx = np.argmax(pspec)

# # print(delay, lag[idx], idx)

# # import matplotlib.pyplot as plt

# # plt.plot(lag, pspec)
# # plt.savefig('/home/landman/testing/test.png', dpi=250)
