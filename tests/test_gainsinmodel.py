
def test_gainsinmodel():
    import numpy as np
    from pyrap.tables import table
    ms = table('/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS', readonly=False)
    spw = table('/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')

    utime = np.unique(ms.getcol('TIME'))
    t = (utime-utime.min())/(utime.max() - utime.min())
    freq = spw.getcol('CHAN_FREQ').squeeze()
    freq0 = np.mean(freq)
    nu = (freq/freq0 - 1.0)

    ntime = t.size
    nchan = nu.size
    nant = np.unique(ms.getcol('ANTENNA1')).size
    ncorr = ms.getcol('FLAG').shape[-1]

    from africanus.gps.utils import abs_diff
    tt = abs_diff(t, t)
    lt = 0.25
    Kt = np.exp(-tt**2/(2*lt**2))
    Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))
    vv = abs_diff(nu, nu)
    lv = 0.1
    Kv = np.exp(-vv**2/(2*lv**2))
    Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
    L = (Lt, Lv)

    from pfb.utils.misc import kron_matvec
    import matplotlib.pyplot as plt

    # gains
    jones = np.zeros((ntime, nant, nchan, 1, 2), dtype=np.complex128)
    for p in range(nant):
        for c in range(2):
            xi = np.random.randn(nchan)
            amp = np.exp(-nu**2 + Lv.dot(xi))
            xi = np.random.randn(nchan)
            phase = Lv.dot(xi)
            jv = amp * np.exp(1.0j*phase)
            xi = np.random.randn(ntime)
            amp = np.exp(Lt.dot(xi))
            xi = np.random.randn(ntime)
            phase = Lt.dot(xi)
            jt = amp * np.exp(1.0j*phase)
            jones[:, p, :, 0, c] = np.kron(jt[:, None], jv[None, :])

    uvw = ms.getcol('UVW')
    nrow = uvw.shape[0]
    u_max = 0.0
    v_max = 0.0
    u_max = np.maximum(u_max, abs(uvw[:, 0]).max())
    v_max = np.maximum(v_max, abs(uvw[:, 1]).max())
    uv_max = np.maximum(u_max, v_max)

    # image size
    from africanus.constants import c as lightspeed
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    cell_rad = cell_N / 2.0
    cell_size = cell_rad * 180 / np.pi
    print("Cell size set to %5.5e arcseconds" % cell_size)

    fov = 2
    npix = int(fov / cell_size)
    if npix % 2:
        npix += 1

    nx = npix
    ny = npix

    print("Image size set to (%i, %i, %i)" % (nchan, nx, ny))

    # model
    model = np.zeros((nchan, nx, ny), dtype=np.float64)
    nsource = 10
    Ix = np.random.randint(0, npix, nsource)
    Iy = np.random.randint(0, npix, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    I0 = 1.0 + np.abs(np.random.randn(nsource))
    for i in range(nsource):
        model[:, Ix[i], Iy[i]] = I0[i] * (freq/freq0) ** alpha[i]


    # model vis
    from ducc0.wgridder import dirty2ms
    model_vis = np.zeros((nrow, nchan, 2), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c:c+1, 0] = dirty2ms(uvw, freq[c:c+1], model[c],
                                      pixsize_x=cell_rad, pixsize_y=cell_rad,
                                      epsilon=1e-8, do_wstacking=True, nthreads=8)
        model_vis[:, c, -1] = model_vis[:, c, 0]

    # corrupted vis
    model_vis = model_vis.reshape(nrow, nchan, 1, 2)
    from africanus.calibration.utils import chunkify_rows
    time = ms.getcol('TIME')
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, 1)
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    from africanus.calibration.utils import corrupt_vis


    print(np.abs(jones).max(), np.abs(jones).min(), np.abs(model_vis).max(), np.abs(model_vis).min())

    # print(jones.dtype, model_vis.dtype)

    vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, model_vis)


    # print(vis.shape)
    # print(np.abs(vis).max(), np.abs(vis).min())

    # quit()


    model_vis[:, :, 0] = 1.0 + 0j
    model_vis[:, :, -1] = 1.0 + 0j
    muellercol = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, model_vis)

    print(np.abs(jones).max(), np.abs(jones).min(), np.abs(model_vis).max(), np.abs(model_vis).min())

    print(muellercol.shape)
    print(np.abs(muellercol).max(), np.abs(muellercol).min())

    quit()

    ms.putcol('DATA', vis.astype(np.complex64))
    ms.putcol('CORRECTED_DATA', muellercol.astype(np.complex64))
    ms.close()

    from pfb.workers.grid.dirty import _dirty
    _dirty(ms='/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS',
           data_column="DATA", weight_column='WEIGHT', imaging_weight_column=None,
           flag_column='FLAG', mueller_column='CORRECTED_DATA',
           row_chunks=None, epsilon=1e-5, wstack=True, mock=False,
           double_accum=True, output_filename='/home/landman/Data/pfb-testing/output/test',
           nband=nchan, field_of_view=2.0, super_resolution_factor=2.0,
           cell_size=None, nx=None, ny=None, output_type='f4', nworkers=1,
           nthreads_per_worker=1, vthreads=8, mem_limit=8, nthreads=8,
           host_address=None)

    from pfb.workers.grid.psf import _psf
    _psf(ms='/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS',
         data_column="DATA", weight_column='WEIGHT', imaging_weight_column=None,
         flag_column='FLAG', mueller_column='CORRECTED_DATA', row_out_chunk=-1,
         row_chunks=None, epsilon=1e-5, wstack=True, mock=False, psf_oversize=2,
         double_accum=True, output_filename='/home/landman/Data/pfb-testing/output/test',
         nband=nchan, field_of_view=2.0, super_resolution_factor=2.0,
         cell_size=None, nx=None, ny=None, output_type='f4', nworkers=1,
         nthreads_per_worker=1, vthreads=8, mem_limit=8, nthreads=8,
         host_address=None)

    # solve for model using pcg and mask
    mask = np.any(model, axis=0)
    from astropy.io import fits
    from pfb.utils.fits import save_fits
    hdr = fits.getheader('/home/landman/Data/pfb-testing/output/test_dirty.fits')
    save_fits('/home/landman/Data/pfb-testing/output/test_mask.fits', mask, hdr)

    from pfb.workers.deconv.forward import _forward
    _forward(residual='/home/landman/Data/pfb-testing/output/test_dirty.fits',
             psf='/home/landman/Data/pfb-testing/output/test_psf.fits',
             mask='/home/landman/Data/pfb-testing/output/test_mask.fits',
             beam_model=None, band='L',
             weight_table='/home/landman/Data/pfb-testing/output/test.zarr',
             output_filename='/home/landman/Data/pfb-testing/output/test',
             nband=nchan, output_type='f4', epsilon=1e-5, sigmainv=1.0,
             wstack=True, double_accum=True, cg_tol=1e-5, cg_minit=10,
             cg_maxit=100, cg_verbose=0, cg_report_freq=10, backtrack=False,
             nworkers=1, nthreads_per_worker=1, vthreads=8, mem_limit=8,
             nthreads=8, host_address=None)







test_gainsinmodel()