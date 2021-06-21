
def test_gainsinmodel():
    import numpy as np
    from pyrap.tables import table
    ms = table('/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS', readonly=False)
    spw = table('/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')

    utime = np.unique(ms.getcol('TIME'))
    t = (utime-utime.min())/(utime.max() - utime.min())
    freq = spw.getcol('CHAN_FREQ').squeeze()
    freq0 = np.mean(freq)
    nu = 5*(freq/freq0 - 1.0)

    ntime = t.size
    nchan = nu.size
    nant = np.unique(ms.getcol('ANTENNA1')).size
    ncorr = ms.getcol('FLAG').shape[-1]

    from africanus.gps.utils import abs_diff
    tt = abs_diff(t, t)
    lt = 0.25
    Kt = 0.1*np.exp(-tt**2/(2*lt**2))
    Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))
    vv = abs_diff(nu, nu)
    lv = 0.1
    Kv = 0.1*np.exp(-vv**2/(2*lv**2))
    Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
    L = (Lt, Lv)

    from pfb.utils.misc import kron_matvec
    import matplotlib.pyplot as plt

    # gains
    jones = np.zeros((ntime, nant, nchan, 1, ncorr), dtype=np.complex128)
    for p in range(nant):
        for c in range(ncorr):
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

    # primary beam
    from katbeam import JimBeam
    beam = JimBeam('MKAT-AA-L-JIM-2020')
    l_coord = np.arange(-(nx//2), nx//2) * cell_size
    m_coord = np.arange(-(ny//2), ny//2) * cell_size
    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
    pbeam = np.zeros((nchan, nx, ny), dtype=np.float64)
    for i in range(nchan):
        pbeam[i] = beam.I(xx, yy, freq[i])


    # model vis
    model_att = pbeam * model
    from ducc0.wgridder import dirty2ms
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c:c+1, 0] = dirty2ms(uvw, freq[c:c+1], model[c],
                                      pixsize_x=cell_rad, pixsize_y=cell_rad,
                                      epsilon=1e-8, do_wstacking=True, nthreads=8)
        model_vis[:, c, -1] = model_vis[:, c, 0]

    # corrupted vis
    model_vis = model_vis.reshape(nrow, nchan, 1, 2, 2)
    from africanus.calibration.utils import chunkify_rows
    time = ms.getcol('TIME')
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, 1)
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    from africanus.calibration.utils import corrupt_vis
    vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, model_vis).reshape(nrow, nchan, ncorr)
    model_vis[...] = 1.0 + 0j
    muellercol = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, model_vis).reshape(nrow, nchan, ncorr)

    ms.putcol('DATA', vis.astype(np.complex64))
    ms.putcol('CORRECTED_DATA', muellercol.astype(np.complex64))
    ms.close()

    from  pfb.workers.grid.dirty import _dirty
    from contextlib import ExitStack

    with ExitStack() as stack:
        return _dirty('/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS', stack,
                      data_column="DATA", weight_column='WEIGHT_SPECTRUM',
                      flag_column='FLAG', mueller_column='CORRECTED_DATA',
                      row_chunks=None, epsilon=1e-5, wstack=True, mock=False,
                      double_accum=True, output_filename='/home/landman/Data/pfb-testing/output/dirty',
                      nband=nchan, field_of_view=2.0, super_resolution_factor=2.0,
                      cell_size=None, nx=None, ny=None, output_type='f4', nworkers=1,
                      nthreads_per_worker=1, ngridder_threads=8, mem_limit=None, nthreads=8,
                      host_address=None)







test_gainsinmodel()