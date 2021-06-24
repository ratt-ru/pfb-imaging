
def test_beaminmodel():
    import numpy as np
    from pyrap.tables import table
    ms = table('/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS', readonly=False)
    spw = table('/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS::SPECTRAL_WINDOW')

    freq = spw.getcol('CHAN_FREQ').squeeze()
    freq0 = np.mean(freq)
    nchan = freq.size

    ncorr = ms.getcol('FLAG').shape[-1]

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

    srf = 2.0
    cell_rad = cell_N / srf
    cell_size = cell_rad * 180 / np.pi
    print("Cell size set to %5.5e degrees" % cell_size)

    fov = 2.0
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
    l_coord = -np.arange(-(nx//2), nx//2) * cell_size
    m_coord = np.arange(-(ny//2), ny//2) * cell_size
    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
    pbeam = np.zeros((nchan, nx, ny), dtype=np.float64)
    for i in range(nchan):
        pbeam[i] = beam.I(xx, yy, freq[i]/1e6)  # freq in MHz

    # model vis
    model_att = pbeam * model
    from ducc0.wgridder import dirty2ms
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c:c+1, 0] = dirty2ms(uvw, freq[c:c+1], model_att[c],
                                      pixsize_x=cell_rad, pixsize_y=cell_rad,
                                      epsilon=1e-8, do_wstacking=True, nthreads=8)
        model_vis[:, c, -1] = model_vis[:, c, 0]


    ms.putcol('DATA', model_vis.astype(np.complex64))
    ms.close()

    from pfb.workers.grid.dirty import _dirty
    _dirty(ms='/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS',
           data_column="DATA", weight_column='WEIGHT', imaging_weight_column=None,
           flag_column='FLAG', mueller_column=None,
           row_chunks=None, epsilon=1e-5, wstack=True, mock=False,
           double_accum=True, output_filename='/home/landman/Data/pfb-testing/output/test',
           nband=nchan, field_of_view=fov, super_resolution_factor=srf,
           cell_size=None, nx=None, ny=None, output_type='f4', nworkers=1,
           nthreads_per_worker=1, nvthreads=8, mem_limit=8, nthreads=8,
           host_address=None)

    from pfb.workers.grid.psf import _psf
    _psf(ms='/home/landman/Data/Simulations/MS/test_ascii_1h60.0s.MS',
         data_column="DATA", weight_column='WEIGHT', imaging_weight_column=None,
         flag_column='FLAG', mueller_column=None, row_out_chunk=-1,
         row_chunks=None, epsilon=1e-5, wstack=True, mock=False, psf_oversize=2,
         double_accum=True, output_filename='/home/landman/Data/pfb-testing/output/test',
         nband=nchan, field_of_view=fov, super_resolution_factor=srf,
         cell_size=None, nx=None, ny=None, output_type='f4', nworkers=1,
         nthreads_per_worker=1, nvthreads=8, mem_limit=8, nthreads=8,
         host_address=None)

    # solve for model using pcg and mask
    mask = np.any(model, axis=0)
    from astropy.io import fits
    from pfb.utils.fits import save_fits
    hdr = fits.getheader('/home/landman/Data/pfb-testing/output/test_dirty.fits')
    save_fits('/home/landman/Data/pfb-testing/output/test_model.fits', model, hdr)
    save_fits('/home/landman/Data/pfb-testing/output/test_mask.fits', mask, hdr)

    from pfb.workers.deconv.forward import _forward
    _forward(residual='/home/landman/Data/pfb-testing/output/test_dirty.fits',
             psf='/home/landman/Data/pfb-testing/output/test_psf.fits',
             mask='/home/landman/Data/pfb-testing/output/test_mask.fits',
             beam_model='JimBeam', band='L',
             weight_table='/home/landman/Data/pfb-testing/output/test.zarr',
             output_filename='/home/landman/Data/pfb-testing/output/test',
             nband=nchan, output_type='f4', epsilon=1e-5, sigmainv=1e-5,
             wstack=True, double_accum=True, cg_tol=1e-5, cg_minit=10,
             cg_maxit=100, cg_verbose=0, cg_report_freq=10, backtrack=False,
             nworkers=1, nthreads_per_worker=1, nvthreads=8, mem_limit=8,
             nthreads=8, host_address=None)

    # get inferred model
    from pfb.utils.fits import load_fits
    model_inferred = load_fits('/home/landman/Data/pfb-testing/output/test_update.fits').squeeze()

    for i in range(nsource):
        print(model_inferred[:, Ix[i], Iy[i]] - model[:, Ix[i], Iy[i]])







test_beaminmodel()