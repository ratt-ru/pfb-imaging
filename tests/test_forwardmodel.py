import packratt
import pytest
from pathlib import Path

pmp = pytest.mark.parametrize

@pmp('do_beam', (False, True))
@pmp('do_gains', (False, True))
def test_forwardmodel(do_beam, do_gains, tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("test_pfb")

    packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from pyrap.tables import table

    ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
    spw = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW'))

    utime = np.unique(ms.getcol('TIME'))

    freq = spw.getcol('CHAN_FREQ').squeeze()
    freq0 = np.mean(freq)


    ntime = utime.size
    nchan = freq.size
    nant = np.maximum(ms.getcol('ANTENNA1').max(), ms.getcol('ANTENNA2').max()) + 1

    ncorr = ms.getcol('FLAG').shape[-1]

    uvw = ms.getcol('UVW')
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    from africanus.constants import c as lightspeed
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
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

    if do_beam:
        # primary beam
        from katbeam import JimBeam
        beam = JimBeam('MKAT-AA-L-JIM-2020')
        refpix = 1 + nx//2
        l_coord = -np.arange(1 - refpix, 1 + npix - refpix) * cell_size
        refpix = ny//2
        m_coord = np.arange(1 - refpix, 1 + npix - refpix) * cell_size
        xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
        pbeam = np.zeros((nchan, nx, ny), dtype=np.float64)
        for i in range(nchan):
            pbeam[i] = beam.I(xx, yy, freq[i]/1e6)  # freq in MHz
        model_att = pbeam * model
        bm = 'JimBeam'
    else:
        model_att = model
        bm = None

    # model vis
    from ducc0.wgridder import dirty2ms
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c:c+1, 0] = dirty2ms(uvw, freq[c:c+1], model_att[c],
                                      pixsize_x=cell_rad, pixsize_y=cell_rad,
                                      epsilon=1e-8, do_wstacking=True, nthreads=8)
        model_vis[:, c, -1] = model_vis[:, c, 0]

    ms.putcol('MODEL_DATA', model_vis.astype(np.complex64))

    if do_gains:
        t = (utime-utime.min())/(utime.max() - utime.min())
        nu = 2.5*(freq/freq0 - 1.0)

        from africanus.gps.utils import abs_diff
        tt = abs_diff(t, t)
        lt = 0.25
        Kt = 0.1 * np.exp(-tt**2/(2*lt**2))
        Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(ntime))
        vv = abs_diff(nu, nu)
        lv = 0.1
        Kv = 0.1 * np.exp(-vv**2/(2*lv**2))
        Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nchan))
        L = (Lt, Lv)

        from pfb.utils.misc import kron_matvec

        jones = np.zeros((ntime, nant, nchan, 1, ncorr), dtype=np.complex128)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_amp = np.random.randn(ntime, nchan)
                amp = np.exp(-nu[None, :]**2 + kron_matvec(L, xi_amp))
                xi_phase = np.random.randn(ntime, nchan)
                phase = kron_matvec(L, xi_phase)
                jones[:, p, :, 0, c] = amp * np.exp(1.0j * phase)

        # corrupted vis
        model_vis = model_vis.reshape(nrow, nchan, 1, 2, 2)
        from africanus.calibration.utils import chunkify_rows
        time = ms.getcol('TIME')
        row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, ntime)
        ant1 = ms.getcol('ANTENNA1')
        ant2 = ms.getcol('ANTENNA2')

        from africanus.calibration.utils import corrupt_vis
        vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, model_vis).reshape(nrow, nchan, ncorr)

        model_vis[:, :, 0, 0, 0] = 1.0 + 0j
        model_vis[:, :, 0, -1, -1] = 1.0 + 0j
        muellercol = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, model_vis).reshape(nrow, nchan, ncorr)

        ms.putcol('DATA', vis.astype(np.complex64))
        ms.putcol('CORRECTED_DATA', muellercol.astype(np.complex64))
        ms.close()
        mcol = 'CORRECTED_DATA'
    else:
        ms.putcol('DATA', model_vis.astype(np.complex64))
        mcol = None

    from pfb.workers.grid.dirty import _dirty
    _dirty(ms=str(test_dir / 'test_ascii_1h60.0s.MS'),
           data_column="DATA", weight_column='WEIGHT', imaging_weight_column=None,
           flag_column='FLAG', mueller_column=mcol,
           row_chunks=None, epsilon=1e-5, wstack=True, mock=False,
           double_accum=True, output_filename=str(test_dir / 'test'),
           nband=nchan, field_of_view=fov, super_resolution_factor=srf,
           cell_size=None, nx=None, ny=None, output_type='f4', nworkers=1,
           nthreads_per_worker=1, nvthreads=8, mem_limit=8, nthreads=8,
           host_address=None)

    from pfb.workers.grid.psf import _psf
    _psf(ms=str(test_dir / 'test_ascii_1h60.0s.MS'),
         data_column="DATA", weight_column='WEIGHT', imaging_weight_column=None,
         flag_column='FLAG', mueller_column=mcol, row_out_chunk=-1,
         row_chunks=None, epsilon=1e-5, wstack=True, mock=False, psf_oversize=2,
         double_accum=True, output_filename=str(test_dir / 'test'),
         nband=nchan, field_of_view=fov, super_resolution_factor=srf,
         cell_size=None, nx=None, ny=None, output_type='f4', nworkers=1,
         nthreads_per_worker=1, nvthreads=8, mem_limit=8, nthreads=8,
         host_address=None)

    # solve for model using pcg and mask
    mask = np.any(model, axis=0)
    from astropy.io import fits
    from pfb.utils.fits import save_fits
    hdr = fits.getheader(str(test_dir / 'test_dirty.fits'))
    save_fits(str(test_dir / 'test_model.fits'), model, hdr)
    save_fits(str(test_dir / 'test_mask.fits'), mask, hdr)

    from pfb.workers.deconv.forward import _forward
    _forward(residual=str(test_dir / 'test_dirty.fits'),
             psf=str(test_dir / 'test_psf.fits'),
             mask=str(test_dir / 'test_mask.fits'),
             beam_model=bm, band='L',
             weight_table=str(test_dir / 'test.zarr'),
             output_filename=str(test_dir / 'test'),
             nband=nchan, output_type='f4', epsilon=1e-5, sigmainv=0.0,
             wstack=True, double_accum=True, cg_tol=1e-6, cg_minit=10,
             cg_maxit=100, cg_verbose=0, cg_report_freq=10, backtrack=False,
             nworkers=1, nthreads_per_worker=1, nvthreads=1, mem_limit=8,
             nthreads=1, host_address=None)

    # get inferred model
    from pfb.utils.fits import load_fits
    model_inferred = load_fits(str(test_dir / 'test_update.fits')).squeeze()

    for i in range(nsource):
        # LB - only matches in apparent scale?
        if do_beam:
            beam = pbeam[:, Ix[i], Iy[i]]
            assert_allclose(0.0, beam * (model_inferred[:, Ix[i], Iy[i]] -
                            model[:, Ix[i], Iy[i]]), atol=1e-4)
        else:
            assert_allclose(0.0, model_inferred[:, Ix[i], Iy[i]] -
                            model[:, Ix[i], Iy[i]], atol=1e-4)
