import packratt
import pytest
from pathlib import Path
from daskms import Dataset
from collections import namedtuple
import dask
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
pmp = pytest.mark.parametrize

@pmp('do_beam', (False, True))
@pmp('do_gains', (False, True))
def test_forwardmodel(do_beam, do_gains, epsilon=1e-10):# tmp_path_factory):
    #test_dir = tmp_path_factory.mktemp("test_pfb")
    test_dir = Path('/home/landman/data/')
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
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600
    print("Cell size set to %5.5e arcseconds" % cell_size)

    fov = 2.0
    npix = int(fov / cell_deg)
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
                                      epsilon=epsilon, do_wstacking=True, nthreads=8)
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

        jones = np.zeros((ntime, nchan, nant, 1, ncorr), dtype=np.complex128)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_amp = np.random.randn(ntime, nchan)
                amp = np.exp(-nu[None, :]**2 + kron_matvec(L, xi_amp))
                xi_phase = np.random.randn(ntime, nchan)
                phase = kron_matvec(L, xi_phase)
                jones[:, :, p, 0, c] = amp * np.exp(1.0j * phase)

        # corrupted vis
        model_vis = model_vis.reshape(nrow, nchan, 1, 2, 2)
        from africanus.calibration.utils import chunkify_rows
        time = ms.getcol('TIME')
        row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, ntime)
        ant1 = ms.getcol('ANTENNA1')
        ant2 = ms.getcol('ANTENNA2')

        from africanus.calibration.utils import corrupt_vis
        vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                          np.swapaxes(jones, 1, 2), model_vis).reshape(nrow, nchan, ncorr)
        ms.putcol('DATA', vis.astype(np.complex64))

        # cast gain to QuartiCal format
        g = da.from_array(jones)
        gflags = da.zeros((ntime, nchan, nant, 1))
        data_vars = {
            'gains':(('gain_t', 'gain_f', 'ant', 'dir', 'corr'), g),
            'gain_flags':(('gain_t', 'gain_f', 'ant', 'dir'), gflags)
        }
        gain_spec_tup = namedtuple('gains_spec_tup', 'tchunk fchunk achunk dchunk cchunk')
        attrs = {
            'NAME': 'NET',
            'TYPE': 'complex',
            'FIELD_NAME': '00',
            'SCAN_NUMBER': int(1),
            'FIELD_ID': int(0),
            'DATA_DESC_ID': int(0),
            'GAIN_SPEC': gain_spec_tup(tchunk=(int(ntime),),
                                       fchunk=(int(nchan),),
                                       achunk=(int(nant),),
                                       dchunk=(int(1),),
                                       cchunk=(int(ncorr),)),
            'GAIN_AXES': ('gain_t', 'gain_f', 'ant', 'dir', 'corr')
        }
        coords = {
            'gain_f': (('gain_f',), freq)

        }
        net_xds_list = Dataset(data_vars, coords=coords, attrs=attrs)
        gain_path = str(test_dir / Path("gains.qc"))
        out_path = f'{gain_path}::NET'
        dask.compute(xds_to_zarr(net_xds_list, out_path))

    else:
        ms.putcol('DATA', model_vis.astype(np.complex64))
        gain_path = None

    from pfb.workers.grid import _grid
    outname = str(test_dir / 'test')
    _grid(ms=str(test_dir / 'test_ascii_1h60.0s.MS'),
          data_column="DATA", weight_column=None, imaging_weight_column=None,
          flag_column='FLAG', gain_table=gain_path, product='I',
          utimes_per_chunk=-1, row_out_chunk=10000, epsilon=epsilon,
          precision='double', group_by_field=True, group_by_scan=True,
          group_by_ddid=True, wstack=True, double_accum=True,
          fits_mfs=True, no_fits_cubes=True, psf=True, dirty=True,
          weights=True, bda_weights=False, do_beam=do_beam,
          output_filename=outname, nband=nchan,
          field_of_view=fov, super_resolution_factor=srf,
          psf_oversize=2, cell_size=float(cell_size), nx=nx, ny=ny, nworkers=1,
          nthreads_per_worker=1, nvthreads=8, mem_limit=8, nthreads=8,
          host_address=None, scheduler='single-threaded')

    # place mask in mds
    mask = np.any(model, axis=0)
    basename = f'{outname}_I'
    mds_name = f'{basename}.mds.zarr'
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    mds = mds.assign(**{
                'MASK': (('x', 'y'), da.from_array(mask, chunks=(-1, -1)))
        })
    dask.compute(xds_to_zarr(mds, mds_name, columns='ALL'))


    from pfb.workers.deconv.forward import _forward
    _forward(output_filename=outname, residual_name='DIRTY',
             mask='mds', nband=nchan, product='I', row_chunk=-1,
             epsilon=epsilon, sigmainv=0.0, wstack=True, double_accum=True,
             use_psf=False, fits_mfs=True, no_fits_cubes=True,
             do_residual=False, cg_tol=1e-10, cg_minit=0,
             cg_maxit=100, cg_verbose=2, cg_report_freq=1, backtrack=False,
             nworkers=1, nthreads_per_worker=1, nvthreads=8, mem_limit=8, nthreads=8,
             host_address=None, scheduler='single-threaded')

    # get inferred model
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    model_inferred = mds.UPDATE.values

    # import ipdb; ipdb.set_trace()

    for i in range(nsource):
        # LB - only matches in apparent scale?
        print(np.abs(model_inferred[:, Ix[i], Iy[i]]/model[:, Ix[i], Iy[i]]))
        # if do_beam:
        #     beam = pbeam[:, Ix[i], Iy[i]]
        #     assert_allclose(0.0, beam * (model_inferred[:, Ix[i], Iy[i]] -
        #                     model[:, Ix[i], Iy[i]]), atol=1e-4)
        # else:
        #     assert_allclose(0.0, model_inferred[:, Ix[i], Iy[i]] -
        #                     model[:, Ix[i], Iy[i]], atol=1e-4)


test_forwardmodel(False, True)
