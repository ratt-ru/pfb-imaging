import pytest
from pathlib import Path
from xarray import Dataset
from collections import namedtuple
import dask
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
pmp = pytest.mark.parametrize

@pmp('do_gains', (True, False))
def test_klean(do_gains, ms_name):
    '''
    Here we test that clean correctly infers the fluxes of point sources
    placed at the centers of pixels in the presence of the wterm and DI gain
    corruptions.
    TODO - add per scan PB variations
    '''

    # we need the client for the init step
    from dask.distributed import LocalCluster, Client
    cluster = LocalCluster(processes=False,
                           n_workers=1,
                           threads_per_worker=1,
                           memory_limit=0,  # str(mem_limit/nworkers)+'GB'
                           asynchronous=False)
    client = Client(cluster, direct_to_workers=False)

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from daskms import xds_from_ms, xds_from_table, xds_to_table
    from daskms.experimental.zarr import xds_to_zarr
    from pfb.utils.naming import xds_from_url
    from africanus.constants import c as lightspeed
    from ducc0.wgridder.experimental import dirty2vis
    from pfb.operators.gridder import wgridder_conventions


    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]

    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    freq0 = np.mean(freq)

    ntime = utime.size
    nchan = freq.size
    nant = np.maximum(xds.ANTENNA1.values.max(), xds.ANTENNA2.values.max()) + 1

    ncorr = xds.corr.size

    uvw = xds.UVW.values
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600
    print("Cell size set to %5.5e arcseconds" % cell_size)

    from ducc0.fft import good_size
    # the test will fail in intrinsic if sources fall near beam sidelobes
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

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
    epsilon = 1e-7
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c:c+1, 0] = dirty2vis(uvw=uvw,
                                           freq=freq[c:c+1],
                                           dirty=model[c],
                                           pixsize_x=cell_rad,
                                           pixsize_y=cell_rad,
                                           epsilon=epsilon,
                                           flip_u=flip_u,
                                           flip_v=flip_v,
                                           flip_w=flip_w,
                                           do_wgridding=True,
                                           nthreads=8)
        model_vis[:, c, -1] = model_vis[:, c, 0]


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
        jones = np.zeros((ntime, nchan, nant, 1, 2), dtype=np.complex128)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_amp = np.random.randn(ntime, nchan)
                amp = np.exp(-nu[None, :]**2 + kron_matvec(L, xi_amp))
                xi_phase = np.random.randn(ntime, nchan)
                phase = kron_matvec(L, xi_phase)
                jones[:, :, p, 0, c] = amp * np.exp(1.0j * phase)

        # corrupted vis
        model_vis = model_vis.reshape(nrow, nchan, 1, 2, 2)
        from pfb.utils.misc import chunkify_rows
        time = xds.TIME.values
        row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, ntime)
        ant1 = xds.ANTENNA1.values
        ant2 = xds.ANTENNA2.values

        from africanus.calibration.utils import corrupt_vis
        gains = np.swapaxes(jones, 1, 2).copy()
        vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                          gains, model_vis).reshape(nrow, nchan, ncorr)


        xds['DATA'] = (('row','chan','corr'),
                       da.from_array(vis, chunks=(-1,-1,-1)))
        dask.compute(xds_to_table(xds, ms_name, columns='DATA'))

        # cast gain to QuartiCal format
        g = da.from_array(jones)
        gflags = da.zeros((ntime, nchan, nant, 1))
        data_vars = {
            'gains':(('gain_time', 'gain_freq', 'antenna', 'direction', 'correlation'), g),
            'gain_flags':(('gain_time', 'gain_freq', 'antenna', 'direction'), gflags)
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
                                       cchunk=(int(2),)),
            'GAIN_AXES': ('gain_time', 'gain_freq', 'antenna', 'direction', 'correlation')
        }
        coords = {
            'gain_freq': (('gain_freq',), freq),
            'gain_time': (('gain_time',), utime)

        }
        net_xds_list = Dataset(data_vars, coords=coords, attrs=attrs)
        gain_path = str(test_dir / Path("gains.qc"))
        dask.compute(xds_to_zarr(net_xds_list, f'{gain_path}::NET'))
        gain_path = [f'{gain_path}/NET']

    else:
        xds['DATA'] = (('row','chan','corr'),
                       da.from_array(model_vis, chunks=(-1,-1,-1)))
        dask.compute(xds_to_table(xds, ms_name, columns='DATA'))
        gain_path = None

    outname = str(test_dir / 'test_I')
    dds_name = f'{outname}_main.dds'
    # set defaults from schema
    from pfb.parser.schemas import schema
    init_args = {}
    for key in schema.init["inputs"].keys():
        init_args[key.replace("-", "_")] = schema.init["inputs"][key]["default"]
    # overwrite defaults
    init_args["ms"] = [str(test_dir / 'test_ascii_1h60.0s.MS')]
    init_args["output_filename"] = outname
    init_args["data_column"] = "DATA"
    init_args["flag_column"] = 'FLAG'
    init_args["gain_table"] = gain_path
    init_args["max_field_of_view"] = fov*1.1
    init_args["overwrite"] = True
    init_args["channels_per_image"] = 1
    init_args["bda_decorr"] = 1.0
    from pfb.workers.init import _init
    _init(**init_args)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key.replace("-", "_")] = schema.grid["inputs"][key]["default"]
    # overwrite defaults
    grid_args["output_filename"] = outname
    grid_args["field_of_view"] = fov
    grid_args["fits_mfs"] = True
    grid_args["psf"] = True
    grid_args["residual"] = False
    grid_args["nthreads"] = 8
    grid_args["overwrite"] = True
    grid_args["robustness"] = 0.0
    grid_args["do_wgridding"] = True
    grid_args["psf_oversize"] = 2.0
    from pfb.workers.grid import _grid
    _grid(**grid_args)

    # run klean
    klean_args = {}
    for key in schema.klean["inputs"].keys():
        klean_args[key.replace("-", "_")] = schema.klean["inputs"][key]["default"]
    klean_args["output_filename"] = outname
    klean_args["dirosion"] = 0
    klean_args["do_residual"] = False
    klean_args["niter"] = 100
    threshold = 1e-1
    klean_args["threshold"] = threshold
    klean_args["gamma"] = 0.1
    klean_args["peak_factor"] = 0.75
    klean_args["sub_peak_factor"] = 0.75
    klean_args["nthreads"] = 8
    klean_args["do_wgridding"] = True
    klean_args["epsilon"] = epsilon
    klean_args["mop_flux"] = True
    klean_args["fits_mfs"] = False
    from pfb.workers.klean import _klean
    _klean(**klean_args)

    # get inferred model
    dds = xds_from_url(dds_name)
    model_inferred = np.zeros((nchan, nx, ny))
    for ds in dds:
        # import ipdb; ipdb.set_trace()
        b = int(ds.bandid)
        model_inferred[b] = ds.MODEL.values

    # we actually reconstruct I/n(l,m) so we need to correct for that
    l, m = np.meshgrid(dds[0].x.values, dds[0].y.values,
                       indexing='ij')
    eps = l**2+m**2
    n = -eps/(np.sqrt(1.-eps)+1.) + 1  # more stable form
    for i in range(nsource):
        assert_allclose(1.0 + model_inferred[:, Ix[i], Iy[i]] * n[Ix[i], Iy[i]] -
                        model[:, Ix[i], Iy[i]], 1.0,
                        atol=5*threshold)
