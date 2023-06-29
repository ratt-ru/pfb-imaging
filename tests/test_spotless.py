import packratt
import pytest
from pathlib import Path
from xarray import Dataset
from collections import namedtuple
import dask
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
pmp = pytest.mark.parametrize


def test_spotless(tmp_path_factory):
    '''
    Here we test that the workers involved in a typical spotless pipeline
    all perform as expected.
    '''
    test_dir = tmp_path_factory.mktemp("test_pfb")
    # test_dir = Path('/home/landman/data/')
    packratt.get('/test/ms/2021-06-24/elwood/test_ascii_1h60.0s.MS.tar', str(test_dir))

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from pyrap.tables import table
    from pfb.utils.misc import Gaussian2D, give_edges
    import matplotlib.pyplot as plt

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
    nsource = 25
    border = np.maximum(int(0.15*nx), int(0.15*ny))
    Ix = np.random.randint(border, npix-border, nsource)
    Iy = np.random.randint(border, npix-border, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    I0 = 1.0 + np.exp(np.random.randn(nsource))
    extentx = np.random.randint(3, int(0.1*nx), nsource)
    extenty = np.random.randint(3, int(0.1*nx), nsource)
    pas = np.random.random(nsource) * 180
    x = -(nx/2) + np.arange(nx)
    y = -(nx/2) + np.arange(ny)
    xin, yin = np.meshgrid(x, y, indexing='ij')
    for i in range(nsource):
        emaj = np.maximum(extentx[i], extenty[i])
        emin = np.minimum(extentx[i], extenty[i])
        gauss = Gaussian2D(xin, yin, GaussPar=(emaj, emin, pas[i]))
        mx, my, gx, gy = give_edges(Ix[i], Iy[i], nx, ny, nx, ny)
        spectrum = I0[i] * (freq/freq0) ** alpha[i]
        model[:, mx, my] += spectrum[:, None, None] * gauss[None, gx, gy]

    # model vis
    epsilon = 1e-7
    from ducc0.wgridder.experimental import dirty2vis
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c:c+1, 0] = dirty2vis(uvw=uvw,
                                           freq=freq[c:c+1],
                                           dirty=model[c],
                                           pixsize_x=cell_rad,
                                           pixsize_y=cell_rad,
                                           epsilon=epsilon,
                                           do_wgridding=True,
                                           divide_by_n=False,
                                           flip_v=False,
                                           nthreads=8,
                                           sigma_min=1.1,
                                           sigma_max=3.0)
        model_vis[:, c, -1] = model_vis[:, c, 0]

    model_vis += (np.random.randn(nrow, nchan, ncorr) +
                  1.0j*np.random.randn(nrow, nchan, ncorr))

    ms.putcol('DATA', model_vis)

    # set defaults from schema
    from pfb.parser.schemas import schema
    init_args = {}
    for key in schema.init["inputs"].keys():
        init_args[key] = schema.init["inputs"][key]["default"]
    # overwrite defaults
    outname = str(test_dir / 'test')
    init_args["ms"] = str(test_dir / 'test_ascii_1h60.0s.MS')
    init_args["output_filename"] = outname
    init_args["data_column"] = "DATA"
    # init_args["weight_column"] = 'WEIGHT_SPECTRUM'
    init_args["flag_column"] = 'FLAG'
    init_args["gain_table"] = None
    init_args["max_field_of_view"] = fov
    init_args["overwrite"] = True
    init_args["channels_per_image"] = 1
    from pfb.workers.init import _init
    _init(**init_args)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key] = schema.grid["inputs"][key]["default"]
    # overwrite defaults
    grid_args["output_filename"] = outname
    grid_args["nband"] = nchan
    grid_args["field_of_view"] = fov
    grid_args["fits_mfs"] = False
    grid_args["psf"] = True
    grid_args["residual"] = False
    grid_args["nthreads_dask"] = 1
    grid_args["nvthreads"] = 8
    grid_args["overwrite"] = True
    grid_args["robustness"] = 0.0
    grid_args["wstack"] = True
    from pfb.workers.grid import _grid
    _grid(**grid_args)

    # run clean
    spotless_args = {}
    for key in schema.spotless["inputs"].keys():
        spotless_args[key] = schema.spotless["inputs"][key]["default"]
    spotless_args["output_filename"] = outname
    spotless_args["nband"] = nchan
    spotless_args["niter"] = 10
    tol = 1e-5
    spotless_args["tol"] = tol
    spotless_args["gamma"] = 1.0
    spotless_args["pd_tol"] = 5e-4
    spotless_args["rmsfactor"] = 0.85
    spotless_args["l1reweight_from"] = 5
    spotless_args["bases"] = 'self,db1,db2,db3'
    spotless_args["nlevels"] = 3
    spotless_args["nthreads_dask"] = 1
    spotless_args["nvthreads"] = 8
    spotless_args["scheduler"] = 'sync'
    spotless_args["wstack"] = True
    spotless_args["epsilon"] = epsilon
    spotless_args["fits_mfs"] = False
    from pfb.workers.spotless import _spotless
    _spotless(**spotless_args)

    # model2comps_args = {}
    # for key in schema.model2comps["inputs"].keys():
    #     model2comps_args[key] = schema.model2comps["inputs"][key]["default"]
    # model2comps_args["output_filename"] = outname
    # model2comps_args["spectral_poly_order"] = nchan
    # model2comps_args["fit_mode"] = 'poly'
    # model2comps_args["overwrite"] = True
    # from pfb.workers.model2comps import _model2comps
    # _model2comps(**model2comps_args)



    # mds_name = f'{basename}_main.coeffs.zarr'
    # mds = xds_from_zarr(mds_name)[0]

    # # grid spec
    # cell_rad = mds.cell_rad_x
    # cell_deg = np.rad2deg(cell_rad)
    # nx = mds.npix_x
    # ny = mds.npix_y
    # x0 = mds.center_x
    # y0 = mds.center_y
    # radec = (mds.ra, mds.dec)

    # # model func
    # ref_freq = mds.ref_freq
    # ref_time = mds.ref_time
    # params = sm.symbols(('t','f'))
    # params += sm.symbols(tuple(mds.params.values))
    # symexpr = parse_expr(mds.parametrisation)
    # modelf = lambdify(params, symexpr)

    # # model coeffs
    # coeffs = mds.coefficients.values
    # locx = mds.location_x.values
    # locy = mds.location_y.values

    # model_test = np.zeros((nchan, nx, ny), dtype=float)
    # tout = np.mean(utime)/ref_time
    # for b in range(nchan):
    #     fout = freq[b]/ref_freq
    #     model_test[b,locx,locy] = modelf(tout, fout, *coeffs[:, :])

    # import pdb; pdb.set_trace()

    # # we actually reconstruct I/n(l,m) so we need to correct for that
    # l, m = np.meshgrid(dds[0].x.values, dds[0].y.values,
    #                    indexing='ij')
    # eps = l**2+m**2
    # n = -eps/(np.sqrt(1.-eps)+1.) + 1  # more stable form
    # for i in range(nsource):
    #     assert_allclose(1.0 + model_inferred[:, Ix[i], Iy[i]] * n[Ix[i], Iy[i]] -
    #                     model[:, Ix[i], Iy[i]], 1.0,
    #                     atol=5*threshold)
    # TODO - currently we just check that this runs through.
    # What should the passing criteria be?
