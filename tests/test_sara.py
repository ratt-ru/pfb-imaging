import pytest
from pathlib import Path
from xarray import Dataset
from collections import namedtuple
import dask
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
pmp = pytest.mark.parametrize


def test_sara(ms_name):
    '''
    # TODO - currently we just check that this runs through.
    # What should the passing criteria be?
    '''
    robustness = None
    do_wgridding = True

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
    import dask
    import xarray as xr
    from daskms import xds_from_ms, xds_from_table, xds_to_table
    from pfb.utils.naming import xds_from_url
    from pfb.utils.misc import Gaussian2D, give_edges
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from ducc0.wgridder.experimental import dirty2vis
    from pfb.operators.gridder import wgridder_conventions
    from pfb.parser.schemas import schema
    from pfb.workers.init import _init
    from pfb.workers.grid import _grid
    from pfb.workers.sara import _sara
    from pfb.workers.model2comps import _model2comps
    from pfb.workers.degrid import _degrid
    import sympy as sm
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr


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
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    epsilon = 1e-7
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c:c+1, 0] = dirty2vis(uvw=uvw,
                                           freq=freq[c:c+1],
                                           dirty=model[c],
                                           pixsize_x=cell_rad,
                                           pixsize_y=cell_rad,
                                           epsilon=epsilon,
                                           do_wgridding=do_wgridding,
                                           divide_by_n=False,
                                           flip_u=flip_u,
                                           flip_v=flip_v,
                                           flip_w=flip_w,
                                           nthreads=8,
                                           sigma_min=1.1,
                                           sigma_max=3.0)
        model_vis[:, c, -1] = model_vis[:, c, 0]

    model_vis += (np.random.randn(nrow, nchan, ncorr) +
                  1.0j*np.random.randn(nrow, nchan, ncorr))

    model_vis = da.from_array(model_vis, chunks=(-1,-1,-1))
    xds['DATA'] = (('row','chan','coor'), model_vis)
    writes = [xds_to_table(xds, ms_name, columns='DATA')]
    dask.compute(writes)

    # set defaults from schema
    init_args = {}
    for key in schema.init["inputs"].keys():
        init_args[key.replace("-", "_")] = schema.init["inputs"][key]["default"]
    # overwrite defaults
    outname = str(test_dir / 'test_I')
    init_args["ms"] = [str(test_dir / 'test_ascii_1h60.0s.MS')]
    init_args["output_filename"] = outname
    init_args["data_column"] = "DATA"
    # init_args["weight_column"] = 'WEIGHT_SPECTRUM'
    init_args["flag_column"] = 'FLAG'
    init_args["gain_table"] = None
    init_args["max_field_of_view"] = fov*1.1
    init_args["overwrite"] = True
    init_args["channels_per_image"] = 1
    _init(**init_args)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key.replace("-", "_")] = schema.grid["inputs"][key]["default"]
    # overwrite defaults
    grid_args["output_filename"] = outname
    grid_args["field_of_view"] = fov
    grid_args["fits_mfs"] = False
    grid_args["psf"] = True
    grid_args["residual"] = False
    grid_args["nthreads"] = 8
    grid_args["overwrite"] = True
    grid_args["robustness"] = robustness
    grid_args["do_wgridding"] = do_wgridding
    _grid(**grid_args)

    dds_name = f'{outname}_main.dds'

    # run sara
    sara_args = {}
    for key in schema.sara["inputs"].keys():
        sara_args[key.replace("-", "_")] = schema.sara["inputs"][key]["default"]
    sara_args["output_filename"] = outname
    sara_args["niter"] = 2
    tol = 1e-5
    sara_args["tol"] = tol
    sara_args["gamma"] = 1.0
    sara_args["pd_tol"] = [1e-3]
    sara_args["rmsfactor"] = 1.0
    sara_args["epsfactor"] = 4.0
    sara_args["l1_reweight_from"] = 5
    sara_args["bases"] = 'self,db1'
    sara_args["nlevels"] = 3
    sara_args["nthreads"] = 8
    sara_args["do_wgridding"] = do_wgridding
    sara_args["epsilon"] = epsilon
    sara_args["fits_mfs"] = False
    _sara(**sara_args)


    # the residual computed by the grid worker should be identical
    # to that computed in sara when transferring model
    dds, _ = xds_from_url(dds_name)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key.replace("-", "_")] = schema.grid["inputs"][key]["default"]
    # overwrite defaults
    grid_args["output_filename"] = outname
    grid_args["field_of_view"] = fov
    grid_args["fits_mfs"] = False
    grid_args["psf"] = False
    grid_args["weight"] = False
    grid_args["noise"] = False
    grid_args["residual"] = True
    grid_args["nthreads"] = 8
    grid_args["overwrite"] = True
    grid_args["robustness"] = robustness
    grid_args["do_wgridding"] = do_wgridding
    grid_args["transfer_model_from"] = f'{outname}_main_model.mds'
    grid_args["suffix"] = 'subtract'
    _grid(**grid_args)

    dds2, _ = xds_from_url(f'{outname}_subtract.dds')

    for ds, ds2 in zip(dds, dds2):
        wsum = ds.WSUM.values
        assert_allclose(1 + np.abs(ds.RESIDUAL.values)/wsum,
                        1 + np.abs(ds2.RESIDUAL.values)/wsum)

    # residuals also need to be the same if we do the
    # subtraction in visibility space
    degrid_args = {}
    for key in schema.degrid["inputs"].keys():
        degrid_args[key.replace("-", "_")] = schema.degrid["inputs"][key]["default"]
    degrid_args["ms"] = [str(test_dir / 'test_ascii_1h60.0s.MS')]
    degrid_args["mds"] = f'{outname}_main_model.mds'
    degrid_args["dds"] = f'{outname}_main.dds'
    degrid_args["channels_per_image"] = 1
    degrid_args["nthreads"] = 8
    degrid_args["do_wgridding"] = do_wgridding
    _degrid(**degrid_args)

    init_args = {}
    for key in schema.init["inputs"].keys():
        init_args[key.replace("-", "_")] = schema.init["inputs"][key]["default"]
    # overwrite defaults
    outname = str(test_dir / 'test2_I')
    init_args["ms"] = [str(test_dir / 'test_ascii_1h60.0s.MS')]
    init_args["output_filename"] = outname
    init_args["data_column"] = "DATA-MODEL_DATA"
    # init_args["weight_column"] = 'WEIGHT_SPECTRUM'
    init_args["flag_column"] = 'FLAG'
    init_args["gain_table"] = None
    init_args["max_field_of_view"] = fov*1.1
    init_args["bda_decorr"] = 1.0
    init_args["overwrite"] = True
    init_args["channels_per_image"] = 1
    _init(**init_args)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key.replace("-", "_")] = schema.grid["inputs"][key]["default"]
    # overwrite defaults
    grid_args["output_filename"] = outname
    grid_args["field_of_view"] = fov
    grid_args["fits_mfs"] = False
    grid_args["psf"] = False
    grid_args["residual"] = False
    grid_args["nthreads"] = 8
    grid_args["overwrite"] = True
    grid_args["robustness"] = robustness
    grid_args["do_wgridding"] = do_wgridding
    _grid(**grid_args)

    dds_name = f'{outname}_main.dds'

    dds2, _ = xds_from_url(dds_name)

    for ds, ds2 in zip(dds, dds2):
        wsum = ds.WSUM.values
        assert_allclose(1 + np.abs(ds.RESIDUAL.values)/wsum,
                        1 + np.abs(ds2.DIRTY.values)/wsum)

# test_sara()
