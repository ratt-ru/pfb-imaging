import pytest
from pathlib import Path
from xarray import Dataset
from collections import namedtuple
import dask
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
pmp = pytest.mark.parametrize


def test_spotless(ms_name):
    '''
    # TODO - currently we just check that this runs through.
    # What should the passing criteria be?
    '''

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    import dask
    from daskms import xds_from_ms, xds_from_table, xds_to_table
    from pfb.utils.misc import Gaussian2D, give_edges
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from ducc0.wgridder.experimental import dirty2vis
    from pfb.parser.schemas import schema
    from pfb.workers.init import _init
    from pfb.workers.grid import _grid
    from pfb.workers.spotless import _spotless
    from pfb.workers.model2comps import _model2comps
    from pfb.workers.degrid import _degrid
    import sympy as sm
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr


    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
    # ms = table(str(test_dir / 'test_ascii_1h60.0s.MS'), readonly=False)
    # spw = table(str(test_dir / 'test_ascii_1h60.0s.MS::SPECTRAL_WINDOW'))

    # utime = np.unique(ms.getcol('TIME'))
    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    # freq = spw.getcol('CHAN_FREQ').squeeze()
    freq0 = np.mean(freq)

    ntime = utime.size
    nchan = freq.size
    # nant = np.maximum(ms.getcol('ANTENNA1').max(), ms.getcol('ANTENNA2').max()) + 1
    nant = np.maximum(xds.ANTENNA1.values.max(), xds.ANTENNA2.values.max()) + 1

    # ncorr = ms.getcol('FLAG').shape[-1]
    ncorr = xds.corr.size

    # uvw = ms.getcol('UVW')
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
    epsilon = 1e-7
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

    model_vis = da.from_array(model_vis, chunks=(-1,-1,-1))
    xds['DATA'] = (('row','chan','coor'), model_vis)
    writes = [xds_to_table(xds, ms_name, columns='DATA')]
    dask.compute(writes)

    # set defaults from schema
    init_args = {}
    for key in schema.init["inputs"].keys():
        init_args[key.replace("-", "_")] = schema.init["inputs"][key]["default"]
    # overwrite defaults
    outname = str(test_dir / 'test')
    init_args["ms"] = str(test_dir / 'test_ascii_1h60.0s.MS')
    init_args["output_filename"] = outname
    init_args["data_column"] = "DATA"
    # init_args["weight_column"] = 'WEIGHT_SPECTRUM'
    init_args["flag_column"] = 'FLAG'
    init_args["gain_table"] = None
    init_args["max_field_of_view"] = fov*1.1
    init_args["overwrite"] = True
    init_args["channels_per_image"] = 1
    xdso = _init(**init_args)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key.replace("-", "_")] = schema.grid["inputs"][key]["default"]
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
    grid_args["do_wgridding"] = True
    dds = _grid(xdsi=xdso, **grid_args)

    # LB - does this avoid duplicate gridding?
    dds_name = f'{outname}_I_main.dds'
    dds = dask.compute(dds)[0]
    writes = xds_to_zarr(dds, dds_name, columns='ALL')
    dask.compute(writes)

    # run clean
    spotless_args = {}
    for key in schema.spotless["inputs"].keys():
        spotless_args[key.replace("-", "_")] = schema.spotless["inputs"][key]["default"]
    spotless_args["output_filename"] = outname
    spotless_args["nband"] = nchan
    spotless_args["niter"] = 2
    tol = 1e-5
    spotless_args["tol"] = tol
    spotless_args["gamma"] = 1.0
    spotless_args["pd_tol"] = 5e-4
    spotless_args["rmsfactor"] = 0.1
    spotless_args["l1reweight_from"] = 5
    spotless_args["bases"] = 'self,db1,db2,db3'
    spotless_args["nlevels"] = 3
    spotless_args["nthreads_dask"] = 1
    spotless_args["nvthreads"] = 8
    spotless_args["scheduler"] = 'sync'
    spotless_args["do_wgridding"] = True
    spotless_args["epsilon"] = epsilon
    spotless_args["fits_mfs"] = False
    _spotless(ddsi=dds, **spotless_args)


    # get the inferred model
    dds = xds_from_zarr(f'{outname}_I_main.dds')
    freqs_dds = []
    times_dds = []
    for ds in dds:
        freqs_dds.append(ds.freq_out)
        times_dds.append(ds.time_out)

    freqs_dds = np.array(freqs_dds)
    times_dds = np.array(times_dds)
    freqs_dds = np.unique(freqs_dds)
    times_dds = np.unique(times_dds)
    ntime_dds = times_dds.size
    nfreq_dds = freqs_dds.size

    model_inferred = np.zeros((ntime_dds, nfreq_dds, nx, ny))
    for ds in dds:
        model_inferred[ds.timeid, ds.bandid, :, :] = ds.MODEL.values

    model2comps_args = {}
    for key in schema.model2comps["inputs"].keys():
        model2comps_args[key.replace("-", "_")] = schema.model2comps["inputs"][key]["default"]
    model2comps_args["output_filename"] = outname
    model2comps_args["nbasisf"] = nchan
    model2comps_args["fit_mode"] = 'Legendre'
    model2comps_args["overwrite"] = True
    model2comps_args["use_wsum"] = False
    model2comps_args["sigmasq"] = 1e-14
    _model2comps(**model2comps_args)

    mds_name = f'{outname}_I_main_model.mds'
    mds = xds_from_zarr(mds_name)[0]

    # grid spec
    cell_rad = mds.cell_rad_x
    cell_deg = np.rad2deg(cell_rad)
    nx = mds.npix_x
    ny = mds.npix_y
    x0 = mds.center_x
    y0 = mds.center_y
    radec = (mds.ra, mds.dec)

    # model func
    params = sm.symbols(('t','f'))
    params += sm.symbols(tuple(mds.params.values))
    symexpr = parse_expr(mds.parametrisation)
    modelf = lambdify(params, symexpr)
    texpr = parse_expr(mds.texpr)
    tfunc = lambdify(params[0], texpr)
    fexpr = parse_expr(mds.fexpr)
    ffunc = lambdify(params[1], fexpr)

    # model coeffs
    coeffs = mds.coefficients.values
    locx = mds.location_x.values
    locy = mds.location_y.values

    model_test = np.zeros((ntime_dds, nfreq_dds, nx, ny), dtype=float)
    for i in range(ntime_dds):
        tout = tfunc(times_dds[i])
        for j in range(nchan):
            fout = ffunc(freqs_dds[j])
            model_test[i,j,locx,locy] = modelf(tout, fout, *coeffs)

    # models need to match exactly
    assert_allclose(1 + model_test, 1 + model_inferred)

    # degrid from coeffs populating MODEL_DATA
    degrid_args = {}
    for key in schema.degrid["inputs"].keys():
        degrid_args[key.replace("-", "_")] = schema.degrid["inputs"][key]["default"]
    degrid_args["ms"] = str(test_dir / 'test_ascii_1h60.0s.MS')
    degrid_args["mds"] = f'{outname}_I_main_model.mds'
    degrid_args["channels_per_image"] = 1
    degrid_args["nthreads_dask"] = 1
    degrid_args["nvthreads"] = 8
    degrid_args["do_wgridding"] = True
    _degrid(**degrid_args)

    # manually place residual in CORRECTED_DATA
    resid = xds.DATA.data - xds.MODEL_DATA.data
    xds['CORRECTED_DATA'] = (('row','chan','coor'), resid)
    writes = [xds_to_table(xds, ms_name, columns='CORRECTED_DATA')]
    dask.compute(writes)

    # gridding CORRECTED_DATA should return identical residuals
    init_args = {}
    for key in schema.init["inputs"].keys():
        init_args[key.replace("-", "_")] = schema.init["inputs"][key]["default"]
    # overwrite defaults
    outname = str(test_dir / 'test2')
    init_args["ms"] = str(test_dir / 'test_ascii_1h60.0s.MS')
    init_args["output_filename"] = outname
    init_args["data_column"] = "CORRECTED_DATA"
    # init_args["weight_column"] = 'WEIGHT_SPECTRUM'
    init_args["flag_column"] = 'FLAG'
    init_args["gain_table"] = None
    init_args["max_field_of_view"] = fov*1.1
    init_args["overwrite"] = True
    init_args["channels_per_image"] = 1
    xdso2 = _init(**init_args)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key.replace("-", "_")] = schema.grid["inputs"][key]["default"]
    # overwrite defaults
    grid_args["output_filename"] = outname
    grid_args["nband"] = nchan
    grid_args["field_of_view"] = fov
    grid_args["fits_mfs"] = False
    grid_args["psf"] = False
    grid_args["residual"] = False
    grid_args["nthreads_dask"] = 1
    grid_args["nvthreads"] = 8
    grid_args["overwrite"] = True
    grid_args["robustness"] = 0.0
    grid_args["do_wgridding"] = True
    dds2 = _grid(xdsi=xdso2, **grid_args)

    dds2 = dask.compute(dds2)[0]

    for ds, ds2 in zip(dds, dds2):
        wsum = ds.WSUM.values
        assert_allclose(1 + np.abs(ds.RESIDUAL.values)/wsum,
                        1 + np.abs(ds2.DIRTY.values)/wsum)

# test_spotless()
