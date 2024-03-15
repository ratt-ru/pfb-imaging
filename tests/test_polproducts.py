import pytest
from pathlib import Path
from xarray import Dataset
from collections import namedtuple
import dask
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
pmp = pytest.mark.parametrize

@pmp('do_gains', (False,True))
def test_polproducts(do_gains, ms_name):
    '''
    Tests polarisation products
    '''

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from daskms import xds_from_ms, xds_from_table, xds_to_table
    from daskms.experimental.zarr import xds_to_zarr
    from africanus.constants import c as lightspeed


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

    # first axis is Stokes
    model = np.zeros((4, nchan, nx, ny), dtype=np.float64)
    flux = {}
    flux['I'] = 1.0
    flux['Q'] = 0.6
    flux['U'] = 0.3
    flux['V'] = 0.1
    locx = int(3*npix//4)
    locy = int(npix//4)
    model[0, :, locx, locy] = flux['I']
    model[1, :, locx, locy] = flux['Q']
    model[2, :, locx, locy] = flux['U']
    model[3, :, locx, locy] = flux['V']

    # model vis
    epsilon = 1e-7
    from ducc0.wgridder.experimental import dirty2vis
    model_vis_I = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_I[:, c:c+1] = dirty2vis(uvw=uvw,
                                        freq=freq[c:c+1],
                                        dirty=model[0, c],
                                        pixsize_x=cell_rad,
                                        pixsize_y=cell_rad,
                                        epsilon=epsilon,
                                        do_wgridding=True,
                                        nthreads=8)
    model_vis_Q = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_Q[:, c:c+1] = dirty2vis(uvw=uvw,
                                        freq=freq[c:c+1],
                                        dirty=model[1, c],
                                        pixsize_x=cell_rad,
                                        pixsize_y=cell_rad,
                                        epsilon=epsilon,
                                        do_wgridding=True,
                                        nthreads=8)
    model_vis_U = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_U[:, c:c+1] = dirty2vis(uvw=uvw,
                                        freq=freq[c:c+1],
                                        dirty=model[2, c],
                                        pixsize_x=cell_rad,
                                        pixsize_y=cell_rad,
                                        epsilon=epsilon,
                                        do_wgridding=True,
                                        nthreads=8)
    model_vis_V = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_V[:, c:c+1] = dirty2vis(uvw=uvw,
                                        freq=freq[c:c+1],
                                        dirty=model[3, c],
                                        pixsize_x=cell_rad,
                                        pixsize_y=cell_rad,
                                        epsilon=epsilon,
                                        do_wgridding=True,
                                        nthreads=8)

    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    model_vis[:, :, 0] = model_vis_I + model_vis_Q
    model_vis[:, :, 1] = model_vis_U + 1j*model_vis_V
    model_vis[:, :, 2] = model_vis_U - 1j*model_vis_V
    model_vis[:, :, 3] = model_vis_I - model_vis_Q

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
        from pfb.utils.misc import chunkify_rows
        from africanus.calibration.utils import corrupt_vis
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
        time = xds.TIME.values
        row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, ntime)
        ant1 = xds.ANTENNA1.values
        ant2 = xds.ANTENNA2.values

        gains = np.swapaxes(jones, 1, 2).copy()
        vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                          gains, model_vis).reshape(nrow, nchan, ncorr)

        # add iid noise
        # vis += (np.random.randn(nrow, nchan, ncorr) +
        #         1j*np.random.randn(nrow, nchan, ncorr))/np.sqrt(2)

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
        out_path = f'{gain_path}::NET'
        dask.compute(xds_to_zarr(net_xds_list, out_path))

    else:
        # # add iid noise
        # model_vis += (np.random.randn(nrow, nchan, ncorr) +
        #               1j*np.random.randn(nrow, nchan, ncorr))/np.sqrt(2)

        xds['DATA'] = (('row','chan','corr'),
                       da.from_array(model_vis, chunks=(-1,-1,-1)))
        dask.compute(xds_to_table(xds, ms_name, columns='DATA'))
        gain_path = None

    postfix = "main"
    outname = str(test_dir / 'test')
    from pfb.parser.schemas import schema
    for p in ['I', 'Q', 'U', 'V']:
        basename = f'{outname}_{p}'
        dds_name = f'{basename}_{postfix}.dds'
        # set defaults from schema
        init_args = {}
        for key in schema.init["inputs"].keys():
            init_args[key.replace("-", "_")] = schema.init["inputs"][key]["default"]
        # overwrite defaults
        init_args["ms"] = str(test_dir / 'test_ascii_1h60.0s.MS')
        init_args["output_filename"] = outname
        init_args["data_column"] = "DATA"
        init_args["flag_column"] = 'FLAG'
        init_args["gain_table"] = gain_path
        init_args["max_field_of_view"] = fov*1.1
        init_args["overwrite"] = True
        init_args["channels_per_image"] = 1
        init_args["product"] = p
        from pfb.workers.init import _init
        xds = _init(**init_args)

        # grid data to produce dirty image
        grid_args = {}
        for key in schema.grid["inputs"].keys():
            grid_args[key.replace("-", "_")] = schema.grid["inputs"][key]["default"]
        # overwrite defaults
        grid_args["output_filename"] = outname
        grid_args["postfix"] = postfix
        grid_args["nband"] = nchan
        grid_args["field_of_view"] = fov
        grid_args["fits_mfs"] = True
        grid_args["psf"] = True
        grid_args["residual"] = False
        grid_args["nthreads"] = 8  # has to be set when calling _grid
        grid_args["nvthreads"] = 8
        grid_args["overwrite"] = True
        grid_args["robustness"] = 0.0
        grid_args["do_wgridding"] = True
        grid_args["product"] = p
        from pfb.workers.grid import _grid
        dds = _grid(xdsi=xds, **grid_args)

        dds = dask.compute(dds)[0]

        for ds in dds:
            wsum = ds.WSUM.values
            comp = ds.DIRTY.values[locx, locy]
            # print(flux[p], comp/wsum)
            assert_allclose(flux[p], comp/wsum, rtol=1e-4, atol=1e-4)


