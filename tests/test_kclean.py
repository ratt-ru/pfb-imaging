from collections import namedtuple
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
from africanus.calibration.utils import corrupt_vis
from africanus.constants import c as lightspeed
from africanus.gps.utils import abs_diff
from daskms import xds_from_ms, xds_from_table, xds_to_table
from daskms.experimental.zarr import xds_to_zarr
from ducc0.fft import good_size
from ducc0.wgridder import dirty2vis
from numpy.testing import assert_allclose
from xarray import Dataset

from pfb_imaging.core.grid import grid as grid_core
from pfb_imaging.core.init import init as init_core
from pfb_imaging.core.kclean import kclean as kclean_core
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.misc import chunkify_rows, kron_matvec
from pfb_imaging.utils.naming import xds_from_url

pmp = pytest.mark.parametrize


@pmp("do_gains", (True, False))
def test_kclean(do_gains, ms_name):
    """
    Here we test that clean correctly infers the fluxes of point sources
    placed at the centers of pixels in the presence of the wterm and DI gain
    corruptions.
    TODO - add per scan PB variations
    """
    np.random.seed(420)

    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1})[0]
    spw = xds_from_table(f"{ms_name}::SPECTRAL_WINDOW")[0]

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
    cell_n = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_n / srf
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
    nsource = 10
    x_index = np.random.randint(0, npix, nsource)
    y_index = np.random.randint(0, npix, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    ref_flux = 1.0 + np.abs(np.random.randn(nsource))
    for i in range(nsource):
        model[:, x_index[i], y_index[i]] = ref_flux[i] * (freq / freq0) ** alpha[i]

    # model vis
    epsilon = 1e-7
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c : c + 1, 0] = dirty2vis(
            uvw=uvw,
            freq=freq[c : c + 1],
            dirty=model[c],
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            epsilon=epsilon,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            do_wgridding=True,
            nthreads=2,
        )
        model_vis[:, c, -1] = model_vis[:, c, 0]

    if do_gains:
        t = (utime - utime.min()) / (utime.max() - utime.min())
        nu = 2.5 * (freq / freq0 - 1.0)

        tt = abs_diff(t, t)
        lt = 0.25
        cov_t = 0.1 * np.exp(-(tt**2) / (2 * lt**2))
        chol_t = np.linalg.cholesky(cov_t + 1e-10 * np.eye(ntime))
        vv = abs_diff(nu, nu)
        lv = 0.1
        cov_nu = 0.1 * np.exp(-(vv**2) / (2 * lv**2))
        chol_nu = np.linalg.cholesky(cov_nu + 1e-10 * np.eye(nchan))
        chol_tnu = (chol_t, chol_nu)

        jones = np.zeros((ntime, nchan, nant, 1, 2), dtype=np.complex128)
        for p in range(nant):
            for c in [0, -1]:  # for now only diagonal
                xi_amp = np.random.randn(ntime, nchan)
                amp = np.exp(-(nu[None, :] ** 2) + kron_matvec(chol_tnu, xi_amp))
                xi_phase = np.random.randn(ntime, nchan)
                phase = kron_matvec(chol_tnu, xi_phase)
                jones[:, :, p, 0, c] = amp * np.exp(1.0j * phase)

        # corrupted vis
        model_vis = model_vis.reshape(nrow, nchan, 1, 2, 2)
        time = xds.TIME.values
        row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, ntime)
        ant1 = xds.ANTENNA1.values
        ant2 = xds.ANTENNA2.values

        gains = np.swapaxes(jones, 1, 2).copy()
        vis = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, gains, model_vis).reshape(nrow, nchan, ncorr)

        xds["DATA"] = (("row", "chan", "corr"), da.from_array(vis, chunks=(-1, -1, -1)))
        dask.compute(xds_to_table(xds, ms_name, columns="DATA"))

        # cast gain to QuartiCal format
        g = da.from_array(jones)
        gflags = da.zeros((ntime, nchan, nant, 1))
        data_vars = {
            "gains": (("gain_time", "gain_freq", "antenna", "direction", "correlation"), g),
            "gain_flags": (("gain_time", "gain_freq", "antenna", "direction"), gflags),
        }
        gain_spec_tup = namedtuple("gains_spec_tup", "tchunk fchunk achunk dchunk cchunk")
        attrs = {
            "NAME": "NET",
            "TYPE": "complex",
            "FIELD_NAME": "00",
            "SCAN_NUMBER": int(1),
            "FIELD_ID": int(0),
            "DATA_DESC_ID": int(0),
            "GAIN_SPEC": gain_spec_tup(
                tchunk=(int(ntime),), fchunk=(int(nchan),), achunk=(int(nant),), dchunk=(int(1),), cchunk=(int(2),)
            ),
            "GAIN_AXES": ("gain_time", "gain_freq", "antenna", "direction", "correlation"),
        }
        coords = {"gain_freq": (("gain_freq",), freq), "gain_time": (("gain_time",), utime)}
        net_xds_list = Dataset(data_vars, coords=coords, attrs=attrs)
        gain_path = str(test_dir / Path("gains.qc"))
        dask.compute(xds_to_zarr(net_xds_list, f"{gain_path}::NET"))
        gain_path = [f"{gain_path}/NET"]

    else:
        xds["DATA"] = (("row", "chan", "corr"), da.from_array(model_vis, chunks=(-1, -1, -1)))
        dask.compute(xds_to_table(xds, ms_name, columns="DATA"))
        gain_path = None

    outname = str(test_dir / "test")
    dds_name = f"{outname}_I_main.dds"

    # initialize Stokes visibilities
    init_core(
        [test_dir / "test_ascii_1h60.0s.MS"],
        outname,
        data_column="DATA",
        flag_column="FLAG",
        gain_table=gain_path,
        max_field_of_view=fov * 1.1,
        overwrite=True,
        channels_per_image=1,
        bda_decorr=1.0,
        keep_ray_alive=True,
    )

    grid_core(
        outname,
        field_of_view=fov,
        fits_mfs=False,
        fits_cubes=False,
        psf=True,
        residual=False,
        nthreads=2,
        overwrite=True,
        robustness=0.0,
        do_wgridding=True,
        psf_oversize=2.0,
        keep_ray_alive=True,
    )

    # run kclean
    threshold = 1e-4
    kclean_core(
        outname,
        dirosion=0,
        niter=100,
        threshold=threshold,
        gamma=0.1,
        peak_factor=0.75,
        sub_peak_factor=0.75,
        nthreads=2,
        do_wgridding=True,
        epsilon=epsilon,
        mop_flux=True,
        fits_mfs=False,
        fits_cubes=False,
    )

    # get inferred model
    dds, _ = xds_from_url(dds_name)
    model_inferred = np.zeros((nchan, nx, ny))
    for ds in dds:
        b = int(ds.bandid)
        model_inferred[b] = ds.MODEL.values

    # we actually reconstruct I/n(l,m) so we need to correct for that
    l_coord, m_coord = np.meshgrid(dds[0].x.values, dds[0].y.values, indexing="ij")
    eps = l_coord**2 + m_coord**2
    n = -eps / (np.sqrt(1.0 - eps) + 1.0) + 1  # more stable form
    for i in range(nsource):
        assert_allclose(
            1.0
            + model_inferred[:, x_index[i], y_index[i]] * n[x_index[i], y_index[i]]
            - model[:, x_index[i], y_index[i]],
            1.0,
            atol=5 * threshold,
        )
