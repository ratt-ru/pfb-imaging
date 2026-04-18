from collections import namedtuple
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
from africanus.calibration.utils import corrupt_vis
from daskms import xds_to_table
from daskms.experimental.zarr import xds_to_zarr
from ducc0.wgridder.experimental import dirty2vis
from numpy.testing import assert_allclose
from xarray import Dataset

from pfb_imaging.core.grid import grid as grid_core
from pfb_imaging.core.init import init as init_core
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.misc import kron_matvec
from pfb_imaging.utils.naming import xds_from_url

pmp = pytest.mark.parametrize


@pmp("do_gains", (False, True))
def test_polproducts(do_gains, ms_name, ms_meta, image_geometry, gain_cholesky, time_chunks):
    """
    Tests polarisation products
    """
    np.random.seed(420)
    test_dir = Path(ms_name).resolve().parent
    xds = ms_meta.xds
    utime = ms_meta.utime
    freq = ms_meta.freq
    ntime = ms_meta.ntime
    nchan = ms_meta.nchan
    nant = ms_meta.nant
    ncorr = ms_meta.ncorr
    uvw = ms_meta.uvw
    nrow = ms_meta.nrow

    fov = image_geometry.fov
    cell_rad = image_geometry.cell_rad
    nx = image_geometry.nx
    ny = image_geometry.ny
    npix = nx
    print("Cell size set to %5.5e arcseconds" % image_geometry.cell_size)
    print("Image size set to (%i, %i, %i)" % (nchan, nx, ny))

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)

    # first axis is Stokes
    model = np.zeros((4, nchan, nx, ny), dtype=np.float64)
    flux = {}
    flux["I"] = 1.0
    flux["Q"] = 0.6
    flux["U"] = 0.3
    flux["V"] = 0.1
    flux["IQUV"] = np.array([flux["I"], flux["Q"], flux["U"], flux["V"]])
    locx = int(3 * npix // 4)
    locy = int(npix // 4)
    model[0, :, locx, locy] = flux["I"]
    model[1, :, locx, locy] = flux["Q"]
    model[2, :, locx, locy] = flux["U"]
    model[3, :, locx, locy] = flux["V"]

    # model vis
    epsilon = 1e-7
    model_vis_i = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_i[:, c : c + 1] = dirty2vis(
            uvw=uvw,
            freq=freq[c : c + 1],
            dirty=model[0, c],
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=epsilon,
            do_wgridding=True,
            nthreads=2,
        )
    model_vis_q = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_q[:, c : c + 1] = dirty2vis(
            uvw=uvw,
            freq=freq[c : c + 1],
            dirty=model[1, c],
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=epsilon,
            do_wgridding=True,
            nthreads=2,
        )
    model_vis_u = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_u[:, c : c + 1] = dirty2vis(
            uvw=uvw,
            freq=freq[c : c + 1],
            dirty=model[2, c],
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=epsilon,
            do_wgridding=True,
            nthreads=2,
        )
    model_vis_v = np.zeros((nrow, nchan), dtype=np.complex128)
    for c in range(nchan):
        model_vis_v[:, c : c + 1] = dirty2vis(
            uvw=uvw,
            freq=freq[c : c + 1],
            dirty=model[3, c],
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=epsilon,
            do_wgridding=True,
            nthreads=2,
        )

    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    model_vis[:, :, 0] = model_vis_i + model_vis_q
    model_vis[:, :, 1] = model_vis_u + 1j * model_vis_v
    model_vis[:, :, 2] = model_vis_u - 1j * model_vis_v
    model_vis[:, :, 3] = model_vis_i - model_vis_q

    if do_gains:
        chol_tnu = (gain_cholesky.chol_t, gain_cholesky.chol_nu)
        nu = gain_cholesky.nu

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
        ant1 = ms_meta.ant1
        ant2 = ms_meta.ant2

        gains = np.swapaxes(jones, 1, 2).copy()
        vis = corrupt_vis(time_chunks.tbin_idx, time_chunks.tbin_counts, ant1, ant2, gains, model_vis).reshape(
            nrow, nchan, ncorr
        )

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

    # test each polarisation product separately
    outname = str(test_dir / "test")
    for p in ["I", "Q", "U", "V"]:
        basename = f"{outname}"
        dds_name = f"{basename}_{p}_main.dds"

        # initialize Stokes visibilities
        init_core(
            [str(test_dir / "test_ascii_1h60.0s.MS")],
            basename,
            data_column="DATA",
            flag_column="FLAG",
            gain_table=gain_path,
            max_field_of_view=fov * 1.1,
            bda_decorr=1.0,
            overwrite=True,
            channels_per_image=1,
            product=p,
            keep_ray_alive=True,
        )

        # grid data to produce dirty image
        grid_core(
            basename,
            field_of_view=fov,
            fits_mfs=False,
            fits_cubes=False,
            psf=False,
            residual=False,
            beam=False,
            noise=False,
            nthreads=2,
            overwrite=True,
            robustness=0.0,
            do_wgridding=True,
            product=p,
            keep_ray_alive=True,
        )

        dds, _ = xds_from_url(dds_name)

        for ds in dds:
            wsum = ds.WSUM.values[0]
            comp = ds.DIRTY.values[0, locx, locy]
            # print(flux[p], comp/wsum)
            assert_allclose(flux[p], comp / wsum, rtol=1e-4, atol=1e-4)

    # test IQUV
    outname = str(test_dir / "test")
    for p in ["IQUV"]:
        basename = f"{outname}"
        dds_name = f"{basename}_{p}_main.dds"
        # initialize Stokes visibilities
        init_core(
            [str(test_dir / "test_ascii_1h60.0s.MS")],
            basename,
            data_column="DATA",
            flag_column="FLAG",
            gain_table=gain_path,
            max_field_of_view=fov * 1.1,
            bda_decorr=1.0,
            overwrite=True,
            channels_per_image=1,
            product=p,
            keep_ray_alive=True,
        )

        # grid data to produce dirty image
        grid_core(
            basename,
            field_of_view=fov,
            fits_mfs=False,
            fits_cubes=False,
            psf=False,
            residual=False,
            beam=False,
            noise=False,
            nthreads=2,
            overwrite=True,
            robustness=0.0,
            do_wgridding=True,
            product=p,
            keep_ray_alive=True,
        )

        dds, _ = xds_from_url(dds_name)

        for ds in dds:
            wsum = ds.WSUM.values[:]
            comp = ds.DIRTY.values[:, locx, locy]
            # print(flux[p], comp/wsum)
            assert_allclose(flux[p], comp / wsum, rtol=1e-4, atol=1e-4)
