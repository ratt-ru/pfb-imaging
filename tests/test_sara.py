from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
from africanus.constants import c as lightspeed
from daskms import xds_from_ms, xds_from_table, xds_to_table
from ducc0.fft import good_size
from ducc0.wgridder.experimental import dirty2vis
from numpy.testing import assert_allclose

from pfb_imaging.core.degrid import degrid as degrid_core
from pfb_imaging.core.grid import grid as grid_core
from pfb_imaging.core.init import init as init_core
from pfb_imaging.core.sara import sara as sara_core
from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.misc import Gaussian2D, give_edges
from pfb_imaging.utils.naming import xds_from_url

pmp = pytest.mark.parametrize


def test_sara(ms_name):
    """
    # TODO - currently we just check that this runs through.
    # What should the passing criteria be?
    """
    robustness = None
    do_wgridding = True

    np.random.seed(420)

    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1})[0]
    spw = xds_from_table(f"{ms_name}::SPECTRAL_WINDOW")[0]

    freq = spw.CHAN_FREQ.values.squeeze()
    freq0 = np.mean(freq)

    nchan = freq.size
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
    nsource = 25
    border = np.maximum(int(0.15 * nx), int(0.15 * ny))
    x_index = np.random.randint(border, npix - border, nsource)
    y_index = np.random.randint(border, npix - border, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    ref_flux = 1.0 + np.exp(np.random.randn(nsource))
    extentx = np.random.randint(3, int(0.1 * nx), nsource)
    extenty = np.random.randint(3, int(0.1 * nx), nsource)
    pas = np.random.random(nsource) * 180
    x = -(nx / 2) + np.arange(nx)
    y = -(nx / 2) + np.arange(ny)
    xin, yin = np.meshgrid(x, y, indexing="ij")
    for i in range(nsource):
        emaj = np.maximum(extentx[i], extenty[i])
        emin = np.minimum(extentx[i], extenty[i])
        gauss = Gaussian2D(xin, yin, GaussPar=(emaj, emin, pas[i]))
        mx, my, gx, gy = give_edges(x_index[i], y_index[i], nx, ny, nx, ny)
        spectrum = ref_flux[i] * (freq / freq0) ** alpha[i]
        model[:, mx, my] += spectrum[:, None, None] * gauss[None, gx, gy]

    # model vis
    flip_u, flip_v, flip_w, _, _ = wgridder_conventions(0.0, 0.0)
    epsilon = 1e-7
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model_vis[:, c : c + 1, 0] = dirty2vis(
            uvw=uvw,
            freq=freq[c : c + 1],
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
            sigma_max=3.0,
        )
        model_vis[:, c, -1] = model_vis[:, c, 0]

    model_vis += np.random.randn(nrow, nchan, ncorr) + 1.0j * np.random.randn(nrow, nchan, ncorr)

    model_vis = da.from_array(model_vis, chunks=(-1, -1, -1))
    xds["DATA"] = (("row", "chan", "corr"), model_vis)
    writes = [xds_to_table(xds, ms_name, columns="DATA")]
    dask.compute(writes)

    outname = str(test_dir / "test")
    dds_name = f"{outname}_I_main.dds"

    # Initialise Stokes visibilities
    init_core(
        [str(test_dir / "test_ascii_1h60.0s.MS")],
        outname,
        data_column="DATA",
        flag_column="FLAG",
        gain_table=None,
        max_field_of_view=fov * 1.1,
        overwrite=True,
        channels_per_image=1,
    )

    # grid data to produce dirty image
    grid_core(
        outname,
        field_of_view=fov,
        fits_mfs=False,
        psf=True,
        residual=False,
        noise=False,
        nthreads=8,
        overwrite=True,
        robustness=robustness,
        do_wgridding=do_wgridding,
    )

    # run sara
    tol = 1e-5
    sara_core(
        outname,
        niter=2,
        tol=tol,
        gamma=1.0,
        pd_tol=[1e-3],
        rmsfactor=1.0,
        epsfactor=4.0,
        l1_reweight_from=5,
        bases="self,db1",
        nlevels=3,
        nthreads=8,
        do_wgridding=do_wgridding,
        epsilon=epsilon,
        fits_mfs=False,
    )

    # the residual computed by the grid worker should be identical
    # to that computed in sara when transferring model
    dds, _ = xds_from_url(dds_name)

    # grid data to produce dirty image
    grid_core(
        outname,
        field_of_view=fov,
        fits_mfs=False,
        psf=False,
        weight=False,
        noise=False,
        residual=True,
        nthreads=8,
        overwrite=True,
        robustness=robustness,
        do_wgridding=do_wgridding,
        transfer_model_from=f"{outname}_main_model.mds",
        suffix="subtract",
    )

    dds2, _ = xds_from_url(f"{outname}_I_subtract.dds")

    for ds, ds2 in zip(dds, dds2):
        wsum = ds.WSUM.values
        assert_allclose(1 + np.abs(ds.RESIDUAL.values) / wsum, 1 + np.abs(ds2.RESIDUAL.values) / wsum)

    # residuals also need to be the same if we do the subtraction in visibility space
    degrid_core(
        [str(test_dir / "test_ascii_1h60.0s.MS")],
        outname,
        mds=f"{outname}_I_main_model.mds",
        dds=f"{outname}_I_main.dds",
        channels_per_image=1,
        nthreads=8,
        do_wgridding=do_wgridding,
    )

    # Initialise Stokes visibilities with DATA-MODEL_DATA
    init_core(
        [str(test_dir / "test_ascii_1h60.0s.MS")],
        outname,
        data_column="DATA-MODEL_DATA",
        flag_column="FLAG",
        gain_table=None,
        max_field_of_view=fov * 1.1,
        bda_decorr=1.0,
        overwrite=True,
        channels_per_image=1,
    )

    # grid data to produce dirty image
    grid_core(
        outname,
        field_of_view=fov,
        fits_mfs=False,
        psf=False,
        residual=False,
        nthreads=8,
        overwrite=True,
        robustness=robustness,
        do_wgridding=do_wgridding,
    )

    dds_name = f"{outname}_main.dds"

    dds2, _ = xds_from_url(dds_name)

    for ds, ds2 in zip(dds, dds2):
        wsum = ds.WSUM.values
        assert_allclose(1 + np.abs(ds.RESIDUAL.values) / wsum, 1 + np.abs(ds2.DIRTY.values) / wsum)


# test_sara()
