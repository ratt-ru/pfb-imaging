"""Per-Stokes correctness through the MSv4 imager (replaces test_polproducts).

NOTE: only the gain-free path is covered. The legacy test also parametrised
do_gains=True through init+grid, but the imager's pass 1 currently applies an
identity jones stub (utils/stokes2vis_msv4.stokes_vis: "Fake jones for now"),
so a gain-corrupted variant cannot pass until gain application is implemented
on the MSv4 path. Add the do_gains=True parametrisation when it is.
"""

from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr
from daskms import xds_to_table
from ducc0.wgridder.experimental import dirty2vis
from numpy.testing import assert_allclose

from pfb_imaging.core.imager import imager as imager_core
from pfb_imaging.operators.gridder import wgridder_conventions


def test_imager_polproducts(ms_name, ms_meta, image_geometry, tmp_path):
    """A polarised point source is recovered in each Stokes product."""
    np.random.seed(420)
    xds = ms_meta.xds
    freq = ms_meta.freq
    nchan = ms_meta.nchan
    ncorr = ms_meta.ncorr
    uvw = ms_meta.uvw
    nrow = ms_meta.nrow

    fov = image_geometry.fov
    cell_rad = image_geometry.cell_rad
    nx = image_geometry.nx
    ny = image_geometry.ny
    npix = nx

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

    # model vis per Stokes, combined onto linear feeds
    epsilon = 1e-7
    stokes_vis = np.zeros((4, nrow, nchan), dtype=np.complex128)
    for s in range(4):
        for c in range(nchan):
            stokes_vis[s, :, c : c + 1] = dirty2vis(
                uvw=uvw,
                freq=freq[c : c + 1],
                dirty=model[s, c],
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
    model_vis[:, :, 0] = stokes_vis[0] + stokes_vis[1]
    model_vis[:, :, 1] = stokes_vis[2] + 1j * stokes_vis[3]
    model_vis[:, :, 2] = stokes_vis[2] - 1j * stokes_vis[3]
    model_vis[:, :, 3] = stokes_vis[0] - stokes_vis[1]

    # write DATA and clear flags so every sample participates (flag handling
    # is covered by the sky_truth-based tests)
    flag = np.zeros((nrow, nchan, ncorr), dtype=bool)
    xds_w = xds.assign(
        DATA=(("row", "chan", "corr"), da.from_array(model_vis, chunks=(-1, -1, -1))),
        FLAG=(("row", "chan", "corr"), da.from_array(flag, chunks=(-1, -1, -1))),
        FLAG_ROW=(("row",), da.zeros(nrow, dtype=bool, chunks=-1)),
    )
    dask.compute(xds_to_table(xds_w, ms_name, columns=["DATA", "FLAG", "FLAG_ROW"]))

    outname = str(tmp_path / "pol")
    for p in ["I", "Q", "U", "V", "IQUV"]:
        imager_core(
            [Path(ms_name)],
            outname,
            data_column="DATA",
            channels_per_image=1,
            integrations_per_image=-1,
            product=p,
            nx=nx,
            ny=ny,
            cell_size=image_geometry.cell_size,
            robustness=None,
            max_field_of_view=fov * 1.1,
            bda_decorr=1.0,
            fits_mfs=False,
            fits_cubes=False,
            overwrite=True,
            keep_ray_alive=True,
        )
        dt = xr.open_datatree(outname + f"_{p}.dt", engine="zarr", chunks=None)
        for n in dt.children:
            if not n.startswith("band"):
                continue
            ds = dt[n].ds
            wsum = ds.WSUM.values
            comp = ds.DIRTY.isel(x=locx, y=locy).values  # dims-aware: survives (Y, X)
            expected = flux["IQUV"] if p == "IQUV" else np.atleast_1d(flux[p])
            assert_allclose(expected, comp / wsum, rtol=1e-4, atol=1e-4)
