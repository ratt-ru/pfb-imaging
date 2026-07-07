"""Tier-3 end-to-end equivalence test: new .dt deconv vs legacy .dds sara.

Runs both pipelines (legacy ``init``+``grid``+``sara`` on the ``.dds`` and the
new ``imager``+``deconv`` on the ``.dt``) on the SAME simulated visibilities,
with natural weighting and a pinned shared ``hess_norm`` (removes power-method
nondeterminism), one major cycle, and reweighting disabled. This isolates the
comparison to the forward/backward kernels themselves rather than convergence
trajectories, giving a tight tolerance on the resulting model images.
"""

import inspect
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr
from daskms import xds_to_table
from ducc0.wgridder.experimental import dirty2vis

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.misc import gaussian2d, give_edges


def _simulate_data(ms_name, ms_meta, image_geometry):
    """Simulate point-source model visibilities into the MS DATA column.

    Copied (same seed/model) from ``tests/test_sara.py::test_sara`` so the
    legacy and new deconv paths are compared on identical simulated data.
    """
    do_wgridding = True

    np.random.seed(420)

    xds = ms_meta.xds
    freq = ms_meta.freq
    freq0 = ms_meta.freq0
    nchan = ms_meta.nchan
    ncorr = ms_meta.ncorr
    uvw = ms_meta.uvw
    nrow = ms_meta.nrow

    fov = image_geometry.fov
    cell_rad = image_geometry.cell_rad
    nx = image_geometry.nx
    ny = image_geometry.ny

    # model
    npix = nx
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
        gauss = gaussian2d(xin, yin, gausspar=(emaj, emin, pas[i]))
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
            nthreads=2,
            sigma_min=1.1,
            sigma_max=3.0,
        )
        model_vis[:, c, -1] = model_vis[:, c, 0]

    model_vis += np.random.randn(nrow, nchan, ncorr) + 1.0j * np.random.randn(nrow, nchan, ncorr)

    model_vis = da.from_array(model_vis, chunks=(-1, -1, -1))
    xds["DATA"] = (("row", "chan", "corr"), model_vis)
    writes = [xds_to_table(xds, ms_name, columns="DATA")]
    dask.compute(writes)

    return fov


def test_deconv_matches_legacy_sara(ms_name, ms_meta, image_geometry, tmp_path):
    """New .dt deconv (sara/PD) vs legacy .dds sara: same data, pinned hess_norm."""
    from pfb_imaging.core.deconv import deconv as deconv_core
    from pfb_imaging.core.grid import grid as grid_core
    from pfb_imaging.core.imager import imager as imager_core
    from pfb_imaging.core.init import init as init_core
    from pfb_imaging.core.sara import sara as sara_core
    from pfb_imaging.utils.naming import xds_from_url

    fov = _simulate_data(ms_name, ms_meta, image_geometry)

    # shared options; each side filters this down to what its own signature
    # actually accepts (sara has no nworkers/opt_backend, deconv has no
    # hess_approx) -- use the real signatures rather than guessing.
    common = dict(
        niter=1,
        gamma=1.0,
        eta=0.5,
        rmsfactor=1.0,
        init_factor=1.0,
        l1_reweight_from=100,  # disabled within one major cycle
        bases="self,db1",
        nlevels=2,
        positivity=1,
        hess_norm=None,  # legacy run estimates it; then we pin it (below)
        pd_tol=1e-6,
        pd_maxit=5000,
        cg_tol=1e-6,
        cg_maxit=3000,
        pm_tol=1e-4,
        pm_maxit=200,
        # deconv's PsiNocopytRay/HessTreeRay are long-lived Ray *actors*: all
        # nband actors must fit simultaneously within the session Ray
        # cluster's num_cpus=1 (conftest.py), unlike the legacy/imager paths
        # where nthreads only sizes an OS thread pool or short-lived Ray
        # *tasks*. nthreads=1 keeps the aggregate actor CPU claim (==
        # nthreads, independent of nband) at the cluster's capacity.
        nthreads=1,
        do_wgridding=True,
        epsilon=1e-7,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
    )

    sara_params = set(inspect.signature(sara_core).parameters)
    deconv_params = set(inspect.signature(deconv_core).parameters)

    # --- legacy path: init + grid + sara on the .dds ---
    out_legacy = str(tmp_path / "legacy")
    init_core(
        [ms_name],
        out_legacy,
        data_column="DATA",
        flag_column="FLAG",
        max_field_of_view=fov * 1.1,
        overwrite=True,
        # single band (nband=1): deconv's HessTreeRay/PsiNocopytRay both fall
        # back to their local (non-Ray) code path only when nband==1. For
        # nband>1 the two long-lived actor pools are alive simultaneously,
        # and each independently claims up to `nthreads` aggregate CPUs
        # (see the nthreads comment on `common` below) -- together that
        # deadlocks ray.get(warmup) on the session's single-CPU test
        # cluster. The Ray-actor distribution itself is already covered by
        # tests/test_hess_tree_ray.py and tests/test_psi_operator.py; this
        # test's job is the algorithmic equivalence, not re-proving that.
        channels_per_image=-1,
        keep_ray_alive=True,
        nthreads=2,
    )
    grid_core(
        out_legacy,
        field_of_view=fov,
        fits_mfs=False,
        fits_cubes=False,
        psf=True,
        residual=False,
        noise=False,
        nthreads=2,
        overwrite=True,
        robustness=None,
        do_wgridding=True,
        keep_ray_alive=True,
    )
    sara_core(out_legacy, **{k: v for k, v in common.items() if k in sara_params})
    dds, _ = xds_from_url(f"{out_legacy}_I_main.dds")
    hess_norm = float(dds[0].hess_norm)  # pin for the new path
    model_legacy = np.stack([ds.MODEL.values[0] for ds in sorted(dds, key=lambda d: d.freq_out)])

    # --- new path: imager + deconv on the .dt ---
    out_new = str(tmp_path / "new")
    imager_core(
        [Path(ms_name)],
        out_new,
        channels_per_image=-1,  # single band, see the init_core call above
        product="I",
        field_of_view=fov,
        robustness=None,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )
    deconv_core(
        out_new,
        minor_cycle="sara",
        opt_backend="primal-dual",
        bases=["self", "db1"],
        hess_norm=hess_norm,
        **{k: v for k, v in common.items() if k in deconv_params and k not in ("bases", "hess_norm")},
    )

    dt = xr.open_datatree(f"{out_new}_I.dt", engine="zarr", chunks=None)
    nodes = sorted(
        (n for n in dt.children if n.startswith("band")),
        key=lambda n: int(dt[n].ds.attrs["bandid"]),
    )
    model_new = np.stack([dt[n].ds.MODEL.values[0] for n in nodes])
    for n in nodes:
        assert "MODEL" in dt[n].ds and "UPDATE" in dt[n].ds
        assert dt[n].ds.attrs["niters"] == 1

    rdiff = np.linalg.norm(model_new - model_legacy) / np.linalg.norm(model_legacy)
    print(f"[test_deconv] model rdiff = {rdiff:.3e}", flush=True)
    # one major cycle, pinned hess_norm, natural weights: the paths share the
    # same kernels; residual slack covers CG/PD convergence and the distinct
    # (legacy Psi vs new PsiNocopytRay) wavelet-operator implementations.
    assert rdiff < 1e-2, f"model mismatch: rdiff = {rdiff:.3e}"
