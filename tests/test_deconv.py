"""Tests for the ``.dt``-native ``pfb deconv`` driver.

``test_deconv_groundtruth`` runs ``imager``+``deconv`` on simulated
visibilities predicted from an injected ``sky_truth`` sky and checks recovery
against that ground truth directly (no legacy oracle). The remaining tests
build a synthetic ``.dt`` store in-process (``_write_synthetic_dt``, no
MS/imager needed) for fast smoke coverage: ``test_deconv_two_band_smoke``
guards the nband>1 Ray actor-pool deadlock, and
``test_band_workers_load_matches_driver_side`` checks that band-worker-side
loading of vis-scale inputs from the store reproduces driver-side reads
exactly.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def test_deconv_groundtruth(sky_truth, ms_name, tmp_path):
    """deconv on the noiseless predicted sky recovers the injected fluxes.

    Replaces test_deconv_matches_legacy_sara: the reference is the injected
    truth itself rather than the legacy sara implementation. The wavelet
    model legitimately spreads a point source over neighbouring pixels, so
    flux is asserted as a +/-4-pixel box sum (measured recovery ~1.08-1.12x
    after 5 cycles at eta=0.001; single-pixel values plateau near ~40%),
    position as the box argmax, and convergence as the normalised residual
    peak dropping well below the faintest source.

    Single band (channels_per_image=-1) and nthreads=1 on purpose: for
    nband==1 the Hessian/Psi pools use their local in-process path, keeping
    the long-lived Ray-actor CPU claims within the session cluster's
    num_cpus=1 (see tests/conftest.py) -- multi-band actor distribution is
    covered by test_hess_tree_ray.py/test_psi_operator.py.
    """
    from pfb_imaging.core.deconv import deconv as deconv_core
    from pfb_imaging.core.imager import imager as imager_core

    outname = str(tmp_path / "gtdeconv")
    imager_core(
        [Path(ms_name)],
        outname,
        channels_per_image=-1,
        integrations_per_image=-1,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )
    deconv_core(
        outname,
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=5,
        gamma=1.0,
        eta=0.001,
        rmsfactor=1.0,
        init_factor=1.0,
        l1_reweight_from=100,  # disabled within these few major cycles
        bases=["self", "db1"],
        nlevels=2,
        positivity=1,
        pd_tol=1e-6,
        pd_maxit=5000,
        cg_tol=1e-6,
        cg_maxit=3000,
        pm_tol=1e-4,
        pm_maxit=200,
        nthreads=1,
        do_wgridding=True,
        epsilon=1e-7,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    for n in nodes:
        assert "MODEL" in dt[n].ds and "UPDATE" in dt[n].ds
    model_mean = sum(dt[n].ds.MODEL[0] for n in nodes) / len(nodes)
    residual = sum(dt[n].ds.RESIDUAL[0] for n in nodes).values
    wsum = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)

    # converged: normalised residual peak well below the faintest source
    assert np.abs(residual).max() / wsum < 0.1 * sky_truth.ref_flux.min()

    half = 4  # box half-width for the flux sums
    for s in range(sky_truth.lpix.size):
        ixm = sky_truth.nx // 2 - int(sky_truth.lpix[s])
        iym = sky_truth.ny // 2 + int(sky_truth.mpix[s])
        box = model_mean.isel(x=slice(ixm - half, ixm + half + 1), y=slice(iym - half, iym + half + 1))
        # the n-term is folded into the stored BEAM (D22), so the model is in
        # intrinsic flux units -- no legacy *n correction
        got = float(box.values.sum())
        want = sky_truth.ref_flux[s]
        assert abs(got - want) < 0.2 * want, f"source {s}: box flux {got} vs {want}"
        # the box is centred on the source in both axes, so the argmax must
        # sit at its centre regardless of the ('x','y')/('y','x') dim order
        i0, i1 = np.unravel_index(int(np.argmax(box.values)), box.shape)
        assert (i0, i1) == (half, half), f"source {s}: model peak off-centre ({i0},{i1})"


def _write_synthetic_dt(store, nx, ny, nrow, nchan, rng, parts_per_band=(1, 1)):
    """Build a minimal synthetic 2-band .dt store matching what core/deconv.py reads.

    No MS / imager pipeline involved -- just the native DataTree groups the
    driver's ``deconv()`` opens directly (see architecture.md §8 tree layout).
    ``parts_per_band`` sets the number of ``part####`` children per band --
    bands legitimately carry different partition counts when a field's chunk
    is fully flagged (stokes_vis writes no scratch piece for it).
    """
    nx_psf, ny_psf = 2 * nx, 2 * ny
    xo2 = nx + 1
    freqs = [1e9, 1.1e9]

    for b, freq in enumerate(freqs):
        bandname = f"band{b:04d}_time0000"
        dirty = rng.standard_normal((1, ny, nx))
        band_ds = xr.Dataset(
            data_vars={
                "DIRTY": (("corr", "y", "x"), dirty),
                # unit partition beams -> BDIRTY == DIRTY (D23)
                "BDIRTY": (("corr", "y", "x"), dirty.copy()),
                "RESIDUAL": (("corr", "y", "x"), rng.standard_normal((1, ny, nx))),
                "PSF": (("corr", "y_psf", "x_psf"), np.ones((1, ny_psf, nx_psf))),
                "WSUM": (("corr",), np.array([1.0])),
            },
            coords={"corr": ["I"]},
            attrs={
                "bandid": b,
                "timeid": 0,
                "freq_out": freq,
                "time_out": 1.7e9,
                "ra": 0.0,
                "dec": 0.0,
                "cell_rad": 2.5e-6,
                "niters": 0,
            },
        )
        band_ds.to_zarr(store, group=bandname, mode="a")

        for pid in range(parts_per_band[b]):
            uvw = rng.uniform(-50.0, 50.0, size=(nrow, 3))
            part_ds = _make_part(uvw, nrow, nchan, nx, ny, ny_psf, xo2, freq, parts_per_band[b])
            part_ds.to_zarr(store, group=f"{bandname}/part{pid:04d}", mode="a")


def _make_part(uvw, nrow, nchan, nx, ny, ny_psf, xo2, freq, nparts):
    return xr.Dataset(
        data_vars={
            # delta-function PSF -> Fourier-domain magnitude is all ones
            # (scaled by the per-part wsum share so the band's Hessian stays
            # the identity), matching the abs()'d PSFHAT convention
            # core/deconv.py expects.
            "PSFHAT": (("corr", "y_psf", "xo2"), np.full((1, ny_psf, xo2), 1.0 / nparts)),
            "BEAM": (("corr", "y", "x"), np.ones((1, ny, nx))),
            "UVW": (("row", "three"), uvw),
            "WEIGHT": (("corr", "row", "chan"), np.ones((1, nrow, nchan))),
            "MASK": (("row", "chan"), np.ones((nrow, nchan), dtype=np.uint8)),
            "FREQ": (("chan",), np.array([freq])),
        },
        attrs={
            "wsum": [1.0 / nparts],
            "l0": 0.0,
            "m0": 0.0,
            "msid": 0,
            "field_name": "f0",
            "spw_name": "s0",
            "baseline_group": "all",
        },
    )


@pytest.mark.timeout(120)
def test_deconv_two_band_smoke(tmp_path):
    """Multi-band driver smoke test: regression guard for the nband>1 Ray deadlock.

    Builds a synthetic 2-band .dt store directly (no MS/imager needed) and
    runs the deconv driver with default-ish worker settings (nworkers=1):
    this MUST NOT hang (see the HessTreeRay/PsiNocopytRay actor-pool CPU
    claim fix -- previously the aggregate actor CPU claim could exceed the
    driver's Ray cluster capacity for nband > 1).
    """
    from pfb_imaging.core.deconv import deconv as deconv_core

    rng = np.random.default_rng(42)
    nx = ny = 32
    nrow, nchan = 64, 1

    output_filename = str(tmp_path / "synth")
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, nx, ny, nrow, nchan, rng)

    deconv_core(
        output_filename,
        product="I",
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=1,
        hess_norm=1.0,
        pd_maxit=20,
        cg_maxit=20,
        bases=["self"],
        nlevels=1,
        l1_reweight_from=100,
        nthreads=2,
        nworkers=1,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
    )

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    assert len(nodes) == 2
    for n in nodes:
        ds = dt[n].ds
        assert "MODEL" in ds and "UPDATE" in ds
        assert "BRESIDUAL" in ds  # gradient residual written back (D23)
        assert ds.attrs["niters"] == 1
        assert np.isfinite(ds.MODEL.values).all()


@pytest.mark.timeout(120)
def test_band_workers_load_matches_driver_side(tmp_path):
    """Worker-side load_bands reproduces driver-side reads exactly.

    The band workers read their own vis-scale inputs (PSFHAT/BEAM/wsum for
    the Hessian, UVW/WEIGHT/MASK/FREQ/BEAM/DIRTY for the exact residual)
    straight from the .dt store; this checks the resulting operators against
    the same data loaded in the test process.
    """
    from numpy.testing import assert_allclose

    from pfb_imaging.operators.band_worker import BandWorkerPool
    from pfb_imaging.operators.gridder import residual_from_partitions
    from pfb_imaging.operators.hessian import HessianTree, HessTreeRay

    rng = np.random.default_rng(99)
    nx = ny = 16
    nrow, nchan = 32, 1
    dt_name = str(tmp_path / "synth_I.dt")
    _write_synthetic_dt(dt_name, nx, ny, nrow, nchan, rng)

    # randomise the PSF magnitudes so the Hessian check is non-trivial
    import zarr

    root = zarr.open_group(dt_name, mode="a")
    for _, grp in root.groups():
        # non-trivial BDIRTY so the gradient-residual check is meaningful
        grp["BDIRTY"][:] = rng.standard_normal(grp["BDIRTY"].shape)
        for _, child in grp.groups():
            child["PSFHAT"][:] = rng.uniform(0.5, 2.0, size=child["PSFHAT"].shape)
            # distinct non-unit per-partition beams: the exact gradient is the
            # per-partition beam-weighted sum, not a band-average (D23)
            child["BEAM"][:] = rng.uniform(0.3, 1.0, size=child["BEAM"].shape)
    zarr.consolidate_metadata(dt_name)

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    nband = len(nodes)
    cell_rad = dt[nodes[0]].ds.attrs["cell_rad"]

    pool = BandWorkerPool(nband, nthreads=1)
    pool.load_bands(dt_name, nodes)
    hess = HessTreeRay(None, nx, ny, 2 * nx, 2 * ny, etas=0.1, wsums=1.0, workers=pool)

    x = rng.standard_normal((nband, nx, ny))
    out_pool = hess.dot(x)
    model = rng.standard_normal((nband, 1, nx, ny))
    res_pool, bres_pool = pool.residual(model, cell_rad)

    for b, n in enumerate(nodes):
        band = dt[n]
        parts, hess_parts = [], []
        for cname in sorted(band.children):
            child = band[cname].ds
            pds = child[["UVW", "WEIGHT", "MASK", "FREQ", "BEAM"]].load()
            pds.attrs.update(child.attrs)
            hess_parts.append(
                {
                    "psfhat": np.abs(child.PSFHAT.values),
                    "beam": pds.BEAM.values,
                    "wsum": np.asarray(child.attrs["wsum"]),
                }
            )
            parts.append(pds)
        ref_hess = HessianTree(hess_parts, nx, ny, 2 * nx, 2 * ny, eta=0.1, wsum=1.0)
        assert_allclose(out_pool[b], ref_hess.dot(x[b])[0], rtol=1e-12, atol=1e-12)
        ref_res, ref_bres = residual_from_partitions(
            band.ds.DIRTY.values, parts, model[b], cell_rad, bdirty=band.ds.BDIRTY.values
        )
        assert_allclose(res_pool[b], ref_res, rtol=1e-12, atol=1e-12)
        assert_allclose(bres_pool[b], ref_bres, rtol=1e-12, atol=1e-12)


@pytest.mark.timeout(120)
def test_deconv_unequal_partition_counts(tmp_path):
    """Bands legitimately carry different partition counts (a fully flagged
    field chunk writes no scratch piece, so its band node has fewer part####
    children -- e.g. a mosaic field flagged out of one band only). The deconv
    driver and band workers must be partition-count-agnostic per band.
    """
    from pfb_imaging.core.deconv import deconv as deconv_core

    rng = np.random.default_rng(7)
    nx = ny = 32
    nrow, nchan = 64, 1

    output_filename = str(tmp_path / "unequal")
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, nx, ny, nrow, nchan, rng, parts_per_band=(1, 3))

    deconv_core(
        output_filename,
        product="I",
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=1,
        hess_norm=1.0,
        pd_maxit=20,
        cg_maxit=20,
        bases=["self"],
        nlevels=1,
        l1_reweight_from=100,
        nthreads=2,
        nworkers=1,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
    )

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    assert len(nodes) == 2
    assert len(dt[nodes[0]].children) == 1 and len(dt[nodes[1]].children) == 3
    for n in nodes:
        ds = dt[n].ds
        assert "MODEL" in ds and "UPDATE" in ds
        assert "BRESIDUAL" in ds  # gradient residual written back (D23)
        assert ds.attrs["niters"] == 1
        assert np.isfinite(ds.MODEL.values).all()
        assert np.isfinite(ds.RESIDUAL.values).all()


@pytest.mark.timeout(120)
def test_deconv_requires_bdirty(tmp_path):
    """A .dt without BDIRTY (pre-D23 imager) is refused with a clear error."""
    from pfb_imaging.core.deconv import deconv as deconv_core

    rng = np.random.default_rng(5)
    output_filename = str(tmp_path / "nobd")
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, 16, 16, 32, 1, rng)

    import shutil

    import zarr

    root = zarr.open_group(dt_name, mode="a")
    for gname, _ in root.groups():
        shutil.rmtree(f"{dt_name}/{gname}/BDIRTY")
    zarr.consolidate_metadata(dt_name)

    with pytest.raises(ValueError, match="BDIRTY"):
        deconv_core(
            output_filename, product="I", nthreads=1, fits_mfs=False, fits_cubes=False, log_directory=str(tmp_path)
        )
