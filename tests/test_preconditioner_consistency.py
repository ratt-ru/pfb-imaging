"""The PSF-convolution preconditioner is asymptotically consistent.

With ``--rmsfactor 0 --positivity 0`` the ``pfb deconv`` major cycle collapses
to preconditioned Richardson on the exact normal equations::

    m_{k+1} = m_k + gamma * M^{-1} ( b - A m_k )

with ``A = sum_p B_p GtWG B_p / wsum + eta I`` the exact (degrid/grid) Hessian
-- "the full Hessian" -- and ``M = HessianTree`` the PSF-convolution
preconditioner (wiki design-decisions D22/D23; deconv-primer forward/backward).
Its fixed point is ``A^{-1} b`` for ANY nonsingular ``M``; the preconditioner
only sets the convergence rate. So solving the preconditioned problem yields the
SAME solution as solving with the full Hessian. On a self-consistent forward
model there is no floor above gridder epsilon -- the residual keeps descending
toward it (rate-limited by the weakly-measured large-scale modes), so a residual
that *plateaus* above that flags a forward-model (beam/rephasing) inconsistency,
the mosaic diagnostic.

These are the guards that this holds in code, including where the preconditioner
is a poor approximation of the exact operator (off-axis / w-term) and in the
distinct-beam mosaic regime where a D23-style bias (feeding the apparent instead
of the beam-attenuated residual) would move the fixed point off ``A^{-1} b``.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ducc0.fft import r2c
from ducc0.wgridder.experimental import dirty2vis, vis2dirty

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.operators.hessian import HessianTree
from pfb_imaging.opt.pcg import pcg_numba

ifftshift = np.fft.ifftshift

FREQ = np.array([1.0e9])
LAM = 299792458.0 / FREQ[0]
FLIP_U, FLIP_V, FLIP_W, X0, Y0 = wgridder_conventions(0.0, 0.0)
EPS = 1e-8
NTHREADS = 2


def _gtwg(m_yx, uvw, wgt, cell):
    """Exact ``Gt W G m`` for a (ny, nx) image via ducc degrid/grid.

    Image-space is (Y, X); ducc's x-major buffers are reached through zero-copy
    ``.T`` views (wiki D19/D20). ``divide_by_n=False`` throughout (D22).
    """
    ny, nx = m_yx.shape
    vis = dirty2vis(
        uvw=uvw,
        freq=FREQ,
        dirty=np.ascontiguousarray(m_yx.T),
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=X0,
        center_y=Y0,
        flip_u=FLIP_U,
        flip_v=FLIP_V,
        flip_w=FLIP_W,
        epsilon=EPS,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=NTHREADS,
    )
    out = np.zeros((ny, nx))
    vis2dirty(
        uvw=uvw,
        freq=FREQ,
        vis=vis,
        wgt=wgt,
        npix_x=nx,
        npix_y=ny,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=X0,
        center_y=Y0,
        flip_u=FLIP_U,
        flip_v=FLIP_V,
        flip_w=FLIP_W,
        epsilon=EPS,
        do_wgridding=True,
        divide_by_n=False,
        double_precision_accumulation=True,
        nthreads=NTHREADS,
        dirty=out.T,
    )
    return out


def _psfhat(uvw, wgt, cell, nx, ny):
    """abs(FFT) of the double-sized dirty PSF, HessianTree's ``PSFHAT`` convention."""
    nxp, nyp = 2 * nx, 2 * ny
    psf = np.zeros((nyp, nxp))
    vis2dirty(
        uvw=uvw,
        freq=FREQ,
        vis=np.ones((uvw.shape[0], 1), dtype=np.complex128),
        wgt=wgt,
        npix_x=nxp,
        npix_y=nyp,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=X0,
        center_y=Y0,
        flip_u=FLIP_U,
        flip_v=FLIP_V,
        flip_w=FLIP_W,
        epsilon=EPS,
        do_wgridding=True,
        divide_by_n=False,
        double_precision_accumulation=True,
        nthreads=NTHREADS,
        dirty=psf.T,
    )
    return np.abs(r2c(ifftshift(psf), axes=(0, 1), forward=True, inorm=0))


def _make_partition(rng, cell, nx, ny, beam):
    """A data partition: centrally-condensed uv coverage with a real w-spread.

    The PSF-convolution preconditioner is a genuinely imperfect approximation of
    the exact degrid/grid operator -- finite-support/aliasing in the periodic
    convolution, the ``abs(PSFHAT)`` rectification and the w-term (a convolution
    cannot represent it; wiki D22) -- so the consistency assertion is non-vacuous
    (~3% operator gap here, seed-stable and dominated by the aliasing term).
    """
    nrow = 2000
    umax = 1.0 / (2 * cell)
    uvw = np.zeros((nrow, 3))
    uvw[:, 0] = np.clip(rng.standard_normal(nrow) * umax / 3.0, -umax, umax) * LAM
    uvw[:, 1] = np.clip(rng.standard_normal(nrow) * umax / 3.0, -umax, umax) * LAM
    uvw[:, 2] = rng.uniform(-1.0, 1.0, nrow) * 0.3 * umax * LAM  # real w-term
    wgt = rng.uniform(0.5, 1.5, (nrow, 1))
    part = {
        "uvw": uvw,
        "wgt": wgt,
        "beam": beam,
        "psfhat": _psfhat(uvw, wgt, cell, nx, ny)[None],
        "wsum": float(wgt.sum()),
    }
    return part


def _gaussian_beam(nx, ny, cell, fwhm_deg, cx=0.0, cy=0.0):
    """A (ny, nx) Gaussian beam centred at image offset (cx, cy) pixels."""
    x = (-(nx / 2) + np.arange(nx) - cx) * cell
    y = (-(ny / 2) + np.arange(ny) - cy) * cell
    yy, xx = np.meshgrid(y, x, indexing="ij")
    sig = np.deg2rad(fwhm_deg) / 2.355
    return np.exp(-0.5 * (xx**2 + yy**2) / sig**2)


def _build_problem(parts, nx, ny, cell, eta, wsum):
    """Exact operator ``A``, preconditioner ``M`` and their dense matrices.

    ``A(m) = sum_p B_p GtWG(B_p m) / wsum + eta m`` (the full Hessian);
    ``M = HessianTree`` (PSF-convolution). Both share beams, wsum and eta.
    """

    def aop(m_yx):  # the exact operator A
        out = np.zeros((ny, nx))
        for p in parts:
            b = p["beam"]
            out += b * _gtwg(b * m_yx, p["uvw"], p["wgt"], cell)
        return out / wsum + eta * m_yx

    hess_parts = [{"psfhat": p["psfhat"], "beam": p["beam"][None], "wsum": np.array([p["wsum"]])} for p in parts]
    mop = HessianTree(hess_parts, nx, ny, 2 * nx, 2 * ny, eta=eta, wsum=np.array([wsum]), nthreads=NTHREADS)

    npix = nx * ny
    a_dense = np.zeros((npix, npix))
    m_dense = np.zeros((npix, npix))
    e = np.zeros((ny, nx))
    for i in range(npix):
        e.flat[i] = 1.0
        a_dense[:, i] = aop(e).ravel()
        m_dense[:, i] = mop.dot(e[None])[0].ravel()
        e.flat[i] = 0.0
    return aop, mop, a_dense, m_dense


CONFIGS = {
    # single partition, centred source, unit beam: the cleanest statement of
    # the theorem (M still differs from A: PSF-convolution aliasing + abs rect.)
    "single_centered": dict(fov_deg=3.0, seed=11, source=(2, -1), beams=[None]),
    # mosaic: two partitions with DISTINCT beams and an off-centre source where
    # the PSF-convolution preconditioner is a poor approximation -- it must
    # still converge to the exact-Hessian solution
    "mosaic_offcenter": dict(
        fov_deg=4.0,
        seed=23,
        source=(5, 4),
        beams=[dict(fwhm_deg=3.5, cx=1.5, cy=-1.0), dict(fwhm_deg=2.8, cx=-2.0, cy=1.5)],
    ),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_preconditioned_solve_matches_full_hessian(name):
    cfg = CONFIGS[name]
    nx = ny = 16
    eta = 1e-2
    cell = np.deg2rad(cfg["fov_deg"]) / nx
    rng = np.random.default_rng(cfg["seed"])

    beams = []
    for spec in cfg["beams"]:
        beams.append(np.ones((ny, nx)) if spec is None else _gaussian_beam(nx, ny, cell, **spec))
    parts = [_make_partition(rng, cell, nx, ny, beam) for beam in beams]
    wsum = sum(p["wsum"] for p in parts)

    aop, mop, a_dense, m_dense = _build_problem(parts, nx, ny, cell, eta, wsum)

    # the preconditioner is a genuine approximation of the exact operator, not a
    # copy of it (otherwise consistency would be vacuous)
    op_gap = np.linalg.norm(a_dense - m_dense) / np.linalg.norm(a_dense)
    assert op_gap > 0.02, f"{name}: preconditioner too close to exact operator ({op_gap:.3e})"

    # a smooth off-centred source; b = A m_true is the noiseless data-term gradient
    yyp, xxp = np.mgrid[0:ny, 0:nx]
    sl, sm = cfg["source"]
    m_true = np.exp(-0.5 * ((xxp - (nx / 2 + sl)) ** 2 + (yyp - (ny / 2 + sm)) ** 2) / (nx / 12) ** 2)
    b = aop(m_true)

    # "solve with the full Hessian": eta makes A SPD, so the solution is unique
    x_full = np.linalg.solve(a_dense, b.ravel()).reshape(ny, nx)
    np.testing.assert_allclose(x_full, m_true, rtol=0, atol=1e-6 * np.abs(m_true).max())

    # preconditioned Richardson with M^{-1} applied by the repo CG (mirrors
    # forward_alg.solve). The step size only affects the RATE, never the fixed
    # point: gamma_opt = 2/(lam_min+lam_max) of M^{-1}A guarantees convergence
    # for any nonsingular M (the driver's fixed gamma=0.95 is the M~=A case).
    evals = np.linalg.eigvals(np.linalg.solve(m_dense, a_dense)).real
    gamma = 2.0 / (evals.min() + evals.max())

    m = np.zeros((1, ny, nx))
    b3 = b[None]
    rel = np.inf
    for _ in range(400):
        grad = b3 - aop(m[0])[None]
        z = pcg_numba(mop.dot, grad, tol=1e-9, maxit=200, minit=1, verbosity=0)
        m = m + gamma * z
        rel = np.linalg.norm(m[0] - x_full) / np.linalg.norm(x_full)
        if rel < 1e-5:
            break

    assert rel < 1e-5, f"{name}: preconditioned solve did not reach the full-Hessian solution (rel={rel:.3e})"
    np.testing.assert_allclose(m[0], x_full, rtol=0, atol=1e-4 * np.abs(x_full).max())


def _resid_peak(store):
    """Normalised peak of the MFS residual in the ``.dt`` store."""
    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    residual = sum(dt[n].ds.RESIDUAL[0] for n in nodes).values
    wsum = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)
    return np.abs(residual).max() / wsum


@pytest.mark.timeout(600)
def test_deconv_unregularised_residual_keeps_descending(sky_truth, ms_name, tmp_path):
    """End-to-end: with rmsfactor=0 and positivity off the major cycle is the
    preconditioned Richardson above. On noiseless predicted vis (a
    self-consistent forward model) the data misfit does not plateau -- it keeps
    descending toward the gridder-accuracy floor, because the fixed point is the
    exact least-squares solution ``H_exact^{-1} bdirty`` (image residual -> 0:
    ``bdirty`` lies in ``range(H_exact)`` and, noiseless, the injected sky solves
    the normal equations exactly). There is no noise/regularisation floor above
    gridder epsilon; the RATE is limited by the weakly-measured (large-scale /
    short-baseline) modes.

    The test runs two segments and asserts the residual **keeps decreasing** on
    resume -- the invariant that actually characterises asymptotic consistency,
    and the one a spurious floor would break. A D23-style bias (feeding the
    apparent instead of the beam-attenuated gradient) shifts the fixed point off
    ``H_exact^{-1} bdirty``, stalling the residual at a nonzero floor: this test
    would then see the second segment fail to make progress.

    Asserts on the RESIDUAL, not the model -- without a prior the unmeasured
    modes are legitimately unconstrained, so only the data-space misfit vanishes.

    Single band + nthreads=1 on purpose (nband==1 in-process pool path, keeping
    the Ray-actor CPU claims within the session cluster's num_cpus=1; see
    test_deconv_groundtruth).
    """
    from pfb_imaging.core.deconv import deconv as deconv_core
    from pfb_imaging.core.imager import imager as imager_core

    outname = str(tmp_path / "unregdeconv")
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
    deconv_kw = dict(
        minor_cycle="sara",
        opt_backend="primal-dual",
        # damped step: without the prior/positivity to stabilise it, a full
        # gamma=1 step overshoots and diverges here -- the abs(PSFHAT)
        # preconditioner underestimates the curvature so lam_max(M^-1 A) > 2,
        # and preconditioned Richardson needs gamma < 2/lam_max (the regularised
        # groundtruth run tolerates gamma=1 only because the prox pulls the
        # model back each cycle). gamma only sets the RATE, not the fixed point.
        gamma=0.5,
        eta=1e-3,
        rmsfactor=0.0,  # no SARA prior -> pure preconditioned least squares
        positivity=0,  # no positivity prox
        l1_reweight_from=-1,  # reweighting disabled
        tol=1e-8,  # do not early-stop on the model-change criterion
        bases=["self"],
        nlevels=1,
        pd_tol=1e-6,
        pd_maxit=100,
        cg_tol=1e-7,
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

    # segment 1 (fresh); segment 2 resumes from the .dt (MODEL/BRESIDUAL/niters)
    deconv_core(outname, niter=30, **deconv_kw)
    resid1 = _resid_peak(outname + "_I.dt")
    deconv_core(outname, niter=30, **deconv_kw)
    resid2 = _resid_peak(outname + "_I.dt")

    # both far below the regularised run's 0.1 * min_flux bound, and still
    # descending on resume: no plateau/floor, so the fixed point is the exact
    # LS solution (an apparent-vs-beam-attenuated bias would stall resid2~resid1)
    # measured: resid1 ~ 1.1e-3, resid2 ~ 4.4e-4 (ratio ~0.40) of min_flux --
    # still descending, far below the regularised 0.1 * min_flux bound
    min_flux = sky_truth.ref_flux.min()
    assert resid1 < 0.02 * min_flux, f"segment 1 residual peak {resid1:.3e} too high"
    assert resid2 < 0.7 * resid1, f"residual plateaued: resid2 {resid2:.3e} vs resid1 {resid1:.3e}"
