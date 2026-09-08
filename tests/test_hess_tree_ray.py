"""HessTreeRay: Ray band-actor Hessian vs local HessianTree/HessPSF (tier 2)."""

import numpy as np
import pytest
from ducc0.fft import r2c
from numpy.testing import assert_allclose

from pfb_imaging.operators import LinearOperator
from pfb_imaging.operators.hessian import HessianTree, HessPSF, HessTreeRay

pmp = pytest.mark.parametrize


def _rand_part(rng, nx, ny, nx_psf, ny_psf, wsum=1.0):
    """Partition dict with a positive real psfhat (|FT of a random psf|)."""
    psf = rng.uniform(0.0, 1.0, size=(nx_psf, ny_psf))
    psfhat = np.abs(r2c(psf, axes=(0, 1), forward=True, inorm=0))[None]
    beam = np.ones((1, nx, ny))
    return {"psfhat": psfhat, "beam": beam, "wsum": np.array([wsum])}


def test_wsum_override():
    rng = np.random.default_rng(0)
    nx = ny = 8
    part = _rand_part(rng, nx, ny, 2 * nx, 2 * ny, wsum=4.0)
    x = rng.standard_normal((1, nx, ny))
    default = HessianTree([part], nx, ny, 2 * nx, 2 * ny).dot(x)
    overridden = HessianTree([part], nx, ny, 2 * nx, 2 * ny, wsum=8.0).dot(x)
    assert_allclose(overridden, default / 2.0, rtol=1e-13)


@pmp("nband", [1, 3])
def test_dot_matches_local_hessian_tree(nband):
    rng = np.random.default_rng(1)
    nx = ny = 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny) for _ in range(2)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.01)
    assert isinstance(hess, LinearOperator)

    x = rng.standard_normal((nband, nx, ny))
    got = hess.dot(x)
    want = np.zeros_like(x)
    for b in range(nband):
        local = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=0.01)
        want[b] = local.dot(x[b])[0]
    assert_allclose(got, want, rtol=1e-12)


def test_dot_matches_hess_psf_single_partition():
    """Single partition, unit wsum, no beam, eta=0: HessTreeRay == HessPSF."""
    rng = np.random.default_rng(2)
    nband, nx, ny = 2, 16, 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    abspsf = np.concatenate([p[0]["psfhat"] for p in parts], axis=0)

    tree = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.0)
    # taper_width only affects HessPSF.idot (unused here); the default (32)
    # errors for nx=ny=16 (taperf slices [:taper_width] on a length-nx axis),
    # so shrink it as tests/test_protocols.py already does for small images.
    ref = HessPSF(nx, ny, abspsf, eta=0.0, taper_width=2)

    x = rng.standard_normal((nband, nx, ny))
    assert_allclose(tree.dot(x), ref.dot(x).copy(), rtol=1e-12)


@pmp("nband", [1, 3])
def test_cg_matches_local_pcg(nband):
    from pfb_imaging.opt.pcg import pcg_numba

    rng = np.random.default_rng(3)
    nx = ny = 16
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.5, cg_tol=1e-8, cg_maxit=200)

    rhs = rng.standard_normal((nband, nx, ny))
    got = hess.cg(rhs)
    for b in range(nband):
        local = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=0.5)
        want_b = pcg_numba(lambda z: local.dot(z)[0], rhs[b], tol=1e-8, maxit=200, minit=1, verbosity=0)
        assert_allclose(got[b], want_b, rtol=1e-6, atol=1e-9)


def test_prior_term_matches_the_dense_congruence():
    """M_gp x == M_data x + eta * (Cinv @ x) when eta is uniform.

    With eta_mode unset the congruence D^.5 Cinv D^.5 collapses to eta*Cinv, so
    the driver-side term is exactly eta*(Cinv - I) applied over the band axis.
    """
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(3)
    nband, nx, ny = 3, 8, 8
    eta = 1e-2
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)

    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=eta, freq_prec=kinv)

    x = rng.standard_normal((nband, nx, ny))
    # the M_data + eta*I reference is built with local HessianTrees rather than a
    # second HessTreeRay: it halves the Ray actor count and keeps this test -- the
    # one that pins the prior term's sign and value -- inside the fast loop
    base_dot = np.zeros_like(x)
    for b in range(nband):
        base_dot[b] = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=eta).dot(x[b])[0]
    want = base_dot - eta * x + eta * np.einsum("bc,cyx->byx", kinv, x)
    assert_allclose(gp.dot(x), want, rtol=1e-11, atol=1e-13)


def test_prior_is_a_no_op_when_freq_prec_is_none():
    """The default path must be bit-identical to today (regression guard)."""
    rng = np.random.default_rng(4)
    nband, nx, ny = 2, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2, freq_prec=None)
    x = rng.standard_normal((nband, nx, ny))
    want = np.zeros_like(x)
    for b in range(nband):
        want[b] = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=1e-2).dot(x[b])[0]
    assert_allclose(hess.dot(x), want, rtol=0, atol=0)


def test_prior_hessian_stays_symmetric_and_positive_definite():
    """CG requires both; the driver-side correction alone is only NSD."""
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(5)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 1.0, cap=50.0)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2, freq_prec=kinv)

    for _ in range(10):
        u = rng.standard_normal((nband, nx, ny))
        v = rng.standard_normal((nband, nx, ny))
        assert_allclose(np.vdot(u, gp.dot(v)), np.vdot(gp.dot(u), v), rtol=1e-10)
        assert np.vdot(u, gp.dot(u)) > 0.0


def test_wrong_freq_prec_shape_is_rejected():
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(6)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, 4), 0.5)  # 4 bands, not 3
    with pytest.raises(ValueError, match="freq_prec"):
        HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2, freq_prec=kinv)


def test_cg_solves_the_coupled_system_when_the_prior_is_on():
    """The real contract: whichever branch runs, cg must invert the operator dot applies."""
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(7)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-1, freq_prec=kinv, cg_tol=1e-10, cg_maxit=500)

    rhs = rng.standard_normal((nband, nx, ny))
    u = gp.cg(rhs)
    assert_allclose(gp.dot(u), rhs, rtol=1e-5, atol=1e-7)


def test_cg_without_the_prior_still_uses_the_band_parallel_pool_path():
    """The in-worker fast path is one Ray dispatch per solve; do not lose it."""
    rng = np.random.default_rng(8)
    nband, nx, ny = 2, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-1)

    calls = []
    original = hess._pool.hess_cg

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    hess._pool.hess_cg = spy
    u = hess.cg(rng.standard_normal((nband, nx, ny)))
    assert calls == [1], "the band-parallel pool path was bypassed"
    assert u.shape == (nband, nx, ny)


def test_cg_with_the_prior_bypasses_the_band_parallel_pool_path():
    """Band-parallel CG cannot solve a band-coupled operator."""
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(9)
    nband, nx, ny = 3, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-1, freq_prec=kinv)

    def boom(*args, **kwargs):
        raise AssertionError("hess_cg must not be called when bands are coupled")

    gp._pool.hess_cg = boom
    gp.cg(rng.standard_normal((nband, nx, ny)))


def test_prior_stats_are_none_when_the_prior_is_off():
    rng = np.random.default_rng(10)
    nband, nx, ny = 2, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2)
    assert hess.get_freq_prior_stats() is None


def test_prior_stats_report_the_spectrum_and_its_contribution_to_m():
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(11)
    nband, nx, ny = 3, 8, 8
    eta, cap = 1e-2, 10.0
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 1.0, cap=cap)
    gp = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=eta, freq_prec=kinv)

    stats = gp.get_freq_prior_stats()
    assert_allclose(stats["prec_max"], 1.0, rtol=1e-12)
    assert stats["prec_min"] >= 1.0 / cap - 1e-12
    assert_allclose(stats["eta_max"], eta, rtol=1e-12)
    # the prior's top contribution to M equals eta: lambda_max(M) is unchanged
    assert_allclose(stats["lam_max"], eta, rtol=1e-12)
    assert_allclose(stats["lam_min"], stats["prec_min"] * eta, rtol=1e-12)


# --- what the prior does to M's spectrum, and why CG gets slower (issue #307) ---


def _shared_parts(rng, nband, nx, ny, uv_hole=False):
    """One partition shared by every band, so ``M_data = I_nband (x) A``.

    With the same ``A`` in every band the data term and the prior commute and
    the coupled operator's whole spectrum is available in closed form.
    ``uv_hole`` zeroes three quarters of ``psfhat`` -- unsampled uv cells, where
    the data term has NO curvature at all (``lambda_min(A) == 0`` to roundoff)
    and ``eta`` is the only thing holding ``M`` up. Real data lives in that
    regime over much of the uv plane: ``scripts/max_gamma.py`` reports 25-30% of
    Fourier modes below ``eta`` on ``subset_withbeam_I.dt``.
    """
    psf = rng.uniform(0.0, 1.0, size=(2 * nx, 2 * ny))
    psfhat = np.abs(r2c(psf, axes=(0, 1), forward=True, inorm=0))[None]
    if uv_hole:
        psfhat[:, nx // 2 :, :] = 0.0
    part = {"psfhat": psfhat, "beam": np.ones((1, nx, ny)), "wsum": np.array([1.0])}
    return part, [[part] for _ in range(nband)]


def _pair(parts, nx, ny, eta, kinv, **kw):
    """``(uncoupled, coupled)`` facades over ONE shared worker pool.

    Both constructors call ``init_hess`` with identical arguments, so the
    worker-side operator is the same for both and only the driver-side coupling
    differs -- which is also what the deconv driver does, and halves the Ray
    actor startup these tests would otherwise pay twice over.
    """
    from pfb_imaging.operators.band_worker import BandWorkerPool

    common = dict(etas=eta, workers=BandWorkerPool(len(parts), 1), **kw)
    off = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, **common)
    on = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, freq_prec=kinv, **common)
    return off, on


def _dense(op, shape):
    """Materialise a cube operator as a matrix, column by column."""
    n = int(np.prod(shape))
    out = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        out[:, i] = np.asarray(op(e.reshape(shape))).ravel()
    return out


def test_prior_couples_bands_and_never_pixels():
    """The coupling is exactly ``eta*(Cinv - I)`` down the band axis, pixel by pixel.

    ``dot`` applies it as ``(nband, nband) @ (nband, npix)`` over a
    ``reshape(nband, -1)`` view, so an axis mix-up would smear one pixel across
    its neighbours -- checked directly with a single-pixel delta, whose response
    must stay in that pixel.
    """
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(15)
    nband, nx, ny = 3, 6, 6
    eta = 1e-2
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)
    off, on = _pair(parts, nx, ny, eta, kinv)
    dprec = kinv - np.eye(nband)

    x = rng.standard_normal((nband, nx, ny))
    assert_allclose(on.dot(x) - off.dot(x), eta * np.einsum("bc,cyx->byx", dprec, x), rtol=1e-10, atol=1e-14)

    d = np.zeros((nband, nx, ny))
    d[0, 2, 3] = 1.0
    resp = on.dot(d) - off.dot(d)
    elsewhere = np.ones((nx, ny), dtype=bool)
    elsewhere[2, 3] = False
    assert_allclose(resp[:, elsewhere], 0.0, atol=1e-15)
    assert_allclose(resp[:, 2, 3], eta * dprec[:, 0], rtol=1e-10)


def test_prior_congruence_with_a_spatially_varying_eta():
    """The ``eta_mode`` branch of ``_s``: D is a per-band, per-pixel profile, not a scalar.

    ``test_prior_term_matches_the_dense_congruence`` covers only uniform eta,
    where ``D^.5 Cinv D^.5`` collapses to ``eta*Cinv``. Under ``--eta-mode`` the
    congruence is the whole point and ``_s`` is an ``(nband, ny, nx)`` cube
    fetched back from the workers, so a band or axis mix-up is possible in a way
    it is not for a scalar. Per-band beams make the profiles differ, so reusing
    one band's profile for all of them fails here.
    """
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(16)
    nband, nx, ny = 3, 6, 6
    eta, eta_mode, eta_cap = 1e-2, "invbeam", 20.0
    parts = []
    for _ in range(nband):
        p = _rand_part(rng, nx, ny, 2 * nx, 2 * ny)
        p["beam"] = rng.uniform(0.2, 1.0, size=(1, nx, ny))
        parts.append([p])
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)
    _, on = _pair(parts, nx, ny, eta, kinv, eta_mode=eta_mode, eta_cap=eta_cap)

    x = rng.standard_normal((nband, nx, ny))
    base = np.zeros_like(x)
    s = np.zeros_like(x)
    for b in range(nband):
        local = HessianTree(parts[b], nx, ny, 2 * nx, 2 * ny, eta=eta, eta_mode=eta_mode, eta_cap=eta_cap)
        base[b] = local.dot(x[b])[0]
        s[b] = np.sqrt(np.asarray(local.eta).reshape(-1, ny, nx)[0])
    assert s.std(axis=(1, 2)).min() > 0.0, "the profiles must vary spatially or this tests nothing"
    assert np.abs(np.diff(s, axis=0)).max() > 0.0, "the profiles must differ between bands"

    want = base + s * np.einsum("bc,cyx->byx", kinv - np.eye(nband), s * x)
    assert_allclose(on.dot(x), want, rtol=1e-10, atol=1e-14)


@pytest.mark.slow
def test_prior_matches_the_kronecker_spectrum():
    """M_gp's entire spectrum is ``alpha_k + eta*p_j`` -- data eigenvalue plus prior eigenvalue.

    Every band shares one partition, so ``M_data = I_nband (x) A`` and
    ``P = eta*Cinv_n (x) I`` commute and their eigenvalues add pairwise. Checking
    the whole sorted spectrum against that closed form pins the driver-side term
    far harder than one dot product can: a sign flip, a transposed apply, a
    band/pixel mix-up, or the wrong normalisation each break it, and none of them
    are visible to the fixed-point test (wiki D22).
    """
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(12)
    nband, nx, ny = 3, 6, 6
    eta, cap = 1e-2, 10.0
    part, parts = _shared_parts(rng, nband, nx, ny, uv_hole=True)
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=cap)

    _, on = _pair(parts, nx, ny, eta, kinv)
    got = np.linalg.eigvalsh(_dense(on.dot, (nband, nx, ny)))

    a = _dense(lambda z: HessianTree([part], nx, ny, 2 * nx, 2 * ny, eta=0.0).dot(z)[0], (nx, ny))
    want = np.sort(np.add.outer(np.linalg.eigvalsh(a), eta * np.linalg.eigvalsh(kinv)).ravel())
    assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.slow
@pmp("uv_hole", [True, False])
def test_prior_lowers_lambda_min_only_where_the_data_lacks_curvature(uv_hole):
    """``lambda_max`` is untouched; ``lambda_min`` drops by the precision's smallest eigenvalue.

    This is the designed trade, not a defect. The precision is normalised so the
    ROUGHEST frequency mode sits at exactly 1 and smoother ones are *relaxed*
    down to ``1/cap`` (wiki D30), so the prior only ever removes curvature from
    ``M``. Where the data term has curvature of its own the removal is invisible;
    where it has none, ``lambda_min(M)`` falls by the full factor and
    ``cond(M)`` rises by it. Contrast ``eta_profile``, whose modes are normalised
    so that ``lambda_min(M)`` cannot degrade -- these two knobs pull opposite ways
    on the CG iteration count, by design.
    """
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(13)
    nband, nx, ny = 3, 6, 6
    eta, cap = 1e-2, 10.0
    _, parts = _shared_parts(rng, nband, nx, ny, uv_hole=uv_hole)
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=cap)
    p_min = float(np.linalg.eigvalsh(kinv).min())

    shape = (nband, nx, ny)
    off, on = _pair(parts, nx, ny, eta, kinv)
    w_off = np.linalg.eigvalsh(_dense(off.dot, shape))
    w_on = np.linalg.eigvalsh(_dense(on.dot, shape))

    # the prec.max() == 1 normalisation, read off the operator rather than the
    # matrix: lambda_max(M) -- hence hess_norm and the PD step sizes -- is fixed
    assert_allclose(w_on.max(), w_off.max(), rtol=1e-10)
    assert w_on.min() <= w_off.min() * (1.0 + 1e-10), "the prior may only remove curvature from M"
    if uv_hole:
        # lambda_min(A) == 0 there, so lambda_min(M) is eta and eta*p_min exactly
        assert_allclose(w_off.min(), eta, rtol=1e-6)
        assert_allclose(w_on.min() / w_off.min(), p_min, rtol=1e-6)
    else:
        # the data term dominates eta everywhere, so the prior is nearly free
        assert w_on.min() / w_off.min() > 0.9


@pytest.mark.slow
def test_prior_needs_more_cg_iterations_and_that_is_expected():
    """CG gets SLOWER with the prior on. Pinned here so it is not read as a regression.

    ``lambda_min(M)`` drops by up to ``gp_cap`` (previous test) while
    ``lambda_max(M)`` is fixed, so ``cond(M)`` rises by up to ``gp_cap`` and CG
    needs of order ``sqrt(gp_cap)`` more iterations. Measured on real data:
    95 -> >150 applications for one forward solve
    (``scripts/profile_freq_correlated_hessian.py``). Expecting a prior to *help*
    CG is reading it as added regularisation; this one is a relaxation.
    """
    from pfb_imaging.operators.hessian import freq_precision
    from pfb_imaging.opt.pcg import pcg_numba

    rng = np.random.default_rng(14)
    nband, nx, ny = 3, 6, 6
    eta, cap = 1e-2, 10.0
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=cap)
    rhs = rng.standard_normal((nband, nx, ny))

    def applications(op):
        n = [0]

        def counted(z):
            n[0] += 1
            return op.dot(z)

        pcg_numba(counted, rhs.copy(), tol=1e-8, maxit=5000, minit=1, verbosity=0)
        return n[0]

    _, holed = _shared_parts(rng, nband, nx, ny, uv_hole=True)
    off, on = _pair(holed, nx, ny, eta, kinv)
    it_off, it_on = applications(off), applications(on)
    assert it_on > 1.5 * it_off, f"expected the prior to cost iterations, got {it_off} -> {it_on}"

    # where eta is not load-bearing the same prior is nearly free: the cost
    # tracks how much of M's spectrum eta is holding up, not the prior itself
    _, full = _shared_parts(rng, nband, nx, ny, uv_hole=False)
    off_f, on_f = _pair(full, nx, ny, eta, kinv)
    it_full, it_full_gp = applications(off_f), applications(on_f)
    assert it_full_gp < 1.3 * it_full, f"expected the prior to be nearly free here, got {it_full} -> {it_full_gp}"


# ---------------------------------------------------------------------------
# prior_dot: the non-data part of M, exposed so the gradient can carry it
# (issue #310, --eta-in-grad). M itself is only ever applied as a whole.
# ---------------------------------------------------------------------------


@pmp("length_scale", [None, 0.5])
def test_prior_dot_is_exactly_the_non_data_part_of_m(length_scale):
    """``dot(x) == M_data x + prior_dot(x)``, prior on or off.

    This is the contract --eta-in-grad rests on: the gradient must gain
    precisely the term the preconditioner adds, so building it from anything
    but M itself (a hardcoded ``eta * x``, say) would drift from M the moment
    the prior or eta_mode is on.
    """
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(20)
    nband, nx, ny = 3, 8, 8
    eta = 1e-2
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = None if length_scale is None else freq_precision(np.linspace(1.0e9, 1.4e9, nband), length_scale)

    data_only = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=0.0)
    full = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=eta, freq_prec=kinv)

    x = rng.standard_normal((nband, nx, ny))
    assert_allclose(full.dot(x) - data_only.dot(x), full.prior_dot(x), rtol=1e-11, atol=1e-13)


def test_prior_dot_is_eta_times_x_when_the_prior_is_off():
    """The uncorrelated limit, written out: K^-1 = eta * I."""
    rng = np.random.default_rng(21)
    nband, nx, ny = 2, 8, 8
    eta = 3e-2
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=eta)

    x = rng.standard_normal((nband, nx, ny))
    assert_allclose(hess.prior_dot(x), eta * x, rtol=1e-12, atol=0)


def test_prior_dot_couples_bands_through_the_correlation_matrix():
    """With the prior on and uniform eta: K^-1 x == eta * (Cinv @ x)."""
    from pfb_imaging.operators.hessian import freq_precision

    rng = np.random.default_rng(22)
    nband, nx, ny = 3, 8, 8
    eta = 1e-2
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    kinv = freq_precision(np.linspace(1.0e9, 1.4e9, nband), 0.5, cap=10.0)
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=eta, freq_prec=kinv)

    x = rng.standard_normal((nband, nx, ny))
    want = eta * np.einsum("bc,cyx->byx", kinv, x)
    assert_allclose(hess.prior_dot(x), want, rtol=1e-11, atol=1e-13)


def test_prior_dot_does_not_mutate_its_argument():
    """The driver subtracts it from a residual it still needs."""
    rng = np.random.default_rng(23)
    nband, nx, ny = 2, 8, 8
    parts = [[_rand_part(rng, nx, ny, 2 * nx, 2 * ny)] for _ in range(nband)]
    hess = HessTreeRay(parts, nx, ny, 2 * nx, 2 * ny, etas=1e-2)

    x = rng.standard_normal((nband, nx, ny))
    before = x.copy()
    hess.prior_dot(x)
    assert_allclose(x, before, rtol=0, atol=0)
