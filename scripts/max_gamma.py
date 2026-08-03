#!/usr/bin/env python
"""Measure the largest stable ``pfb deconv --gamma`` for a given ``.dt``.

With ``--rmsfactor 0 --positivity 0`` the major cycle is preconditioned
Richardson on the exact normal equations (wiki ``deconv-primer.md``; issue #287)::

    m_{k+1} = m_k + gamma * M^{-1} ( bdirty - H_exact m_k )

which converges iff ``gamma < 2 / lambda_max(M^-1 H_exact)``, where ``M`` is the
``HessianTree`` PSF-convolution preconditioner and ``H_exact = sum_p B_p GtWG B_p``
is the exact degrid/grid Hessian.  ``M`` is only an approximation of ``H_exact``
(``abs(PSFHAT)`` rectification, PSF truncation at ``psf_oversize < 2``, and the
w-term, which makes ``H_exact`` non-convolutional over a wide field), so
``lambda_max`` can exceed 1 and the driver's fixed ``gamma`` can silently cross
into divergence.  Issue #287 measured ``lambda_max`` from 1.00 at
``psf_oversize=2`` to 9.9 at ``psf_oversize=1`` on a small coplanar test problem;
this script measures it on real data.

``M^-1 H_exact`` is self-adjoint in the ``M`` inner product, so power iteration
converges to ``lambda_max`` and the generalised Rayleigh quotient
``v'Hv / v'Mv`` is a monotone lower bound.  Each iteration costs one exact
degrid/grid sweep plus one CG solve -- roughly one deconv major cycle.

Also reports the Rayleigh quotient along the directions the solver actually
moves in (the stored ``UPDATE`` and ``DIRTY``).  Those give the practically
binding bound: a large ``lambda_max`` in a direction the iteration never excites
does not destabilise it, whereas a large quotient along ``UPDATE`` does.

Interpreting a large ``lambda_max`` needs to know where ``eta`` sits in ``M``'s
own spectrum, so the header reports ``lambda_max(M)``, the percentiles of ``M``'s
convolution multiplier ``|PSFHAT|/wsum_tot``, and the fraction of Fourier modes
below ``eta``.  Wherever the multiplier falls below ``eta`` the preconditioner is
effectively ``eta*I`` and any ``H`` response there is divided by ``eta``, so
``lambda ~ h/eta``: a sub-percent operator mismatch landing on those modes is
enough to put ``lambda_max`` in the tens.  ``psf_oversize`` and the ``BEAM``
range are reported for the same reason -- truncation and the beam floor are the
other two ways ``M`` loses curvature that ``H`` still has.

The beam enters ``H`` on BOTH sides (``sum_p B_p GtWG B_p``, wiki D23): the
gradient sweep's second return value carries the outer per-partition beam, which
is what this script consumes.  Using the once-attenuated apparent residual
instead would make ``H`` non-symmetric -- ``--check-adjoint`` asserts it is not.

Run (from the repo root)::

    uv run python scripts/max_gamma.py /path/to/out_I.dt --eta 1e-3 --nthreads 8

Writes ``<dt>_max_gamma.json`` and, unless ``--no-fits``, the dominant
eigenvector as ``<dt>_max_gamma_eigvec.fits`` -- inspect it to see where the
instability lives (field edge and high frequency implicates the w-term /
off-axis PSF mismatch; large-scale implicates the short-baseline hole).
"""

import argparse
import json

import numpy as np
import psutil
import xarray as xr
from ducc0.misc import resize_thread_pool

from pfb_imaging import init_ray, set_envs, setup_ray_worker
from pfb_imaging.deconv.presets import _build_hess
from pfb_imaging.operators.band_worker import BandWorkerPool
from pfb_imaging.operators.hessian import ETA_MODES
from pfb_imaging.opt.power_method import power_method_numba as power_method
from pfb_imaging.utils.fits import save_fits, set_wcs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dt", help="Path to the imager output <output-filename>_<PRODUCT>.dt")
    p.add_argument("--eta", type=float, default=1e-3, help="Tikhonov term of the preconditioner (deconv --eta)")
    p.add_argument(
        "--eta-mode",
        default=None,
        choices=list(ETA_MODES),
        help="Shape of a spatially varying eta (deconv --eta-mode); default is uniform",
    )
    p.add_argument("--eta-cap", type=float, default=1e2, help="Dynamic range of the --eta-mode profile")
    p.add_argument("--niter", type=int, default=15, help="Maximum power iterations")
    p.add_argument("--tol", type=float, default=5e-3, help="Stop when the relative change in lambda falls below this")
    p.add_argument("--nthreads", type=int, default=4, help="Threads per band worker")
    p.add_argument("--nworkers", type=int, default=None, help="Ray workers (default: nband)")
    p.add_argument("--ray-address", default="local")
    p.add_argument("--cg-tol", type=float, default=1e-8, help="CG tolerance for the M^-1 applications")
    p.add_argument("--cg-maxit", type=int, default=3000)
    p.add_argument("--epsilon", type=float, default=1e-7, help="Gridder accuracy (match the deconv run)")
    p.add_argument("--no-wgridding", action="store_true", help="Disable w-gridding (match the deconv run)")
    p.add_argument("--no-double-accum", action="store_true")
    p.add_argument("--safety", type=float, default=1.8, help="Recommend gamma = safety / lambda_max (< 2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fits", action="store_true", help="Skip writing the dominant eigenvector")
    p.add_argument("--pm-tol", type=float, default=1e-3, help="Tolerance for the lambda_max(M) power method")
    p.add_argument("--pm-maxit", type=int, default=200)
    p.add_argument(
        "--check-adjoint",
        action="store_true",
        help="Assert H is symmetric and M^-1 H is self-adjoint in the M inner product, then exit. "
        "Catches a beam applied only once (which makes H non-symmetric).",
    )
    return p.parse_args()


def load_tree(dt_name):
    """Band nodes, geometry and per-band wsums, mirroring core/deconv.py."""
    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    names = sorted(n for n in dt.children if n.startswith("band"))
    if not names:
        raise ValueError(f"No band nodes found in {dt_name}")
    nodes = sorted(names, key=lambda n: int(dt[n].ds.attrs["bandid"]))
    first = dt[nodes[0]].ds
    if first.corr.size > 1:
        raise NotImplementedError("Joint polarisation not supported (matches pfb deconv)")
    part = next(iter(dt[nodes[0]].children.values()), None)
    if part is None or "PSFHAT" not in part.ds:
        raise ValueError(f"{dt_name} has no per-partition PSFHAT -- re-run pfb imager with --psf")
    if "BDIRTY" not in first:
        raise ValueError(f"{dt_name} has no BDIRTY -- re-run pfb imager to regenerate the .dt")
    wsums = np.array([float(dt[n].ds.WSUM.values[0]) for n in nodes])
    geometry = {
        "nx": first.x.size,
        "ny": first.y.size,
        "nx_psf": first.x_psf.size,
        "ny_psf": first.y_psf.size,
    }
    meta = {
        "cell_rad": float(first.attrs["cell_rad"]),
        "radec": [float(first.attrs["ra"]), float(first.attrs["dec"])],
        "l0": float(first.attrs.get("l0", 0.0)),
        "m0": float(first.attrs.get("m0", 0.0)),
        "freq_out": np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes]),
    }
    cubes = {}
    for key in ("UPDATE", "DIRTY"):
        if key in first:
            cubes[key] = np.stack([dt[n].ds[key].values[0] for n in nodes])
    return dt, nodes, geometry, wsums, meta, cubes


def spectrum_report(dt, nodes, wsum_tot, wsums, eta):
    """Where ``eta`` sits in ``M``'s spectrum, plus psf_oversize and the beam.

    ``M``'s convolution part has Fourier multiplier ``sum_p |PSFHAT_p| / wsum_tot``
    (exact only for a unit beam, indicative otherwise -- with a beam ``M`` is not
    diagonal in Fourier).  Modes below ``eta`` are ones where the preconditioner
    has no curvature of its own, so any ``H`` response there is amplified by
    ``1/eta``.  The beam floor is the image-space analogue: pixels where
    ``B^2 * wsum_b / wsum_tot < eta`` are effectively pure Tikhonov.
    """
    out = {}
    print("\nM's convolution multiplier |PSFHAT|/wsum_tot (indicative when a beam is present):")
    for name in nodes:
        band = dt[name]
        mult = None
        for cname in sorted(band.children):
            a = np.abs(band[cname].ds.PSFHAT.values[0])
            mult = a if mult is None else mult + a
            del a
        mult /= wsum_tot
        pct = float((mult < eta).mean()) * 100.0
        # eta's percentile in the multiplier distribution
        med = float(np.median(mult))
        print(
            f"  {name}: max {mult.max():.3e}  median {med:.3e}  min {mult.min():.3e}  | {pct:5.1f}% of modes below eta"
        )
        out[name] = {"mult_max": float(mult.max()), "mult_median": med, "pct_modes_below_eta": pct}
        del mult

    for b, name in enumerate(nodes):
        ds = dt[name].ds
        if "BEAM" not in ds:
            continue
        beam = ds.BEAM.values[0]
        # image-space curvature of the convolution part is ~ B^2 * psf(0)/wsum_tot
        floor = float((beam**2 * wsums[b] / wsum_tot < eta).mean()) * 100.0
        print(
            f"  {name} BEAM: min {beam.min():.4f} max {beam.max():.4f} median {np.median(beam):.4f}"
            f"  | {floor:5.1f}% of pixels below the eta beam floor"
        )
        out[name]["beam_min"] = float(beam.min())
        out[name]["beam_max"] = float(beam.max())
        out[name]["pct_pixels_below_beam_floor"] = floor
        del beam
    return out


def eigvec_report(v):
    """Where the dominant eigenvector's power lives -- radially and in frequency."""
    p = v**2
    tot = p.sum()
    if tot == 0:
        return {}
    _, ny, nx = v.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt(((yy - ny / 2) / (ny / 2)) ** 2 + ((xx - nx / 2) / (nx / 2)) ** 2)
    outer = float(p.sum(axis=0)[r > 0.5].sum() / tot)
    f = np.abs(np.fft.fftshift(np.fft.fft2(v.mean(axis=0)))) ** 2
    hf = float(f[r > 0.5].sum() / f.sum()) if f.sum() > 0 else float("nan")
    print(f"\ndominant eigenvector: {100 * outer:.1f}% of power outside half the field radius, ")
    print(f"                      high-frequency fraction {hf:.3f} (power above half-Nyquist)")
    print("  field-edge + high-frequency implicates the w-term / off-axis PSF mismatch;")
    print("  large-scale implicates the short-baseline hole.")
    return {"frac_power_outside_half_radius": outer, "high_freq_fraction": hf}


def denominator_report(v, hv, den, hess, workers, geometry, nband, wsum_tot, args):
    """Split ``v'Mv`` along the dominant eigenvector into its two terms.

    ``lambda = v'Hv / (v'BCBv/wsum + v'e v)``. The share of the denominator that
    the Tikhonov term supplies bounds what ANY eta profile can buy: scaling
    ``e`` by ``R`` gives ``lambda >= v'Hv/(a + R*e_term)``, a lower bound
    because the maximiser then moves to a mode with more curvature. So the
    required ``R`` printed here is optimistic -- if it is already large, no
    profile with a sane dynamic range will stabilise the run.

    Costs one extra Hessian application: the pool is re-initialised with
    ``eta=0`` to isolate the beam-convolution term, then restored.
    """
    nx, ny = geometry["nx"], geometry["ny"]
    nx_psf, ny_psf = geometry["nx_psf"], geometry["ny_psf"]
    wtot = np.full(nband, wsum_tot)
    workers.init_hess(None, nx, ny, nx_psf, ny_psf, np.zeros(nband), wtot)
    a = float(np.vdot(v, workers.hess_dot(v)).real)
    workers.init_hess(None, nx, ny, nx_psf, ny_psf, np.full(nband, args.eta), wtot, args.eta_mode, args.eta_cap)
    h = float(np.vdot(v, hv).real)
    e_term = den - a
    print("\ndenominator along the dominant eigenvector (v'Mv = a + e_term):")
    print(f"  v'Hv (exact curvature)      = {h:.4e}")
    print(f"  a    = v'BCBv/wsum          = {a:.4e}  ({100 * a / den:.1f}% of v'Mv)")
    print(f"  e_term = v'e v              = {e_term:.4e}  ({100 * e_term / den:.1f}% of v'Mv)")
    needs = {}
    if e_term > 0:
        if a > 0:
            print(f"  M under-estimates the curvature along this mode by {h / a:.1f}x")
        print("  eta multiplier needed at this mode (optimistic; the maximiser moves):")
        for gam, lam in ((1.0, 2.0), (0.5, 4.0)):
            # lambda = h/(a + R*e_term) <= lam  =>  R >= (h/lam - a)/e_term.
            # R <= 1 means the current e already suffices along this mode (R <= 0
            # means a alone does) -- the binding constraint is then another mode.
            need = (h / lam - a) / e_term
            needs[gam] = need
            verdict = f"{need:.1f}x" if need > 1.0 else f"satisfied along this mode ({need:.2f}x of the current e)"
            print(f"    gamma = {gam:<4} (lambda <= {lam})  ->  {verdict}")
    return {
        "vHv": h,
        "a_conv": a,
        "e_term": e_term,
        "eta_share_of_denominator": e_term / den if den else np.nan,
        "eta_multiplier_needed": needs,
    }


def check_adjoint(h_exact, hess, shape, seed=0):
    """H must be symmetric; M^-1 H self-adjoint in the M inner product, not the Euclidean one.

    A beam applied only once (the apparent instead of the beam-attenuated
    gradient) shows up here immediately as a non-symmetric H.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    y = rng.standard_normal(shape)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

    def dot(a, b):
        return float(np.vdot(a, b).real)

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-300)

    hx, hy = h_exact(x), h_exact(y)
    r_h = rel(dot(hx, y), dot(x, hy))
    print(f"  <Hx,y> vs <x,Hy>            : rel {r_h:.2e}")
    mx, my = hess.dot(x), hess.dot(y)
    r_m = rel(dot(mx, y), dot(x, my))
    print(f"  <Mx,y> vs <x,My>            : rel {r_m:.2e}")
    ax, ay = hess.cg(hx), hess.cg(hy)
    print(f"  <Ax,y> vs <x,Ay>  (Euclid)  : rel {rel(dot(ax, y), dot(x, ay)):.2e}  (expected to FAIL)")
    r_am = rel(dot(ax, hess.dot(y)), dot(x, hess.dot(ay)))
    print(f"  <Ax,y>_M vs <x,Ay>_M        : rel {r_am:.2e}")
    assert r_h < 1e-10, f"H is not symmetric (rel {r_h:.2e}) -- is the beam applied only once?"
    assert r_m < 1e-10, f"M is not symmetric (rel {r_m:.2e})"
    print("  H and M symmetric, M^-1 H self-adjoint in the M inner product: OK")


def main():
    args = parse_args()
    dt_name = args.dt.rstrip("/")
    dt, nodes, geometry, wsums, meta, cubes = load_tree(dt_name)
    nband = len(nodes)
    ny, nx = geometry["ny"], geometry["nx"]
    wsum_tot = float(wsums.sum())
    nworkers = args.nworkers or nband

    resize_thread_pool(args.nthreads)
    ncpu = int(np.minimum(args.nthreads, psutil.cpu_count(logical=False)))
    env_vars = set_envs(args.nthreads, ncpu)
    init_ray(
        nworkers,
        ray_address=args.ray_address,
        runtime_env={"env_vars": env_vars, "worker_process_setup_hook": setup_ray_worker},
    )

    workers = BandWorkerPool(nband, args.nthreads)
    workers.load_bands(dt_name, nodes)

    # identical construction to the deconv driver so the measured lambda applies
    # to the run being diagnosed (total-wsum normalisation, wiki D4)
    opts = {
        "eta": args.eta,
        "eta_mode": args.eta_mode,
        "eta_cap": args.eta_cap,
        "nthreads": args.nthreads,
        "cg_tol": args.cg_tol,
        "cg_maxit": args.cg_maxit,
        "cg_verbose": 0,
    }
    hess = _build_hess(None, geometry, opts, workers=workers, wsums=wsums)

    grid_kw = dict(
        cell_rad=meta["cell_rad"],
        epsilon=args.epsilon,
        do_wgridding=not args.no_wgridding,
        double_accum=not args.no_double_accum,
    )

    # r(m) = bdirty - H_exact m is affine in m, so H_exact v = r(0) - r(v).
    # Going through workers.residual (rather than reading BDIRTY) keeps this on
    # exactly the code path the major cycle uses.
    def bresidual(m):
        return workers.residual(m[:, None], **grid_kw)[1][:, 0]

    br0 = bresidual(np.zeros((nband, ny, nx)))

    def h_exact(v):
        return (br0 - bresidual(v)) / wsum_tot

    def rayleigh(v):
        """v'Hv / v'Mv -- a lower bound on lambda_max(M^-1 H_exact)."""
        num = float(np.vdot(v, h_exact(v)).real)
        den = float(np.vdot(v, hess.dot(v)).real)
        return num / den if den != 0 else np.nan

    nx_psf, ny_psf = geometry["nx_psf"], geometry["ny_psf"]
    print(
        f"{nband} band(s), {ny}x{nx}, psf {ny_psf}x{nx_psf} (psf_oversize {nx_psf / nx:.3f}), wsum_tot = {wsum_tot:.4e}"
    )

    if args.check_adjoint:
        print("\nadjointness checks:")
        check_adjoint(h_exact, hess, (nband, ny, nx), seed=args.seed)
        return

    # lambda_max(M): sets the scale eta has to be judged against. Aliasing-exact
    # convolution needs psf_oversize >= 2; below that M degrades (issue #287).
    lam_m, _ = power_method(hess.dot, (nband, ny, nx), tol=args.pm_tol, maxit=args.pm_maxit, verbosity=0)
    emode = "uniform" if args.eta_mode is None else f"{args.eta_mode} (cap {args.eta_cap:g})"
    print(f"eta = {args.eta:.3e} [{emode}]   lambda_max(M) = {lam_m:.4e}   eta/lambda_max(M) = {args.eta / lam_m:.2e}")
    spectrum = spectrum_report(dt, nodes, wsum_tot, wsums, args.eta)
    print()

    # directions the solver actually moves in
    probes = {}
    for key, cube in cubes.items():
        if np.any(cube):
            probes[key] = rayleigh(cube / np.linalg.norm(cube))
            print(f"Rayleigh quotient along {key:<7} = {probes[key]:.4f}")

    rng = np.random.default_rng(args.seed)
    v = rng.standard_normal((nband, ny, nx))
    v /= np.linalg.norm(v)

    lam, lam_prev, history = np.nan, np.inf, []
    # the (v, Hv, v'Mv) triple lam was computed from: v advances once more below,
    # so keeping the consistent triple lets denominator_report reuse it instead of
    # paying for another exact degrid/grid sweep
    vlam = hvlam = None
    denlam = np.nan
    for k in range(args.niter):
        hv = h_exact(v)
        den = float(np.vdot(v, hess.dot(v)).real)
        lam = float(np.vdot(v, hv).real) / den
        vlam, hvlam, denlam = v, hv, den
        history.append(lam)
        rel = abs(lam - lam_prev) / max(abs(lam), 1e-30)
        print(f"  power iter {k + 1:>3}: lambda = {lam:.5f}   (rel change {rel:.2e})")
        if rel < args.tol and k > 1:
            break
        lam_prev = lam
        w = hess.cg(hv, tol=args.cg_tol, maxit=args.cg_maxit)
        nrm = np.linalg.norm(w)
        if nrm == 0:
            print("  M^-1 H v vanished -- H_exact is singular along this direction; stopping")
            break
        v = w / nrm

    lam_max = float(max([x for x in history if np.isfinite(x)] + [max(probes.values(), default=0.0)]))
    gamma_max = 2.0 / lam_max if lam_max > 0 else np.inf
    gamma_rec = args.safety / lam_max if lam_max > 0 else np.inf

    print("\n" + "=" * 62)
    print(f"lambda_max(M^-1 H_exact) >= {lam_max:.4f}")
    print(f"  divergence threshold : gamma >= {gamma_max:.4f}")
    print(f"  recommended          : --gamma {gamma_rec:.3f}   (safety {args.safety})")
    if lam_max > 2.0:
        print("  WARNING: even gamma=1 diverges on this data")
    print("=" * 62)
    print("Lower bound only: power iteration converges from below, and a truncated")
    print("run underestimates. Treat the recommendation as an upper limit on gamma.")

    eig = eigvec_report(v)
    denom = (
        denominator_report(vlam, hvlam, denlam, hess, workers, geometry, nband, wsum_tot, args)
        if vlam is not None
        else {}
    )

    record = {
        "dt": dt_name,
        "nband": nband,
        "nx": nx,
        "ny": ny,
        "nx_psf": nx_psf,
        "psf_oversize": nx_psf / nx,
        "wsum_tot": wsum_tot,
        "eta": args.eta,
        "eta_mode": args.eta_mode,
        "eta_cap": args.eta_cap,
        "denominator": denom,
        "lambda_max_M": float(lam_m),
        "eta_over_lambda_max_M": args.eta / float(lam_m),
        "spectrum": spectrum,
        "eigvec": eig,
        "probe_rayleigh": probes,
        "power_history": history,
        "lambda_max": lam_max,
        "gamma_divergence_threshold": gamma_max,
        "gamma_recommended": gamma_rec,
    }
    out = f"{dt_name}_max_gamma.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwritten: {out}")

    if not args.no_fits:
        cell_deg = np.rad2deg(meta["cell_rad"])
        hdr = set_wcs(
            cell_deg,
            cell_deg,
            nx,
            ny,
            meta["radec"],
            float(np.mean(meta["freq_out"])),
            casambm=False,
            l0=meta["l0"],
            m0=meta["m0"],
        )
        name = f"{dt_name}_max_gamma_eigvec.fits"
        save_fits(np.mean(v, axis=0), name, hdr, yx_order=True)
        print(f"written: {name}")

    dt.close()


if __name__ == "__main__":
    main()
