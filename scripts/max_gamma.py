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
from pfb_imaging.utils.fits import save_fits, set_wcs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dt", help="Path to the imager output <output-filename>_<PRODUCT>.dt")
    p.add_argument("--eta", type=float, default=1e-3, help="Tikhonov term of the preconditioner (deconv --eta)")
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

    print(f"{nband} band(s), {ny}x{nx}, wsum_tot = {wsum_tot:.4e}")
    print(f"hess_norm lambda_max(M) is data-dependent; eta = {args.eta:.3e}")

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
    for k in range(args.niter):
        hv = h_exact(v)
        lam = float(np.vdot(v, hv).real) / float(np.vdot(v, hess.dot(v)).real)
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

    record = {
        "dt": dt_name,
        "nband": nband,
        "nx": nx,
        "ny": ny,
        "wsum_tot": wsum_tot,
        "eta": args.eta,
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
