#!/usr/bin/env python
"""Profile the deconv preconditioner with and without the GP frequency prior (issue #307).

Without the prior ``M`` is block diagonal over bands: a ``dot`` fans one image
out to each of the nband workers and gathers nband images back, and a *forward
solve* costs exactly ONE such round trip -- every worker runs its own CG to
convergence in process (``BandWorkerPool.hess_cg``).

The prior couples the bands through the driver-side remainder
``D^.5 (Cinv - I) D^.5`` (wiki D30), which costs three separate things:

1. the band-parallel CG fast path is no longer valid: ``HessTreeRay.cg`` falls
   back to a cube-level CG on the driver, so the round-trip count per forward
   solve goes from 1 to one per CG *iteration*;
2. every one of those ``dot``\\ s pays an ``(nband, nband) @ (nband, npix)``
   matmul plus two elementwise multiplies on the driver, O(nband^2 * npix); and
3. ``lambda_min(M)`` drops by up to ``gp_cap``, so CG needs more iterations to
   reach the same tolerance -- an effect on the *count*, not the cost, of the
   round trips, and the one a per-``dot`` benchmark cannot see.

The script separates them by timing three forward solves on the same rhs and the
same worker pool::

    A. hess_off.cg(rhs)            band-parallel in-worker CG (production, prior off)
    B. pcg_numba(hess_off.dot)     driver-side CG, no coupling term
    C. hess_on.cg(rhs)             driver-side CG + the coupling term (production, prior on)

``B - A`` is (1); ``C - B`` is (2) and (3) together and is split between them
using the measured iteration counts. B is not a code path anything runs; it
exists only to make that split possible. C is run as ``pcg_numba(counted.dot,
...)``, which is exactly what ``HessTreeRay.cg`` does on the coupled branch, so
that the ``dot`` calls can be counted and timed.

A per-``dot`` breakdown says where one round trip goes: pure FFT compute (timed
in a driver-local single-band ``HessianTree``), Ray dispatch latency (an
argument-free round trip to every worker), and the coupling term on its own.

**Read the driver's CPU column, not just its wall time.** This is how the
coupling term's original numpy form was caught: it measured 1.9 ms with the
driver otherwise idle but added 63 ms to a ``dot`` in situ, because BLAS spread
its matmul over every core and those threads then busy-polled for ~100 ms --
straight through the next ``ray.get``, while the workers needed the cores. The
term is now ``gauss.eta_freq_mul``, a fused numba kernel, and the forward solve
went 20.0 s -> 12.7 s on this tree. The check below stays as a regression guard:
a cheap-looking kernel that leaves threads spinning is more expensive than a
slower one that does not, and only the CPU column shows it.

The image size and the band count both matter and only one of them can be
varied from a given ``.dt``, so a synthetic sweep of the driver-side coupling
term over nband is reported alongside (``--coupling-nbands``).

Run (from the repo root)::

    uv run python scripts/profile_freq_correlated_hessian.py /path/to/out_I.dt \
        --gp-length-scale 0.5 --nthreads 8

Writes ``<dt>_freq_prior_profile_gp<ls>_cap<cap>.json``.
"""

import argparse
import json
import os
from time import process_time, time

# Ray's uv_run_runtime_env hook (on under `uv run`) relaunches workers via
# `uv run --frozen python`, rebuilding a venv that lacks the [full] extra where
# ray itself lives -- workers then die on `import ray` and ray.wait() blocks
# forever. Same workaround, and same reason, as tests/conftest.py. The constant
# is read when ray is imported, so this must precede the pfb_imaging imports.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

# Captured before numpy is imported and before set_envs overwrites them. BLAS
# sizes its thread pool when it loads, so these -- not what set_envs writes
# later -- are the values that actually apply to the driver's coupling matmul.
BLAS_ENV = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import xarray as xr  # noqa: E402
from ducc0.misc import resize_thread_pool  # noqa: E402

from pfb_imaging import init_ray, set_envs, setup_ray_worker  # noqa: E402
from pfb_imaging.deconv.presets import _build_hess  # noqa: E402
from pfb_imaging.operators.band_worker import BandWorkerPool  # noqa: E402
from pfb_imaging.operators.gauss import eta_freq_mul  # noqa: E402
from pfb_imaging.operators.hessian import ETA_MODES, HessianTree, freq_precision  # noqa: E402
from pfb_imaging.opt.pcg import pcg_numba  # noqa: E402


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
    p.add_argument(
        "--gp-length-scale",
        type=float,
        default=0.5,
        help="GP frequency prior length scale as a fraction of the band span (deconv --gp-length-scale)",
    )
    p.add_argument("--gp-cap", type=float, default=10.0, help="Max relaxation of the smoothest mode (deconv --gp-cap)")
    p.add_argument("--nthreads", type=int, default=4, help="Threads per band worker")
    p.add_argument("--nworkers", type=int, default=None, help="Ray workers (default: nband)")
    p.add_argument("--ray-address", default="local")
    p.add_argument("--cg-tol", type=float, default=1e-3, help="CG tolerance (deconv --cg-tol)")
    p.add_argument("--cg-maxit", type=int, default=150, help="CG iteration cap (deconv --cg-maxit)")
    p.add_argument("--cg-minit", type=int, default=1, help="CG iteration floor (deconv --cg-minit)")
    p.add_argument(
        "--cg-verbose",
        type=int,
        default=1,
        help="CG reporting. At 1 each solve prints its outcome, which is the only way to see the per-band "
        "iteration counts of the in-worker path (the workers print them; Ray forwards them with a pid prefix)",
    )
    p.add_argument("--repeat", type=int, default=5, help="Timed repeats of each dot measurement")
    p.add_argument("--warmup", type=int, default=2, help="Untimed warmup calls before each dot measurement")
    p.add_argument(
        "--no-local-probe",
        action="store_true",
        help="Skip the driver-local single-band HessianTree. That probe is what separates FFT compute from Ray "
        "overhead, but it holds one band's PSFHAT and BEAM in the driver -- skip it if that does not fit.",
    )
    p.add_argument("--probe-band", type=int, default=0, help="Band index for the driver-local probe")
    p.add_argument(
        "--coupling-nbands",
        default="2,4,8,16",
        help="Synthetic sweep of the driver-side coupling term over band count at this image size. "
        "Empty string disables it.",
    )
    p.add_argument(
        "--coupling-max-gb",
        type=float,
        default=2.0,
        help="Skip synthetic band counts whose driver buffers would exceed this",
    )
    p.add_argument("--no-solve", action="store_true", help="Skip the three forward solves (dot breakdown only)")
    return p.parse_args()


def fmt(t):
    """Wall time as ms below a second, else seconds."""
    return f"{1e3 * t:8.2f} ms" if t < 1.0 else f"{t:8.3f} s "


def timeit(fn, repeat, warmup):
    """Median/min/max wall time of ``fn`` over ``repeat`` calls after ``warmup`` untimed ones.

    Also records driver CPU time (``process_time`` sums every thread in the
    process), because the driver competes with the band workers for cores:
    ``cpu/wall`` is how many cores the driver kept busy for the call, and a
    number well above 1 on a step that looks cheap in wall time is the signature
    of a multithreaded BLAS call stealing cores from the workers.
    """
    for _ in range(warmup):
        fn()
    ts, cs = [], []
    for _ in range(repeat):
        t0, c0 = time(), process_time()
        fn()
        ts.append(time() - t0)
        cs.append(process_time() - c0)
    wall = float(np.median(ts))
    cpu = float(np.median(cs))
    return {
        "median": wall,
        "min": float(np.min(ts)),
        "max": float(np.max(ts)),
        "cpu": cpu,
        "cores_busy": cpu / wall if wall > 0 else np.nan,
        "n": repeat,
    }


def load_tree(dt_name):
    """Band nodes, geometry, per-band wsums and a realistic rhs, mirroring core/deconv.py.

    The rhs is the beam-attenuated gradient the forward solver actually consumes
    (D23): ``BRESIDUAL`` when a previous deconv run left one, else ``BDIRTY``
    (the major-cycle-0 value), both normalised by the total wsum as the driver
    does. CG's stopping test is relative, so the scale only matters for realism.
    """
    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    names = [n for n in dt.children if n.startswith("band")]
    if not names:
        raise ValueError(f"No band nodes found in {dt_name}")
    nodes = sorted(names, key=lambda n: int(dt[n].ds.attrs["bandid"]))
    first = dt[nodes[0]].ds
    if first.corr.size > 1:
        raise NotImplementedError("Joint polarisation not supported (matches pfb deconv)")
    part = next(iter(dt[nodes[0]].children.values()), None)
    if part is None or "PSFHAT" not in part.ds:
        raise ValueError(f"{dt_name} has no per-partition PSFHAT -- re-run pfb imager with --psf")
    wsums = np.array([float(dt[n].ds.WSUM.values[0]) for n in nodes])
    geometry = {
        "nx": first.x.size,
        "ny": first.y.size,
        "nx_psf": first.x_psf.size,
        "ny_psf": first.y_psf.size,
        "freq_out": np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes]),
    }
    key = "BRESIDUAL" if "BRESIDUAL" in first else "BDIRTY"
    if key not in first:
        raise ValueError(f"{dt_name} has neither BRESIDUAL nor BDIRTY -- re-run pfb imager to regenerate the .dt")
    rhs = np.stack([dt[n].ds[key].values[0] for n in nodes]) / wsums.sum()
    dt.close()
    return nodes, geometry, wsums, rhs, key


def local_band_tree(dt_name, node, geometry, wsum_tot, args):
    """Driver-local copy of one band's HessianTree: pure FFT compute, no Ray.

    Mirrors the Hessian inputs of ``_BandWorkerImpl.load_band``, including the
    ``abs`` on the stored complex ``PSFHAT`` (the ``HessianTree`` convention).
    This is the memory-expensive part of the script -- it holds one band's
    ``PSFHAT`` and ``BEAM`` in the driver process.

    Returns:
        ``(tree, nbytes)``: the operator and the bytes it pinned in the driver.
    """
    band = xr.open_datatree(dt_name, engine="zarr", chunks=None)[node]
    parts, nbytes = [], 0
    for cname in sorted(band.children):
        child = band[cname].ds
        psfhat = np.abs(child.PSFHAT.values)
        beam = child.BEAM.values
        nbytes += psfhat.nbytes + beam.nbytes
        parts.append({"psfhat": psfhat, "beam": beam, "wsum": np.asarray(child.attrs["wsum"])})
    tree = HessianTree(
        parts,
        geometry["nx"],
        geometry["ny"],
        geometry["nx_psf"],
        geometry["ny_psf"],
        eta=args.eta,
        nthreads=args.nthreads,
        wsum=wsum_tot,
        eta_mode=args.eta_mode,
        eta_cap=args.eta_cap,
    )
    tree.dot(np.zeros((geometry["ny"], geometry["nx"])))  # warm the FFT plans, as init_hess does
    return tree, nbytes


class Counted:
    """``dot`` facade that counts and times the calls a driver-side CG makes."""

    def __init__(self, hess):
        self._hess = hess
        self.ncalls = 0
        self.time = 0.0

    def dot(self, x):
        t0 = time()
        out = self._hess.dot(x)
        self.time += time() - t0
        self.ncalls += 1
        return out


def coupling_apply(hess):
    """The driver-side prior remainder on its own, as ``HessTreeRay.dot`` applies it.

    Uses the operator's own matrix, profile and chunk count rather than
    rebuilding them, so this times the shipped kernel and not a lookalike. The
    accumulator is scratch owned by this script -- the operator no longer holds
    any (that was the point of the fused kernel).
    """
    dcinv, s, nchunk = hess._dC, hess._s, hess._nchunk
    acc = np.zeros((hess.nband, hess.ny, hess.nx))

    def apply(x):
        acc.fill(0.0)
        return eta_freq_mul(acc, dcinv, s, x, nchunk=nchunk)

    return apply


def coupling_scaling(nbands, ny, nx, eta, max_gb, repeat, warmup):
    """Driver-side coupling cost vs band count at fixed image size.

    The term is O(nband^2 * npix) in flops and O(nband * npix) in memory, and a
    single ``.dt`` only exercises one band count -- this sweeps it synthetically
    on a representative dense precision matrix (a squared-exponential over
    evenly spaced frequencies, which is what a real band plan gives).
    """
    rows = []
    rng = np.random.default_rng(0)
    for nb in nbands:
        gb = 3 * nb * ny * nx * 8 / 2**30  # x plus the two driver buffers
        if gb > max_gb:
            print(f"  nband {nb:>3}: skipped, would need {gb:.2f} GB of driver buffers (--coupling-max-gb {max_gb})")
            continue
        prec = freq_precision(np.linspace(1.0e9, 1.4e9, nb), 0.5, 10.0)
        dcinv = prec - np.eye(nb)
        s = np.full((nb, 1, 1), np.sqrt(eta))
        x = rng.standard_normal((nb, ny, nx))
        buf, buf2 = np.empty_like(x), np.empty_like(x)

        def apply(dcinv=dcinv, s=s, x=x, buf=buf, buf2=buf2, nb=nb):
            np.multiply(x, s, out=buf)
            np.matmul(dcinv, buf.reshape(nb, -1), out=buf2.reshape(nb, -1))
            np.multiply(buf2, s, out=buf2)

        t = timeit(apply, repeat, warmup)
        per_band = t["median"] / nb
        print(f"  nband {nb:>3}: {fmt(t['median'])}   {1e3 * per_band:6.2f} ms per band   ({gb:.2f} GB of buffers)")
        rows.append({"nband": nb, "seconds": t["median"], "seconds_per_band": per_band, "buffer_gb": gb})
        del x, buf, buf2
    return rows


def main():
    args = parse_args()
    dt_name = args.dt.rstrip("/")
    nodes, geometry, wsums, rhs, rhs_key = load_tree(dt_name)
    nband = len(nodes)
    ny, nx = geometry["ny"], geometry["nx"]
    wsum_tot = float(wsums.sum())
    cube_gb = nband * ny * nx * 8 / 2**30

    if nband < 2:
        raise SystemExit(
            f"{dt_name} has {nband} band: the frequency prior needs at least 2 (freq_precision returns None)"
        )

    resize_thread_pool(args.nthreads)
    ncpu = int(np.minimum(args.nthreads, psutil.cpu_count(logical=False)))
    env_vars = set_envs(args.nthreads, ncpu)
    env_vars["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"  # see the module-level note
    init_ray(
        args.nworkers or nband,
        ray_address=args.ray_address,
        runtime_env={"env_vars": env_vars, "worker_process_setup_hook": setup_ray_worker},
    )
    workers = BandWorkerPool(nband, args.nthreads)
    workers.load_bands(dt_name, nodes)

    # both facades share the pool: the worker-side operator is identical (same
    # eta, same eta_mode), so only the driver-side coupling differs and building
    # one does not disturb the other
    opts = {
        "eta": args.eta,
        "eta_mode": args.eta_mode,
        "eta_cap": args.eta_cap,
        "gp_length_scale": None,
        "gp_cap": args.gp_cap,
        "nthreads": args.nthreads,
        "cg_tol": args.cg_tol,
        "cg_maxit": args.cg_maxit,
        "cg_verbose": args.cg_verbose,
    }
    hess_off = _build_hess(None, geometry, opts, workers=workers, wsums=wsums)
    hess_on = _build_hess(
        None, geometry, dict(opts, gp_length_scale=args.gp_length_scale), workers=workers, wsums=wsums
    )
    prior = hess_on.get_freq_prior_stats()
    if prior is None:
        raise SystemExit(
            f"the prior is inactive on this tree (freq_out = {geometry['freq_out']}): "
            "it needs at least two distinct band frequencies"
        )

    emode = "uniform" if args.eta_mode is None else f"{args.eta_mode} (cap {args.eta_cap:g})"
    print(f"\n{dt_name}")
    print(f"  {nband} bands, {ny}x{nx}, psf {geometry['ny_psf']}x{geometry['nx_psf']}, {args.nthreads} threads/worker")
    print(f"  eta = {args.eta:.3e} [{emode}], rhs = {rhs_key}/wsum_tot, cube = {cube_gb * 1e3:.1f} MB")
    print(
        f"  prior: length_scale={args.gp_length_scale:g} cap={args.gp_cap:g}, "
        f"precision spectrum [{prior['prec_min']:.4f}, {prior['prec_max']:.4f}]"
    )
    print(
        f"  cores: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical; "
        f"the workers alone claim {nband} x {args.nthreads} = {nband * args.nthreads} FFT threads"
    )
    print("  driver BLAS env at import: " + " ".join(f"{k}={v}" for k, v in BLAS_ENV.items()))

    # --- one Hessian application -------------------------------------------
    print("\n" + "=" * 78)
    print(f"ONE HESSIAN APPLICATION (median of {args.repeat} after {args.warmup} warmup)")
    print("=" * 78)
    x = rhs.copy()
    dots = {}
    # dispatch floor: an argument-free round trip to every worker. get_eta
    # returns a scalar for uniform eta and this band's (ncorr, ny, nx) profile
    # under --eta-mode, so with a profile it measures dispatch plus a one-way
    # cube transfer rather than dispatch alone.
    dots["ray_dispatch_floor"] = timeit(lambda: workers._map("get_eta", [()] * nband), args.repeat, args.warmup)
    dots["pool_hess_dot"] = timeit(lambda: workers.hess_dot(x), args.repeat, args.warmup)
    dots["dot_prior_off"] = timeit(lambda: hess_off.dot(x), args.repeat, args.warmup)
    dots["dot_prior_on"] = timeit(lambda: hess_on.dot(x), args.repeat, args.warmup)
    dots["coupling_term"] = timeit(lambda: coupling_apply(hess_on)(x), args.repeat, args.warmup)

    local_bytes = 0
    if not args.no_local_probe:
        tree, local_bytes = local_band_tree(dt_name, nodes[args.probe_band], geometry, wsum_tot, args)
        xb = rhs[args.probe_band]
        dots["one_band_fft_local"] = timeit(lambda: tree.dot(xb), args.repeat, args.warmup)

    print(f"  {'':<22} {'wall':>11}  {'driver cpu':>11}  cores busy on the driver")
    for k in (
        "ray_dispatch_floor",
        "one_band_fft_local",
        "pool_hess_dot",
        "coupling_term",
        "dot_prior_off",
        "dot_prior_on",
    ):
        if k in dots:
            d = dots[k]
            print(f"  {k:<22} {fmt(d['median'])}  {fmt(d['cpu'])}  {d['cores_busy']:5.1f}")
    overhead = dots["pool_hess_dot"]["median"] - dots.get("one_band_fft_local", {}).get("median", np.nan)
    overhead_str = "?" if np.isnan(overhead) else fmt(overhead)
    if not np.isnan(overhead):
        print(
            f"\n  Ray overhead per dot   {fmt(overhead)}  "
            f"({100 * overhead / dots['pool_hess_dot']['median']:.0f}% of the fan-out; "
            f"{fmt(dots['ray_dispatch_floor']['median'])} of it is bare dispatch latency)"
        )
        print("    = pool fan-out minus the same FFT work done locally on one band;")
        print("      the bands run concurrently, so one band's compute is the ideal parallel wall time.")
    extra = dots["dot_prior_on"]["median"] - dots["dot_prior_off"]["median"]
    ratio_dot = dots["dot_prior_on"]["median"] / dots["dot_prior_off"]["median"]
    coupling = dots["coupling_term"]["median"]
    print(f"\n  dot with the prior costs {ratio_dot:.2f}x one without it (+{fmt(extra)})")
    print(f"    the coupling arithmetic alone, driver otherwise idle:  {fmt(coupling)}")
    print(f"    the same term measured in situ (prior on minus off):   {fmt(extra)}")
    # regression guard: a kernel that is cheap in isolation but expensive in
    # situ is leaving threads spinning into the next ray.get. Thresholds are set
    # above the few ms of run-to-run noise on the fan-out measurement.
    if extra > 5 * coupling and extra > 0.2 * dots["dot_prior_off"]["median"]:
        print(
            f"\n  ^ the gap is CONTENTION, not arithmetic: the coupling kernel runs on\n"
            f"    ~{dots['coupling_term']['cores_busy']:.0f} driver threads which are still spinning during the next\n"
            f"    ray.get, when the {nband} x {args.nthreads} worker FFT threads need the cores. This is what the\n"
            f"    numpy matmul used to do (~100 ms of OpenBLAS THREAD_TIMEOUT spin per 3 ms call).\n"
            f"    Check what the coupling term dispatches to; numba's TBB pool does not do this."
        )
    print(f"\n  per dot the driver ships {2e3 * cube_gb:.1f} MB through the object store (out and back)")

    solves = {}
    if not args.no_solve:
        # --- the forward solve ---------------------------------------------
        print("\n" + "=" * 78)
        print("ONE FORWARD SOLVE")
        print("=" * 78)
        cg_kw = dict(tol=args.cg_tol, maxit=args.cg_maxit, minit=args.cg_minit)
        print(f"  (cg_tol {args.cg_tol:g}, cg_maxit {args.cg_maxit}; at cg-verbose > 0 each CG reports its outcome)")

        t0 = time()
        xa = hess_off.cg(rhs.copy(), **cg_kw)
        ta = time() - t0
        print(f"  A  prior off, in-worker band-parallel CG  {fmt(ta)}    1 round trip")

        counted_off = Counted(hess_off)
        t0 = time()
        xb_sol = pcg_numba(counted_off.dot, rhs.copy(), verbosity=args.cg_verbose, **cg_kw)
        tb = time() - t0
        dot_b = counted_off.time / max(counted_off.ncalls, 1)
        print(
            f"  B  prior off, driver-side CG               {fmt(tb)}  {counted_off.ncalls:>3} round trips"
            f"   ({fmt(dot_b)}/dot)"
        )

        counted_on = Counted(hess_on)
        t0 = time()
        xc = pcg_numba(counted_on.dot, rhs.copy(), verbosity=args.cg_verbose, **cg_kw)
        tc = time() - t0
        dot_c = counted_on.time / max(counted_on.ncalls, 1)
        capped = counted_on.ncalls > args.cg_maxit
        print(
            f"  C  prior on,  driver-side CG               {fmt(tc)}  {counted_on.ncalls:>3} round trips"
            f"   ({fmt(dot_c)}/dot){'   [hit cg-maxit]' if capped else ''}"
        )
        if capped:
            print("     C did not converge inside cg-maxit, so its cost is a floor, not a like-for-like solve.")

        # A and B solve the same system by different routes, so they must agree
        # to the CG tolerance; C is a different operator and is expected not to
        d_ab = float(np.linalg.norm(xb_sol - xa) / np.linalg.norm(xa))
        d_ac = float(np.linalg.norm(xc - xa) / np.linalg.norm(xa))
        print(
            f"\n  ||B-A||/||A|| = {d_ab:.2e} (same system, must match to tol)   "
            f"||C-A||/||A|| = {d_ac:.2e} (the prior's effect on the step)"
        )

        # exact split of C - B into "more iterations" and "costlier iterations",
        # both measured, plus whatever is left in the driver's own CG algebra
        t_iters = (counted_on.ncalls - counted_off.ncalls) * dot_b
        t_perdot = counted_on.ncalls * (dot_c - dot_b)
        t_algebra = (tc - tb) - t_iters - t_perdot
        arith = counted_on.ncalls * dots["coupling_term"]["median"]
        print("\n  where the extra time goes:")
        print(f"    A -> B  the band-parallel CG fast path is lost   +{fmt(tb - ta)}  ({tb / ta:5.2f}x)")
        print(f"            {counted_off.ncalls} round trips instead of 1, at ~{overhead_str} of Ray overhead each")
        print(f"    B -> C  the prior's two costs                    +{fmt(tc - tb)}  ({tc / tb:5.2f}x)")
        print(
            f"      conditioning: {counted_off.ncalls} -> {counted_on.ncalls} CG iterations  "
            f"+{fmt(t_iters)}   (lambda_min(M) drops by up to gp_cap)"
        )
        print(f"      costlier dots: {fmt(dot_b)} -> {fmt(dot_c)} each     +{fmt(t_perdot)}")
        print(f"        of which real coupling arithmetic          ~{fmt(arith)}   (the rest is BLAS contention)")
        print(
            f"      driver CG vector algebra                     {'+' if t_algebra >= 0 else '-'}{fmt(abs(t_algebra))}"
        )
        print(f"\n  FORWARD SOLVE: {fmt(ta)} -> {fmt(tc)}   ({tc / ta:.2f}x)")
        print(f"  object store traffic: {2e3 * cube_gb:.1f} MB -> {2e3 * cube_gb * counted_on.ncalls:.1f} MB")
        solves = {
            "A_prior_off_in_worker": {"seconds": ta, "round_trips": 1},
            "B_prior_off_driver_cg": {
                "seconds": tb,
                "round_trips": counted_off.ncalls,
                "dot_seconds": counted_off.time,
                "seconds_per_dot": dot_b,
            },
            "C_prior_on_driver_cg": {
                "seconds": tc,
                "round_trips": counted_on.ncalls,
                "dot_seconds": counted_on.time,
                "seconds_per_dot": dot_c,
                "hit_cg_maxit": capped,
            },
            "cost_of_losing_fast_path": tb - ta,
            "cost_of_prior": tc - tb,
            "cost_of_extra_iterations": t_iters,
            "cost_of_costlier_dots": t_perdot,
            "coupling_arithmetic_est": arith,
            "cost_of_driver_algebra": t_algebra,
            "total_ratio": tc / ta,
            "rel_diff_B_A": d_ab,
            "rel_diff_C_A": d_ac,
        }

    scaling = []
    if args.coupling_nbands:
        print("\n" + "=" * 78)
        print(f"DRIVER-SIDE COUPLING TERM vs BAND COUNT (synthetic, at {ny}x{nx})")
        print("=" * 78)
        nbs = [int(v) for v in args.coupling_nbands.split(",") if v.strip()]
        scaling = coupling_scaling(nbs, ny, nx, args.eta, args.coupling_max_gb, args.repeat, args.warmup)
        print("  O(nband^2 * npix) in flops: the per-band figure is what grows.")

    record = {
        "dt": dt_name,
        "nband": nband,
        "nx": nx,
        "ny": ny,
        "nx_psf": geometry["nx_psf"],
        "nthreads_per_worker": args.nthreads,
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "blas_env_at_import": BLAS_ENV,
        "eta": args.eta,
        "eta_mode": args.eta_mode,
        "eta_cap": args.eta_cap,
        "gp_length_scale": args.gp_length_scale,
        "gp_cap": args.gp_cap,
        "freq_prior": prior,
        "rhs": rhs_key,
        "cube_gb": cube_gb,
        "local_probe_bytes": local_bytes,
        "cg": {"tol": args.cg_tol, "maxit": args.cg_maxit, "minit": args.cg_minit},
        "dot": dots,
        "solve": solves,
        "coupling_scaling": scaling,
    }
    out = f"{dt_name}_freq_prior_profile_gp{args.gp_length_scale:g}_cap{args.gp_cap:g}.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
