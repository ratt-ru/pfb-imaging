"""How much does a time-averaged beam bias the deconv gradient? (#401, step 0)

Deconv's gradient applies one beam per partition, the weighted mean of its scans'
beams (``core/imager._concat_pieces``, wiki D28). This script reproduces the
``wsum_nat``-weighted mean that was measured (docs/experiments/beam_time_gradient.md);
since D65 ``_concat_pieces`` weights the scans by their imaging wsum instead. The
exact gradient applies each scan's own beam:

    g_exact = sum_s B_s G_s^T W_s (V_s - G_s(B_s x))
    g_avg   = sum_s Bbar G_s^T W_s (V_s - G_s(Bbar x))

This script computes both from a kept imager ``.scratch`` (one piece per scan
per baseline group, each with its own beam) and the run's ``.dt``
(``MODEL`` from ``--transfer-model-from``), one band per call. It re-applies
pass 2's robust weights itself, so the scratch must come from a
``--concat-row`` run with ``--weight-grouping per-band`` (the default
``concat_row`` coercion). Everything is reported wsum-normalised, the units of
the deconv gradient.

It also measures the beams themselves on a coarse field-centred grid, with no
gridding: the spread of per-scan beams about their mean (time) against the
change across the band's channel extent (frequency, issue #401 Q1).

Usage (one band per process; run bands in parallel):

    python beam_time_gradient.py gradient --scratch X.scratch --dt X.dt --band 1 --out out/ --nthreads 16
    python beam_time_gradient.py beams --scratch X.scratch --dt X.dt --band 1 --out out/ --nthreads 16

Writes ``<out>/band####_{gradient,beams}.json`` and, for ``gradient``, FITS of
``g_avg`` and ``g_exact - g_avg`` (and its data- and model-term parts).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from pfb_imaging import set_envs
from pfb_imaging.core.imager import _load_piece
from pfb_imaging.operators.gridder import grid_partition, residual_from_partitions
from pfb_imaging.utils.fits import save_fits, set_wcs
from pfb_imaging.utils.stokes2vis import partition_beam


def rms_mad(x):
    """Robust rms (1.4826 * MAD), insensitive to sources."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def band_pieces(scratch, bandid):
    """The band's scratch piece paths, grouped by baseline group, in scratch order."""
    groups = {}
    for name in sorted(scratch.children):
        if not name.startswith(f"band{bandid:04d}_"):
            continue
        for cname, child in scratch[name].children.items():
            groups.setdefault(child.ds.attrs["baseline_group"], []).append(f"{name}/{cname}")
    return groups


def counts_path(scratch, bandid):
    """``per-band`` grouping: groups are numbered in sorted (bandid,) order (write_group_counts)."""
    bands = sorted({int(scratch[n].ds.attrs.get("bandid", n[4:8])) for n in scratch.children if n.startswith("band")})
    return f"counts/group{bands.index(bandid):04d}"


def save(arr, path, meta):
    hdr = set_wcs(
        np.rad2deg(meta["cell_rad"]),
        np.rad2deg(meta["cell_rad"]),
        meta["nx"],
        meta["ny"],
        (meta["ra"], meta["dec"]),
        np.atleast_1d(meta["freq"]),
        unit="Jy/beam",
    )
    save_fits(arr.astype(np.float32), str(path), hdr, yx_order=True)


def gradient(args):
    t0 = time.time()
    scratch = xr.open_datatree(args.scratch, engine="zarr", chunks=None, cache=False)
    dt = xr.open_datatree(args.dt, engine="zarr", chunks=None, cache=False)
    root = dt.attrs
    nx, ny, cell_rad = int(root["nx"]), int(root["ny"]), float(root["cell_rad"])
    bnode = next(n for n in dt.children if int(dt[n].ds.attrs["bandid"]) == args.band)
    bds = dt[bnode].ds
    if "MODEL" in bds:
        model = bds.MODEL.values
    else:
        # smoke-test fallback (no --transfer-model-from): one point at the dirty peak
        model = np.zeros(bds.DIRTY.shape, dtype=bds.DIRTY.dtype)
        model[(0, *np.unravel_index(np.argmax(bds.DIRTY.values[0]), (ny, nx)))] = 1.0
    robustness = bds.attrs.get("robustness")
    counts = scratch[counts_path(scratch, args.band)].ds.COUNTS.values if robustness is not None else None
    nx_pad, ny_pad = (counts.shape[1], counts.shape[2]) if counts is not None else (None, None)
    groups = band_pieces(scratch, args.band)
    grid = np.array([bds.attrs["ra"], bds.attrs["dec"]])
    # the offset pass 2 placed and gridded at (--target); a moving target is not supported here
    pattrs = dt[bnode][sorted(dt[bnode].children)[0]].ds.attrs
    l0, m0 = float(pattrs.get("l0", 0.0)), float(pattrs.get("m0", 0.0))

    def placed(ds):
        """A piece's image-grid B/n: stored by older scratches, else placed as pass 2 does."""
        if "BEAM" in ds:
            return ds.BEAM.values
        return partition_beam(ds, grid, cell_rad, nx, ny, l0=l0, m0=m0, dtype=model.dtype)

    g_exact = np.zeros_like(model)
    g_avg = np.zeros_like(model)
    dg_data = np.zeros_like(model)
    dirty_tot = np.zeros_like(model)
    wsum = np.zeros(model.shape[0])
    per_group = {}
    for bg, paths in groups.items():
        # the partition's mean beam, as _concat_pieces formed it before D65
        w = np.array([float(scratch[p].ds.attrs["wsum_nat"]) for p in paths])
        bbar = np.zeros_like(model)
        for wi, p in zip(w, paths):
            bbar += wi * placed(scratch[p].ds)
        bbar /= w.sum()
        gx = np.zeros_like(model)
        ga = np.zeros_like(model)
        spread = []
        for p in paths:
            piece = _load_piece(scratch[p])
            if "BEAM" not in piece:
                piece = piece.drop_vars(["BEAM_SMALL", "l_beam", "m_beam"], errors="ignore").assign(
                    BEAM=(("corr", "y", "x"), placed(piece))
                )
            prod = grid_partition(
                piece,
                counts,
                nx,
                ny,
                None,
                None,
                cell_rad,
                robustness=robustness,
                nx_pad=nx_pad,
                ny_pad=ny_pad,
                l0=l0,
                m0=m0,
                nthreads=args.nthreads,
                epsilon=args.epsilon,
                do_psf=False,
                overwrite_weight=True,
            )
            dirty, beam = prod["DIRTY"], prod["BEAM"]
            part = piece.assign(WEIGHT=(("corr", "row", "chan"), prod["WEIGHT"]))
            kw = {"cell_rad": cell_rad, "nthreads": args.nthreads, "epsilon": args.epsilon}
            _, bx = residual_from_partitions(dirty, [part], model, bdirty=beam * dirty, **kw)
            part_avg = part.assign(BEAM=(("corr", "y", "x"), bbar))
            _, ba = residual_from_partitions(dirty, [part_avg], model, bdirty=bbar * dirty, **kw)
            gx += bx
            ga += ba
            dg_data += (beam - bbar) * dirty
            dirty_tot += dirty
            wsum += prod["WSUM"]
            inner = bbar[0] > 0.1
            spread.append(float(np.abs(beam[0] - bbar[0])[inner].max()))
            print(f"{bg} {piece.attrs['scan_name']}: {time.time() - t0:.0f} s", flush=True)
            del piece, part, part_avg, prod, dirty, beam
        g_exact += gx
        g_avg += ga
        per_group[bg] = {
            "npieces": len(paths),
            "max_abs_beam_minus_mean_where_mean_gt_0.1": spread,
            "rel_norm_dg": float(np.linalg.norm(gx - ga) / np.linalg.norm(ga)),
        }

    gn = g_avg / wsum[:, None, None]
    dgn = (g_exact - g_avg) / wsum[:, None, None]
    dgd = dg_data / wsum[:, None, None]
    rms = rms_mad(gn[0])
    imax = np.unravel_index(np.argmax(np.abs(dgn[0])), dgn[0].shape)
    out = {
        "band": args.band,
        "freq_out": float(bds.attrs["freq_out"]),
        "nx": nx,
        "ny": ny,
        "groups": per_group,
        # consistency with what pass 2 stored (gridding is linear in rows)
        "check_dirty_vs_dt": float(np.abs(dirty_tot - bds.DIRTY.values).max() / np.abs(bds.DIRTY.values).max()),
        "check_bres_vs_dt": float(np.abs(g_avg - bds.BRESIDUAL.values).max() / np.abs(bds.BRESIDUAL.values).max())
        if "BRESIDUAL" in bds
        else None,
        "rms_g_avg_mad": rms,
        "max_abs_g_avg": float(np.abs(gn).max()),
        "rel_norm_dg": float(np.linalg.norm(dgn) / np.linalg.norm(gn)),
        "max_abs_dg": float(np.abs(dgn).max()),
        "max_abs_dg_over_rms": float(np.abs(dgn).max() / rms),
        "rms_dg": float(np.sqrt(np.mean(dgn**2))),
        "argmax_dg_yx": [int(i) for i in imax],
        "max_abs_dg_data_over_rms": float(np.abs(dgd).max() / rms),
        "max_abs_dg_model_over_rms": float(np.abs(dgn - dgd).max() / rms),
        "seconds": time.time() - t0,
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"band{args.band:04d}_gradient.json").write_text(json.dumps(out, indent=2))
    meta = {"cell_rad": cell_rad, "nx": nx, "ny": ny, "ra": bds.attrs["ra"], "dec": bds.attrs["dec"]}
    meta["freq"] = bds.attrs["freq_out"]
    for name, arr in (("g_avg", gn), ("dg", dgn), ("dg_data", dgd), ("dg_model", dgn - dgd)):
        save(arr, outdir / f"band{args.band:04d}_{name}.fits", meta)
    print(json.dumps(out, indent=2))


def beams(args):
    # deferred: only this subcommand needs the beam stack
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from meerkat_beams.utils import BeamWizard

    from pfb_imaging.utils.beam import BEAM_FREQ_EXTRAPOLATION, BEAM_PA_TOLERANCE_DEG
    from pfb_imaging.utils.misc import to_mjd_time

    scratch = xr.open_datatree(args.scratch, engine="zarr", chunks=None, cache=False)
    # band frequencies from the pieces (the .dt need not exist yet): the extreme
    # channel centres, and the wsum_nat-weighted mean as _concat_pieces formed it before D65
    pattrs = [scratch[p].ds.attrs for ps in band_pieces(scratch, args.band).values() for p in ps]
    flo = min(float(a["freq_min"]) for a in pattrs)
    fhi = max(float(a["freq_max"]) for a in pattrs)
    wn = np.array([float(a["wsum_nat"]) for a in pattrs])
    battrs = {"freq_out": float(np.dot(wn, [float(a["freq_out"]) for a in pattrs]) / wn.sum())}
    if args.bandwidth_mhz is not None:
        # a narrower band centred on this one's frequency, to size a production band
        flo = battrs["freq_out"] - 0.5e6 * args.bandwidth_mhz
        fhi = battrs["freq_out"] + 0.5e6 * args.bandwidth_mhz
    out = {"band": args.band, "freq_out": float(battrs["freq_out"]), "freq_bounds": [flo, fhi], "groups": {}}
    for bg, paths in band_pieces(scratch, args.band).items():
        attrs = [scratch[p].ds.attrs for p in paths]
        wiz = BeamWizard(band=args.beam_band, group=bg, freq_extrapolation=BEAM_FREQ_EXTRAPOLATION)
        a0 = attrs[0]
        wiz.set_pointing_centre(SkyCoord(ra=a0["ra0"] * u.rad, dec=a0["dec0"] * u.rad))
        half = args.half_width_deg
        lm = np.linspace(-half, half, args.npix)

        def ev(times, freq):
            t = Time(to_mjd_time(np.atleast_1d(times)) / 86400.0, format="mjd")
            b, _ = wiz.get_rotation_averaged_beam(
                l=lm,
                m=lm,
                times=t,
                freq=np.atleast_1d(freq),
                time_stepping=1,
                pixel_stepping=1,
                var="nstokes",
                i="I",
                j="I",
                verbose=0,
                nthreads=args.nthreads,
                pa_tolerance=BEAM_PA_TOLERANCE_DEG,
            )
            return np.real(np.squeeze(b))

        # one beam per scan at its own time, at the band frequency
        w = np.array([float(a["wsum_nat"]) for a in attrs])
        tscan = np.array([float(a["time_out"]) for a in attrs])
        fband = out["freq_out"]
        per_scan = np.stack([ev([t], fband) for t in tscan])
        bbar = np.tensordot(w / w.sum(), per_scan, axes=1)
        # the whole track at the band's channel-extent edges
        b_lo, b_hi, b_mid = ev(tscan, flo), ev(tscan, fhi), ev(tscan, fband)
        res = {}
        for lo, hi, label in ((0.5, 2.0, "main_lobe"), (0.1, 0.5, "shoulder"), (-1.0, 0.1, "sidelobes")):
            sel = (bbar > lo) & (bbar <= hi)
            dt_ = np.abs(per_scan - bbar)[:, sel]
            df = np.abs(b_hi - b_lo)[sel]
            res[label] = {
                "npix": int(sel.sum()),
                "time_max": float(dt_.max()),
                "time_rms": float(np.sqrt(np.mean(dt_**2))),
                "freq_edge_to_edge_max": float(df.max()),
                "freq_edge_to_edge_rms": float(np.sqrt(np.mean(df**2))),
                "freq_flat_vs_chord_mid_max": float(np.abs(b_mid - 0.5 * (b_lo + b_hi))[sel].max()),
            }
        out["groups"][bg] = res
        print(bg, json.dumps(res), flush=True)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.bandwidth_mhz is None else f"_{args.bandwidth_mhz:g}MHz"
    (outdir / f"band{args.band:04d}_beams{tag}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gradient", "beams"])
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--dt", help="the run's .dt (gradient mode)")
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=1e-5)
    ap.add_argument("--beam-band", default="L", help="meerkat-beams band (beams mode)")
    ap.add_argument("--npix", type=int, default=128, help="coarse beam grid size (beams mode)")
    ap.add_argument("--half-width-deg", type=float, default=1.5, help="coarse beam grid half width (beams mode)")
    ap.add_argument(
        "--bandwidth-mhz", type=float, help="beams mode: measure a band this wide about freq_out instead of the band"
    )
    args = ap.parse_args()
    set_envs(args.nthreads, args.nthreads)
    gradient(args) if args.mode == "gradient" else beams(args)


if __name__ == "__main__":
    main()
