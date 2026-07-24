#!/usr/bin/env python
"""Diagnostic plots for ``pfb deconv --debug`` output (mosaic misfit QA).

Reads the ``<fits_oname>_<suffix>_debug.json`` written by ``pfb deconv --debug``
(and, when present, the ``<fits_oname>_<suffix>_partitions/`` FITS written by
``--fits-per-partition``) and produces:

1. ``<stem>_chi2.png`` -- per-partition reduced chi2 trajectories over major
   iterations, absolute and normalised to the starting model. A partition whose
   chi2 rises while the others fall is being traded off against them: the
   signature of inter-pointing inconsistency (flux scale, beam, astrometry).
2. ``<stem>_uvprofile.png`` -- baseline-length-binned weighted residual power
   per partition. Flat = noise-like; excess at short baselines = large-scale
   disagreement (beam / flux scale); excess at long baselines = small-scale /
   positional disagreement (astrometry, rephasing).
3. ``<stem>_residuals.png`` -- grid of per-partition residual images (bands x
   partitions, shared symmetric colour scale) when the partitions FITS
   directory is found.

Colour identifies the field (fixed assignment, colourblind-safe Okabe-Ito
palette); linestyle distinguishes bands.

Run (from the repo root):

    uv run python scripts/deconv_qa.py /path/to/out_I_main_debug.json
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito: colourblind-safe categorical palette, fixed assignment order
FIELD_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
BAND_STYLES = ["-", "--", ":", "-."]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("debug_json", help="Path to the <stem>_debug.json written by pfb deconv --debug")
    p.add_argument(
        "--partitions-dir",
        default=None,
        help="Per-partition FITS directory from --fits-per-partition. "
        "Defaults to <stem>_partitions next to the JSON; the residual-image "
        "grid is skipped when it does not exist.",
    )
    p.add_argument("--outdir", default=None, help="Directory for the PNGs (default: next to the JSON)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def style_maps(record):
    """Fixed colour-per-field and linestyle-per-band assignments."""
    fields, bands = set(), set()
    for entry in record.get("iterations", []):
        for bnd in entry["bands"]:
            bands.add(bnd["band"])
            for part in bnd["partitions"]:
                fields.add(part["field"])
    for band, plist in (record.get("uv_profiles") or {}).items():
        bands.add(band)
        for part in plist:
            fields.add(part["field"])
    field_colour = {f: FIELD_COLOURS[i % len(FIELD_COLOURS)] for i, f in enumerate(sorted(fields))}
    band_style = {b: BAND_STYLES[i % len(BAND_STYLES)] for i, b in enumerate(sorted(bands))}
    return field_colour, band_style


def _tidy(ax):
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_chi2(record, field_colour, band_style, out, dpi):
    """Reduced-chi2 trajectories, absolute and normalised to iteration 0."""
    # series[(band, part_name)] = (field, iters, rchi2)
    series = {}
    for entry in record["iterations"]:
        for bnd in entry["bands"]:
            for part in bnd["partitions"]:
                key = (bnd["band"], part["name"])
                it, rc = series.setdefault(key, (part["field"], [], []))[1:]
                it.append(entry["iter"])
                rc.append(part["chi2"][0] / max(part["ndata"], 1.0))
    if not series:
        print("no chi2 iterations in the record; skipping chi2 plot")
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    seen_fields = set()
    for (band, pname), (field, it, rc) in sorted(series.items()):
        label = field if field not in seen_fields else None
        seen_fields.add(field)
        for ax, y in ((ax0, rc), (ax1, np.asarray(rc) / rc[0] if rc[0] > 0 else np.asarray(rc))):
            ax.plot(it, y, band_style[band], color=field_colour[field], label=label if ax is ax0 else None, lw=1.5)
    ax0.set_yscale("log")
    ax0.set_ylabel(r"$\chi^2$ / N$_{\rm data}$")
    ax0.set_title("per-partition reduced chi2")
    ax1.set_ylabel(r"$\chi^2(k)\ /\ \chi^2(0)$")
    ax1.set_title("normalised to starting model")
    ax1.axhline(1.0, color="0.6", lw=0.8)
    for ax in (ax0, ax1):
        ax.set_xlabel("major iteration")
        _tidy(ax)
    ax0.legend(title="field", frameon=False, fontsize=8)
    if len({b for b, _ in series}) > 1:
        ax1.text(
            0.02,
            0.02,
            "linestyle = band: " + ", ".join(f"{s} {b}" for b, s in sorted({(b, band_style[b]) for b, _ in series})),
            transform=ax1.transAxes,
            fontsize=7,
            color="0.4",
        )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")


def plot_uvprofiles(record, field_colour, band_style, out, dpi):
    """Weighted residual power vs baseline length, per partition."""
    profs = record.get("uv_profiles") or {}
    if not profs:
        print("no uv_profiles in the record; skipping uv-profile plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seen_fields = set()
    for band in sorted(profs):
        for part in profs[band]:
            edges = np.asarray(part["uvdist_edges_lambda"]) / 1e3  # klambda
            centres = 0.5 * (edges[:-1] + edges[1:])
            power = np.asarray(part["resid_power"])
            count = np.asarray(part["count"])
            sel = count > 0
            label = part["field"] if part["field"] not in seen_fields else None
            seen_fields.add(part["field"])
            ax.plot(centres[sel], power[sel], band_style[band], color=field_colour[part["field"]], label=label, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"baseline length (k$\lambda$)")
    ax.set_ylabel(r"$\langle w\,|V - G(Bm)|^2 \rangle$ per bin")
    ax.set_title("residual power vs baseline length (flat = noise-like)")
    _tidy(ax)
    ax.legend(title="field", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")


def plot_residual_grid(pdir, out, dpi):
    """Bands x partitions grid of per-partition residual images, shared scale."""
    from astropy.io import fits  # needed by this panel only

    hits = sorted(glob.glob(f"{pdir}/residual_band*_part*.fits"))
    if not hits:
        print(f"no residual FITS under {pdir}; skipping residual grid")
        return
    # organise into rows = bands, cols = partitions
    imgs = {}
    for h in hits:
        base = os.path.basename(h)  # residual_band####_time####_part####_<field>.fits
        toks = base.split("_")
        band, part = toks[1], toks[3]
        field = "_".join(toks[4:]).removesuffix(".fits")
        with fits.open(h) as hdul:
            img = np.squeeze(hdul[0].data).astype(np.float64)
            rchi2 = hdul[0].header.get("RCHI2", np.nan)
        imgs[(band, part)] = (field, img, rchi2)
    bands = sorted({b for b, _ in imgs})
    parts = sorted({p for _, p in imgs})
    vmax = max(np.percentile(np.abs(im), 99.5) for _, im, _ in imgs.values())

    fig, axes = plt.subplots(len(bands), len(parts), figsize=(3.2 * len(parts) + 1.2, 3.2 * len(bands)), squeeze=False)
    for i, band in enumerate(bands):
        for j, part in enumerate(parts):
            ax = axes[i][j]
            if (band, part) not in imgs:
                ax.axis("off")
                continue
            field, img, rchi2 = imgs[(band, part)]
            m = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{field}  rms={np.std(img):.2e}\n{band} {part}  rchi2={rchi2:.3e}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(m, ax=axes, shrink=0.85, label="residual (Jy/beam)")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    args = parse_args()
    with open(args.debug_json) as f:
        record = json.load(f)

    stem = args.debug_json.removesuffix(".json").removesuffix("_debug")
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.debug_json))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, os.path.basename(stem))

    field_colour, band_style = style_maps(record)
    plot_chi2(record, field_colour, band_style, f"{base}_chi2.png", args.dpi)
    plot_uvprofiles(record, field_colour, band_style, f"{base}_uvprofile.png", args.dpi)

    pdir = args.partitions_dir or f"{stem}_partitions"
    if os.path.isdir(pdir):
        plot_residual_grid(pdir, f"{base}_residuals.png", args.dpi)
    else:
        print(f"partitions dir {pdir} not found; run pfb deconv with --fits-per-partition for the image grid")


if __name__ == "__main__":
    main()
