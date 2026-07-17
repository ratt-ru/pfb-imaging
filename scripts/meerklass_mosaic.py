#!/usr/bin/env python
"""Joint-imaging harness for MeerKLASS OTF pointings (mosaic acceptance test).

Images N adjacent OTF pointings jointly with ``pfb imager`` and each pointing
separately with identical parameters, then reports:

1. mechanics -- partitions per band node and per-partition wsum bookkeeping
   against the single-field runs (valid regardless of rephasing);
2. alignment -- the brightest source of each single-field image is located on
   the sky via its FITS WCS and searched for in the joint image, both at its
   true position and at the position displaced by the phase-centre difference.

Until the pass-1 rephasing lands (#281), the imager grids each partition about
its OWN phase centre and sums the images centre-on-centre, so each source
appears displaced by (centre_ref - centre_k): this script MEASURES that
baseline. Once rephasing is in, the displaced copy must vanish and the true
position must win -- flip --expect-aligned to turn the report into hard
assertions (exit code 1 on failure).

Run (uv, from the repo root; the env guard below handles Ray under uv):

    uv run python scripts/meerklass_mosaic.py \
        --data-dir /home/bester/data/mkat_test_subsets/MeerKLASS/test_subset \
        --pointings 600 601 602 --outdir /tmp/mk_mosaic
"""

import os

# Under `uv run`, Ray's uv runtime-env hook relaunches workers via
# `uv run --frozen python`, whose fresh venv lacks the [full] extra and the
# workers crash on `import ray`. Must be set before ray is imported
# (see tests/conftest.py).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory holding otf_pointing_no_*.ms")
    p.add_argument("--pointings", type=int, nargs="+", default=[600, 601, 602])
    p.add_argument("--data-column", default="CORRECTED_DATA")
    p.add_argument("--field-of-view", type=float, default=2.0, help="deg")
    p.add_argument("--channels-per-image", type=int, default=-1)
    p.add_argument("--robustness", type=float, default=None)
    p.add_argument("--nthreads", type=int, default=8)
    p.add_argument("--outdir", required=True)
    p.add_argument("--skip-singles", action="store_true", help="Reuse existing single-field runs in outdir")
    p.add_argument(
        "--expect-aligned",
        action="store_true",
        help="Assert sources land at their true positions (post-rephasing acceptance mode)",
    )
    return p.parse_args()


def run_imager(ms_list, outname, args):
    from pfb_imaging.core.imager import imager

    imager(
        [Path(m) for m in ms_list],
        outname,
        data_column=args.data_column,
        channels_per_image=args.channels_per_image,
        integrations_per_image=-1,
        product="I",
        field_of_view=args.field_of_view,
        robustness=args.robustness,
        fits_mfs=True,
        fits_cubes=False,
        overwrite=True,
        nworkers=1,
        nthreads=args.nthreads,
    )


def load_mfs(outname):
    """(image (ny, nx), celestial WCS) of the MFS dirty written by run_imager."""
    import glob

    from astropy.io import fits
    from astropy.wcs import WCS

    hits = sorted(glob.glob(f"{outname}_I_dirty_time*_mfs.fits"))
    assert hits, f"no MFS dirty FITS found for {outname}"
    with fits.open(hits[0]) as hdul:
        img = hdul[0].data.squeeze()
        w = WCS(hdul[0].header).celestial
    return img, w


def peak_sky(img, w):
    """Sky coordinate and value of the image maximum."""
    iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)
    return w.pixel_to_world(ix, iy), float(img[iy, ix]), (iy, ix)


def value_near(img, w, sky, half=15):
    """Max value and pixel offset within a (2*half+1)^2 box around a sky position."""
    px, py = w.world_to_pixel(sky)
    ix, iy = int(round(float(px))), int(round(float(py)))
    ny, nx = img.shape
    if not (0 <= ix < nx and 0 <= iy < ny):
        return np.nan, (np.nan, np.nan)
    box = img[max(0, iy - half) : iy + half + 1, max(0, ix - half) : ix + half + 1]
    by, bx = np.unravel_index(int(np.argmax(box)), box.shape)
    return float(box[by, bx]), (by - min(half, iy), bx - min(half, ix))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ms_paths = [f"{args.data_dir}/otf_pointing_no_{n}.ms" for n in args.pointings]
    for m in ms_paths:
        assert os.path.exists(m), f"missing {m}"

    from astropy.coordinates import SkyCoord
    from casacore.tables import table

    centres = []
    for m in ms_paths:
        fld = table(f"{m}::FIELD", readonly=True, ack=False)
        ra, dec = np.rad2deg(fld.getcol("PHASE_DIR").squeeze())
        fld.close()
        centres.append(SkyCoord(ra=ra, dec=dec, unit="deg"))
    print("phase centres:")
    for n, c in zip(args.pointings, centres):
        print(f"  {n}: {c.to_string('hmsdms')}")

    # --- single-field reference runs ---
    singles = []
    for n, m in zip(args.pointings, ms_paths):
        out = f"{args.outdir}/p{n}"
        if not args.skip_singles:
            print(f"\n=== imaging pointing {n} alone ===", flush=True)
            run_imager([m], out, args)
        singles.append(out)

    # --- joint run ---
    out_joint = f"{args.outdir}/joint"
    print(f"\n=== imaging {len(ms_paths)} pointings jointly ===", flush=True)
    run_imager(ms_paths, out_joint, args)

    # --- mechanics: partitions and wsum bookkeeping ---
    import xarray as xr

    print("\n--- mechanics ---")
    dt = xr.open_datatree(out_joint + "_I.dt", engine="zarr", chunks=None)
    bands = sorted(n for n in dt.children if n.startswith("band"))
    ok = True
    single_wsums = []
    for out in singles:
        sdt = xr.open_datatree(out + "_I.dt", engine="zarr", chunks=None)
        sbands = sorted(n for n in sdt.children if n.startswith("band"))
        single_wsums.append({b: float(sdt[b].ds.WSUM.values[0]) for b in sbands})
    for b in bands:
        parts = sorted(dt[b].children)
        wsum_parts = [float(np.asarray(dt[b][p].ds.attrs["wsum"]).ravel()[0]) for p in parts]
        wsum_band = float(dt[b].ds.WSUM.values[0])
        expect_parts = len(ms_paths)
        line = f"{b}: {len(parts)} partitions, wsum {wsum_band:.6g}"
        if len(parts) != expect_parts:
            line += f"  [FAIL: expected {expect_parts} partitions]"
            ok = False
        if not np.isclose(sum(wsum_parts), wsum_band, rtol=1e-6):
            line += "  [FAIL: partition wsums do not sum to band wsum]"
            ok = False
        singles_sum = sum(sw.get(b, 0.0) for sw in single_wsums)
        if singles_sum > 0 and not np.isclose(singles_sum, wsum_band, rtol=1e-3):
            line += f"  [FAIL: singles wsum sum {singles_sum:.6g} != joint {wsum_band:.6g}]"
            ok = False
        print(line)

    # --- alignment: brightest source of each single, hunted in the joint ---
    print("\n--- alignment (joint frame) ---")
    img_j, w_j = load_mfs(out_joint)
    # the joint tangent point: pre-rephasing it was the first piece's centre,
    # post-rephasing it is the barycentre -- read it from the tree
    jdt = xr.open_datatree(out_joint + "_I.dt", engine="zarr", chunks=None)
    jband = sorted(n for n in jdt.children if n.startswith("band"))[0]
    tangent = SkyCoord(ra=np.rad2deg(jdt[jband].ds.attrs["ra"]), dec=np.rad2deg(jdt[jband].ds.attrs["dec"]), unit="deg")
    cell_deg_j = np.rad2deg(float(jdt[jband].ds.attrs["cell_rad"]))
    aligned_ok = True
    for n, c, out in zip(args.pointings, centres, singles):
        img_s, w_s = load_mfs(out)
        sky, val, _ = peak_sky(img_s, w_s)
        # where the source should be (true sky) and where centre-stacking
        # (no rephasing) would put it: displaced by (tangent - field centre)
        dra = tangent.ra - c.ra
        ddec = tangent.dec - c.dec
        displaced = SkyCoord(ra=sky.ra + dra, dec=sky.dec + ddec)
        disp_px = np.hypot(dra.deg * np.cos(c.dec.rad), ddec.deg) / cell_deg_j
        v_true, off_true = value_near(img_j, w_j, sky)
        v_disp, off_disp = value_near(img_j, w_j, displaced)
        print(
            f"pointing {n}: single peak {val:.4f} Jy/beam at {sky.to_string('hmsdms')}\n"
            f"    joint @ true position: {v_true:.4f} (box-peak offset {off_true})\n"
            f"    joint @ centre-stacked position: {v_disp:.4f} (box-peak offset {off_disp}, "
            f"{disp_px:.1f} px from true)"
        )
        if args.expect_aligned:
            # true position must carry at least half the single-field peak
            # (the joint value is a PB-weighted average across pointings --
            # full mosaic gain needs the #281 beam weighting); the displaced
            # ghost must be gone. Skip the ghost check when the displacement
            # is too small to separate the two probes.
            ok_here = v_true > 0.5 * val
            if disp_px > 3 * 2 * args.field_of_view:  # ~3 psf widths in px at srf 2
                ok_here = ok_here and abs(v_disp) < 0.2 * abs(v_true)
            if not ok_here:
                print(f"    [FAIL: pointing {n} not aligned at its true position]")
                aligned_ok = False

    if args.expect_aligned:
        print("\nacceptance:", "PASS" if (ok and aligned_ok) else "FAIL")
        sys.exit(0 if (ok and aligned_ok) else 1)
    else:
        print("\nmechanics:", "PASS" if ok else "FAIL", "(alignment reported, not asserted -- pre-rephasing baseline)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
