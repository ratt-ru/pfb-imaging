"""What beam cell is safe for bilinear placement onto the image grid? (#401, #365)

Pass 1 evaluates the rotation-averaged beam on a small field-centred grid and pass 2
places it on the image grid by bilinear interpolation. This measures the
interpolation error against the beam evaluated directly at a fine reference cell,
for candidate cells that are fractions of the beam cube's own spacing ``dx``.

The candidate grids are exact subsamples of the reference grid, so every reference
point that is not a candidate node tests the interpolation. Errors are reported
per region of the reference beam (main lobe > 0.5, shoulder 0.1-0.5, sidelobes
< 0.1) and per halving of the cell, which should quarter for bilinear error.

Usage:
    python beam_cell_study.py --scratch X.scratch --band 3 --half-width-deg 1.5 --out cell_study.json
"""

import argparse
import json

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.time import Time
from meerkat_beams.utils import BeamWizard
from scipy.interpolate import RegularGridInterpolator

from pfb_imaging import set_envs
from pfb_imaging.utils.beam import BEAM_FREQ_EXTRAPOLATION, BEAM_PA_TOLERANCE_DEG
from pfb_imaging.utils.misc import to_mjd_time

STRIDES = (16, 8, 4, 2)  # candidate cells dx, dx/2, dx/4, dx/8 on a dx/16 reference


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", required=True, help="kept imager scratch (piece times and pointing)")
    ap.add_argument("--band", type=int, default=0, help="scratch band whose pieces give the times")
    ap.add_argument("--freqs-mhz", type=float, nargs="+", default=[1000.0, 1711.0])
    ap.add_argument("--half-width-deg", type=float, default=1.5)
    ap.add_argument("--beam-band", default="L")
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_envs(args.nthreads, args.nthreads)

    scratch = xr.open_datatree(args.scratch, engine="zarr", chunks=None, cache=False)
    attrs = [
        child.ds.attrs
        for name in sorted(scratch.children)
        if name.startswith(f"band{args.band:04d}_")
        for child in scratch[name].children.values()
    ]
    groups = sorted({a["baseline_group"] for a in attrs})
    a0 = attrs[0]
    centre = SkyCoord(ra=a0["ra0"] * u.rad, dec=a0["dec0"] * u.rad)
    results = []
    for bg in groups:
        wiz = BeamWizard(band=args.beam_band, group=bg, freq_extrapolation=BEAM_FREQ_EXTRAPOLATION)
        wiz.set_field_centre(centre)
        dx = float(wiz.bds.attrs["dx"])
        href = dx / max(STRIDES)
        # odd point count divisible by every stride: each candidate is an exact subsample
        nseg = int(np.ceil(2 * args.half_width_deg / (href * max(STRIDES)))) * max(STRIDES)
        lm = (np.arange(nseg + 1) - nseg // 2) * href
        times = sorted({float(a["time_out"]) for a in attrs if a["baseline_group"] == bg})
        tracks = {"one_scan": [times[len(times) // 2]], "full_track": times}
        for freq in args.freqs_mhz:
            for tname, tt in tracks.items():
                t = Time(to_mjd_time(np.asarray(tt)) / 86400.0, format="mjd")
                ref, _ = wiz.get_rotation_averaged_beam(
                    l=lm,
                    m=lm,
                    times=t,
                    freq=np.atleast_1d(freq * 1e6),
                    time_stepping=1,
                    pixel_stepping=1,
                    var="nstokes",
                    i="I",
                    j="I",
                    verbose=0,
                    nthreads=args.nthreads,
                    pa_tolerance=BEAM_PA_TOLERANCE_DEG,
                )
                ref = np.real(np.squeeze(ref))
                mm, ll = np.meshgrid(lm, lm, indexing="ij")
                # the square's corners reach past the inscribed disk, and past
                # the cube's own extent for wide grids
                disk = np.hypot(ll, mm) <= args.half_width_deg
                regions = {
                    "main_lobe": ref > 0.5,
                    "shoulder": (ref > 0.1) & (ref <= 0.5),
                    "sidelobes": ref <= 0.1,
                    "sidelobes_in_disk": (ref <= 0.1) & disk,
                    "corners": ~disk,
                }
                lam_over_d = 299792458.0 / (freq * 1e6) / (15.0 if bg == "MPMP" else 13.5)
                row = {"group": bg, "freq_mhz": freq, "times": tname, "dx_deg": dx, "cells": []}
                for s in STRIDES:
                    coarse = ref[::s, ::s]
                    interp = RegularGridInterpolator((lm[::s], lm[::s]), coarse, method="linear")
                    err = np.abs(interp((mm, ll)) - ref)
                    row["cells"].append(
                        {
                            "cell_deg": href * s,
                            "cell_over_dx": s / max(STRIDES),
                            "cell_over_lambda_d": float(np.deg2rad(href * s) / lam_over_d),
                            "npix_for_3deg": int(np.ceil(3.0 / (href * s))) + 1,
                            **{f"max_{k}": float(err[v].max()) for k, v in regions.items()},
                        }
                    )
                results.append(row)
                print(json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
