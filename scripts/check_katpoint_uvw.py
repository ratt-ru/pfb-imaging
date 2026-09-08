"""Recompute an MS's UVW with katpoint and compare to the recorded coordinates.

Motivation (ratt-ru/pfb-imaging#280): the rephasing path synthesizes UVW with
casacore *measures* (`utils/astrometry.synthesize_uvw`), which differs slightly
from the observatory's own UVW (MeerKAT SDP uses *katpoint*), making the rephase
not perfectly reversible. This script checks whether katpoint recovers the
recorded UVW, and quantifies katpoint-vs-measures, using the antenna positions,
times and phase centre straight out of the MS.

Antennas are rebuilt from the MS ANTENNA::POSITION (ITRF ECEF) via
katpoint.ecef_to_lla, so the comparison isolates the UVW *algorithm* (katpoint
vs measures vs whatever wrote the MS) rather than differences in station
coordinates.

Conventions
-----------
* MS TIME is UTC MJD seconds; katpoint wants UTC unix seconds
  (unix = mjd_sec - 3506716800).
* katpoint.Target.uvw(a2, t, antenna=a1) is the baseline a1->a2 in metres. Its
  sign is opposite to CASA (xarray-kat negates for the "casa" convention). We
  therefore try both sign/ordering variants and report the best-matching one.
* pfb's synthesize_uvw forms the row baseline as station[ant1] - station[ant2];
  we mirror that ordering for the katpoint station vectors.

Usage
-----
    python scripts/check_katpoint_uvw.py [MS] [--rows N]
"""

import argparse

import katpoint
import numpy as np
from daskms import xds_from_ms, xds_from_table

from pfb_imaging.utils.astrometry import synthesize_uvw

# UTC MJD seconds -> UTC unix seconds (MJD 40587 == unix epoch)
MJD_UNIX_OFFSET_S = 40587.0 * 86400.0

DEFAULT_MS = "/home/bester/data/mkat_test_subsets/MeerKLASS/test_subset/otf_pointing_no_600.ms"


def _report(tag, a, b, blmax):
    """Print how well array `a` matches reference `b` (both (nrow, 3))."""
    d = a - b
    per_axis_max = np.abs(d).max(axis=0)
    per_axis_rms = np.sqrt((d**2).mean(axis=0))
    corr = [np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(3)]
    print(f"  {tag}")
    print(f"    max abs diff/axis (m) : [{per_axis_max[0]:.4g}, {per_axis_max[1]:.4g}, {per_axis_max[2]:.4g}]")
    print(f"    rms diff/axis (m)     : [{per_axis_rms[0]:.4g}, {per_axis_rms[1]:.4g}, {per_axis_rms[2]:.4g}]")
    print(f"    max|diff| / max|bl|   : {np.abs(d).max() / blmax:.3g}")
    print(f"    per-axis correlation  : [{corr[0]:.10f}, {corr[1]:.10f}, {corr[2]:.10f}]")
    return np.abs(d).max()


def katpoint_uvw(antpos, names, diam, time, ant1, ant2, phase_ref):
    """Row UVW (nrow, 3) via katpoint, ordered station[ant1] - station[ant2]."""
    antennas = []
    for i in range(antpos.shape[0]):
        lat, lon, alt = katpoint.ecef_to_lla(*antpos[i])
        antennas.append(katpoint.Antenna(str(names[i]), lat, lon, alt, float(diam[i])))
    target = katpoint.construct_radec_target(float(phase_ref[0]), float(phase_ref[1]))

    uvw = np.zeros((time.size, 3))
    for t in np.unique(time):
        ts = t - MJD_UNIX_OFFSET_S  # MJD s -> unix s
        # baseline ref->antenna_i for every antenna, in this timestamp's uvw frame
        u, v, w = target.uvw(antennas, timestamp=ts, antenna=antennas[0])
        station = np.stack([np.asarray(u), np.asarray(v), np.asarray(w)], axis=-1)  # (nant, 3)
        rows = np.where(time == t)[0]
        uvw[rows] = station[ant1[rows]] - station[ant2[rows]]
    return uvw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ms", nargs="?", default=DEFAULT_MS)
    ap.add_argument("--rows", type=int, default=None, help="limit to first N rows (debug)")
    args = ap.parse_args()

    ds = xds_from_ms(args.ms, chunks={"row": -1}, group_cols=["FIELD_ID", "DATA_DESC_ID"])[0]
    ant = xds_from_table(f"{args.ms}::ANTENNA")[0]
    field = xds_from_table(f"{args.ms}::FIELD")[0]

    uvw_ms = ds.UVW.values
    time = ds.TIME.values
    ant1 = ds.ANTENNA1.values
    ant2 = ds.ANTENNA2.values
    if args.rows is not None:
        sl = slice(0, args.rows)
        uvw_ms, time, ant1, ant2 = uvw_ms[sl], time[sl], ant1[sl], ant2[sl]

    antpos = ant.POSITION.values
    names = ant.NAME.values
    diam = ant.DISH_DIAMETER.values
    phase_ref = np.asarray(field.PHASE_DIR.values.squeeze())

    blmax = np.sqrt((uvw_ms[:, :2] ** 2).sum(axis=1)).max()
    print(f"MS: {args.ms}")
    print(
        f"nrow={uvw_ms.shape[0]}, nant={antpos.shape[0]}, ntime={np.unique(time).size}, "
        f"max|uv|={blmax:.1f} m, phase_dir(rad)={phase_ref}"
    )
    print(f"MJD={time.min() / 86400.0:.5f}")

    # katpoint
    uvw_kp = katpoint_uvw(antpos, names, diam, time, ant1, ant2, phase_ref)
    # measures (what pfb uses today) — same station[ant1]-station[ant2] ordering
    uvw_me = synthesize_uvw(antpos, time, ant1, ant2, phase_ref)

    print("\n=== katpoint vs MS (both sign/ordering variants) ===")
    e_pos = _report("station[ant1]-station[ant2]  (as pfb orders measures)", uvw_kp, uvw_ms, blmax)
    print()
    e_neg = _report("station[ant2]-station[ant1]  (negated / CASA flip)", -uvw_kp, uvw_ms, blmax)
    best = "ant1-ant2" if e_pos < e_neg else "ant2-ant1 (negated)"
    uvw_kp_best = uvw_kp if e_pos < e_neg else -uvw_kp
    print(f"\n  -> katpoint best match: {best}")

    print("\n=== measures (synthesize_uvw) vs MS ===")
    _report("station[ant1]-station[ant2]", uvw_me, uvw_ms, blmax)

    print("\n=== katpoint vs measures (best-sign katpoint) ===")
    _report("katpoint - measures", uvw_kp_best, uvw_me, blmax)


if __name__ == "__main__":
    main()
