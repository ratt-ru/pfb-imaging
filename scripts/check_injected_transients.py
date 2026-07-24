"""Localise a transient-injection astrometry error along the full recovery chain.

For each injected source the recovery goes
    injected (ra,dec)
      --[pfb imaging]-->        where it peaks in the cube (a pixel)
      --[breifast detection]--> the detected pixel (x, y)
      --[WCS pixel->sky]-->     the reported (ra, dec) in the detections file
and the localisation error is injected-vs-reported. This tool splits that error
into its three links so the culprit is unambiguous (ratt-ru/breifast#263):

* place"  - injected  vs  cube-truth position  (pfb places the source correctly?)
* detdpix - cube-truth pixel  vs  detected (x,y)  (breifast detects the right pixel?)
* conv"   - reported pos  vs  WCS applied to the detected (x,y)  (breifast's
            pixel->sky conversion; expected ~0 if it uses the FITS WCS properly)
* net"    - injected  vs  reported pos  (the end-to-end error)

Inputs (all mandatory, positional):
  1 INJECTION_YAML  - the transient injection file (positions in degrees)
  2 ZARR_CUBE       - the raw hci zarr cube
  3 DETECTIONS      - the unified detections file (astropy QTable / ECSV);
                      uses its `pos` SkyCoord and `x`/`y` pixel columns
  4 FITS_CUBE       - the FITS cube written by hci (its SIN WCS is ground truth)

Usage
-----
    python scripts/check_injected_transients.py \\
        INJECTION_YAML  ZARR_CUBE  DETECTIONS  FITS_CUBE \\
        [--match-radius ARCSEC] [--window PIX]
"""

import argparse

import numpy as np
import xarray as xr
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS


def load_injected(path):
    """Return (names, SkyCoord, peak_times) from the injection YAML.

    Positions are in degrees; ``peak_times`` are seconds from the start of the
    observation (utils/transients subtracts times[0] before evaluating the
    light curve), one per source (None when a source has no time block).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    names, ras, decs, peak_times = [], [], [], []
    for t in cfg["transients"]:
        names.append(t.get("name", f"src{len(names)}"))
        ras.append(float(t["position"]["ra"]))
        decs.append(float(t["position"]["dec"]))
        tblock = t.get("time") or {}
        peak_times.append(float(tblock["peak_time"]) if "peak_time" in tblock else None)
    # pfb builds injection coords as FK5 (utils/stokes2im uses SkyCoord(..., frame="fk5"))
    return names, SkyCoord(ras * u.deg, decs * u.deg, frame="fk5"), peak_times


def load_detections(path):
    """Return (SkyCoord positions, (N,2) pixel array or None) from a detections QTable."""
    tbl = QTable.read(path)
    coord = None
    for col in tbl.colnames:  # a serialised SkyCoord column (e.g. breifast's `pos`)
        if isinstance(tbl[col], SkyCoord):
            print(f"  detections: sky column '{col}' ({len(tbl)} rows)")
            coord = tbl[col]
            break
    if coord is None:
        ra_col = _find_col(tbl, ("ra", "raj2000", "ra_deg", "pos.ra", "alpha"))
        dec_col = _find_col(tbl, ("dec", "decj2000", "dec_deg", "pos.dec", "delta"))
        if ra_col is None or dec_col is None:
            raise ValueError(f"no sky coordinates in {path}; columns: {tbl.colnames}")
        coord = SkyCoord(u.Quantity(tbl[ra_col], u.deg), u.Quantity(tbl[dec_col], u.deg), frame="fk5")
        print(f"  detections: sky columns '{ra_col}'/'{dec_col}' ({len(tbl)} rows)")

    xcol, ycol = _find_col(tbl, ("x", "x_pix", "xpix", "col")), _find_col(tbl, ("y", "y_pix", "ypix", "row"))
    xy = None
    if xcol and ycol:
        xy = np.column_stack([np.asarray(tbl[xcol], float), np.asarray(tbl[ycol], float)])
        print(f"  detections: pixel columns '{xcol}'/'{ycol}' (assumed 0-indexed)")
    else:
        print("  detections: no pixel columns found — pixel-chain checks disabled")
    return coord, xy


def _find_col(tbl, candidates):
    lower = {c.lower(): c for c in tbl.colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _collapse_to_yx(arr):
    """Reduce an (..., Y, X) cube to a 2D (Y, X) peak map (max over leading axes)."""
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = np.nanmax(arr, axis=0)
    return arr


def _yx_at_time(data, t):
    """2D (Y, X) image of a single time slice from a (STOKES, FREQ, TIME, Y, X)
    array (or any leading singleton axes + TIME + Y + X). Indexes lazily so a
    memmapped/dask cube is not fully materialised."""
    arr = data
    while arr.ndim > 3:  # drop leading STOKES/FREQ (singletons)
        arr = arr[0]
    if arr.ndim == 3:  # (TIME, Y, X)
        arr = arr[t]
    return np.asarray(arr)


def _cube_times(zds):
    """Return the cube's TIME coordinate (seconds) or None if it has no time axis."""
    for key in ("TIME", "time"):
        if key in zds.coords or key in zds.variables:
            return np.asarray(zds[key].values, dtype=float)
    return None


def _peak_pixel(img, wcs, coord, window):
    """0-indexed (ix, iy) of the peak in a `window`-half-width box about `coord`."""
    x0, y0 = wcs.world_to_pixel(coord)
    xc, yc = int(round(float(x0))), int(round(float(y0)))
    ny, nx = img.shape
    if not (0 <= xc < nx and 0 <= yc < ny):
        return None, None
    lo_x, hi_x = max(0, xc - window), min(nx, xc + window + 1)
    lo_y, hi_y = max(0, yc - window), min(ny, yc + window + 1)
    sub = img[lo_y:hi_y, lo_x:hi_x]
    ly, lx = np.unravel_index(np.nanargmax(sub), sub.shape)
    return lo_x + lx, lo_y + ly


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("injection_yaml")
    ap.add_argument("zarr_cube")
    ap.add_argument("detections")
    ap.add_argument("fits_cube")
    ap.add_argument("--match-radius", type=float, default=600.0, help="detection match radius, arcsec (default 600)")
    ap.add_argument("--window", type=int, default=8, help="peak-search half-width, pixels (default 8)")
    args = ap.parse_args()

    names_inj, inj, peak_times = load_injected(args.injection_yaml)
    print(f"injected: {len(inj)} sources from {args.injection_yaml}")

    hdul = fits.open(args.fits_cube, memmap=True)
    fhdr = hdul[0].header
    fdata = hdul[0].data  # (STOKES, FREQ, TIME, Y, X); indexed lazily per slice
    wcs = WCS(fhdr).celestial
    nx, ny = int(fhdr["NAXIS1"]), int(fhdr["NAXIS2"])
    centre = wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    cell = abs(float(fhdr["CDELT1"])) * 3600.0
    print(
        f"FITS: {fhdr.get('CTYPE1')}/{fhdr.get('CTYPE2')} ({ny}, {nx}), "
        f'centre {centre.to_string("hmsdms", sep=":", precision=2)}, cell {cell:.2f}"'
    )

    zds = xr.open_zarr(args.zarr_cube)
    zvar = "cube" if "cube" in zds else ("cube_mean" if "cube_mean" in zds else None)
    if zvar is None:
        raise ValueError(f"no 'cube'/'cube_mean' in {args.zarr_cube}; vars: {list(zds.data_vars)}")
    ctimes = _cube_times(zds)  # absolute seconds, or None (no time axis)
    rel_times = None if ctimes is None else ctimes - ctimes[0]  # peak_time is from obs start
    ntime = None if rel_times is None else rel_times.size
    if rel_times is not None:
        print(f"  cube TIME axis: {ntime} slices spanning {rel_times[-1]:.0f} s from start")
    else:
        print("  cube has no TIME axis — peaks searched in the time-collapsed image")

    det, det_xy = load_detections(args.detections)
    print()

    head = "{:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>6} {:>7}".format(
        "name", "dist_deg", 'net"', 'place"', "detdpix", 'conv"', "zvf_px", "t_slice"
    )
    print(head)
    print("-" * len(head))
    dists, nets, places, detdpixs, convs = [], [], [], [], []
    posrows = []  # (name, injected SkyCoord, recovered SkyCoord|None, place, t_slice)
    fimg_collapsed = None
    for name, c, pk in zip(names_inj, inj, peak_times):
        dist_deg = float(centre.separation(c).to_value(u.deg))

        # expected time slice from the source's peak_time (nearest cube integration)
        if rel_times is not None and pk is not None:
            t_exp = int(np.argmin(np.abs(rel_times - pk)))
        else:
            t_exp = None

        # the (Y, X) image the source should peak in: its own time slice if we
        # know it, else the time-collapsed cube (computed once)
        if t_exp is not None:
            fimg = _yx_at_time(fdata, t_exp)
            zsl = _yx_at_time(zds[zvar].data, t_exp)
        else:
            if fimg_collapsed is None:
                fimg_collapsed = _collapse_to_yx(fdata)
            fimg = fimg_collapsed
            zsl = None
        tslice = "-" if t_exp is None else str(t_exp)

        # where the source actually peaks (FITS, and zarr as a cross-check)
        fp = _peak_pixel(fimg, wcs, c, args.window)
        zp = _peak_pixel(zsl, wcs, c, args.window) if zsl is not None else (None, None)
        if fp[0] is None:
            print(f"{name:>8} {dist_deg:>8.3f}   <off-grid>{'':>40}{tslice:>7}")
            posrows.append((name, c, None, np.nan, tslice))
            continue
        truth_pos = wcs.pixel_to_world(*fp)
        place = float(truth_pos.separation(c).to_value(u.arcsec))  # pfb placement error
        zvf = "-" if zp[0] is None else str(int(abs(zp[0] - fp[0]) + abs(zp[1] - fp[1])))
        posrows.append((name, c, truth_pos, place, tslice))

        # nearest detection to this injected source
        sep = c.separation(det)
        j = int(np.argmin(sep.to_value(u.arcsec)))
        if float(sep[j].to_value(u.arcsec)) > args.match_radius:
            print(f"{name:>8} {dist_deg:>8.3f} {'-':>8} {place:>8.3f} {'(no match)':>17} {zvf:>6} {tslice:>7}")
            dists.append(dist_deg)
            places.append(place)
            nets.append(np.nan)
            detdpixs.append(np.nan)
            convs.append(np.nan)
            continue
        net = float(sep[j].to_value(u.arcsec))  # injected -> reported (end-to-end)

        detdpix = conv = np.nan
        if det_xy is not None:
            dx, dy = det_xy[j]
            detdpix = float(np.hypot(dx - fp[0], dy - fp[1]))  # true peak vs detected pixel
            conv = float(det[j].separation(wcs.pixel_to_world(dx, dy)).to_value(u.arcsec))  # breifast pix->sky

        dd = f"{detdpix:>8.2f}" if np.isfinite(detdpix) else f"{'-':>8}"
        cv = f"{conv:>7.3f}" if np.isfinite(conv) else f"{'-':>7}"
        print(f"{name:>8} {dist_deg:>8.3f} {net:>8.3f} {place:>8.3f} {dd} {cv} {zvf:>6} {tslice:>7}")
        dists.append(dist_deg)
        nets.append(net)
        places.append(place)
        detdpixs.append(detdpix)
        convs.append(conv)

    print("-" * len(head))

    # injected vs recovered positions in hms/dms, to paste into a FITS viewer
    print("\ninjected vs recovered peak (paste into your FITS viewer):")
    phead = "{:>8} {:>28} {:>28} {:>8} {:>7}".format("name", "injected", "recovered", 'place"', "t_slice")
    print(phead)
    print("-" * len(phead))
    for name, c, rec, place, tslice in posrows:
        inj_str = c.to_string("hmsdms", sep=":", precision=2)
        rec_str = "-" if rec is None else rec.to_string("hmsdms", sep=":", precision=2)
        pl = "-" if not np.isfinite(place) else f"{place:.3f}"
        print(f"{name:>8} {inj_str:>28} {rec_str:>28} {pl:>8} {tslice:>7}")

    hdul.close()
    if not dists:
        print("\nno injected sources landed on the grid")
        return
    dists, nets, places = np.array(dists), np.array(nets), np.array(places)
    detdpixs, convs = np.array(detdpixs), np.array(convs)
    print(
        f'\npfb placement (place"): max {np.nanmax(places):.3f}", median {np.nanmedian(places):.3f}" (cell {cell:.2f}")'
    )
    m = np.isfinite(nets)
    if m.sum() >= 2:
        rn = float(np.corrcoef(dists[m], nets[m])[0, 1])
        rp = float(np.corrcoef(dists[m], places[m])[0, 1]) if np.nanstd(places[m]) > 0 else float("nan")
        print(
            f'end-to-end (net")     : max {np.nanmax(nets[m]):.3f}", '
            f"corr(net, dist)={rn:+.3f}, corr(place, dist)={rp:+.3f}"
        )
        _verdict(places, nets, detdpixs, convs, m, cell)


def _verdict(places, nets, detdpixs, convs, m, cell):
    """Attribute the end-to-end error to a single link in the recovery chain."""
    if np.nanmax(places) > 2 * cell:
        print(
            '\n=> the source peaks are already displaced in the cube (place" grows) -> the error is\n'
            "   in pfb imaging / rephasing / the cube WCS, upstream of breifast."
        )
        return
    if np.nanmax(nets[m]) <= 2 * cell:
        print("\n=> injected and recovered positions agree to ~a pixel at all distances — no error.")
        return
    conv_bad = np.isfinite(convs[m]).any() and np.nanmedian(convs[m]) > 2 * cell
    dpix_bad = np.isfinite(detdpixs[m]).any() and np.nanmedian(detdpixs[m]) * cell > 2 * cell
    if conv_bad:
        print(
            "\n=> peaks are placed correctly and detected on the right pixel, but the reported\n"
            "   (ra, dec) disagrees with the WCS of that pixel (conv\" grows) -> the detections'\n"
            "   pixel->sky conversion is the culprit (not using the FITS WCS properly)."
        )
    elif dpix_bad:
        print(
            "\n=> peaks are placed correctly and the pixel->sky conversion is exact, but the\n"
            "   detected pixel is offset from the true peak (detdpix grows) -> the error is in\n"
            "   the detection/centroiding step upstream of the coordinate conversion."
        )
    else:
        print(
            "\n=> peaks, detected pixels and conversion all check out per-source, yet net error is\n"
            "   large -> likely a detection<->injection MATCHING problem (wrong associations)."
        )


if __name__ == "__main__":
    main()
