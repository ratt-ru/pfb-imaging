#!/usr/bin/env python
"""Diagnose the ripples in a per-pixel spectral index fit of a restored ``.dt``.

Runs where the tree lives and writes a few MB of cutouts and summary statistics
that can be pulled back for inspection.  Nothing here modifies the tree.

The hypothesis under test (pfb-imaging issue #312, wiki D31/D32).  ``pfb restore``
builds the intrinsic band image as::

    IMAGE_b = MODEL_b (x) G  +  (RESIDUAL_b / WSUM_b) (x) [G / PSFPARSN_b] / BEAM_b

The second term reconvolves the residual from the band's *fitted Gaussian*
``PSFPARSN_b`` to the common restoring beam ``G``.  But the residual's actual
resolution is the dirty beam -- the uv sampling function -- which is Gaussian
only in its core.  A Gaussian reconvolution matches the cores across bands and
cannot move the sidelobes, whose angular scale goes as ``1/nu``.  Each pixel
therefore sees a flux that oscillates with frequency on top of the true power
law, and a per-pixel fit reads that as spectral index: coherent, spatially
periodic ripples, strongest where the residual carries a large fraction of the
flux (the diffuse emission), weakest on the bright deconvolved structure.

Six diagnostics, in decreasing order of how decisive they are:

1. **Sidelobes survive homogenisation.**  Convolve each band's stored PSF from
   ``PSFPARSN_b`` to ``G`` -- exactly what restore does to the residual -- and
   compare bands.  The cores must agree by construction; if the sidelobes do
   not, the mechanism above is demonstrated on the actual data.
2. **The dirty beam scales as 1/nu.**  Correlate band b's PSF sidelobes with
   band 0's, raw and after rescaling band 0 radially by ``nu_b / nu_0``.  A big
   jump confirms the frequency dependence that drives the ripples.
3. **alpha with and without the residual.**  Fit the per-pixel power law on
   ``MODEL (x) G`` alone and on the full intrinsic image.  Their difference map
   isolates what the residual term does to the spectral index; if it carries the
   striping, the case is closed.
4. **Restoration flux units.**  The main-lobe volume of the dirty beam over the
   clean beam volume, per band.  It is not 1 and it is not constant in
   frequency, which biases the restored spectrum wherever the residual matters.
5. **FFT padding.**  ``restore_products`` convolves at ``pfrac=0.2``.  For a
   source filling the frame that wraps, and it wraps differently per band.
   Refit with ``pfrac=1.0`` and difference.
6. **Beam model.**  Azimuthal profile of alpha about the pointing centre, plus
   the angular and radial power spectra of the alpha maps.  A primary beam error
   gives concentric structure; a PSF error gives the striping.

Run (from the repo root, on the box holding the tree)::

    uv run python scripts/test_spi_ripples.py /path/to/out_I.dt --nthreads 16

Writes ``<outdir>/spi_ripples_report.txt`` (read this first),
``spi_ripples_diagnostics.npz`` and ``spi_ripples_summary.png``.  Default
``--size 512`` keeps the bundle under ~15 MB; shrink it if that is too much.
"""

import argparse
import os
import textwrap

import numpy as np
import xarray as xr
from ducc0.misc import resize_thread_pool
from scipy.ndimage import map_coordinates

from pfb_imaging.utils.misc import convolve2gaussres, gaussian2d

FWHM = 2 * np.sqrt(2 * np.log(2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dt", help="Path to the restored tree, <output-filename>_<PRODUCT>.dt")
    p.add_argument("--outdir", default=None, help="Where to write the bundle (default: alongside the tree)")
    p.add_argument("--size", type=int, default=512, help="Side of the analysed cutout, in pixels")
    p.add_argument(
        "--centre",
        default=None,
        help="Cutout centre as 'y,x' in pixels. Default is the image centre; aim it at a "
        "diffuse region where the ripples are visible",
    )
    p.add_argument("--psf-size", type=int, default=256, help="Side of the PSF window used for the sidelobe tests")
    p.add_argument("--timeid", type=int, default=None, help="Which timeid to analyse (default: the first)")
    p.add_argument("--drop-bands", type=int, nargs="*", default=(), help="Band ids to exclude, as spifit would")
    p.add_argument("--threshold", type=float, default=2.5, help="SNR cut, matching the spifit run being diagnosed")
    p.add_argument("--pb-min", type=float, default=0.15, help="Beam cut, matching the spifit run being diagnosed")
    p.add_argument(
        "--rms",
        type=float,
        default=None,
        help="Override the noise estimate (Jy/beam) with the value the spifit run reported, "
        "so the mask matches the alpha map being diagnosed",
    )
    p.add_argument("--model-name", default="MODEL")
    p.add_argument("--residual-name", default="RESIDUAL")
    p.add_argument("--image-name", default="IMAGE", help="Stored restored product, used as a consistency check")
    p.add_argument("--nthreads", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=50, help="Gauss-Newton iterations in the alpha fit")
    p.add_argument("--no-png", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def grids(ny, nx):
    """The (X, Y)-ordered coordinate grids convolve2gaussres expects (wiki D19)."""
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    return np.meshgrid(x, y, indexing="ij")


def conv(image, gaussparf, gausspari=None, nthreads=1, pfrac=0.2):
    """restore_products' convolution, on a single (ny, nx) plane."""
    ny, nx = image.shape
    xx, yy = grids(ny, nx)
    out = convolve2gaussres(
        image[None],
        xx,
        yy,
        np.asarray(gaussparf, dtype=float),
        nthreads=nthreads,
        gausspari=None if gausspari is None else np.asarray(gausspari, dtype=float),
        pfrac=pfrac,
        norm_kernel=False,
        yx_order=True,
    )[0]
    return np.ascontiguousarray(out)


def fit_alpha(cube, weights, freqs, nu_ref, maxiter=50, tol=1e-6):
    """Weighted Gauss-Newton fit of I(nu) = I0 (nu/nu_ref)**alpha, per pixel.

    Mirrors spimple.utils.fit_spi._fit_spi_components_impl (beam folded into the
    weights, which is the intrinsic-scale form), vectorised over pixels.

    Args:
        cube: (nband, npix) fluxes.
        weights: (nband, npix) or (nband,) inverse-variance weights.
        freqs: (nband,) frequencies.
        nu_ref: reference frequency.

    Returns:
        (alpha, i0, converged) each (npix,); converged is a bool array.
    """
    cube = np.asarray(cube, dtype=np.float64)
    nband, npix = cube.shape
    w = (freqs / nu_ref).astype(np.float64)[:, None]  # (nband, 1)
    wgt = np.broadcast_to(np.atleast_2d(weights.T).T if weights.ndim > 1 else weights[:, None], (nband, npix))

    # initialise from a log-space linear fit where every band is positive, which
    # is most of the mask; elsewhere fall back to a flat spectrum
    pos = np.all(cube > 0, axis=0)
    alpha = np.full(npix, -0.7)
    lw = np.log(w[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        ly = np.log(np.where(cube > 0, cube, 1.0))
    sw = wgt.sum(axis=0)
    mx = (wgt * lw[:, None]).sum(axis=0) / sw
    my = (wgt * ly).sum(axis=0) / sw
    cov = (wgt * (lw[:, None] - mx) * (ly - my)).sum(axis=0)
    var = (wgt * (lw[:, None] - mx) ** 2).sum(axis=0)
    alpha[pos] = np.where(var > 0, cov / np.maximum(var, 1e-30), -0.7)[pos]
    alpha = np.clip(alpha, -6.0, 6.0)
    i0 = np.maximum((wgt * cube).sum(axis=0) / sw, 1e-12)

    converged = np.zeros(npix, dtype=bool)
    for _ in range(maxiter):
        jac1 = w**alpha  # (nband, npix)
        model = i0 * jac1
        jac0 = model * np.log(w)
        res = cube - model
        h00 = (jac0 * wgt * jac0).sum(axis=0)
        h01 = (jac0 * wgt * jac1).sum(axis=0)
        h11 = (jac1 * wgt * jac1).sum(axis=0)
        jr0 = (jac0 * wgt * res).sum(axis=0)
        jr1 = (jac1 * wgt * res).sum(axis=0)
        det = h00 * h11 - h01**2
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        da = (h11 * jr0 - h01 * jr1) / det
        di = (-h01 * jr0 + h00 * jr1) / det
        step = np.maximum(np.abs(da), np.abs(di / np.maximum(np.abs(i0), 1e-12)))
        alpha = np.clip(alpha + da, -8.0, 8.0)
        i0 = i0 + di
        converged = step < tol
        if converged.all():
            break
    return alpha, i0, converged


def radial_profile(image, nbins=120, centre=None):
    """Azimuthal mean and the radii it was taken at, ignoring NaN."""
    ny, nx = image.shape
    cy, cx = (ny // 2, nx // 2) if centre is None else centre
    y, x = np.mgrid[:ny, :nx]
    r = np.hypot(y - cy, x - cx)
    bins = np.linspace(0, r.max(), nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    flat = image.ravel()
    prof = np.full(nbins, np.nan)
    for b in range(nbins):
        sel = (idx == b) & np.isfinite(flat)
        if sel.any():
            prof[b] = flat[sel].mean()
    return 0.5 * (bins[1:] + bins[:-1]), prof


def power_spectra(image, mask, smooth=8):
    """Radial and angular power spectra of the fine structure in a masked map.

    Returns (wavelengths_px, radial_power, pa_deg, angular_power). The map is
    high-passed mask-aware first, so the astrophysical gradient does not swamp
    the ripples, and apodised so the mask edge does not ring.
    """
    from scipy.ndimage import gaussian_filter

    m = mask.astype(float)
    filled = np.where(mask, image, 0.0)
    num = gaussian_filter(filled * m, smooth)
    den = gaussian_filter(m, smooth)
    hp = (filled - np.where(den > 1e-3, num / np.maximum(den, 1e-9), 0.0)) * m

    n = min(hp.shape)
    hp = hp[:n, :n]
    win = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    hat = np.fft.fftshift(np.abs(np.fft.fft2(hp * win)) ** 2)
    ky = np.fft.fftshift(np.fft.fftfreq(n))[:, None] * np.ones((1, n))
    kx = np.ones((n, 1)) * np.fft.fftshift(np.fft.fftfreq(n))[None, :]
    kr = np.hypot(kx, ky)
    hat[kr < 3.0 / n] = 0.0

    bins = np.geomspace(3.0 / n, 0.5, 41)
    idx = np.digitize(kr.ravel(), bins)
    rad = np.array([hat.ravel()[idx == b].mean() if (idx == b).any() else np.nan for b in range(1, len(bins))])
    kmid = 0.5 * (bins[1:] + bins[:-1])

    pa = np.rad2deg(np.arctan2(ky, kx)) % 180
    band = (kr > 0.01) & (kr < 0.25)
    edges = np.arange(0, 181, 10)
    ang = np.array([hat[band & (pa >= lo) & (pa < lo + 10)].mean() for lo in edges[:-1]])
    return 1.0 / kmid, rad, 0.5 * (edges[1:] + edges[:-1]), ang


def psf_diagnostics(dt, keep, freqs, wsums, psfparsn, gcommon, psf_size, nthreads, say):
    """Diagnostics 1, 2 and 4: sidelobes, their 1/nu scaling, and flux units.

    Returns a dict of arrays for the bundle.
    """
    ref = dt[keep[0]].ds
    _, nyp, nxp = ref.PSF.shape
    h = int(min(psf_size, nyp, nxp)) // 2
    sy = slice(nyp // 2 - h, nyp // 2 + h)
    sx = slice(nxp // 2 - h, nxp // 2 + h)
    n = 2 * h

    psfs, homog = [], []
    for b, node in enumerate(keep):
        p = np.asarray(dt[node].ds.PSF.isel(corr=0, y_psf=sy, x_psf=sx).values, dtype=np.float64) / wsums[b]
        psfs.append(p)
        # exactly what restore does to the residual: PSFPARSN_b -> G
        homog.append(conv(p, gcommon, gausspari=psfparsn[b], nthreads=nthreads))
    psfs = np.stack(psfs)
    homog = np.stack(homog)

    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n // 2, xx - n // 2)
    # "main lobe" = inside the common restoring beam's major axis; everything
    # beyond it is sidelobe as far as the restoration is concerned
    core = r <= gcommon[0]
    wings = r > 1.5 * gcommon[0]

    # --- diagnostic 1: does homogenisation equalise the sidelobes? ----------
    say("DIAGNOSTIC 1 -- do the sidelobes survive homogenisation to the common beam?")
    say("  Each band's PSF is convolved from PSFPARSN_b to PSFPARSF, which is exactly what")
    say("  restore does to RESIDUAL. The cores must agree by construction. If the wings do")
    say("  not, a Gaussian reconvolution cannot equalise them and the residual carries")
    say("  band-dependent structure into the fit.")
    say(f"{'band':>5} {'core rms diff':>15} {'wing rms diff':>15} {'wing rms':>12} {'ratio wing/core':>16}")
    core_diff, wing_diff, wing_rms = [], [], []
    p0 = homog[0]
    p0c = np.sqrt(np.mean(p0[core] ** 2))
    for b in range(len(keep)):
        d = homog[b] - p0
        cd = np.sqrt(np.mean(d[core] ** 2)) / p0c
        wd = np.sqrt(np.mean(d[wings] ** 2)) / p0c
        wr = np.sqrt(np.mean(homog[b][wings] ** 2)) / p0c
        core_diff.append(cd)
        wing_diff.append(wd)
        wing_rms.append(wr)
        say(f"{b:5d} {cd:15.3e} {wd:15.3e} {wr:12.3e} {wd / max(cd, 1e-30):16.1f}")
    say("  Read: wing rms diff >> core rms diff means homogenisation matched the cores and")
    say("  left the sidelobes mismatched -- the mechanism behind the ripples.")
    say()

    # --- diagnostic 2: do the sidelobes scale as 1/nu? ---------------------
    say("DIAGNOSTIC 2 -- do the dirty beam's sidelobes scale as 1/nu?")
    say("  Correlation of band b's PSF wings with band 0's, raw and after rescaling band 0")
    say("  radially by nu_b/nu_0. A jump from raw to rescaled confirms the frequency")
    say("  dependence that turns a static sidelobe pattern into a spectral signal.")
    say(f"{'band':>5} {'nu_b/nu_0':>10} {'corr raw':>10} {'corr rescaled':>15}")
    corr_raw, corr_scaled = [], []
    cy = cx = n // 2
    for b in range(len(keep)):
        s = freqs[b] / freqs[0]
        # PSF_b(x) ~ PSF_0(x * nu_b/nu_0) if the beam shrinks as 1/nu
        coords = np.array([(yy - cy) * s + cy, (xx - cx) * s + cx])
        p0s = map_coordinates(psfs[0], coords, order=1, mode="constant", cval=0.0)
        a, braw, bsc = psfs[b][wings], psfs[0][wings], p0s[wings]

        def cc(u, v):
            u = u - u.mean()
            v = v - v.mean()
            d = np.sqrt((u * u).sum() * (v * v).sum())
            return float((u * v).sum() / d) if d > 0 else np.nan

        corr_raw.append(cc(a, braw))
        corr_scaled.append(cc(a, bsc))
        say(f"{b:5d} {s:10.4f} {corr_raw[-1]:10.4f} {corr_scaled[-1]:15.4f}")
    say()

    # --- diagnostic 4: restoration flux units ------------------------------
    say("DIAGNOSTIC 4 -- clean beam volume vs dirty beam main-lobe volume.")
    say("  MODEL (x) G is Jy per clean beam; RESIDUAL is Jy per dirty beam. The restored sum")
    say("  is only consistent if the ratio below is 1 and constant in frequency. It is")
    say("  neither, so the residual enters the spectrum with a band-dependent scale.")
    xxg, yyg = grids(n, n)
    say(f"{'band':>5} {'dirty mainlobe':>16} {'clean volume':>14} {'ratio':>10} {'vs band 0':>11}")
    eps = []
    for b in range(len(keep)):
        gk = gaussian2d(xxg, yyg, psfparsn[b], normalise=False).T
        dirty_vol = float(psfs[b][core].sum())
        clean_vol = float(gk[core].sum())
        eps.append(dirty_vol / clean_vol if clean_vol else np.nan)
        say(f"{b:5d} {dirty_vol:16.4e} {clean_vol:14.4e} {eps[-1]:10.5f} {eps[-1] / eps[0]:11.5f}")
    spread = (max(eps) - min(eps)) / np.mean(eps)
    say(f"  spread across bands: {100 * spread:.2f} percent of the mean")
    say()

    prof = np.stack([radial_profile(p, nbins=120)[1] for p in psfs])
    rad = radial_profile(psfs[0], nbins=120)[0]
    return {
        "psf_radial_r": rad,
        "psf_radial": prof,
        "psf_homog": homog.astype(np.float32),
        "psf_core_diff": np.array(core_diff),
        "psf_wing_diff": np.array(wing_diff),
        "psf_wing_rms": np.array(wing_rms),
        "psf_corr_raw": np.array(corr_raw),
        "psf_corr_scaled": np.array(corr_scaled),
        "beam_volume_ratio": np.array(eps),
    }


def main():
    args = parse_args()
    resize_thread_pool(args.nthreads)
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.dt.rstrip("/"))) or "."
    os.makedirs(outdir, exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    say(f"spi ripple diagnostics for {args.dt}")
    say("=" * 78)

    # ---- tree, bands, geometry -------------------------------------------
    dt = xr.open_datatree(args.dt, engine="zarr", chunks={})
    band_nodes = [n for n in dt.children if n.startswith("band")]
    if not band_nodes:
        raise SystemExit(f"{args.dt} has no band nodes")
    tids = sorted({int(dt[n].ds.attrs["timeid"]) for n in band_nodes})
    tid = tids[0] if args.timeid is None else args.timeid
    nodes = sorted(
        (n for n in band_nodes if int(dt[n].ds.attrs["timeid"]) == tid),
        key=lambda n: int(dt[n].ds.attrs["bandid"]),
    )
    dropped = set(args.drop_bands or ())
    keep = [n for n in nodes if int(dt[n].ds.attrs["bandid"]) not in dropped]
    keep = [n for n in keep if float(np.sum(dt[n].ds.WSUM.values)) > 0]
    if len(keep) < 2:
        raise SystemExit("Need at least two live bands")

    ref = dt[keep[0]].ds
    ncorr, ny, nx = ref[args.model_name].shape
    cell_deg = np.rad2deg(float(ref.attrs["cell_rad"]))
    nband = len(keep)
    freqs = np.array([float(dt[n].ds.attrs["freq_out"]) for n in keep])
    wsums = np.array([float(np.atleast_1d(dt[n].ds.WSUM.values)[0]) for n in keep])
    psfparsn = np.stack([np.asarray(dt[n].ds.PSFPARSN.values, dtype=float)[0] for n in keep])
    has_psfparsf = "PSFPARSF" in ref
    gcommon = np.asarray(ref.PSFPARSF.values, dtype=float)[0] if has_psfparsf else psfparsn.max(axis=0)

    say(f"timeid {tid}: {nband} live bands of {len(nodes)}, image {ny} x {nx}, cell {cell_deg * 3600:.3f} arcsec")
    say(f"frequencies {freqs[0] / 1e6:.1f} - {freqs[-1] / 1e6:.1f} MHz (ratio {freqs[-1] / freqs[0]:.3f})")
    say(
        f"common restoring beam PSFPARSF = ({gcommon[0]:.3f}, {gcommon[1]:.3f}) px, pa {np.rad2deg(gcommon[2]):.2f} deg"
    )
    if not has_psfparsf:
        say("  WARNING: no PSFPARSF -- the tree was not homogenised; using max axes as the target")
    say()
    say("per band: native beam PSFPARSN, and how far it is from the common target")
    say(f"{'band':>5} {'freq/MHz':>10} {'emaj':>8} {'emin':>8} {'pa/deg':>8} {'wsum':>11} {'emaj_G/emaj_b':>14}")
    for b, n in enumerate(keep):
        say(
            f"{int(dt[n].ds.attrs['bandid']):5d} {freqs[b] / 1e6:10.2f} {psfparsn[b, 0]:8.3f} {psfparsn[b, 1]:8.3f} "
            f"{np.rad2deg(psfparsn[b, 2]):8.2f} {wsums[b]:11.4e} {gcommon[0] / psfparsn[b, 0]:14.4f}"
        )
    say()

    # ---- cutout window ----------------------------------------------------
    if args.centre:
        cy, cx = (int(v) for v in args.centre.split(","))
    else:
        cy, cx = ny // 2, nx // 2
    # clamp so a cutout larger than the image, or a centre near an edge, still works
    size = int(min(args.size, ny, nx))
    half = size // 2
    size = 2 * half
    cy = int(np.clip(cy, half, ny - half))
    cx = int(np.clip(cx, half, nx - half))
    margin = int(np.ceil(4 * gcommon[0]))
    y0, y1 = max(0, cy - half - margin), min(ny, cy + half + margin)
    x0, x1 = max(0, cx - half - margin), min(nx, cx + half + margin)
    ty0, tx0 = cy - half - y0, cx - half - x0
    trim = (slice(ty0, ty0 + size), slice(tx0, tx0 + size))
    say(f"cutout centred on (y={cy}, x={cx}), {size} px analysed with a {margin} px convolution margin")
    if size < args.size:
        say(f"  (--size {args.size} was clamped to the image)")
    say()

    bundle = psf_diagnostics(dt, keep, freqs, wsums, psfparsn, gcommon, args.psf_size, args.nthreads, say)

    # ---- rebuild the restored image, term by term -------------------------
    say("Rebuilding the restored cutout term by term (this repeats what restore did).")
    sel = dict(corr=0, y=slice(y0, y1), x=slice(x0, x1))
    mconv, rconv, rconv1, beams = [], [], [], []
    for b, node in enumerate(keep):
        ds = dt[node].ds
        model = np.asarray(ds[args.model_name].isel(**sel).values, dtype=np.float64)
        resid = np.asarray(ds[args.residual_name].isel(**sel).values, dtype=np.float64) / wsums[b]
        beams.append(np.asarray(ds.BEAM.isel(**sel).values, dtype=np.float64)[trim])
        mconv.append(conv(model, gcommon, nthreads=args.nthreads)[trim])
        rconv.append(conv(resid, gcommon, gausspari=psfparsn[b], nthreads=args.nthreads)[trim])
        rconv1.append(conv(resid, gcommon, gausspari=psfparsn[b], nthreads=args.nthreads, pfrac=1.0)[trim])
    mconv, rconv, rconv1 = np.stack(mconv), np.stack(rconv), np.stack(rconv1)
    beams = np.stack(beams)

    with np.errstate(invalid="ignore", divide="ignore"):
        image_full = mconv + np.where(beams > 0, rconv / beams, 0.0)
        image_p1 = mconv + np.where(beams > 0, rconv1 / beams, 0.0)

    # consistency: does the rebuild match what restore stored?
    if args.image_name in ref:
        stored = np.stack(
            [np.asarray(dt[n].ds[args.image_name].isel(**sel).values, dtype=np.float64)[trim] for n in keep]
        )
        err = np.nanmax(np.abs(image_full - stored)) / np.nanmax(np.abs(stored))
        say(f"  rebuild vs stored {args.image_name}: max relative difference {err:.3e}")
        if err > 1e-3:
            say("  WARNING: the rebuild does not match the tree. Was restore run at a different")
            say("  --gausspar, or with a different pfrac? Interpret the alpha maps with care.")
        del stored
    say()

    # ---- mask, matching what spifit would cut on --------------------------
    # spifit takes the rms as std(RESIDUAL/WSUM); reproduce that, but report a
    # MAD estimate too -- over a cutout full of source the std is signal, not noise
    rms_b = np.array([float(np.std(rconv[b])) for b in range(nband)])
    mad_b = np.array([float(1.4826 * np.median(np.abs(rconv[b] - np.median(rconv[b])))) for b in range(nband)])
    weights_b = wsums / wsums.max()
    rms_mfs = float(np.sum(rms_b * weights_b) / weights_b.sum())
    mad_mfs = float(np.sum(mad_b * weights_b) / weights_b.sum())
    if args.rms is not None:
        rms_mfs = args.rms
        say(f"mask: rms {rms_mfs:.4e} Jy/beam (from --rms)")
    else:
        say(f"mask: rms {rms_mfs:.4e} Jy/beam (std, as spifit computes it); MAD estimate {mad_mfs:.4e}")
        if rms_mfs > 3 * mad_mfs:
            say("      NOTE: std is much larger than MAD, so it is measuring source not noise.")
            say("      Pass --rms with the value your spifit run reported to match its mask.")
    thresh = args.threshold * rms_mfs
    apparent = image_full * beams  # spifit compares on the apparent scale
    mask = (beams.min(axis=0) > args.pb_min) & np.isfinite(apparent).all(axis=0) & (apparent.min(axis=0) > thresh)
    say(f"      threshold {args.threshold} x rms = {thresh:.4e} Jy/beam")
    say(f"      {int(mask.sum())} of {mask.size} pixels fitted ({100 * mask.mean():.1f} percent)")
    if mask.sum() < 500:
        say("      WARNING: very few pixels. Lower --threshold, pass --rms, or move --centre.")
    say()

    # ---- diagnostics 3 and 5: alpha with and without the residual ---------
    idx = np.argwhere(mask)
    nu_ref = float(np.sum(freqs * wsums) / np.sum(wsums))
    pb = beams[:, idx[:, 0], idx[:, 1]]
    wfit = weights_b[:, None] * pb**2  # intrinsic-scale weights, as spifit uses

    say("DIAGNOSTIC 3 -- what does the residual term do to alpha?")
    say(f"  Reference frequency {nu_ref / 1e6:.2f} MHz. Fitting {len(idx)} pixels three ways.")
    maps, good = {}, np.ones(len(idx), dtype=bool)
    for name, cube in (("full", image_full), ("model", mconv), ("pfrac1", image_p1)):
        a, i0, ok = fit_alpha(cube[:, idx[:, 0], idx[:, 1]], wfit, freqs, nu_ref, maxiter=args.maxiter)
        m = np.full(mask.shape, np.nan, dtype=np.float32)
        m[idx[:, 0], idx[:, 1]] = np.where(ok, a, np.nan)
        maps[name] = m
        good &= ok
        say(
            f"  alpha[{name:6}] median {np.nanmedian(a[ok]) if ok.any() else np.nan:7.3f}  "
            f"std {np.nanstd(a[ok]) if ok.any() else np.nan:7.3f}  "
            f"unconverged {100 * (1 - ok.mean()):5.1f} percent"
        )
    # compare only where every fit converged, else the statistic is contaminated
    # by pixels the model-only fit could not constrain (no model flux there)
    dalpha = maps["full"] - maps["model"]
    dg = dalpha[idx[good, 0], idx[good, 1]]
    say(f"  comparing on the {int(good.sum())} pixels ({100 * good.mean():.1f} percent) where all three converged")
    say(f"  alpha[full] - alpha[model]: median {np.nanmedian(dg):.4f}, std {np.nanstd(dg):.4f}")
    a_model = maps["model"][idx[good, 0], idx[good, 1]]
    say(f"  for reference, alpha[model] over the same pixels has std {np.nanstd(a_model):.4f}")
    say("  The first std is the spectral index the residual term invents; the second is the")
    say("  spread the deconvolved sky actually has. If the first is comparable to or larger")
    say("  than the second, the ripples are the residual and not the sky.")
    say()

    resfrac = np.full(mask.shape, np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        rf = np.abs(rconv / np.maximum(beams, 1e-12)) / (np.abs(mconv) + np.abs(rconv / np.maximum(beams, 1e-12)))
    resfrac[mask] = np.nanmedian(rf, axis=0)[mask]
    say(f"  median residual flux fraction over the mask: {np.nanmedian(resfrac[mask]):.3f}")
    say("  (the fraction of each pixel's restored flux that comes from the residual term)")
    say()

    say("DIAGNOSTIC 5 -- is the pfrac=0.2 padding in restore_products contributing?")
    dp = maps["pfrac1"] - maps["full"]
    say(f"  alpha[pfrac=1] - alpha[pfrac=0.2]: median {np.nanmedian(dp):.4f}, std {np.nanstd(dp):.4f}")
    say("  A std comparable to the ripple amplitude means FFT wrap-around is a real")
    say("  contributor and restore_products should pad more.")
    say()

    # ---- diagnostic 6: geometry of the ripples ---------------------------
    say("DIAGNOSTIC 6 -- geometry: is it striping (PSF) or rings (primary beam)?")
    spectra = {}
    for name in ("full", "model"):
        wl, rad, pa, ang = power_spectra(np.nan_to_num(maps[name]), mask)
        spectra[name] = (wl, rad, pa, ang)
    wl, rad_full, pa, ang_full = spectra["full"]
    peak = int(np.nanargmax(rad_full))
    say(
        f"  alpha[full] fine-structure power peaks at {wl[peak]:.1f} px = "
        f"{wl[peak] / gcommon[0]:.2f} restoring beams = {wl[peak] * cell_deg * 3600:.1f} arcsec"
    )
    aniso = np.nanmax(ang_full) / np.nanmin(ang_full)
    say(f"  angular anisotropy {aniso:.2f} (peak at PA {pa[int(np.nanargmax(ang_full))]:.0f} deg)")
    say("  A high anisotropy at a fixed PA is PSF striping. A flat angular spectrum with a")
    say("  strong radial signal about the pointing centre is a primary beam error.")
    rr, aprof = radial_profile(np.where(mask, maps["full"], np.nan), nbins=80)
    say()

    # ---- write the bundle -------------------------------------------------
    bundle.update(
        alpha_full=maps["full"],
        alpha_model=maps["model"],
        alpha_pfrac1=maps["pfrac1"],
        dalpha_residual=dalpha.astype(np.float32),
        dalpha_pfrac=dp.astype(np.float32),
        residual_fraction=resfrac,
        mask=mask,
        beam_band0=beams[0].astype(np.float32),
        freqs=freqs,
        wsums=wsums,
        psfparsn=psfparsn,
        gcommon=gcommon,
        cell_deg=np.array(cell_deg),
        nu_ref=np.array(nu_ref),
        rms_per_band=rms_b,
        centre=np.array([cy, cx]),
        image_shape=np.array([ny, nx]),
        ps_wavelength=wl,
        ps_radial_full=rad_full,
        ps_radial_model=spectra["model"][1],
        ps_pa=pa,
        ps_angular_full=ang_full,
        ps_angular_model=spectra["model"][3],
        alpha_radial_r=rr,
        alpha_radial=aprof,
    )
    npz = os.path.join(outdir, "spi_ripples_diagnostics.npz")
    np.savez_compressed(npz, **bundle)
    say(f"wrote {npz} ({os.path.getsize(npz) / 1e6:.1f} MB)")

    if not args.no_png:
        write_png(os.path.join(outdir, "spi_ripples_summary.png"), bundle, say)

    report = os.path.join(outdir, "spi_ripples_report.txt")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {report}")
    print(
        textwrap.dedent(f"""
        Pull these back:
          {report}
          {npz}
          {os.path.join(outdir, "spi_ripples_summary.png")}
    """)
    )


def write_png(path, bundle, say):
    """Panels: the alpha maps, what the residual adds, and the spectra."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    mask = bundle["mask"]

    def show(a, key, title, cmap="plasma", robust=True):
        arr = np.where(mask, bundle[key], np.nan)
        lo, hi = np.nanpercentile(arr, [2, 98]) if robust else (None, None)
        im = a.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(title, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046)

    show(ax[0, 0], "alpha_full", "alpha from the full restored image")
    show(ax[0, 1], "alpha_model", "alpha from MODEL (x) G only")
    show(ax[0, 2], "dalpha_residual", "what the residual term adds to alpha", cmap="RdBu_r")
    show(ax[0, 3], "residual_fraction", "residual fraction of the restored flux", cmap="viridis")

    ax[1, 0].plot(bundle["ps_wavelength"], bundle["ps_radial_full"], label="alpha full")
    ax[1, 0].plot(bundle["ps_wavelength"], bundle["ps_radial_model"], label="alpha model")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("wavelength [px]")
    ax[1, 0].set_title("radial power spectrum of alpha", fontsize=10)
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(bundle["ps_pa"], bundle["ps_angular_full"], label="alpha full")
    ax[1, 1].plot(bundle["ps_pa"], bundle["ps_angular_model"], label="alpha model")
    ax[1, 1].set_xlabel("PA of the wavevector [deg]")
    ax[1, 1].set_title("angular power (striping direction)", fontsize=10)
    ax[1, 1].legend(fontsize=8)

    # band 0 is the reference and identically zero; plotting it flattens the axis
    nb = len(bundle["freqs"])
    ax[1, 2].semilogy(range(1, nb), bundle["psf_core_diff"][1:], "o-", label="core")
    ax[1, 2].semilogy(range(1, nb), bundle["psf_wing_diff"][1:], "s-", label="wings")
    ax[1, 2].set_xlabel("band")
    ax[1, 2].set_title("PSF difference from band 0 after homogenisation", fontsize=10)
    ax[1, 2].legend(fontsize=8)

    ax[1, 3].plot(bundle["freqs"] / 1e6, bundle["psf_corr_raw"], "o-", label="raw")
    ax[1, 3].plot(bundle["freqs"] / 1e6, bundle["psf_corr_scaled"], "s-", label="rescaled by nu")
    ax[1, 3].set_xlabel("frequency [MHz]")
    ax[1, 3].set_title("sidelobe correlation with band 0", fontsize=10)
    ax[1, 3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    say(f"wrote {path}")


if __name__ == "__main__":
    main()
