#!/usr/bin/env python
"""Compare ``pfb imager`` against WSClean on the same data with matched settings.

Runs (or reuses) both imagers from a *single* parameter set, aligns their FITS
products via the WCS, and reports how closely they agree.  Deliberate
differences that the comparison has to absorb:

1. **PSF size.**  WSClean writes the PSF at the image size; ``pfb imager``
   writes it at ``psf_oversize * image size``.  Products are aligned on
   ``CRPIX`` and cropped to their common extent, so only the shared central
   region is compared.
2. **Half-pixel coordinate conventions.**  The reference pixel is located from
   ``CRPIX`` in each header rather than assumed, and any residual ``CRVAL``
   disagreement is reported as a pixel offset instead of silently shifting the
   comparison.  A sub-pixel cross-correlation shift is also measured -- that is
   the check that catches a real half-pixel gridding error, as opposed to a
   cosmetic difference in how the header is written.
3. **The wgridder n-term.**  ``pfb imager`` grids with ``divide_by_n=False`` and
   folds ``1/n`` into the stored ``BEAM`` (wiki D22), so ``BEAM`` is ``B/n`` and
   is ``1/n`` exactly when no beam model is used.  ``--beam-convention`` selects
   whether the pfb image is multiplied by (or divided by) that beam before
   comparing; ``all`` reports every convention so the matching one is evident
   from the numbers.

Weighting comparison (``--robustness``, the single knob) adds three more things
that have to be handled explicitly:

4. **Reaching uniform, and switching off both cell filters.**  pfb goes uniform at
   ``robustness <= -2`` (the limit of its Briggs formula) where wsclean takes an
   explicit ``-weight uniform``.  Both imagers protect near-empty uv cells by
   default and must be disabled to compare raw grids: pfb's
   ``--filter-counts-level`` (default 5.0) is set to 0, and wsclean's
   ``-weighting-rank-filter`` -- which is **active out of the box** at level 3.0,
   window 16 -- is set to 0.
5. **Matching the uv grid size.**  ``psf_oversize`` does *not* size pfb's density
   grid: ``core/imager.py`` hardcodes ``min_padding = 1.7`` for the COUNTS grid and
   rounds up to an even number, while wsclean uses the padded image size rounded
   up to a multiple of 4.  For ``nx=462`` that is 786 vs 788, and they cannot be
   made equal by choosing ``-padding``.  Since different grid sizes mean different
   uv cell sizes, the *images* cannot agree either, so this is not cosmetic.
   ``--wsclean-super-weight auto`` shrinks wsclean's grid onto pfb's (wsclean
   builds it at ``round(padded / super_weight)``); the alternative is an image size
   where the two rounding rules coincide, which the warning reports.
6. **fftshift / reflection conventions.**  pfb folds to ``v >= 0`` and offsets the
   v index by ``ny/2``; wsclean folds to ``v >= 0`` at index 0 and saves the
   Hermitian-symmetrised full plane; and pass 1 grids ``(-u, v)`` because
   ``wgridder_conventions`` yields ``usign=-1``, reflecting the u axis.  Instead of
   trusting that chain, ``--grid-transform auto`` scores every flip/roll candidate
   and reports the ranking, so the convention is established from the data.

Multi-band imaging (``--nband``) adds the frequency axis:

7. **Channel slicing.**  wsclean's ``-channels-out`` splits by equal channel
   *count*; pfb has no band-count option -- it slices ``channels_per_image``-wide
   pieces and assigns each to the nearest of ``nband`` centres spaced uniformly in
   *frequency*.  When ``nband`` divides ``nchan`` the two coincide exactly (band
   centres agree to 0 Hz).  When it does not, the imagers put the remainder at
   opposite ends -- pfb's short piece lands in the **last** band, wsclean's in the
   **first** -- so the bands cover different channels.  ``--nband`` therefore exits
   with the list of even splits unless ``--allow-uneven-bands`` is given, and the
   band-alignment table quantifies the damage either way.
8. **Frequency weighting mode.**  pfb's band-resolved groupings (``per-band``, and
   ``per-band-time`` which ``concat_row`` collapses to it) pair with wsclean's
   default ``-no-mf-weighting``: one density grid per output band.  The
   band-collapsed ones (``mfs``, ``per-time``) pair with ``-mf-weighting``: a single
   grid built from every channel, which wsclean then writes as one unsuffixed
   weights file rather than one per band.
9. **Band-integrated wsum.**  wsclean's MFS header copies band 0's ``WSCNORMF``
   verbatim; the band-summed weight is in ``WSCIMGWG``, which is what the
   provenance block compares against pfb's total ``WSUM``.

The grids are compared as imaging weights ``W_k`` (what wsclean's ``-save-weights``
writes), derived from pfb's density grid ``D_k`` with the formula for the
weighting in play.  pfb's reduced counts grid is never written to the ``.dt``, so
it is re-summed from the ``.scratch`` per-piece ``COUNTS`` -- hence
``--keep-scratch`` in the generated pfb command.  Note that natural weighting
makes ``W_k = 1`` on every sampled cell, so a natural-weighted grid comparison
only tests the cell support, not the density.

Metrics per product (computed on the aligned overlap, in float64):

* ``max|diff|`` and ``rms diff``, both absolute and relative to the reference
  peak, plus ``rms diff / rms(ref)`` -- the honest test, since a difference well
  below the image noise is irrelevant.
* ``scale`` = ``sum(pfb*wsc)/sum(wsc^2)``: a normalisation mismatch (wsum
  convention, weight definition) shows up here as a departure from 1 and
  nowhere else as clearly.
* ``shift``: sub-pixel offset of pfb relative to WSClean from a parabolic fit to
  the cross-correlation peak.
* ``wsum``: pfb's ``WSUM`` header against WSClean's ``WSCNORMF``.  These agreeing
  means flagging, weight column and Stokes-I weight combination all agree, which
  is a precondition for the images agreeing at all.

Run (from the repo root)::

    # compare products that already exist on disk
    uv run python scripts/compare_wsclean.py \
        --ms ~/data/mkat_test_subsets/sgra_scan42_700to800MHz.ms \
        --pfb-prefix ~/projects/GC/output/sgra/test_single_precision \
        --wsclean-prefix ~/projects/GC/wsclean/natural_mfs \
        --run none

    # run whichever imager has no output yet, then compare
    uv run python scripts/compare_wsclean.py --ms /path/to.ms --outdir /tmp/cmp

The defaults reproduce the naturally-weighted single-precision comparison on the
``sgra_scan42_700to800MHz.ms`` test subset; ``--dry-run`` prints both commands
without executing them.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from astropy.io import fits

# Okabe-Ito, matching scripts/deconv_qa.py
C_PFB = "#0072B2"
C_WSC = "#E69F00"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--ms", required=True, help="Measurement set (MSv2 layout readable by both imagers)")
    g.add_argument("--data-column", default="DATA")
    g.add_argument(
        "--weight-column",
        default="WEIGHT_SPECTRUM",
        help="pfb --weight-column. WSClean has no equivalent option: it uses "
        "WEIGHT_SPECTRUM when present and WEIGHT otherwise, so anything else "
        "here is flagged as an unmatched setting.",
    )

    g = p.add_argument_group("imaging (mapped onto both imagers)")
    g.add_argument("--cell-size", type=float, default=27.615, help="Cell size in arcsec")
    g.add_argument("--nx", type=int, default=462)
    g.add_argument("--ny", type=int, default=462)
    g.add_argument("--epsilon", type=float, default=1e-5, help="pfb --epsilon / wsclean -wgridder-accuracy")
    g.add_argument("--nthreads", type=int, default=8, help="pfb --nthreads / wsclean -j")
    g.add_argument("--nworkers", type=int, default=1, help="pfb --nworkers (no wsclean equivalent)")
    g.add_argument(
        "--psf-oversize",
        type=float,
        default=1.6,
        help="pfb --psf-oversize. Also used for wsclean -padding unless "
        "--wsclean-padding is given. These are not the same quantity: pfb sizes "
        "the PSF image, wsclean pads its FFT grid (ducc pads the dirty gridding "
        "internally regardless).",
    )
    g.add_argument("--wsclean-padding", type=float, default=None, help="Override wsclean -padding")
    g.add_argument(
        "--nband",
        type=int,
        default=1,
        help="Number of output bands. wsclean takes -channels-out directly; pfb "
        "infers its band count from --channels-per-image, which is set to "
        "nchan/nband and must divide exactly (see the module docstring).",
    )
    g.add_argument(
        "--integrations-per-image",
        type=int,
        default=-1,
        help="pfb --integrations-per-image: how many time integrations each pass-1 "
        "time chunk holds (-1 -> one chunk per scan). concat_row folds every chunk "
        "back into one image, so results must not depend on this. wsclean has no "
        "counterpart: -intervals-out stays at 1 and is the invariant reference.",
    )
    g.add_argument(
        "--ipi-sweep",
        default=None,
        help="Comma-separated --integrations-per-image values to sweep, each pfb run "
        "compared against one shared wsclean reference, plus an invariance summary. "
        "This is the concat_row weighting test. Use --ipi-sweep=-1,56,45 (with the "
        "equals sign) when the list starts with -1, or argparse reads it as a flag.",
    )
    g.add_argument(
        "--allow-uneven-bands",
        action="store_true",
        help="Compare anyway when nband does not divide nchan, instead of exiting. "
        "The band-alignment table then quantifies the disagreement.",
    )
    g.add_argument(
        "--weight-grouping",
        choices=("per-band-time", "mfs", "per-band", "per-time"),
        default="per-band-time",
        help="pfb --weight-grouping. Band-resolved groupings (per-band, and "
        "per-band-time which concat_row collapses to it) pair with wsclean's "
        "default -no-mf-weighting; band-collapsed ones (mfs, per-time) pair with "
        "-mf-weighting.",
    )
    g.add_argument(
        "--robustness",
        type=float,
        default=None,
        help="Briggs robustness, the single weighting knob. Omitted -> natural "
        "(wsclean -weight natural); <= -2 -> uniform (wsclean -weight uniform), "
        "since pfb reaches uniform as the limit of its Briggs formula; anything "
        "else -> wsclean -weight briggs <R>.",
    )
    g.add_argument(
        "--filter-counts-level",
        type=float,
        default=0.0,
        help="pfb --filter-counts-level. Defaults to 0 (off) so the raw density "
        "grids can be compared; pfb's own default is 5.0.",
    )
    g.add_argument(
        "--npix-super",
        type=int,
        default=0,
        help="pfb --npix-super (super-uniform box half-size). 0 -> standard uniform.",
    )
    g.add_argument(
        "--wsclean-rank-filter",
        type=float,
        default=0.0,
        help="wsclean -weighting-rank-filter. Defaults to 0 (off) to match "
        "--filter-counts-level 0; wsclean's own default is 3.0 and IS applied "
        "out of the box, which would clip its weight grid but not pfb's.",
    )
    g.add_argument(
        "--wsclean-super-weight",
        default="auto",
        help="wsclean -super-weight. 'auto' picks the value that makes wsclean's "
        "weight grid the same size as pfb's (see the module docstring); 'none' "
        "omits the flag; a float is passed through.",
    )
    g.add_argument("--precision", choices=("single", "double"), default="single", help="pfb --precision")
    g.add_argument("--double-accum", action="store_true", help="pfb --double-accum (default: --no-double-accum)")
    g.add_argument("--wgt-mode", choices=("l2", "minvar"), default="minvar", help="minvar matches wsclean Stokes I")
    g.add_argument("--product", default="I", help="pfb --product (only I is comparable to wsclean)")

    g = p.add_argument_group("paths")
    g.add_argument("--outdir", default=".", help="Base directory for the default prefixes")
    g.add_argument("--tag", default="cmp", help="Basename used by the default prefixes")
    g.add_argument("--pfb-prefix", default=None, help="pfb --output-filename (default <outdir>/pfb/<tag>)")
    g.add_argument("--wsclean-prefix", default=None, help="wsclean -name (default <outdir>/wsclean/<tag>)")
    g.add_argument("--pfb-bin", default="pfb")
    g.add_argument("--wsclean-bin", default="wsclean")

    g = p.add_argument_group("execution")
    g.add_argument(
        "--run",
        choices=("auto", "both", "pfb", "wsclean", "none"),
        default="auto",
        help="Which imagers to (re)run. auto -> only those whose output is missing.",
    )
    g.add_argument("--dry-run", action="store_true", help="Print the commands and exit")
    g.add_argument("--pfb-extra", default="", help="Extra args appended to the pfb command (quoted string)")
    g.add_argument("--wsclean-extra", default="", help="Extra args appended to the wsclean command (quoted string)")

    g = p.add_argument_group("comparison")
    g.add_argument("--timeid", type=int, default=0, help="pfb time chunk to compare")
    g.add_argument(
        "--beam-convention",
        choices=("none", "multiply", "divide", "all"),
        default="none",
        help="Apply the pfb BEAM (=B/n, wiki D22) to the pfb image before "
        "comparing. Defaults to 'none' because wsclean also grids with "
        "divide_by_n=False, which 'all' established by reporting every "
        "convention side by side; use 'all' to re-check that.",
    )
    g.add_argument("--psf-zoom", type=int, default=128, help="Half-width in pixels of the plotted PSF zoom")
    g.add_argument(
        "--compare-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compare the uv weight grids. Requires wsclean -save-weights "
        "and pfb --keep-scratch (added to the generated pfb command), since the "
        "reduced counts grid is never written to the .dt.",
    )
    g.add_argument(
        "--grid-transform",
        default="auto",
        help="Index transform taking wsclean's weight FITS into pfb's (u, v) "
        "grid layout, as 'flipu:<0|1>,rollv:<int>'. 'auto' scores every candidate "
        "and uses the best, which is also how the fftshift/reflection convention "
        "gets established in the first place.",
    )
    g.add_argument("--plot", default=None, help="Output PNG (default <outdir>/compare_wsclean.png)")
    g.add_argument("--no-plot", action="store_true")
    g.add_argument("--json", default=None, help="Also dump the metrics to this JSON file")
    g.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def prefixes(a):
    """Resolve the pfb and wsclean output prefixes, creating their directories."""
    pfb = a.pfb_prefix or os.path.join(a.outdir, "pfb", a.tag)
    wsc = a.wsclean_prefix or os.path.join(a.outdir, "wsclean", a.tag)
    pfb, wsc = os.path.expanduser(pfb), os.path.expanduser(wsc)
    for pre in (pfb, wsc):
        os.makedirs(os.path.dirname(os.path.abspath(pre)), exist_ok=True)
    return pfb, wsc


# core/imager.py hardcodes the COUNTS uv-grid padding: it is NOT psf_oversize,
# which only sizes the PSF image.  Kept in sync manually; the actual COUNTS shape
# read back from the .scratch store is checked against it.
PFB_MIN_PADDING = 1.7


def pfb_counts_size(npix):
    """Size of pfb's padded COUNTS uv-grid (core/imager.py: min_padding rule)."""
    n = int(np.ceil(PFB_MIN_PADDING * npix))
    return n + n % 2


def wsclean_padded_size(npix, padding):
    """wsclean's padded image size: npix*padding rounded up to a multiple of 4."""
    return 4 * int(np.ceil(npix * padding / 4.0))


def min_padding_for_counts(npix):
    """Smallest wsclean -padding whose grid is at least pfb's COUNTS grid.

    ``-super-weight`` can only make wsclean's weight grid coarser, so its padded
    grid has to start out no smaller than pfb's.  The half-pixel slack keeps the
    round-up-to-a-multiple-of-4 insensitive to floating point.
    """
    p4 = 4 * int(np.ceil(pfb_counts_size(npix) / 4.0))
    return (p4 - 0.5) / npix


def super_weight_for_match(npix, padding):
    """wsclean -super-weight that shrinks its weight grid onto pfb's COUNTS grid.

    wsclean builds the weight grid at ``round(padded / super_weight)``, so the
    exact ratio lands it on pfb's ``nx_pad``.  Without this the two density grids
    live on different uv cell sizes and are not comparable cell-by-cell -- and,
    more importantly, the uniformly weighted *images* cannot agree either.

    Returns ``None`` when the sizes already match.
    """
    target = pfb_counts_size(npix)
    padded = wsclean_padded_size(npix, padding)
    if padded == target:
        return None
    if padded < target:
        raise ValueError(
            f"wsclean's padded grid ({padded}) is smaller than pfb's COUNTS grid "
            f"({target}); raise --wsclean-padding to at least {target / npix:.6f}"
        )
    return padded / target


def compatible_npix(npix, padding, span=40):
    """Image sizes near ``npix`` whose pfb COUNTS grid wsclean can hit exactly.

    pfb rounds its uv grid up to an even number, wsclean to a multiple of 4, so
    the two only coincide for some image sizes.  Reported as an alternative to
    ``-super-weight`` in the size-mismatch warning.
    """
    out = []
    for n in range(max(2, npix - span), npix + span + 1, 2):
        if pfb_counts_size(n) == wsclean_padded_size(n, padding):
            out.append(n)
    return out


def ms_nchan(ms):
    """Channel count of the MS's single spectral window."""
    from casacore.tables import table  # deferred: heavy, and only needed for --nband > 1

    with table(f"{os.path.expanduser(ms)}/SPECTRAL_WINDOW", ack=False) as t:
        cf = t.getcol("CHAN_FREQ")
    if cf.shape[0] != 1:
        sys.exit(f"--nband needs a single-spw MS; this one has {cf.shape[0]}")
    return int(cf.shape[1])


_BAND_PLAN = {}


def band_plan(a):
    """Memoised :func:`_band_plan` -- it reads the MS and warns at most once."""
    key = (os.path.expanduser(a.ms), a.nband, a.allow_uneven_bands)
    if key not in _BAND_PLAN:
        _BAND_PLAN[key] = _band_plan(a)
    return _BAND_PLAN[key]


def _band_plan(a):
    """``(channels_per_image, nchan)`` for ``--nband``, checking both split rules.

    wsclean's ``-channels-out`` divides the channels into groups of equal *count*.
    pfb has two mechanisms that both have to line up with that: pass 1 slices fine
    pieces by channel count (``channels_per_image``), and each piece is assigned to
    the output band with the nearest centre, where the centres come from a
    ``linspace`` over the *frequency* span.  When ``nband`` divides ``nchan`` the
    two coincide exactly; when it does not, the slicing leaves a short final piece
    and the frequency-uniform centres no longer sit at the channel-group centres,
    so band *labels* drift even if the channel sets happen to match.  pfb also
    derives its band count as ``ceil((fmax-fmin)/(width*cpi)) = ceil((nchan-1)/cpi)``,
    which is checked rather than assumed.
    """
    if a.nband == 1:
        return -1, None
    nchan = ms_nchan(a.ms)
    cpi = int(np.ceil(nchan / a.nband))
    implied = int(np.ceil((nchan - 1) / cpi))
    if implied != a.nband:
        sys.exit(f"--channels-per-image {cpi} makes pfb choose {implied} bands, not {a.nband}")
    if nchan % a.nband:
        ok = [d for d in range(1, nchan + 1) if nchan % d == 0]
        msg = (
            f"{nchan} channels do not divide into {a.nband} bands: pfb slices "
            f"{cpi}-channel pieces (the last one short) while its band centres stay "
            f"uniform in frequency, so band labels and possibly channel sets differ "
            f"from wsclean's equal-count split. Even splits: {ok}."
        )
        if not a.allow_uneven_bands:
            sys.exit(f"{msg} Pass --allow-uneven-bands to compare anyway.")
        print(f"  WARNING: {msg}")
    return cpi, nchan


def collapses_bands(grouping):
    """Does this pfb weight-grouping sum counts over the band axis?

    ``concat_row`` (on by default, and never disabled here) collapses the time
    axis, mapping ``per-band-time -> per-band`` and ``per-time -> mfs``
    (``core/imager.py``), so only the band axis distinguishes the groupings.
    """
    return grouping in ("mfs", "per-time")


def weight_mode(robustness):
    """(pfb robustness arg, wsclean -weight args, name) for a robustness value."""
    if robustness is None:
        return None, ["-weight", "natural"], "natural"
    if robustness <= -2:
        return robustness, ["-weight", "uniform"], "uniform"
    return robustness, ["-weight", "briggs", repr(robustness)], f"briggs {robustness}"


def pfb_cmd(a, prefix):
    cmd = [
        a.pfb_bin,
        "imager",
        "--ms",
        os.path.expanduser(a.ms),
        "--output-filename",
        prefix,
        "--data-column",
        a.data_column,
        "--weight-column",
        a.weight_column,
        "--product",
        a.product,
        "--integrations-per-image",
        str(a.integrations_per_image),
        "--channels-per-image",
        str(band_plan(a)[0]),
        "--cell-size",
        repr(a.cell_size),
        "--nx",
        str(a.nx),
        "--ny",
        str(a.ny),
        "--nworkers",
        str(a.nworkers),
        "--nthreads",
        str(a.nthreads),
        "--wgt-mode",
        a.wgt_mode,
        "--precision",
        a.precision,
        "--epsilon",
        repr(a.epsilon),
        "--psf-oversize",
        repr(a.psf_oversize),
        "--double-accum" if a.double_accum else "--no-double-accum",
        "--overwrite",
        # the reduced counts grid only ever lives in the driver's memory, so the
        # weight-grid comparison has to rebuild it from the per-piece COUNTS
        "--keep-scratch" if a.compare_weights else "--no-keep-scratch",
        # the per-band planes are compared from the cube; a single band needs only mfs
        "--fits-cubes" if a.nband > 1 else "--no-fits-cubes",
    ]
    if a.robustness is not None:
        cmd += [
            "--robustness",
            repr(a.robustness),
            "--filter-counts-level",
            repr(a.filter_counts_level),
            "--npix-super",
            str(a.npix_super),
            "--weight-grouping",
            a.weight_grouping,
        ]
    return cmd + a.pfb_extra.split()


def wsclean_padding(a):
    """Resolve wsclean ``-padding``.

    Defaults to ``psf_oversize`` so one knob drives both, but raised to
    :func:`min_padding_for_counts` when the weight grids are being compared --
    otherwise wsclean's grid starts out finer than pfb's and ``-super-weight``,
    which only coarsens, cannot bring them together.
    """
    if a.wsclean_padding is not None:
        return a.wsclean_padding
    pad = a.psf_oversize
    if a.compare_weights and wsclean_padded_size(a.nx, pad) < pfb_counts_size(a.nx):
        pad = min_padding_for_counts(a.nx)
    return pad


def wsclean_super_weight(a):
    """Resolve --wsclean-super-weight, returning None when the flag is omitted."""
    if a.wsclean_super_weight == "none":
        return None
    if a.wsclean_super_weight != "auto":
        return float(a.wsclean_super_weight)
    if not a.compare_weights:
        return None
    return super_weight_for_match(a.nx, wsclean_padding(a))


def wsclean_cmd(a, prefix):
    weight = weight_mode(a.robustness)[1]
    padding = wsclean_padding(a)
    sw = wsclean_super_weight(a)
    extra = ["-weighting-rank-filter", repr(a.wsclean_rank_filter)]
    if sw is not None:
        extra += ["-super-weight", repr(sw)]
    # pfb's band-collapsed groupings build one grid from all channels, which is
    # exactly what -mf-weighting does; the band-resolved ones are wsclean's default
    extra.append("-mf-weighting" if collapses_bands(a.weight_grouping) else "-no-mf-weighting")
    return (
        [
            a.wsclean_bin,
            "-j",
            str(a.nthreads),
            "-scale",
            f"{a.cell_size}asec",
            "-size",
            str(a.nx),
            str(a.ny),
        ]
        + weight
        + [
            "-save-weights",
            "-name",
            prefix,
            "-padding",
            repr(padding),
            "-channels-out",
            str(a.nband),
            "-no-min-grid-resolution",
            "-make-psf",
            "-wgridder-accuracy",
            repr(a.epsilon),
            "-data-column",
            a.data_column,
            "-niter",
            "0",
        ]
        + extra
        + a.wsclean_extra.split()
        + [os.path.expanduser(a.ms)]
    )


def pfb_fits(prefix, product, column, timeid, mfs=True):
    """Path of a pfb FITS product, following utils.naming.set_output_names."""
    base, stem = os.path.split(os.path.abspath(prefix))
    suffix = "_mfs" if mfs else ""
    return f"{base}/fits/{stem}_{product.upper()}_{column.lower()}_time{timeid}{suffix}.fits"


def load2d(path, band=None):
    """A 2D plane plus its header.

    With ``band=None`` the degenerate freq/Stokes axes are squeezed out of an MFS
    image; with ``band=b`` plane ``b`` is taken off the FREQ axis of a pfb cube,
    whose numpy layout is ``(corr, band, ny, nx)`` (``utils/fits.save_fits`` with
    ``yx_order=True``).  Only corr 0 (Stokes I) is compared.
    """
    data, hdr = fits.getdata(path, header=True)
    data = np.asarray(data, dtype=np.float64)
    if band is not None:
        if data.ndim != 4:
            raise ValueError(f"{path}: expected a 4D cube for band {band}, got shape {data.shape}")
        data = data[0, band]
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"{path}: expected a single 2D plane, got shape {data.shape}")
    return data, hdr


def product_list(a, pfb_pre, wsc_pre):
    """``(label, (pfb_path, band), wsc_path)`` for every image product to compare.

    With more than one band, wsclean writes ``-<NNNN>-`` per-band files plus an
    ``-MFS-`` image, while pfb writes one cube plus its ``_mfs`` image, so the
    per-band planes come out of the cube.
    """
    out = []
    for var in ("DIRTY", "PSF"):
        low = var.lower()
        if a.nband > 1:
            cube = pfb_fits(pfb_pre, a.product, var, a.timeid, mfs=False)
            for b in range(a.nband):
                out.append((f"{low} b{b}", (cube, b), f"{wsc_pre}-{b:04d}-{low}.fits"))
            out.append((f"{low} mfs", (pfb_fits(pfb_pre, a.product, var, a.timeid), None), f"{wsc_pre}-MFS-{low}.fits"))
        else:
            out.append((low, (pfb_fits(pfb_pre, a.product, var, a.timeid), None), f"{wsc_pre}-{low}.fits"))

    keep = []
    for label, (ppath, band), wpath in out:
        if os.path.exists(ppath) and os.path.exists(wpath):
            keep.append((label, (ppath, band), wpath))
        else:
            missing = ppath if not os.path.exists(ppath) else wpath
            print(f"\nNOTE: skipping '{label}' ({missing} absent)")
    return keep


def band_report(a, pfb_pre, wsc_pre):
    """Per-band centre frequency and wsum -- the channel-slicing test.

    Frequencies come from pfb's ``.dt`` band-node ``freq_out`` attrs (its
    ``band_edges`` centres) against each wsclean per-band image's ``CRVAL3``;
    wsums from the pfb cube's ``WSUM<n>`` cards against wsclean's ``WSCNORMF``.
    Both agreeing is what says the two imagers put the same channels in the same
    band -- a frequency match with a wsum mismatch would mean equal band centres
    over unequal channel sets.
    """
    import xarray as xr  # deferred: heavy import, only needed for the band check

    dt = xr.open_datatree(f"{os.path.abspath(pfb_pre)}_{a.product.upper()}.dt", engine="zarr", chunks=None)
    nodes = [dt[n].ds for n in dt.children if n.startswith("band")]
    nodes = sorted((ds for ds in nodes if int(ds.attrs["timeid"]) == a.timeid), key=lambda d: d.attrs["freq_out"])
    cube_hdr = fits.getheader(pfb_fits(pfb_pre, a.product, "DIRTY", a.timeid, mfs=False))

    print(f"\n=== band alignment ({a.nband} bands, {band_plan(a)[0]} channels each) ===")
    print(f"  {'band':>4s} {'pfb freq (Hz)':>18s} {'wsclean freq (Hz)':>18s} {'df (Hz)':>10s} {'wsum ratio':>13s}")
    for b, ds in enumerate(nodes):
        fp = float(ds.attrs["freq_out"])
        whdr = fits.getheader(f"{wsc_pre}-{b:04d}-dirty.fits")
        fw = float(whdr["CRVAL3"])
        ratio = float(cube_hdr[f"WSUM{b + 1}"]) / float(whdr["WSCNORMF"])
        # an even split must agree exactly, so the tolerance is tight: a drifting
        # band centre means the labels disagree, a drifting wsum means the data does
        flag = "   <- MISMATCH" if abs(fp - fw) > 1.0 or abs(ratio - 1) > 1e-3 else ""
        print(f"  {b:>4d} {fp:>18.3f} {fw:>18.3f} {fp - fw:>+10.3g} {ratio:>13.9f}{flag}")


def wcs_offset(ha, hb):
    """Reference-coordinate disagreement between two headers, in pixels of a."""
    dra = (float(ha["CRVAL1"]) - float(hb["CRVAL1"]) + 180.0) % 360.0 - 180.0
    ddec = float(ha["CRVAL2"]) - float(hb["CRVAL2"])
    cosdec = np.cos(np.deg2rad(float(ha["CRVAL2"])))
    return dra * cosdec / abs(float(ha["CDELT1"])), ddec / abs(float(ha["CDELT2"]))


def align(a, ha, b, hb):
    """Crop two images to the common extent about their respective CRPIX.

    Handles the PSF size difference (pfb writes ``psf_oversize * nx``, wsclean
    writes ``nx``) and any even/odd centring convention, since the reference
    pixel is read from each header rather than assumed.

    Returns:
        ``(a_crop, b_crop, info)`` with ``info`` carrying the cell-size ratio and
        the CRVAL offset in pixels.
    """
    ax, ay = float(ha["CRPIX1"]) - 1.0, float(ha["CRPIX2"]) - 1.0
    bx, by = float(hb["CRPIX1"]) - 1.0, float(hb["CRPIX2"]) - 1.0
    if not (ax.is_integer() and ay.is_integer() and bx.is_integer() and by.is_integer()):
        raise ValueError("non-integer CRPIX: sub-pixel grid alignment is not supported")
    ax, ay, bx, by = int(ax), int(ay), int(bx), int(by)

    # left/right extents available on both sides of the reference pixel
    lx, rx = min(ax, bx), min(a.shape[1] - 1 - ax, b.shape[1] - 1 - bx)
    ly, ry = min(ay, by), min(a.shape[0] - 1 - ay, b.shape[0] - 1 - by)
    acrop = a[ay - ly : ay + ry + 1, ax - lx : ax + rx + 1]
    bcrop = b[by - ly : by + ry + 1, bx - lx : bx + rx + 1]

    info = {
        "cdelt_ratio": (float(ha["CDELT1"]) / float(hb["CDELT1"]), float(ha["CDELT2"]) / float(hb["CDELT2"])),
        "crval_offset_pix": wcs_offset(ha, hb),
        "crpix": ((ax, ay), (bx, by)),
        "ref_pixel_in_crop": (lx, ly),
    }
    return acrop, bcrop, info


def subpixel_shift(a, b):
    """Offset of ``a`` relative to ``b`` in pixels, from the cross-correlation peak.

    Parabolic interpolation about the integer peak of the (mean-removed) FFT
    cross-correlation.  A real half-pixel gridding error shows up here; a
    difference in how CRVAL/CRPIX are written does not.
    """
    fa = np.fft.rfft2(a - a.mean())
    fb = np.fft.rfft2(b - b.mean())
    cc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    ny, nx = cc.shape
    iy, ix = (int(i) for i in np.unravel_index(np.argmax(cc), cc.shape))
    peak = cc[iy, ix]

    def refine(i, n, neighbours):
        lo, hi = neighbours
        denom = lo - 2.0 * peak + hi
        lag = i + (0.5 * (lo - hi) / denom if denom != 0 else 0.0)
        return lag - n if lag > n / 2 else lag  # unwrap the circular lag

    dy = refine(iy, ny, (cc[(iy - 1) % ny, ix], cc[(iy + 1) % ny, ix]))
    dx = refine(ix, nx, (cc[iy, (ix - 1) % nx], cc[iy, (ix + 1) % nx]))
    return dx, dy


def cleanbeam(hdr):
    """``BMAJ BMIN BPA`` as a string, for information only (the fitters differ)."""
    return " ".join(f"{float(hdr.get(k, np.nan)):.6g}" for k in ("BMAJ", "BMIN", "BPA"))


def mad_rms(x):
    """Median-absolute-deviation rms, robust against the sources in the field.

    Restricted to the non-zero support when the array is mostly zeros, so the
    sparse uv weight grids get a meaningful scale rather than 0.
    """
    v = x[x != 0] if float(np.mean(x == 0)) > 0.5 else x
    if v.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def compare(a, b, label):
    """Metrics for pfb image ``a`` against wsclean reference ``b`` (already aligned)."""
    d = a - b
    peak = float(np.max(np.abs(b)))
    rms_ref = mad_rms(b)
    dx, dy = subpixel_shift(a, b)
    return {
        "product": label,
        "shape": list(a.shape),
        "peak_pfb": float(np.max(a)),
        "peak_wsclean": float(np.max(b)),
        "argmax_pfb": [int(i) for i in np.unravel_index(np.argmax(a), a.shape)],
        "argmax_wsclean": [int(i) for i in np.unravel_index(np.argmax(b), b.shape)],
        "max_abs_diff": float(np.max(np.abs(d))),
        "rms_diff": float(np.sqrt(np.mean(d**2))),
        "max_abs_diff_rel_peak": float(np.max(np.abs(d)) / peak),
        "rms_diff_rel_peak": float(np.sqrt(np.mean(d**2)) / peak),
        "rms_diff_rel_rms": float(np.sqrt(np.mean(d**2)) / rms_ref) if rms_ref > 0 else float("nan"),
        "rms_wsclean": rms_ref,
        "scale": float(np.sum(a * b) / np.sum(b * b)),
        "shift_pix": [dx, dy],
    }


def scratch_time_chunks(prefix, product):
    """Distinct pass-1 ``timeid``s in the ``.scratch`` store.

    Direct evidence of how many time chunks ``--integrations-per-image`` produced,
    all of which ``concat_row`` folds into one output image.
    """
    import xarray as xr  # deferred: heavy import, only for the concat_row check

    store = f"{os.path.abspath(prefix)}_{product.upper()}.scratch"
    if not os.path.exists(store):
        return None
    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    tids = set()
    for name in dt.children:
        if name.startswith("band"):
            tids.add(int(next(iter(dt[name].children.values())).ds.attrs["timeid"]))
    return sorted(tids)


def read_pfb_counts(prefix, product, bandid, grouping, filter_counts_level, npix_super):
    """Rebuild the density grid pfb actually used, from the ``.scratch`` store.

    The reduced counts grid never reaches the ``.dt`` -- the driver holds it in
    memory between passes -- so it is re-summed here from the per-piece
    ``COUNTS`` over the same group the driver would use, then put through pfb's
    own ``filter_extreme_counts`` and ``box_sum_counts`` so the script cannot
    drift from the code it is checking.

    Args:
        bandid: output band whose applied grid is wanted.
        grouping: pfb ``--weight-grouping``; band-collapsed groupings sum every
            band's pieces, band-resolved ones only ``bandid``'s. Time chunks are
            always summed -- see the loop comment.

    Returns:
        ``(counts, nodenames)`` with ``counts`` the ``(nx_pad, ny_pad)`` grid for
        corr 0, indexed ``[u, v]``.

    Raises:
        FileNotFoundError: if the ``.scratch`` store is absent (pfb was run
            without ``--keep-scratch``).
    """
    # deferred: heavy imports on a rarely-taken path (only when comparing grids)
    import xarray as xr

    from pfb_imaging.utils.weighting import box_sum_counts, filter_extreme_counts

    store = f"{os.path.abspath(prefix)}_{product.upper()}.scratch"
    if not os.path.exists(store):
        raise FileNotFoundError(store)

    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    members = []
    for name in dt.children:
        if not name.startswith("band"):
            continue
        attrs = next(iter(dt[name].children.values())).ds.attrs
        # No timeid filter: concat_row (on by default, never disabled here) collapses
        # the time axis, so counts_key uses tid=0 and *every* time chunk of a band
        # belongs to the same weighting group -- summing one chunk would understate
        # the density grid whenever integrations_per_image splits the scan.
        if collapses_bands(grouping) or int(attrs["bandid"]) == bandid:
            members.append(name)
    if not members:
        raise ValueError(f"no scratch band node for bandid {bandid} in {store}")

    counts = None
    for name in sorted(members):
        for _, child in dt[name].children.items():
            c = child.ds.COUNTS.values
            counts = c.copy() if counts is None else counts + c
    counts = filter_extreme_counts(counts, level=filter_counts_level)
    counts = box_sum_counts(counts, npix_super)
    return np.asarray(counts[0], dtype=np.float64), sorted(members)


def counts_to_weight_grid(counts, robustness):
    """Density grid ``D_k`` -> imaging weight grid ``W_k``.

    The same three formulae in both imagers (pfb ``counts_to_weights``, wsclean
    ``ImageWeights::FinishGridding``).  The Briggs expressions are algebraically
    identical: pfb's ``ssq = 25*10^(-2R) * sum(D)/sum(D^2)`` equals wsclean's
    ``25*10^(-2R)/avgW`` with ``avgW = sum(D^2)/sum(w)`` and ``sum(w) = sum(D)``.
    Cells with no samples are left at 0 in both.
    """
    w = np.zeros_like(counts)
    sampled = counts > 0
    if robustness is None:  # natural
        w[sampled] = 1.0
    elif robustness <= -2:  # uniform
        w[sampled] = 1.0 / counts[sampled]
    else:
        ssq = 25.0 * 10.0 ** (-2.0 * robustness) * counts.sum() / (counts**2).sum()
        w[sampled] = 1.0 / (1.0 + counts[sampled] * ssq)
    return w


def wsclean_grid_to_uv(wf, flipu, rollv):
    """wsclean's weight FITS re-indexed into pfb's ``[u, v]`` COUNTS layout."""
    g = wf.T  # FITS is [row=v, col=u]
    if flipu:
        g = g[::-1, :]
    if rollv:
        g = np.roll(g, rollv, axis=1)
    return g


def match_weight_grids(w_pfb, wf, transform="auto"):
    """Align wsclean's weight FITS to pfb's counts grid over index conventions.

    pfb folds to ``v >= 0`` and offsets the v index by ``ny/2``, wsclean folds to
    ``v >= 0`` at index 0 and writes the Hermitian-symmetrised full plane, and
    pass 1 grids ``(-u, v)`` (``wgridder_conventions`` gives ``usign=-1``), so the
    u axis is reflected.  Rather than trusting that chain, every candidate is
    scored and the winner reported: the fftshift/reflection convention is
    *established* here, not assumed.

    Returns:
        ``(pfb_half, wsc_half, best, scores)`` -- both grids cropped to the
        populated ``v >= 0`` half in ``(v, u)`` display order.
    """
    nv = w_pfb.shape[1]
    half = slice(nv // 2, nv)
    ref = w_pfb[:, half]

    if transform != "auto":
        parts = dict(kv.split(":") for kv in transform.split(","))
        cands = [(bool(int(parts["flipu"])), int(parts["rollv"]))]
    else:
        cands = [(f, r) for f in (False, True) for r in (0, -1, 1, nv // 2)]

    scores = []
    for flipu, rollv in cands:
        trial = wsclean_grid_to_uv(wf, flipu, rollv)[:, half]
        denom = mad_rms(ref)
        score = float(np.sqrt(np.mean((trial - ref) ** 2)) / denom) if denom > 0 else np.inf
        scores.append({"flipu": flipu, "rollv": rollv, "score": score})
    scores.sort(key=lambda s: s["score"])
    best = scores[0]
    return (
        ref.T,
        wsclean_grid_to_uv(wf, best["flipu"], best["rollv"])[:, half].T,
        best,
        scores,
    )


def weight_grid_pair(a, pfb_prefix, wsc_wpath, bandid, verbose=True):
    """Aligned ``(pfb, wsclean)`` weight grids plus provenance, or None if skipped.

    Every skip is a printed reason rather than an exception: a missing
    ``-save-weights`` file, pfb run without ``--keep-scratch``, or grids of
    different size (which makes a cell-by-cell comparison meaningless).
    """
    if not os.path.exists(wsc_wpath):
        print(f"  NOTE: skipping band {bandid} ({wsc_wpath} absent; needs wsclean -save-weights)")
        return None
    try:
        counts, nodes = read_pfb_counts(
            pfb_prefix, a.product, bandid, a.weight_grouping, a.filter_counts_level, a.npix_super
        )
    except FileNotFoundError as e:
        print(f"  NOTE: skipping the weight-grid comparison ({e} absent; pfb needs --keep-scratch)")
        return None

    wf = load2d(wsc_wpath)[0]
    print(f"  band {bandid}: pfb COUNTS {counts.shape} from {len(nodes)} node(s), wsclean grid {wf.shape}")
    if counts.shape != wf.shape:
        print(
            "  SKIPPED: grids differ in size, so their uv cells differ. Rerun both "
            "imagers with --wsclean-super-weight auto."
        )
        return None

    pfb_half, wsc_half, best, scores = match_weight_grids(
        counts_to_weight_grid(counts, a.robustness), wf, a.grid_transform
    )
    if verbose:
        print("  transform scores (rms diff / rms, lower is better):")
        for s in scores:
            mark = " <- used" if s is best else ""
            print(f"    flipu={int(s['flipu'])} rollv={s['rollv']:>4d}  {s['score']:.4e}{mark}")
    extra = {
        "uv grid size": str(counts.shape[0]),
        "grid transform": f"flipu={int(best['flipu'])} rollv={best['rollv']}",
        "density sum (pfb)": f"{counts.sum():.9g}",
    }
    return pfb_half, wsc_half, extra


def wsclean_weights_path(a, wsc_pre, bandid):
    """wsclean's weight-grid FITS.

    ``-mf-weighting`` builds a single grid from every channel and writes one
    unsuffixed file; otherwise there is one grid, and one file, per output band.
    """
    if a.nband == 1 or collapses_bands(a.weight_grouping):
        return f"{wsc_pre}-weights.fits"
    return f"{wsc_pre}-{bandid:04d}-weights.fits"


def support_report(a, b):
    """Cell-support agreement -- the diagnostic for uv cell-assignment differences."""
    sa, sb = a != 0, b != 0
    both = sa & sb
    rel = np.abs(a[both] - b[both]) / np.abs(b[both]) if both.any() else np.array([0.0])
    print("\n  weight-grid cell support:")
    print(f"    sampled in both      {int(both.sum())}")
    print(f"    pfb only             {int((sa & ~sb).sum())}")
    print(f"    wsclean only         {int((sb & ~sa).sum())}")
    print(f"    |rel diff| on shared cells: median {np.median(rel):.3e}  max {rel.max():.3e}")


def beam_variants(convention, image, beam, name):
    """(label, image) pairs for the requested n-term/beam conventions."""
    if beam is None:
        return [(name, image)]
    wanted = ("none", "multiply", "divide") if convention == "all" else (convention,)
    out = []
    for conv in wanted:
        if conv == "none":
            out.append((name, image))
        elif conv == "multiply":
            out.append((f"{name} x beam", image * beam))
        else:
            out.append((f"{name} / beam", image / beam))
    return out


def print_report(rows, header_info):
    """Print the provenance block and the metrics table."""
    print("\n=== provenance ===")
    for k, v in header_info.items():
        print(f"  {k:<24s} {v}")

    print("\n=== agreement (pfb vs wsclean, aligned overlap) ===")
    head = f"  {'product':<18s} {'shape':>11s} {'peak(pfb)':>13s} {'peak(wsc)':>13s}"
    head += f" {'max|d|/pk':>11s} {'rms(d)/pk':>11s} {'rms(d)/rms':>11s} {'scale':>12s}"
    print(head)
    for r in rows:
        shape = "x".join(str(s) for s in r["shape"])
        print(
            f"  {r['product']:<18s} {shape:>11s} {r['peak_pfb']:>13.6g} {r['peak_wsclean']:>13.6g}"
            f" {r['max_abs_diff_rel_peak']:>11.3e} {r['rms_diff_rel_peak']:>11.3e}"
            f" {r['rms_diff_rel_rms']:>11.3e} {r['scale']:>12.9f}"
        )

    print("\n  sub-pixel shift of pfb relative to wsclean (from the cross-correlation peak):")
    for r in rows:
        dx, dy = r["shift_pix"]
        print(f"    {r['product']:<18s} dx = {dx:+.4f}  dy = {dy:+.4f} pixels")


def make_plot(panels, outname, psf_zoom, dpi):
    """One row per product: pfb, wsclean, difference, and a cut through the peak."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    nrow = len(panels)
    fig, axes = plt.subplots(nrow, 4, figsize=(17, 4.3 * nrow), squeeze=False)

    for row, (label, a, b) in enumerate(panels):
        # display-only zoom about the reference pixel for the PSF's sidelobe field
        if psf_zoom and "psf" in label.lower() and psf_zoom < min(a.shape) // 2:
            cy, cx = a.shape[0] // 2, a.shape[1] // 2
            sl = (slice(cy - psf_zoom, cy + psf_zoom + 1), slice(cx - psf_zoom, cx + psf_zoom + 1))
            av, bv = a[sl], b[sl]
        else:
            av, bv = a, b

        d = av - bv
        # sequential magnitude: one perceptually ordered, CVD-safe ramp, shared
        # between the two panels so they are directly comparable
        linthresh = max(3.0 * mad_rms(bv), np.abs(bv).max() * 1e-6)
        norm = SymLogNorm(linthresh=linthresh, vmin=bv.min(), vmax=bv.max())
        for col, (img, name) in enumerate(((av, "pfb imager"), (bv, "wsclean"))):
            ax = axes[row][col]
            im = ax.imshow(img, origin="lower", cmap="cividis", norm=norm)
            ax.set_title(f"{name} -- {label}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # signed difference: diverging, two hues about a neutral midpoint
        ax = axes[row][2]
        lim = max(np.abs(d).max(), 1e-30)
        im = ax.imshow(d, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(f"pfb - wsclean (max |d| = {lim:.3g})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        for col in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

        ax = axes[row][3]
        if label == "weights":
            # a row cut through a sparse uv grid is mostly zeros; the sorted
            # non-zero values compare the whole weight distribution instead
            ax.plot(np.sort(av[av != 0]), color=C_PFB, lw=1.8, label="pfb imager")
            ax.plot(np.sort(bv[bv != 0]), color=C_WSC, lw=1.8, ls="--", label="wsclean")
            ax.set_yscale("log")
            ax.set_title("sorted non-zero cell weights", fontsize=10)
            ax.set_xlabel("rank")
        else:
            # cut through the reference peak: the traces should be indistinguishable
            iy = int(np.unravel_index(np.argmax(bv), bv.shape)[0])
            x = np.arange(bv.shape[1])
            ax.plot(x, av[iy], color=C_PFB, lw=1.8, label="pfb imager")
            ax.plot(x, bv[iy], color=C_WSC, lw=1.8, ls="--", label="wsclean")
            ax.set_title(f"cut through peak (row {iy})", fontsize=10)
            ax.set_xlabel("pixel")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(outname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {outname}")


def wsc_dirty(a, wsc_pre):
    """wsclean's band-integrated dirty image: MFS when there is more than one band."""
    return f"{wsc_pre}-dirty.fits" if a.nband == 1 else f"{wsc_pre}-MFS-dirty.fits"


def run_imager(a, name, cmd, expected_output):
    """Run one imager, honouring --run (``auto`` skips it if its output is there)."""
    if not (a.run == "both" or a.run == name or (a.run == "auto" and not os.path.exists(expected_output))):
        return
    exe = shutil.which(cmd[0])
    if exe is None:
        sys.exit(f"{cmd[0]} not found on PATH (use --{name}-bin)")
    print(f"\n=== running {name} ===", flush=True)
    # When this script is itself launched via `uv run`, Ray's uv hook tries to
    # replicate the parent's uv environment onto the pfb workers and they fail
    # to start. The venv console script needs no uv at runtime, so switch the
    # hook off for the child.
    env = {**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}
    subprocess.run([exe] + cmd[1:], check=True, env=env)


def pick(rows, *names):
    """First metrics row whose product matches one of ``names``."""
    for name in names:
        for r in rows:
            if r["product"] == name:
                return r
    return None


def print_sweep_summary(results):
    """Invariance table over ``--integrations-per-image``.

    ``concat_row`` concatenates every pass-1 time chunk back into one image, and
    the weighting group spans them all, so every column here must be flat: a value
    that moves with the chunk count means the time split leaked into the result.
    """
    print("\n=== concat_row invariance over --integrations-per-image ===")
    print(f"  {'ipi':>5s} {'chunks':>7s} {'dirty max|d|/pk':>16s} {'weights max|d|/pk':>18s} {'wsum ratio':>13s}")
    for ipi, chunks, rows, hdr in results:
        d = pick(rows, "dirty", "dirty mfs")
        w = pick(rows, "weights", "weights b0")
        dv = f"{d['max_abs_diff_rel_peak']:.3e}" if d else "n/a"
        wv = f"{w['max_abs_diff_rel_peak']:.3e}" if w else "n/a"
        print(f"  {ipi:>5d} {len(chunks) if chunks else 0:>7d} {dv:>16s} {wv:>18s} {hdr['wsum ratio']:>13s}")


def main():
    a = parse_args()
    pfb_pre, wsc_pre = prefixes(a)
    ipis = [int(x) for x in a.ipi_sweep.split(",")] if a.ipi_sweep else [a.integrations_per_image]
    cmds = {"pfb": pfb_cmd(a, pfb_pre), "wsclean": wsclean_cmd(a, wsc_pre)}

    print("=== commands ===")
    for name, cmd in cmds.items():
        print(f"  {name}: {' '.join(cmd)}")
    print(f"  weighting: {weight_mode(a.robustness)[2]}")
    if len(ipis) > 1:
        print(f"  integrations-per-image sweep: {ipis} (one pfb run each, shared wsclean reference)")
    if a.weight_column != "WEIGHT_SPECTRUM":
        print(f"  WARNING: --weight-column {a.weight_column} cannot be passed to wsclean (it uses WEIGHT_SPECTRUM)")
    if a.compare_weights:
        if a.nx != a.ny:
            sys.exit("--compare-weights needs a square image: -super-weight is a single scalar for both axes")
        target, padded = pfb_counts_size(a.nx), wsclean_padded_size(a.nx, wsclean_padding(a))
        sw = wsclean_super_weight(a)
        print(
            f"  uv grid: pfb COUNTS {target}, wsclean padded {padded} "
            f"(-padding {wsclean_padding(a):.6f}), -super-weight {sw}"
        )
        if sw is None and padded != target:
            alt = compatible_npix(a.nx, wsclean_padding(a))
            print(
                f"  WARNING: weight grids differ in size ({target} vs {padded}) with "
                f"-super-weight off, so their uv cells differ and neither the grids "
                f"nor the uniformly weighted images can agree. Use "
                f"--wsclean-super-weight auto, or an image size that matches "
                f"natively: {alt}"
            )
    if a.dry_run:
        return

    # wsclean is the shared, ipi-invariant reference: -intervals-out stays at 1
    run_imager(a, "wsclean", cmds["wsclean"], wsc_dirty(a, wsc_pre))

    results = []
    for ipi in ipis:
        a.integrations_per_image = ipi
        pre = pfb_pre if len(ipis) == 1 else f"{pfb_pre}_ipi{ipi}"
        run_imager(a, "pfb", pfb_cmd(a, pre), pfb_fits(pre, a.product, "DIRTY", a.timeid))
        if len(ipis) > 1:
            print(f"\n######## integrations-per-image = {ipi} ########")
        rows, header_info = compare_run(a, pre, wsc_pre, cmds, suffix=f"_ipi{ipi}" if len(ipis) > 1 else "")
        results.append((ipi, scratch_time_chunks(pre, a.product), rows, header_info))

    if len(ipis) > 1:
        print_sweep_summary(results)


def compare_run(a, pfb_pre, wsc_pre, cmds, suffix=""):
    """Compare one pfb output against the wsclean reference; report and return metrics."""
    dirty_pfb = pfb_fits(pfb_pre, a.product, "DIRTY", a.timeid)
    dirty_wsc = wsc_dirty(a, wsc_pre)
    for path in (dirty_pfb, dirty_wsc):
        if not os.path.exists(path):
            sys.exit(f"missing product {path} (use --run to generate it)")

    # the stored BEAM carries the folded 1/n term; absent when --no-beam was used
    beam_path = pfb_fits(pfb_pre, a.product, "BEAM", a.timeid)
    beam = load2d(beam_path)[0] if os.path.exists(beam_path) else None

    chunks = scratch_time_chunks(pfb_pre, a.product)
    if chunks and len(chunks) > 1:
        print(f"\nNOTE: pass 1 wrote {len(chunks)} time chunks; concat_row folds them into one image per band")
    if a.nband > 1:
        band_report(a, pfb_pre, wsc_pre)

    rows, panels = [], []
    # provenance is quoted for the band-integrated pair, so that pfb's total WSUM
    # is compared against wsclean's MFS normalisation and not one band's
    info_first, info_ref = None, None
    for label, (ppath, pband), wpath in product_list(a, pfb_pre, wsc_pre):
        pimg, phdr = load2d(ppath, pband)
        wimg, whdr = load2d(wpath)
        pcrop, wcrop, info = align(pimg, phdr, wimg, whdr)
        if info_first is None:
            info_first = (phdr, whdr, info)
        if info_ref is None and os.path.abspath(wpath) == os.path.abspath(dirty_wsc):
            info_ref = (phdr, whdr, info)
        bcrop = None
        if beam is not None:
            bhdr = fits.getheader(beam_path)
            bcrop = align(beam, bhdr, wimg, whdr)[0]
            if bcrop.shape != pcrop.shape:  # beam is image-sized; PSF crop may be larger
                bcrop = None
        for name, variant in beam_variants(a.beam_convention, pcrop, bcrop, label):
            rows.append(compare(variant, wcrop, name))
        # every band is in the table; the figure shows band 0 and the MFS image so
        # it stays readable as nband grows
        if pband in (None, 0):
            panels.append((label, pcrop, wcrop))

    # ---- uv weight grids ----
    wgrid_extra, support = {}, None
    if a.compare_weights:
        print(f"\n=== uv weight grids ({a.weight_grouping}) ===")
        # a band-collapsed grouping gives every band the same grid on both sides,
        # so one comparison covers them all
        single = a.nband == 1 or collapses_bands(a.weight_grouping)
        for b in [0] if single else range(a.nband):
            pair = weight_grid_pair(a, pfb_pre, wsclean_weights_path(a, wsc_pre, b), b, verbose=b == 0)
            if pair is None:
                continue
            pfb_half, wsc_half, wgrid_extra = pair
            label = "weights" if single else f"weights b{b}"
            rows.append(compare(pfb_half, wsc_half, label))
            if b == 0:
                panels.append((label, pfb_half, wsc_half))
                support = (pfb_half, wsc_half)

    phdr, whdr, info = info_ref or info_first
    header_info = {
        "ms": os.path.expanduser(a.ms),
        # WSCIMGWG, not WSCNORMF: wsclean's MFS header copies band 0's WSCNORMF
        # verbatim, while WSCIMGWG carries the band-summed weight (the two are
        # equal for a single band)
        "pfb WSUM": f"{float(fits.getheader(dirty_pfb)['WSUM']):.6g}",
        "wsclean WSCIMGWG": f"{float(whdr['WSCIMGWG']):.6g}",
        "wsum ratio": f"{float(fits.getheader(dirty_pfb)['WSUM']) / float(whdr['WSCIMGWG']):.9f}",
        "wsclean weighting": str(whdr.get("WSCWEIGH", "?")),
        "wsclean nvis (eff)": f"{float(whdr.get('WSCNVIS', np.nan)):.0f} ({float(whdr.get('WSCENVIS', np.nan)):.6g})",
        "cdelt ratio (x, y)": f"{info['cdelt_ratio'][0]:.12g}, {info['cdelt_ratio'][1]:.12g}",
        "crval offset (pix)": f"{info['crval_offset_pix'][0]:+.4g}, {info['crval_offset_pix'][1]:+.4g}",
        # different fitters/fit regions: informational, not a pass/fail comparison
        "pfb clean beam (deg)": cleanbeam(phdr),
        "wsc clean beam (deg)": cleanbeam(whdr),
        "pfb beam (B/n) range": "n/a" if beam is None else f"{beam.min():.6f} to {beam.max():.6f}",
        "weighting": weight_mode(a.robustness)[2],
        "integrations per image": str(a.integrations_per_image),
        "pass-1 time chunks": str(len(chunks) if chunks else "n/a"),
        **wgrid_extra,
    }
    print_report(rows, header_info)
    if support is not None:
        support_report(*support)

    if not a.no_plot:
        outname = a.plot or os.path.join(os.path.expanduser(a.outdir), "compare_wsclean.png")
        if suffix:
            stem, ext = os.path.splitext(outname)
            outname = f"{stem}{suffix}{ext}"
        make_plot(panels, outname, a.psf_zoom, a.dpi)

    if a.json:
        path = a.json
        if suffix:
            stem, ext = os.path.splitext(path)
            path = f"{stem}{suffix}{ext}"
        with open(path, "w") as f:
            json.dump({"provenance": header_info, "commands": cmds, "metrics": rows}, f, indent=2)
        print(f"wrote {path}")
    return rows, header_info


if __name__ == "__main__":
    main()
