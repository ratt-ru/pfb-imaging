#!/opt/casa-6.6.5-31-py3.10.el8/bin/python3
"""Split a frequency range / max-baseline subset out of a measurement set.

Wraps CASA's mstransform, which is the standard, well-tested tool for this
exact job (channel selection + uv-range selection into a fresh, correctly
structured MS with adjusted SPECTRAL_WINDOW/DATA_DESCRIPTION subtables).

Note: mstransform only carries over MS-standard columns. Non-standard
columns present in some of these MSs (BITFLAG, BITFLAG_ROW, MODEL_DATA_PFB,
RESIDUAL) will NOT be present in the output -- DATA, MODEL_DATA,
CORRECTED_DATA, WEIGHT_SPECTRUM, FLAG/FLAG_ROW etc. are all preserved.

This script must be run with a CASA python that has casatasks available,
e.g. the modular CASA 6 distribution's own interpreter:

    /opt/casa-6.6.5-31-py3.10.el8/bin/python3 split_ms.py <ms> \\
        --freq-range 700 900 --max-baseline 1000

or, since the shebang above already points there, simply:

    ./split_ms.py <ms> --freq-range 700 900 --max-baseline 1000
"""

import argparse
import os
import sys
import time


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ms", help="Path to the input measurement set")
    p.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        metavar=("FMIN", "FMAX"),
        default=None,
        help="Frequency range to keep, in MHz (e.g. --freq-range 700 900). Omit to keep the full band.",
    )
    p.add_argument(
        "--max-baseline",
        type=float,
        default=None,
        help="Maximum baseline (uv-distance) to keep, in metres. Omit to keep all baselines.",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output MS path (default: derived from input name and the selection applied)",
    )
    p.add_argument(
        "--datacolumn",
        default="all",
        help="Data column(s) to copy, passed straight to mstransform (default: %(default)s)",
    )
    p.add_argument("--field", default="", help="Optional field selection")
    p.add_argument("--scan", default="", help="Optional scan selection")
    p.add_argument("--overwrite", action="store_true", help="Overwrite the output MS if it already exists")
    return p


def default_output_name(ms, freq_range, max_baseline):
    base = ms.rstrip("/")
    if base.endswith(".ms"):
        base = base[:-3]
    tag = ""
    if freq_range is not None:
        tag += f"_{freq_range[0]:g}-{freq_range[1]:g}MHz"
    if max_baseline is not None:
        tag += f"_bl{max_baseline:g}m"
    if not tag:
        tag = "_subset"
    return base + tag + ".ms"


def main():
    args = make_parser().parse_args()

    if not os.path.exists(args.ms):
        sys.exit(f"Input MS not found: {args.ms}")

    if args.freq_range is None and args.max_baseline is None:
        sys.exit("Nothing to do: specify --freq-range and/or --max-baseline")

    output = args.output or default_output_name(args.ms, args.freq_range, args.max_baseline)

    if os.path.exists(output):
        if not args.overwrite:
            sys.exit(f"Output MS already exists: {output} (use --overwrite to replace)")
        import shutil

        shutil.rmtree(output)

    spw = ""
    if args.freq_range is not None:
        fmin, fmax = args.freq_range
        spw = f"*:{fmin}~{fmax}MHz"

    uvrange = ""
    if args.max_baseline is not None:
        uvrange = f"0~{args.max_baseline}m"

    try:
        from casatasks import mstransform
    except ImportError:
        sys.exit(
            "Could not import casatasks. Run this script with a CASA python "
            "that has casatasks installed, e.g.:\n"
            "  /opt/casa-6.6.5-31-py3.10.el8/bin/python3 split_ms.py ..."
        )

    print(f"Input:        {args.ms}")
    print(f"Output:       {output}")
    print(f"spw selection:     {spw or '(none -- full band)'}")
    print(f"uvrange selection: {uvrange or '(none -- all baselines)'}")
    print(f"datacolumn:        {args.datacolumn}")

    t0 = time.time()
    mstransform(
        vis=args.ms,
        outputvis=output,
        spw=spw,
        uvrange=uvrange,
        field=args.field,
        scan=args.scan,
        datacolumn=args.datacolumn,
        keepflags=True,
    )
    print(f"Done in {time.time() - t0:.1f}s -> {output}")


if __name__ == "__main__":
    main()
