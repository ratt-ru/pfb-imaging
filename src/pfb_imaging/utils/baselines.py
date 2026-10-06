"""Antenna classification and baseline-group masks for mixed MeerKAT arrays.

On a mixed MeerKAT / MeerKAT+ array the primary beam depends on which pair of
dishes forms a baseline, so the imager partitions visibilities into the three
pairings meerkat-beams serves: MM, MPM, MPMP (issue #335).

Everything here is pure: numpy arrays and strings in, numpy arrays out,
ValueError on anything ambiguous. The driver owns logging and I/O. That keeps
every test in tests/test_baselines.py free of an MS, so they run on the
casacore-free CI legs too.
"""

import fnmatch

import numpy as np

# meerkat-beams' own labels: MeerKAT-MeerKAT, MeerKAT-MeerKAT+, MeerKAT+-MeerKAT+.
GROUP_LABELS = ("MM", "MPM", "MPMP")

# OBSERVATION::TELESCOPE_NAME values we accept. xarray-ms broadcasts this one
# value across every antenna, so it identifies the array but cannot
# discriminate dishes -- that is what dish diameter is for.
MEERKAT_TELESCOPE_NAMES = frozenset({"meerkat", "meerkat+"})


def check_telescope_is_meerkat(telescope_name: str, ms_path: str) -> None:
    """Raise unless `telescope_name` names a MeerKAT array.

    Args:
        telescope_name: antenna_xds.attrs["overall_telescope_name"].
        ms_path: the MS this came from, for the error message.

    Raises:
        ValueError: if the array is not MeerKAT.
    """
    if str(telescope_name).strip().lower() not in MEERKAT_TELESCOPE_NAMES:
        raise ValueError(
            f"baseline-group partitioning needs MeerKAT beams, but {ms_path} reports "
            f"telescope {telescope_name!r}. meerkat-beams has no beam model for this "
            f"array; re-run without --baseline-groups, or without --beam-model."
        )


def parse_antenna_groups(spec: str) -> tuple[str, str]:
    """Parse a --antenna-groups value into (meerkat_pattern, meerkat_plus_pattern).

    Args:
        spec: exactly two comma-separated glob patterns. Order is load-bearing:
            the first names the MeerKAT antennas, which is the `p` of
            meerkat-beams' fixed `p = MeerKAT` convention.

    Returns:
        The two patterns, whitespace-stripped.

    Raises:
        ValueError: if `spec` does not hold exactly two non-empty patterns.
    """
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"--antenna-groups needs exactly two comma-separated patterns "
            f"(MeerKAT first, then MeerKAT+), e.g. 'm*,e*'; got {spec!r}"
        )
    return parts[0], parts[1]


def _classify_by_override(antenna_names, override):
    mk_pat, ext_pat = override
    is_mk = np.array([fnmatch.fnmatchcase(n, mk_pat) for n in antenna_names])
    is_ext = np.array([fnmatch.fnmatchcase(n, ext_pat) for n in antenna_names])
    both = antenna_names[is_mk & is_ext]
    if both.size:
        raise ValueError(
            f"--antenna-groups patterns {mk_pat!r} and {ext_pat!r} both match "
            f"{both.tolist()}; the override exists to remove ambiguity, not to add it."
        )
    neither = antenna_names[~(is_mk | is_ext)]
    if neither.size:
        raise ValueError(
            f"--antenna-groups patterns {mk_pat!r} and {ext_pat!r} match neither "
            f"{neither.tolist()}; every selected antenna must fall in exactly one group."
        )
    return is_ext


def classify_antennas(antenna_names, dish_diameters, override=None):
    """Split antennas into MeerKAT and MeerKAT+ classes.

    Classification leads on ANTENNA_DISH_DIAMETER -- a property of the hardware --
    and uses the m/e name prefix only to corroborate it. Anything ambiguous is
    refused rather than guessed; --antenna-groups is the documented escape hatch.

    Args:
        antenna_names: (nant,) array of antenna names.
        dish_diameters: (nant,) array of dish diameters in m, or None.
        override: optional (meerkat_pattern, meerkat_plus_pattern) from
            parse_antenna_groups. When given, diameters are ignored entirely.

    Returns:
        (nant,) bool array; True marks a MeerKAT+ (larger-dish) antenna.

    Raises:
        ValueError: on uniform diameters, more than two diameter classes, a
            diameter/prefix disagreement, missing diameters with no override,
            or an override that leaves an antenna uncovered or doubly covered.
    """
    antenna_names = np.asarray(antenna_names).astype(str)

    if override is not None:
        return _classify_by_override(antenna_names, override)

    if dish_diameters is None:
        raise ValueError(
            "cannot classify antennas: the MS has no ANTENNA_DISH_DIAMETER. "
            "Name the two groups explicitly with --antenna-groups 'm*,e*'."
        )

    # Compared exactly, with no clustering tolerance. MSv2 stores the nominal
    # value (verified on MeerKAT+ data: exactly 13.5 and 15.0 m), and a
    # tolerance would silently merge two genuinely distinct dish classes whose
    # diameters happen to fall close together. Float noise would instead split
    # one class in two, which the m/e cross-check below catches.
    diam = np.asarray(dish_diameters, dtype=float)
    # Refuse non-finite diameters rather than let them fall through. A NaN
    # survives the clustering below (`nan - x > tol` is False) and would be
    # classified as the SMALLER dish class in silence; the m/e cross-check only
    # catches that when the antenna happens to be e-named.
    if not np.isfinite(diam).all():
        bad = antenna_names[~np.isfinite(diam)]
        raise ValueError(
            f"non-finite ANTENNA_DISH_DIAMETER for {bad.tolist()}; cannot classify these "
            f"antennas. Name the two groups explicitly with --antenna-groups 'm*,e*'."
        )
    classes = np.unique(diam)
    if len(classes) == 1:
        raise ValueError(
            f"all antennas report a single dish diameter ({classes[0]:.2f} m), so this "
            f"is not a mixed array. Drop --baseline-groups, or name the groups "
            f"explicitly with --antenna-groups 'm*,e*'."
        )
    if len(classes) > 2:
        raise ValueError(
            f"found {len(classes)} distinct dish diameters "
            f"({', '.join(f'{c:.2f}' for c in classes)} m); baseline groups are defined "
            f"for exactly two dish classes. Use --antenna-groups to name them."
        )

    is_ext = diam == classes[-1]

    # Corroborate against the m/e naming convention. A disagreement means one of
    # the two signals is wrong and we have no way to tell which.
    prefix_ext = np.array([n.lower().startswith("e") for n in antenna_names])
    prefix_mk = np.array([n.lower().startswith("m") for n in antenna_names])
    known = prefix_ext | prefix_mk
    mismatch = known & (is_ext != prefix_ext)
    bad = antenna_names[mismatch]
    if bad.size:
        raise ValueError(
            f"dish diameter and the m/e naming convention disagree for {bad.tolist()}: "
            f"diameter says {'MeerKAT+' if is_ext[mismatch][0] else 'MeerKAT'} "
            f"but the name says otherwise. Resolve with --antenna-groups 'm*,e*'."
        )

    return is_ext


def baseline_group_masks(is_ext, antenna_names, ant1_names, ant2_names):
    """Map each baseline-group label to its indices along the baseline axis.

    A baseline is MPM when its two antennas fall in different classes,
    whichever is antenna1: meerkat-beams fixes `p = MeerKAT` by convention and
    handles the reversed ordering by conjugation, so we must not sub-split.

    Args:
        is_ext: (nant,) bool array from classify_antennas.
        antenna_names: (nant,) antenna names, parallel to is_ext.
        ant1_names: (nbaseline,) baseline_antenna1_name values.
        ant2_names: (nbaseline,) baseline_antenna2_name values.

    Returns:
        {label: int index array}, in GROUP_LABELS order, ascending within each
        array. Groups with no baselines are omitted entirely, so a pure-MeerKAT
        selection returns {"MM": ...} rather than two empty entries.
    """
    lookup = {str(n): bool(e) for n, e in zip(np.asarray(antenna_names).astype(str), is_ext)}
    e1 = np.array([lookup[str(n)] for n in np.asarray(ant1_names).astype(str)])
    e2 = np.array([lookup[str(n)] for n in np.asarray(ant2_names).astype(str)])

    membership = {
        "MM": ~e1 & ~e2,
        "MPM": e1 != e2,
        "MPMP": e1 & e2,
    }
    masks = {}
    for label in GROUP_LABELS:
        idx = np.flatnonzero(membership[label]).astype(np.int64)
        if idx.size:
            masks[label] = idx
    return masks
