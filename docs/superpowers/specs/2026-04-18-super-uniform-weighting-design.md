# Super-uniform weighting — design

**Date:** 2026-04-18
**Status:** approved, pending implementation plan

## Goal

Add super-uniform weighting (Briggs 1995) to the imaging weighting pipeline. Super-uniform generalises uniform weighting: instead of normalising each visibility by the weight count in its own uv-cell, the normaliser is the sum of counts over a (2·`npix_super`+1)² box centred on that cell. `npix_super = 0` recovers standard uniform weighting exactly.

## Non-goals

- Cab definitions: the `update-cabs` workflow regenerates `src/pfb_imaging/cabs/grid.yml` automatically. This spec leaves cab files untouched.
- Plumbing `npix_super` into any command other than `grid`.
- Changes to `counts_to_weights`, `_compute_counts`, or `filter_extreme_counts`.

## Approach

Implement super-uniform weighting as a **box-filter pre-processing step** applied to the counts grid between `filter_extreme_counts` and `counts_to_weights`. The existing Briggs/robust normalisation inside `counts_to_weights` then operates on the smoothed counts, yielding "super-robust" for free when `robustness` is set alongside `npix_super`.

New pipeline order in `image_data_products`:

```
_compute_counts → filter_extreme_counts → box_sum_counts → counts_to_weights
```

When `npix_super = 0`, `box_sum_counts` is a no-op and the pipeline is bit-for-bit identical to the current implementation.

## Components

### 1. `src/pfb_imaging/utils/weighting.py`

Add one function:

```python
def box_sum_counts(counts, npix_super):
    """Box-sum the counts grid for super-uniform weighting.

    Replaces each cell of counts with the sum over a (2*npix_super+1)^2
    window centred on that cell, with zero-padding at image edges.
    Returns counts unchanged when npix_super <= 0.

    Args:
        counts: array of shape (ncorr, nx, ny).
        npix_super: half-size of the box in pixels. 0 => standard uniform.

    Returns:
        Array of shape (ncorr, nx, ny) with box-summed counts.
    """
    if npix_super is None or npix_super <= 0:
        return counts
    from scipy.ndimage import uniform_filter
    size = 2 * npix_super + 1
    out = np.empty_like(counts)
    for c in range(counts.shape[0]):
        out[c] = uniform_filter(counts[c], size=size, mode="constant", cval=0.0) * (size * size)
    return out
```

Notes:
- `scipy` is already a dependency (already imported elsewhere in the module).
- `uniform_filter(..., mode="constant", cval=0.0)` × `size²` yields the windowed sum with zero-padding at edges.
- `scipy.ndimage` is imported inside the function to keep module import cost low.

### 2. `src/pfb_imaging/operators/gridder.py`

- Add `npix_super: int = 0` to `image_data_products`.
- Import `box_sum_counts` alongside the existing `_compute_counts, counts_to_weights, filter_extreme_counts` import.
- Between the existing `filter_extreme_counts` call (line ~560) and the `counts_to_weights` call (line ~561), insert:

```python
counts = box_sum_counts(counts, npix_super)
```

### 3. `src/pfb_imaging/core/grid.py`

- Add `npix_super: int = 0` to the `grid()` signature (alongside `filter_counts_level`).
- Forward it to `rimage_data_products.remote(...)` (alongside `filter_counts_level=filter_counts_level`).

### 4. `src/pfb_imaging/cli/grid.py`

- Add a `npix_super` Typer option with `rich_help_panel="Weighting"`, default `0`:

```python
npix_super: Annotated[
    int,
    typer.Option(
        help="Half-size of the box used for super-uniform weighting. "
        "Each visibility is normalised by the sum of counts over a "
        "(2*npix_super+1)^2 box around its uv-cell. 0 (default) "
        "recovers standard uniform weighting. Combines with robustness "
        "to give super-robust weighting.",
        rich_help_panel="Weighting",
    ),
] = 0,
```

- Pass it through to both call sites: `grid_core(..., npix_super=npix_super, ...)` and the `run_in_container` kwargs dict.

Cab definitions are auto-generated — leave `src/pfb_imaging/cabs/grid.yml` untouched.

## Testing

Extend `tests/test_weighting.py`:

1. Identity test: `box_sum_counts(counts, 0)` returns `counts` unchanged.
2. Correctness test: on a small hand-built counts grid (shape `(1, 5, 5)`), verify that `box_sum_counts(counts, 1)` equals the 3×3 neighbourhood sum at each cell, including zero-padding at edges.

Existing `test_counts` continues to pass because the gridder default keeps `npix_super=0`.

## Acceptance criteria

- `uv run ruff format . && uv run ruff check . --fix` is clean.
- `uv run pytest -v tests/test_weighting.py` passes.
- Running `pfb grid --help` shows the new `--npix-super` option.
- With `npix_super=0`, dirty images and PSFs are bit-for-bit identical to the pre-change output for any value of `robustness`.

## Out-of-scope follow-ups

- Exposing `npix_super` to other CLI commands (`init`, `degrid`, etc.) — can be wired on demand later.
- A pure-numba implementation of `box_sum_counts` — scipy is fast enough; revisit only if profiling shows it as a hot spot.
