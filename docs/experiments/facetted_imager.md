# Faceted `pfb imager`

**Status:** design only; no code. Drafted 2026-10-27, with a task-level plan for Milestone 1
that was never started (folded in under "Milestone 1 build notes" below). Code citations were
checked against main at 0f3084b.

## Intent

Split `pfb imager` by facet, so that:

1. each facet can carry its own direction-dependent (DD) corrections (gains, beam) and its own
   averaging, and
2. the data each facet needs can be **placed** on a chosen node, so that neither imaging, nor
   the Hessian applications, nor the major cycle ever moves visibility-sized arrays between
   nodes. Only image-sized arrays cross node boundaries.

The facet layout comes from a region file. That is either polygons used as given, or point
"seeds" (usually the calibrator directions) tessellated into Voronoi or Delaunay cells.

Success for the full programme is DD-corrected, faceted imaging and deconvolution, with
visibilities that stay node-local on a multi-node Ray cluster. Success for **Milestone 1**
(the only one planned in detail so far) is: a faceted `pfb imager` run whose stitched band
products equal an unfaceted run's to gridder accuracy, with per-facet products stored in the
`.dt`. Every existing consumer (`deconv`, `restore`, FITS) must keep working unchanged on a
faceted tree.

## Global constraints

* Facets **share the tangent point** of the output grid. They differ only in the image-centre
  offset `(l0, m0)` handed to ducc as `center_x/center_y` through
  `wgridder_conventions(l0, m0)`. Visibilities are never rephased per facet, and images are
  never reprojected between facets.
* Facet windows are **pixel-aligned sub-windows of the global grid**: even `nx_f`, `ny_f`,
  facet centre at an integer global pixel. Stitching is then an exact copy, with no
  interpolation.
* Image-space arrays stay `(corr, y, x)` end to end (D19/D20). ducc's x-major world stays
  behind `.T` views at the call sites.
* The tree stays uniformly at `--precision` (D27). The facet label image is `int32`.
* Large arrays never travel as Ray task arguments or return values (D10). The facet label image
  goes through the `.dt` store, not through task args.
* Runs without `--facets` must be **byte-identical** to today's.
* No new runtime dependency. `scipy.spatial`, `scipy.ndimage` and `regions` are already in
  `[full]`.

## 1. Geometry

### 1.1 Inputs

`--facets <region file>` (any format `regions.Regions.read` auto-detects: DS9, CRTF, ...).
* **All point regions** → seeds, tessellated with `--tessellation voronoi|delaunay`
  (default `voronoi`).
* **No point regions** → shapes (polygons, circles, ...), each rasterised as one facet in file
  order. **The first region wins on overlap.**
* **Mixed** → `ValueError` naming the counts of each kind.

Sky regions go through the global grid's WCS
(`utils.fits.set_wcs(..., l0=, m0=, header=False)` with the freq/stokes axes dropped, exactly
as in `utils/degrid.py`). Pixel regions are used as is.

### 1.2 Labelling

Every pixel of the global `(ny, nx)` grid is owned by exactly one facet, recorded in an
`int32` label image `FACET_ID`.

* **Voronoi:** each pixel goes to its nearest seed, measured in pixel coordinates on the
  tangent plane (`scipy.spatial.cKDTree`). The tangent-plane distance is the right one: facets
  are regions of this one grid.
* **Delaunay:** `scipy.spatial.Delaunay(seeds).find_simplex(pixels)`. Each triangle is one
  facet. Pixels outside the convex hull go to the nearest owned pixel. Fewer than 3 seeds, or
  collinear seeds, raise `ValueError("delaunay tessellation needs at least 3 non-collinear
  seeds")`.
* **Shapes:** rasterise in file order (`to_mask(mode="center")`), first wins. Pixels that no
  shape covers go to the nearest owned pixel.
* **Nearest-owned fill** (shared by Delaunay and shapes):
  `scipy.ndimage.distance_transform_edt(labels < 0, return_indices=True)`.
* Facets that end up owning **no pixel** (seed off the grid, a shape fully shadowed by earlier
  ones) are dropped with a warning naming their region index. The labels are then renumbered
  contiguously `0..nfacet-1` in order of first appearance in the file. Zero surviving facets
  → `ValueError`.

**Why Voronoi is the default.** Voronoi puts each seed (calibrator) at the centre of its cell,
which is where a piecewise-constant DD solution is most valid. Delaunay puts the seeds on the
triangle vertices, i.e. on corners shared by about 6 facets, which is exactly where bright
sources should not be. Delaunay's appeal, interpolating gains across a triangle, cannot be
used while one facet applies one gain. It stays as an option to experiment with.

### 1.3 Windows

For each facet, take the bounding box of its owned pixels, `[ylo, yhi) × [xlo, xhi)`, and make
each side even. If a side is odd, extend `hi` by one when `hi < n`, else extend `lo` by one
downwards. The global `nx`, `ny` are always even (`set_image_size`), so this always stays
inside the grid.

Window centre in global pixels: `(cy, cx) = (ylo + ny_f//2, xlo + nx_f//2)`.

Facet offset, adding to the global `(l0_g, m0_g)` (the `--target` offset), from the measured
wgridder ground truth (`docs/wiki/image-and-beam-orientation.md` §2: `ix` runs l-descending,
`iy` m-ascending):

```
l0_f = l0_g + (nx//2 - cx) * cell_rad
m0_f = m0_g + (cy - ny//2) * cell_rad
```

The stitch-equals-global test (build notes, below) is the authority on these signs. If the test disagrees,
change the formula, not the test.

Facet PSF size: same rule as the global one, applied to the facet window. That is
`psf_oversize * max(nx_f, ny_f)`, made even with `good_size`, and at least 128. N full-size
PSFs would cost N× the global PSF gridding, which is prohibitive.

### 1.4 Restrictions

* `--target` naming a moving body (time-dependent offset) together with `--facets` is refused:
  the labels are computed once and must be the same for every output image.
  `HH:MM:SS,DD:MM:SS` targets are fine.

## 2. Tree schema

```
<out>_<PRODUCT>.dt/
  (root attrs) ... existing ..., nfacet            # nfacet = 1 when unfaceted
  facets/                                          # only when --facets
      attrs: nfacet, tessellation ("voronoi"|"delaunay"|"shapes"), facets_file
      vars:  FACET_ID (y, x) int32
             WINDOW (facet, four) int64            # ylo, xlo, ny_f, nx_f
             L0M0  (facet, two) float64            # l0_f, m0_f
  band{b}_time{t}/
      ... band node exactly as today (DIRTY/BDIRTY stitched from the facets) ...
      part{p}/   ... exactly as today ...
      facet{f}/  (only when --facets)
          attrs: facetid, l0, m0, ylo, xlo
          vars:  DIRTY, BDIRTY (corr, y, x)          # the facet window, ny_f × nx_f
                 PSF (corr, y_psf, x_psf), PSFPARSN (corr, bpar)   # only with --psf
                 WSUM (corr,)
```

* The band node is **unchanged in meaning**. DIRTY and BDIRTY are stitched from facet windows
  through `FACET_ID`. BEAM, WSUM, PSF and PSFPARSN are computed exactly as today, with the
  band PSF still gridded at the global centre. So every existing consumer works on a faceted
  tree without modification.
* Partitions stay at band level in M1, because every facet sees the same visibilities.
  Visibilities move down to `facet{f}/part{p}` only in M2, once they differ per facet (DD gains,
  per-facet BDA).
* **The only consumer change:** code that walks `band.children` assuming they are all
  partitions must filter to `part*`. Known sites: `operators/band_worker.py` (`load_bands`, `sorted(band.children)`) and
  `core/deconv.py` (the PSFHAT check on the first child).

## 3. Pass 2 with facets

Per partition (inside the existing `_grid_image_body` loop):

1. Compute the imaging weights **once** (`imaging_weights`). They do not depend on the facet
   offset (`usign`/`vsign` come from the fixed flips).
2. Global PSF (if `do_psf`), BEAM, WSUM and the partition node: exactly as today.
3. For each facet: grid its dirty image (`grid_dirty`) into `(ncorr, ny_f, nx_f)` at
   `(l0_f, m0_f)`, and its PSF (`grid_psf`) at the facet PSF size. Accumulate the per-facet
   DIRTY, BDIRTY (`BEAM_p[window] * dirty_pf`), PSF and WSUM.

After the loop: stitch DIRTY and BDIRTY into the band arrays, write the `facet{f}` nodes, then
the band node as today. `--transfer-model-from` keeps working: `residual_from_partitions` runs
on the band level and the full grid, untouched.

**Cost:** gridding has a per-visibility term, so M1 grids every visibility once per facet. That
is inherent to visibility-domain faceting. M2's per-facet BDA is what pays it back (§5).

## 4. Milestones

Each milestone gets its own plan when it starts.

| | Milestone | Deliverable |
|---|---|---|
| **M1** | Geometry + faceted pass 2 + schema | §1–3 here; build notes below |
| M2 | DD gains + per-facet datagroups | `imager --gain-table` with a `direction` axis (lifting the single-direction refusals in `utils/gains.load_node_gains` and `utils/weighting.weight_data`, D63); facet→direction map (nearest seed, or explicit in the region text); pass 1 applies facet gains and per-facet BDA and writes per-facet pieces; partitions move to `facet{f}/part{p}` |
| M3 | Facet-aware deconvolution | `(band, facet)` worker pool; facet-local approximate Hessian `H_ff` (PSF-based, per-facet PSFHAT); prox on the **stitched** image on the driver (wavelets must cross seams); exact residual = Σ over facets of the degrid in each facet's datagroup |
| M4 | Placement and locality | Ray `NodeAffinitySchedulingStrategy`; time chunks pinned to nodes, balanced by visibility volume; `(time chunk, facet)` datagroups on node-local disk; major cycle as node-local predict-all-facets → residual → per-facet grid, with an image-sized reduce |
| M5 | DD degrid | `pfb degrid` applying facet gains, for calibration feedback |

## 5. Design positions carried into M2–M4

**Data layout is time-major, compute is facet-major, and the two are co-located.** The exact
residual for facet f, `R_fᴴ G_fᴴ W (V − Σ_g G_g R_g x_g)`, needs **every** facet's prediction
summed at full visibility resolution. Facet-major storage on different nodes cannot form that
sum without moving visibility-sized data, short of approximating it away.

So time chunks are pinned to nodes, and a node holds all facets' datagroups for its time chunks.
A major cycle then works like this:
1. broadcast the facet models (image-sized);
2. each node predicts all facets for its own rows;
3. it forms the residual locally and grids per-facet residual images;
4. those images are reduced across nodes (image-sized).

Nothing visibility-sized ever leaves a node. The forward step uses the cheap facet-local
`H_ff`, which is all PFB needs, since only the residual has to be exact.

**Data volume.** The decorrelation limit scales inversely with facet radius. With N facets of
linear size FoV/√N, time and frequency averaging can each grow by about √N, so the total over
facets is roughly the original Nvis, not N×Nvis. Two things cap this: DD solution intervals
limit time averaging, and the number of output bands limits frequency averaging.

## 6. Open questions (not blocking M1)

* Facet overlap or pad for the M3 minor cycle (sidelobes of bright sources near seams). It will
  probably be a `--facet-pad` in PSF widths, cropped away at stitching.
* Whether the per-facet PSF needs a time-varying DD part, or one per `(band, facet)` is enough.
* Ray object store versus node-local zarr for M4 datagroups. Large ones must not go through the
  object store (`docs/wiki/memory-and-ray.md`).
* The break-even facet size below which ducc's single full-field pass is cheaper. This needs a
  benchmark in M2, once BDA is per facet.
* Moving-body `--target` with facets (relabel per output image).

## Milestone 1 build notes

From the M1 plan. Five tasks, each with its tests first:

1. **`utils/facets.py`: labelling and windows.** `label_voronoi`, `label_delaunay`,
   `label_shapes`, the nearest-owned fill, `facet_windows` (the even-window rule of §1.3) and
   `stitch`.
2. **Region-file reader** (`facet_labels`). Reads the region file, classifies it as seeds or
   shapes (§1.1), and converts sky regions through the grid WCS.
3. **Split `operators/gridder.grid_partition` into three pieces.** `grid_partition` keeps its
   signature and output exactly.
   - `imaging_weights(part, counts, robustness, nx_pad, ny_pad, cell_rad, overwrite_weight=False) -> (wgt, wsum)`
   - `grid_dirty(uvw, freq, vis, wgt, mask, nx, ny, cell_rad, l0, m0, ...) -> (corr, ny, nx)`
   - `grid_psf(uvw, freq, wgt, mask, nx_psf, ny_psf, cell_rad, l0, m0, ...) -> (PSF, PSFHAT, PSFPARSN)`
   - `facet_psf_size(n, psf_oversize)`, the PSF-size rule factored out of
     `utils/misc.set_image_size`, which then calls it.
4. **Faceted pass 2** in `core/imager._grid_image_body`, plus the consumer filtering to
   `part*` (§2).
5. **Driver, CLI and docs.** Adds `--facets` and `--tessellation`, and refuses a moving-body
   `--target` (§1.4).

The cases the review should check, each with its test:

1. **A seed that falls off the image** (a catalogue wider than the FoV). Its facet owns no
   pixels, so it is dropped with a warning and the remaining facets are renumbered. No crash,
   no empty window. (`test_seed_off_grid_dropped`)
2. **A single seed, or one polygon covering everything.** This gives one facet equal to the
   whole grid, and the stitched DIRTY equals the unfaceted DIRTY.
   (`test_single_facet_matches_unfaceted`)
3. **Overlapping polygons, or polygons with gaps.** The first region wins and gaps go to the
   nearest owned pixel, so every pixel is owned.
   (`test_shapes_overlap_first_wins_and_gaps_filled`)
4. **A facet touching the grid edge with an odd bbox side.** The even-window rule stays
   inside `[0, n]`. (`test_windows_even_and_inside_grid`)
5. **Robust weighting and single precision.** Weights are computed once, WSUM is identical
   across facets, and the ducc dtype is consistent.
   (`test_facet_stitch_matches_global[single-robust]`)

Left out of M1 on purpose: per-facet PSFHAT and per-facet partitions (M2/M3),
`--facet-pad` (M3), and per-facet FITS output.
