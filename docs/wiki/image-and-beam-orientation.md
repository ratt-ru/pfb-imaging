---
type: Subsystem Notes
title: Image and beam orientation conventions (hci)
description: The measured axis conventions of the wgridder image, the hci cube/FITS header, BeamWizard beam maps and reproject_interp; the post-mortem of the transpose+flip beam hack; and the corrected reprojection construction.
tags: [hci, beam, orientation, wcs, reproject, wgridder, conventions]
timestamp: 2026-07-16T12:00:00Z
last_verified_commit: c4ae544
---

# Image and beam orientation conventions (hci)

Three different array orders coexist on the `hci` beam path. Every historical
alignment bug (including the "hack to get the images to align",
`547458f`/`330bc5d`) came from conflating them. The facts below were **measured**
with synthetic point sources / asymmetric beams, not derived from documentation —
re-run the pinning test (planned, see Status) before "fixing" anything here.

## 1. The three array orders

| Object | Index order | Axis directions |
|--------|-------------|-----------------|
| ducc wgridder images (`vis2dirty`/`dirty2vis` view of the image) | `arr[ix, iy]`, shape `(nx, ny)` | `ix`: l **descending** (East at low `ix`, RA descending); `iy`: m **ascending** (Dec ascending) |
| hci cube / FITS data (`cube` dims `(STOKES, FREQ, TIME, Y, X)`; legacy `save_fits` writes `(..., ny, nx)`); **all `stokes_image` working arrays** (`residual`, `psf`, `pbeam`) | `arr[iy, ix]` | `X` coord `out_ras` descending (header `CDELT1 = -cell`); `Y` coord `out_decs` ascending (`CDELT2 = +cell`) |
| `BeamWizard.get_rotation_averaged_beam` maps (meerkat-beams ≥ `616906b`) and `bds` variables (dims `(..., Y, X)`) | `arr[iy, ix]` — FITS convention | direction follows the l/m grid handed in (or the wizard's image grid, which inherits the header's `CDELT` signs) |

The wgridder image and the cube/FITS layer are exactly each other's transpose
(no flips). On the hci path the (Y, X) order is canonical end to end and **no
data-moving transposes exist**: ducc's x-major world is confined to the
`vis2dirty` call sites, which fill the `(ny, nx)` buffers through zero-copy
transposed views (`dirty=buf.T` — ducc accepts strided output; verified to
1e-12 against the C-ordered return path). The other x-major citizen is
`fitcleanbeam` (shared with the legacy `.dds` path, PA convention defined by
its input axes); `stokes_image` calls it with `yx_order=True`, which adapts via
an internal zero-copy view and returns bitwise-identical parameters.

## 2. Measured ground truth for the wgridder image

With pfb's fixed convention set `wgridder_conventions(0,0) = (flip_u=False,
flip_v=True, flip_w=False)`, a unit point source at offset (l, m) (l East, m
North) whose visibilities follow

    V(u,v,w) = exp(+2πi (u·l + v·m + w(n-1)) ν / c)

peaks at `(ix, iy) = (nx//2 - l/cell, ny//2 + m/cell)` — i.e. the effective sign
convention of real MS data through this pipeline is the **+2πi** one (the −2πi
convention produces a 180°-rotated image). This is what makes the hci coordinate
assignment (`out_ras` descending along `ix`, `out_decs` ascending along `iy`)
and the legacy `save_fits` transpose (`(..., nx, ny) → (..., ny, nx)`, no flips,
`CDELT1 = -cell`) label the sky correctly — long validated in production.

The hci reference header (`core/hci.py`, scaffold construction):
`CTYPE = [RA---SIN, DEC--SIN, ...]`, `CDELT = [-cell_deg, +cell_deg, ...]`,
`CRPIX = [1 + nx//2, 1 + ny//2, ...]`, `CRVAL = radec_new` (the rephased/
barycentre target, **not** the per-scan pointing).

## 3. BeamWizard semantics that matter here

* `BeamWizard(beam_model, output_dataset)` attaches the hci scaffold, so
  `wizard.centre` = the **image** centre (`radec_new`), and the default l/m grid
  is `l_grid = (arange(nx) - (crpix1-1)) * cdelt1` — descending, because
  `cdelt1 < 0`. Parallactic angles are therefore computed at the image centre,
  not the per-scan pointing: a small approximation, fine for rotation averaging.
* `get_rotation_averaged_beam` returns maps in **(Y, X)** order since
  meerkat-beams `616906b` (pinned there by
  `test_rotation_averaged_beam_map_indexes_as_y_x`); with 1D `l`/`m` inputs the
  shape is `(len(m), len(l))`. Before that commit it returned (X, Y) — the era
  in which pfb's `(nx, ny)` buffers were filled from it.
* `bds.X`/`bds.Y` (the MdV beam grid) are **degrees**, centred on 0, ascending
  (`mdv_beams_to_bds`: `degs = margin_deg`, `CRPIX = len//2 + 1`,
  `CDELT = degs[1]-degs[0] > 0`).

## 4. `reproject_interp` semantics (the root of the confusion)

For a 2-axis celestial WCS, astropy/reproject binds the numpy array as
`arr[row, col]` with **col ↔ WCS axis 1 (RA-like)** and **row ↔ WCS axis 2
(Dec-like)** — i.e. arrays must be **(Y, X)-ordered**, and `WCS.array_shape` is
in numpy `(rows, cols)` order. For `RA---SIN` the intermediate coordinate
`(pixel - crpix) * cdelt1` **is l** (a direction cosine, East positive), so a
beam map whose columns run l-ascending is correctly described by a *positive*
`CDELT1` — the output image header has negative `CDELT1` only because its X axis
runs East-to-West.

`reproject_and_interp_beam` / `reproject_and_interp_scat_beam` (pre-fix state at
`9a46876`) violate this three ways:

1. **Transposed feed** — they pass (X, Y)-ordered `(l, m)` maps, so reproject
   reads the l axis as Dec and the m axis as RA. Measured consequence: with
   identical input/output centres the call degenerates to a pure copy (the array
   stays (X, Y)-ordered while downstream treats the result per its own
   bookkeeping), and a rephasing offset is applied **along the wrong array
   axis** (an eastward `radec_new` offset shifted the map along m).
2. **Target `crpix` off by one** — `[nxo//2, nyo//2]` vs the reference's (and the
   real header's) `[1 + n//2, 1 + n//2]`: every reprojected beam is shifted by
   one pixel in both axes. For a σ≈0.2° Gaussian on 36″ pixels this alone is a
   **4.3 % of peak** error.
3. **Target `CDELT1` sign** — `+cell` instead of the output header's `-cell`, so
   even a correctly-fed map would come out East-West mirrored.

## 5. Post-mortem of the transpose+flip hack

The hack (`pbeam = pbeam.transpose(0, 2, 1); pbeam = pbeam[:, ::-1, :]`,
introduced in `547458f`, removed from the wizard branch in `330bc5d`, still
present in the zarr-beam branch at `stokes2im.beam_for_band`) composed with the
buggy reprojection to give, at wgridder pixel offset `(p, q)` from centre, the
beam value at `(l, m) = (q, -p)·cell` where the correct value is at
`(-p, -q)·cell`. Those differ by a 90° rotation ∘ reflection — **identical for a
circularly symmetric beam**, which is why the images "aligned": the MeerKAT
rotation-averaged beam is nearly circular. Measured against an analytic
reference (identical grids, no rephasing):

* circular beam: max error 4.3 % of peak (entirely the 1-pixel `crpix` shift);
* elliptical beam (axial ratio 0.6, PA 30°): max error **21 %** of peak;
* any rephasing offset lands on the wrong axis regardless of beam shape
  (worst case for MeerKLASS drift scans, where per-scan pointings are rephased
  to a common target mostly along RA).

It also silently required `nx == ny` (the transpose swaps the axes' lengths).

The `[:, ::-1, :]` flip is *not* the physical feed-plane→sky flip mentioned in
the zarr branch's "flip the beam upside down" comment — that question (MdV beams
transmissive vs receptive, feed vs sky parity) is still open for the zarr path
and is owned by meerkat-beams for the wizard path (its PR #8 M1 validation).

## 6. The corrected construction (validated)

Feed **(Y, X)-ordered** maps; describe the beam grid honestly; make the target
WCS *equal* the real output header:

```python
# beam[iy, ix] on the bds grid, around the pointing centre radec0
wcs_ref.wcs.cdelt = (l_beam[1] - l_beam[0], m_beam[1] - m_beam[0])  # signed, +ve for bds
wcs_ref.wcs.crval = radec0_deg
wcs_ref.wcs.crpix = (1 + (0 - l_beam[0]) / dl, 1 + (0 - m_beam[0]) / dm)

# target = the hci output header, exactly
wcs_target.wcs.cdelt = (-cell_deg, cell_deg)
wcs_target.wcs.crval = radec_new_deg
wcs_target.wcs.crpix = (1 + nx // 2, 1 + ny // 2)

pbeam = reproject_interp((beam_yx, wcs_ref), wcs_target, shape_out=(ny, nx))[0]
# already cube (Y, X)-ordered — used as-is by stokes_image
```

Validated against the analytic reference: exact (≤ 5e-5, bilinear interpolation
error) for circular and elliptical beams, with and without rephasing, blob
positions land on the predicted output pixels (astropy `world_to_pixel`
cross-check).

## 7. Status

The §6 construction is implemented: `reproject_and_interp_scat_beam` takes the
1D `l_beam`/`m_beam` coordinates (signed cdelt, coord-derived crpix, target WCS
= the hci header) and returns cube-ordered `(nstokes, ny, nx)` maps;
`beam_for_band`'s wizard branch evaluates the beam on the coarse `bds` grid
around the pointing and reprojects `radec → radec_new`. `stokes_image` works in
(Y, X) order throughout, so the beam is used as returned and the cube/psf/
beam_weight are written without transposition — the only x-major seams are the
`vis2dirty` calls (zero-copy `dirty=buf.T` views) and the `fitcleanbeam` call
(`yx_order=True`); see §1 and design-decisions.md D19. Non-square images work.
Pinned by `tests/test_beam_orientation.py` (point-source wgridder orientation,
elliptical-beam reproject vs analytic with rephasing and non-square output,
`fitcleanbeam` order-invariance), and the refactor was verified bitwise-
equivalent in output against the pre-refactor code on the test MS (cube/psf to
single-precision threading noise ~1e-7; `psf_pa` exactly). The interim
no-reprojection state (`a516530`, the jagged-gain sampling experiment for
breifast#208 — which ruled out coarse sampling as the cause) is gone. The
zarr-beam branch and its hack remain untouched, documented debt (§5,
design-decisions.md Known debt).

Sources: `src/pfb_imaging/utils/stokes2im.py` (`beam_for_band`, coordinate
assignment), `src/pfb_imaging/utils/beam.py`, `core/hci.py` (scaffold header),
`utils/fits.py` (`save_fits`/`set_wcs`), commits `547458f`, `330bc5d`,
`a516530`; meerkat-beams `616906b` + PR landmanbester/meerkat-beams#8;
ratt-ru/pfb-imaging#263, ratt-ru/breifast#208.
