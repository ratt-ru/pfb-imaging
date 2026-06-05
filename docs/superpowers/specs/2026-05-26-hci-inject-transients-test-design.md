# Design: `test_hci_inject_transients`

Date: 2026-05-26

## Goal

Add an integration test for the transient-injection feature of the `hci`
sub-command. The test must verify that a transient injected via the
`inject_transients` parameter ends up:

1. at the **correct pixel** in the output cube, and
2. with the **correct flux as a function of time and frequency**.

The test reads these values directly from the raw `cube` variable by placing
the transient on an exact pixel centre, so the flux can be looked up without
PSF-spreading ambiguity.

## Background (how injection works)

- `hci()` reads the YAML config, builds time/frequency profiles over the full
  `(time, freq)` domain via `generate_transient_spectra`, and writes them to a
  sidecar zarr named `<config_basename>.zarr` (next to the YAML).
- Each worker (`stokes2im`) reads that sidecar, converts the transient RA/Dec to
  direction cosines `(l, m)` with `radec_to_lm` relative to the MS phase centre,
  forms `dspec = tprofile[:, None] * fprofile[None, :]`, phases it onto `(l, m)`,
  and adds it to the Stokes-I visibilities (`data[:, :, 0] += dspec`).
- The dirty image is normalised by the PSF peak, so a point source reads its
  flux in Jy/beam at its location.

## Test environment facts (test MS)

- Field phase centre: RA = 0°, Dec = 30°.
- 60 integrations @ 60 s (span 3540 s); 8 channels 1.0–1.7 GHz @ 100 MHz.
- With `field_of_view=0.5`, `super_resolution_factor=2.0`: cell ≈ 8.82″,
  `nx = ny = 210`, half-FOV ≈ 0.257°.
- `wgridder_conventions(0,0)` → `flip_u=False, flip_v=True, flip_w=False`.
- Cube dims `(STOKES, FREQ, TIME, Y, X)`; X/Y transposed in `stokes2im`.

## Decisions

- **Placement:** exact pixel centre derived in `(l, m)` space, plus an
  empirical `argmax` position check. (Not approximate placement; not phase
  centre.)
- **Config source:** generated in `tmp_path` (self-contained; avoids writing the
  sidecar zarr into the repo). The committed `tests/transient1.yaml` is *not*
  used by the test.
- **Weighting:** natural / unity weights (`robustness=None`) so the per-bin flux
  factorises analytically.
- **Profile shape:** broadened Gaussian (peak mid-early, larger sigma) so several
  time bins are clearly populated.
- **Scale factor:** if a trial run shows an unexplained constant between
  recovered and injected flux, **stop and report** rather than encode a tolerance
  or fudge factor.

## Design

### Isolation of the transient

- `data_column="DATA-DATA"` → `dc1=dc2="DATA"`, `operator="-"`, so
  `data = DATA - DATA = 0`; the injected transient is the only signal in the
  cube. Injection happens *after* the column arithmetic and (absent) rephasing.
- `product="I"` (injection is Stokes-I only), `beam_model=None`,
  `robustness=None`, `phase_dir=None`.

### Exact pixel-centre placement

1. Compute geometry in-test from the `ms_meta` fixture:
   `cell_n = 1/(2·max_blength·max_freq/c)`, `cell_rad = cell_n/2.0`,
   `nx = ny` via `good_size` exactly as `set_image_size` (fov=0.5 → 210).
2. Choose integer pixel offsets from centre, e.g. `(di, dj) = (-20, 16)`, well
   inside the half-FOV.
3. Target `(l, m) = (di·cell_rad, dj·cell_rad)`; invert the SIN projection about
   `(ra0, dec0) = (0°, 30°)`:
   - `n = sqrt(1 - l² - m²)`
   - `dec = arcsin(m·cos(dec0) + n·sin(dec0))`
   - `ra  = ra0 + arctan2(l, n·cos(dec0) - m·sin(dec0))`
4. Guard: assert `radec_to_lm([[ra,dec]], [[ra0,dec0]]) ≈ (l, m)`.

### Config (written to `tmp_path`)

- One transient, gaussian time profile: `peak_time≈1200 s` (mid-observation),
  `duration` (σ) `≈600 s` → with the binning below, ~8–10 of the 15 time bins are
  meaningfully populated and trace a smooth Gaussian with good contrast
  (edge ≈ 0.13 → peak 1.0). Exact `peak_time`/σ tuned during implementation so
  several time bins are bright while task count stays moderate.
- Power-law spectrum: `peak_flux=2 Jy`, `reference_freq=1.4e9`,
  `spectral_index=-1.5`.
- `position.ra`, `position.dec` from the inversion above (degrees).

### Binning (≈60 gridding tasks)

- `integrations_per_image=4` → 15 time bins.
- `channels_per_image=2` → 4 freq bands.
- `images_per_chunk=15`, `max_simul_chunks=1`, `nworkers=1`, `nthreads=1`,
  `epsilon=1e-7`, `overwrite=True`, `keep_ray_alive=True`,
  `log_directory=tmp_path/logs`.

### Expected flux

- Call `generate_transient_spectra(utime, freq, transient_dict)` directly to get
  `tprofile` (over `utime`) and `fprofile` (over `freq`) exactly as the code does.
- Per cube cell `(t_bin, f_bin)`:
  `expected = mean(tprofile over integrations in t_bin) × mean(fprofile over channels in f_bin)`,
  using contiguous blocks (`integrations_per_image`, `channels_per_image`).
- Assert cube `TIME`/`FREQ` coords equal the mean of the assumed blocks, making
  the contiguous-binning assumption explicit and checked.

### Assertions

1. **Position:** `argmax` over `(Y, X)` of `cube[0, f_mid, t_peak]` equals the
   pixel predicted from the cube's own coordinate labels
   (`ix = argmin|out_ras − ra|`, `iy = argmin|out_decs − dec|`). Catches
   projection/sign-convention bugs.
2. **Flux:** `cube[0, f, t, iy, ix] ≈ expected[f, t]` for all bins; bright bins
   match the profile, near-zero bins confirm temporal/spectral localisation.
   Tolerance calibrated from a trial run (start strict).

### Read from `cube`, not `cube_mean`

With zeroed base data, the per-bin RMS flag logic (`rms > 1.5·median_rms`) flags
the bright transient bins (median RMS ≈ 0), suppressing them in `cube_mean`. The
raw `cube` is unaffected. A short comment will note this.

## Open item to resolve during implementation

Whether the PSF-peak normalisation returns the injected flux *exactly* for an
off-centre source (w-term / `divide_by_n` interaction). Verify against a trial
run; per the decision above, an unexplained scale factor is reported, not
absorbed.

## Out of scope

- Beam application to transients (`BeamWizard` path is `NotImplementedError`).
- Multiple transients, polarisation other than I, periodicity.
- Rephasing (`phase_dir`) interaction with injection.
