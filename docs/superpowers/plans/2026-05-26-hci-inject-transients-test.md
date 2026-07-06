# test_hci_inject_transients Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `test_hci_inject_transients` in `tests/test_hci.py`, verifying that a transient injected via `inject_transients` lands at the correct pixel and with the correct flux as a function of time and frequency.

**Architecture:** Single integration test. It zeroes the base visibilities with `data_column="DATA-DATA"` so the cube contains only the injected transient, places the transient on an exact pixel centre by choosing integer `(l, m)` offsets and inverting the SIN projection to RA/Dec, runs `hci` once with natural weighting, then asserts (a) the `argmax` of a bright cube slice is the predicted pixel and (b) the per-`(time, freq)` flux at that pixel matches the analytically binned dynamic spectrum.

**Tech Stack:** pytest, numpy, xarray/zarr, daskms, africanus (`radec_to_lm`), ray, ducc0; existing `pfb_imaging.core.hci.hci`, `pfb_imaging.utils.misc.set_image_size`, `pfb_imaging.utils.transients.generate_transient_spectra`.

---

## Context the engineer needs

- `hci` is `from pfb_imaging.core.hci import hci as hci_core` (already imported in `tests/test_hci.py`).
- Fixtures (from `tests/conftest.py`): `ms_name` (path to the test MS), `ms_meta` (SimpleNamespace with `utime` [60 values, ascending], `freq` [8 values, 1.0–1.7 GHz ascending], `ntime=60`, `nchan=8`, `max_blength`, `max_freq`), `tmp_path` (pytest builtin). Ray is started session-wide by the autouse `manage_ray` fixture, so pass `keep_ray_alive=True`.
- Test MS field phase centre: RA=0°, Dec=30°. Read it in-test from the FIELD subtable rather than hardcoding.
- `data_column="DATA-DATA"` parses to `dc1=dc2="DATA"`, `operator="-"`, giving `data = 0`; the transient is added afterwards inside `stokes2im`.
- `set_image_size(max_blength, max_freq, field_of_view, super_resolution_factor)` returns `(nx, ny, nx_psf, ny_psf, cell_n, cell_rad, cell_deg)`. With fov=0.5, srf=2.0 → `nx=ny=210`, `cell_rad≈4.276e-5 rad`. Centre pixel index is `nx//2` (where `l=0`), matching `crpix=1+nx//2`.
- `generate_transient_spectra(times, freqs, transient_dict)` returns `(time_profile, freq_profile)`; it internally subtracts `times[0]` (so `peak_time` is seconds from start). `transient_dict` needs `time` and `frequency` keys (`position` is not used by this function).
- The transient sidecar zarr is written next to the YAML, so writing the YAML into `tmp_path` keeps the repo clean.
- `channels_per_image` = channels per image (cpi=2 → 4 bands); `integrations_per_image` = unique times per image (ipi=4 → 15 time bins). 60 % 4 == 0 and 8 % 2 == 0, so bins are clean contiguous blocks.
- Cube dims are `(STOKES, FREQ, TIME, Y, X)`; `ds.X` holds RA labels, `ds.Y` holds Dec labels.
- Read flux from the raw `cube` variable, **not** `cube_mean`: with zeroed base data the RMS-based flag logic (`rms > 1.5·median_rms`, median≈0) flags the bright transient time bins, which would suppress them in `cube_mean` only.

## File structure

- Modify: `tests/test_hci.py` — replace the stub `test_hci_inject_transients` (currently ends the file at lines 289-293, with no trailing newline) and add a module-level helper `_lm_to_radec`. Add imports `yaml`, `set_image_size`, `generate_transient_spectra`, `radec_to_lm`, `xds_from_table`.

---

## Task 1: Position check (scaffold + exact pixel placement)

**Files:**
- Modify: `tests/test_hci.py` (imports near top; new helper after imports; replace stub `test_hci_inject_transients`)

- [ ] **Step 1: Add imports**

At the top of `tests/test_hci.py`, add `import yaml` next to the other stdlib/third-party imports, and add these alongside the existing `from pfb_imaging.core.hci import ...` block:

```python
import yaml
from africanus.coordinates import radec_to_lm
from daskms import xds_from_table

from pfb_imaging.utils.misc import set_image_size
from pfb_imaging.utils.transients import generate_transient_spectra
```

- [ ] **Step 2: Add the SIN-inversion helper**

Add this module-level helper (e.g. just below the imports, before the first test):

```python
def _lm_to_radec(l, m, ra0, dec0):
    """Invert the SIN (orthographic) projection.

    Args:
        l, m: direction cosines relative to the phase centre.
        ra0, dec0: phase-centre RA/Dec in radians.

    Returns:
        (ra, dec) of the (l, m) point in radians.
    """
    n = np.sqrt(1.0 - l**2 - m**2)
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    ra = ra0 + np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    return ra, dec
```

- [ ] **Step 3: Replace the stub test with the position-only version**

Replace the stub (the `def test_hci_inject_transients(...)` block at the end of the file) with:

```python
def test_hci_inject_transients(ms_name, ms_meta, tmp_path):
    """Injected transient lands at the expected pixel with the expected dynamic spectrum.

    The base visibilities are zeroed via data_column="DATA-DATA" so the cube
    contains only the injected transient. The transient is placed on an exact
    pixel centre by choosing integer (l, m) offsets and inverting the SIN
    projection, with natural weighting so the per-bin flux is analytic.
    """
    out = str(tmp_path / "hci_transients.zarr")

    # --- geometry: must match what hci computes internally ---
    nx, ny, _, _, _, cell_rad, _ = set_image_size(ms_meta.max_blength, ms_meta.max_freq, 0.5, 2.0)

    # phase centre of the test field (radians)
    field = xds_from_table(f"{ms_name}::FIELD")[0]
    ra0, dec0 = (float(v) for v in field.PHASE_DIR.values.squeeze())

    # place the transient an integer number of pixels off centre (well inside
    # the ~0.257 deg half-FOV) so it sits exactly on a pixel centre.
    di, dj = -20, 16  # X (l) and Y (m) pixel offsets from the central pixel
    l = di * cell_rad
    m = dj * cell_rad
    ra, dec = _lm_to_radec(l, m, ra0, dec0)

    # guard the inversion against the projection the code actually uses
    lm_check = radec_to_lm(np.array([[ra, dec]]), np.array([ra0, dec0])).squeeze()
    assert_allclose(lm_check, [l, m], atol=1e-12)

    # --- transient config written into tmp_path (sidecar zarr stays out of repo) ---
    transient = {
        "name": "test_transient",
        "time": {"peak_time": 1200.0, "duration": 600.0, "shape": "gaussian"},
        "frequency": {"peak_flux": 2.0, "reference_freq": 1.4e9, "spectral_index": -1.5},
        "position": {"ra": float(np.rad2deg(ra)), "dec": float(np.rad2deg(dec))},
    }
    config_path = tmp_path / "transient.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"transients": [transient]}, f)

    ipi = 4  # integrations per image -> 60 / 4 = 15 time bins
    cpi = 2  # channels per image    -> 8  / 2 = 4  freq bins
    hci_core(
        [ms_name],
        out,
        product="I",
        data_column="DATA-DATA",
        inject_transients=str(config_path),
        channels_per_image=cpi,
        channels_per_bin=-1,
        integrations_per_image=ipi,
        images_per_chunk=15,
        max_simul_chunks=1,
        field_of_view=0.5,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        beam_model=None,
        robustness=None,
        epsilon=1e-7,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    ds = xr.open_zarr(out)
    assert ds.sizes["FREQ"] == ms_meta.nchan // cpi
    assert ds.sizes["TIME"] == ms_meta.ntime // ipi

    # predicted source pixel from the cube's own RA/Dec coordinate labels
    ra_deg = np.rad2deg(ra)
    dec_deg = np.rad2deg(dec)
    ix = int(np.argmin(np.abs(ds.X.values - ra_deg)))
    iy = int(np.argmin(np.abs(ds.Y.values - dec_deg)))

    # expected time/freq profiles, evaluated on the MS sampling
    tprofile, fprofile = generate_transient_spectra(ms_meta.utime, ms_meta.freq, transient)
    n_tbin = ms_meta.ntime // ipi
    n_fbin = ms_meta.nchan // cpi
    expected_t = tprofile.reshape(n_tbin, ipi).mean(axis=1)
    expected_f = fprofile.reshape(n_fbin, cpi).mean(axis=1)

    # position: argmax of the brightest (time, freq) slice must be the predicted pixel
    t_peak = int(np.argmax(expected_t))
    f_peak = int(np.argmax(expected_f))
    img = ds.cube.values[0, f_peak, t_peak]  # (Y, X)
    peak_iy, peak_ix = np.unravel_index(np.argmax(img), img.shape)
    assert (int(peak_iy), int(peak_ix)) == (iy, ix)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_hci.py::test_hci_inject_transients -v`
Expected: PASS. The injected transient appears at the pixel whose RA/Dec label matches the injection site.

If it FAILS on the `argmax` assertion, the most likely cause is an `(l, m)` sign/axis convention mismatch (X vs Y, or sign of `di`/`dj`). Diagnose by printing `peak_iy, peak_ix` vs `iy, ix` and the central pixel `nx//2`; adjust the sign/axis of `di`/`dj` so the predicted and observed pixels agree, then re-run. Do **not** loosen the assertion to a tolerance — position must be exact to a pixel.

- [ ] **Step 5: Lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors (note: `l`/`m` single-char names may trip `E741` ambiguous-name; if so, rename to `l_off`/`m_off` consistently).

- [ ] **Step 6: Commit**

```bash
git add tests/test_hci.py
git commit -m "test(hci): assert injected transient lands at expected pixel"
```

---

## Task 2: Flux check (dynamic spectrum at the source pixel)

**Files:**
- Modify: `tests/test_hci.py` (append assertions to `test_hci_inject_transients`)

- [ ] **Step 1: Append the binning sanity-check and flux assertion**

Add the following at the end of `test_hci_inject_transients` (after the position assertion). It first verifies the contiguous-block binning assumption against the cube's own coordinates, then checks the recovered flux:

```python
    # verify our contiguous-block binning matches the cube's coordinates
    assert_allclose(ds.TIME.values, ms_meta.utime.reshape(n_tbin, ipi).mean(axis=1), atol=1e-6)
    assert_allclose(ds.FREQ.values, ms_meta.freq.reshape(n_fbin, cpi).mean(axis=1), atol=1e-3)

    # flux at the source pixel as a function of (freq, time)
    src = ds.cube.values[0, :, :, iy, ix]          # (FREQ, TIME)
    expected_ft = expected_f[:, None] * expected_t[None, :]  # (FREQ, TIME)
    peak = float(expected_ft.max())

    assert_allclose(src, expected_ft, rtol=0.05, atol=0.02 * peak)
```

- [ ] **Step 2: Run the test and calibrate**

Run: `uv run pytest tests/test_hci.py::test_hci_inject_transients -v`
Expected: PASS — the recovered dynamic spectrum matches the injected one.

**Calibration / stop-and-report rule (per design decision):** If the flux assertion fails, compute the ratio `src / expected_ft` over the bins where `expected_ft > 0.1 * peak`:

```python
import numpy as np
ratio = src[expected_ft > 0.1 * peak] / expected_ft[expected_ft > 0.1 * peak]
print("flux ratio min/median/max:", ratio.min(), np.median(ratio), ratio.max())
```

- If the ratio is a **consistent constant ≠ 1** (low spread), **STOP**. Do not loosen the tolerance or fold the factor in. Report the observed ratio to the user — it may indicate a normalisation bug in the injection/gridding path.
- If the spread is small and centred on 1 but slightly exceeds `rtol=0.05` (pure gridding numerics), report the observed spread to the user and propose the minimal tolerance that passes, with a one-line comment in the test explaining it.

- [ ] **Step 3: Lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors. (Remove the temporary `import numpy as np`/`print` diagnostics from Step 2 if they were added — `np` is already imported at module level.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_hci.py
git commit -m "test(hci): assert injected transient flux vs time and frequency"
```

---

## Task 3: Final verification and cleanup

**Files:**
- Modify (optional): `tests/transient1.yaml` (the untracked reference file — leave as-is or remove; not used by the test)

- [ ] **Step 1: Run the full hci test module to confirm no regressions**

Run: `uv run pytest tests/test_hci.py -v`
Expected: all tests pass, including `test_hci_inject_transients`.

- [ ] **Step 2: Confirm no stray artifacts were written into the repo**

Run: `git status --porcelain`
Expected: only `tests/test_hci.py` (and possibly the untracked `tests/transient1.yaml`, `tests/MSv4_experiments.ipynb`, `scripts/profile_mappings.py` that pre-existed). There must be **no** `transient*.zarr` or `hci_transients.zarr` inside the repo tree — those should live only under `tmp_path`. If any appear in the repo, a path is wrong; fix and re-run.

- [ ] **Step 3: Final lint**

Run: `uv run ruff format . && uv run ruff check .`
Expected: clean.

---

## Self-review notes

- **Spec coverage:** isolation via `DATA-DATA` (Task 1 Step 3); exact pixel placement + round-trip guard (Task 1 Step 3); config in tmp_path (Task 1 Step 3); natural weighting `robustness=None` (Task 1 Step 3); position assertion (Task 1 Step 3); flux assertion with analytic binning (Task 2 Step 1); read from `cube` not `cube_mean` (used throughout via `ds.cube`); stop-and-report on scale factor (Task 2 Step 2); broadened profile `peak_time=1200, σ=600` (Task 1 Step 3). All covered.
- **Binning divides evenly:** 60/4=15, 8/2=4, so `reshape` blocks are exact and contiguous; validated against `ds.TIME`/`ds.FREQ` in Task 2 Step 1.
- **Names consistent:** `ix`/`iy` (pixel), `expected_t`/`expected_f`/`expected_ft`, `n_tbin`/`n_fbin`, `ipi`/`cpi`, `_lm_to_radec` used consistently across tasks.
- **Known risk:** the `(l, m)`→pixel sign/axis convention is verified empirically in Task 1 Step 4 (argmax must equal the label-predicted pixel); the flux scale is verified/reported in Task 2 Step 2.
