<!--
Salvaged from tmp/degrid_freqacc_report.md (2026-11-02). Measurements from com08 runs at
pfb-imaging ac2459b (Experiments 1-3) and 51e7905 (Experiment 4). Paths below are on com08
under /data7/lbester/MKPLUS/ESO137MKP. Bears on #387 (make --spectral linear the default)
and landmanbester/pfb-model-spec#42 (tent fit through RFI-heavy bands).
-->

# Degrid accuracy on ESO137-006 (MK+): flat vs linear spectral prediction

Branch `degridaccuracy`; com08 ran pfb-imaging at ac2459b. Project dir
`/data7/lbester/MKPLUS/ESO137MKP`. Scripts, logs and analysis live under
`degrid_tests/freqacc/` (Experiments 1–2) and `degrid_tests/freqacc2/` (Experiment 3).

## TL;DR

1. **Degrid is consistent with deconv.** If you image `CORRECTED_DATA4 − MODEL` (flat degrid,
   default chunks), you get the same residual as deconv's own forward model to within 0.1% in
   the core and off-source. The source region is ~10% worse with degrid, which is consistent with
   the #355 east-west beam mirror in the imager's stored BEAM. That mirror affects deconv's
   residual, not degrid's.
2. **Linear prediction looks *worse* on the existing CORRECTED_DATA4, but the data is to
   blame, not the method.** CORRECTED_DATA4 was calibrated against a band-flat (64-ch staircase)
   model. G (16-ch solutions) soaked up the model's spectral error, so the corrected data carries
   the staircase and a flat model matches it best.
3. **Linear chunk size doesn't matter.** 64-, 16- and 8-channel chunks agree to 5e-8 (MFS core
   difference), so the default 64-channel chunks are enough.
4. **Self-cal against linear removes most of the gain staircase.** The median G step across
   model-band edges falls from 0.0167 to 0.0060; the within-band level is ~0.004. The residuals
   after self-cal barely move: the core is within ±1–2%, and the source region improves 3–12% at
   the band edges. Flat's error moves into the gains/corrected data rather than staying in the
   residual.
5. **Recommendation:** use `--spectral linear` (default 64-ch chunks) for the self-cal
   prediction. This supports making linear the default (#387). The remaining steps sit next to
   RFI-heavy bands, where the tent fit is poorly constrained. That is a pfb-model-spec question.

## Setup common to all experiments

- **Model:** `output/c4data_beampart_16b_I_main.mds`, 15 bands from 855.9 to 1658.4 MHz,
  from the c4data deconv.
- **Reference imaging (c4data, Oct 24):** `CORRECTED_DATA4`, `WEIGHT_SPECTRUM`, robust 0.3,
  minvar, 64 ch/image, `--beam-model L --baseline-groups`, cell 0.7434716792793206″,
  8748² px. Every imager run below is pinned to this grid and these settings. Only the data
  column, channels per image and products change.
- **Degrid:** `--product I --beam-model L --baseline-groups`, default
  `--integrations-per-chunk` (one beam per partition).
- **Regions, from c4data's MFS model:**
  - **core:** a 256² box on the model peak (ESO137-006 itself).
  - **src:** model > 1e-3·peak, dilated 20 px, minus the core.
  - **off:** the central 4096², minus model > 1e-4·peak dilated 100 px. This is the noise
    reference.
- **Metric:** RMS of `DIRTY/WSUM` (Jy/beam) per image band. Each image band is also labelled by
  its position inside its parent model band, to expose band-edge effects.

## Experiment 1: is degrid consistent with deconv?

**Subset:** scans 3, 19, 34, 52 and `--freq-range 963.0e6:1176.5e6` (model bands 2–5,
256 ch), on the full MS.

| run | what | core rms | src rms | off rms |
|---|---|---|---|---|
| `r0_64` | imager `--transfer-model-from` (deconv's forward model) | 1.513e-4 | 8.98e-5 | 3.833e-5 |
| `a_64` | image `CORRECTED_DATA4 − MODEL_FLAT` (degrid flat, default) | 1.514e-4 | 9.85e-5 | 3.844e-5 |

- MFS core rms(a_64 − r0_64) = 9.3e-6, against rms(r0_64) = 1.19e-4 (8%, mostly structure
  near the source).
- The core and off-source agree. The src region is ~10% worse with degrid.
- **Interpretation:** degrid reproduces deconv's forward model. The src gap matches #355: the
  imager's single-field BEAM is mirrored E–W, so deconv and degrid apply different beams away
  from the pointing centre. This doesn't test whether degrid is *better*. Nothing that
  outperforms deconv is possible until #355 is fixed, because deconv's model was fitted through
  the mirrored beam.

## Experiment 2: does frequency resolution in the prediction help?

Same subset. Degrid flat, then linear with 64/16/8-channel chunks. Each result was imaged at
8 ch/image (`CORRECTED_DATA4 − MODEL_*`).

| run | core rms | src rms | off rms |
|---|---|---|---|
| `flat_8` | 2.579e-4 | 2.28e-4 | 1.168e-4 |
| `lin64_8` | 2.718e-4 | 4.30e-4 | 1.287e-4 |
| `lin16_8` | 2.720e-4 | 4.31e-4 | 1.288e-4 |
| `lin8_8` | 2.720e-4 | 4.31e-4 | 1.288e-4 |

Core rms relative to flat, by 8-ch sub-band position inside the model band (0 = low edge):

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| lin64 / flat | 1.099 | 1.096 | 0.984 | 1.005 | 1.016 | 1.021 | 1.104 | 1.110 |

- **Linear is ~10% worse in the core at the band edges, and 2–3× worse in src.** It is
  marginally better near band centre.
- **Chunk size is irrelevant:** MFS core rms(lin8 − lin16) = 5.0e-8, against
  rms(lin64 − flat) = 9.6e-6.
- **The model itself is smooth:** a 4–5% change per band, consistent with α ≈ −0.7 to −1. Linear
  interpolation is therefore a priori a better description of the sky than a staircase.
- **Why linear loses here:** CORRECTED_DATA4 came from QuartiCal (Oct 23, `gains4.qc`, G every
  16 ch) against an Oct 21 MODEL_DATA that was a 64-ch *staircase* degrid of the cdata model.
  G absorbed the staircase. Measured in the existing gain tables, the median |Δlog|G|| across
  model-band edges vs within bands is:
  - `gains4.qc`: 0.0151 vs 0.0063
  - `gains2.qc`: 0.0163 vs 0.0045

  The corrected data therefore carries the flat model's spectral shape, and only a flat model
  can subtract it cleanly. This is circular, so it can't judge the prediction. Hence
  Experiment 3.

## Experiment 3: self-cal against each prediction

- **Subset MS:** `msdir/1790414245_sdp_l0.full_1024ch_ESO137-006-scan-3-and-5.ms` (scans 3
  and 5, all 1024 ch).
- **Steps:**
  1. Degrid the c4data model flat → `MODEL_C4FLAT`, and linear with 64-ch chunks →
     `MODEL_C4LIN`.
  2. Run QuartiCal with `qcselfcal_amp.yaml` settings. Only the MS, the model recipe, the
     output columns (`CORRECTED_{DATA,RESIDUAL,WEIGHT}_C4{FLAT,LIN}`) and the gain/log dirs
     change, and `flags: false`.
  3. Measure G steps.
  4. Image `CORRECTED_RESIDUAL_C4*` at 16 ch/image.

### Gains (`gjumps.py`)

| prediction | median \|Δlog\|G\|\| across band edges | within bands | antenna/scan-mean step within bands |
|---|---|---|---|
| flat | **0.0167** | 0.0040 | 0.0026 |
| linear | **0.0060** | 0.0033 | 0.0013 |

Antenna/scan-mean step at each of the 14 model-band edges:

```
flat   -0.0295 +0.0113 +0.0180 +0.0143 +0.0166 +0.0360 +0.0085 +0.0331 +0.0096 +0.0116 +0.0096 -0.0055 +0.0076 +0.0298
linear -0.0285 +0.0049 +0.0029 -0.0021 +0.0002 +0.0170 +0.0026 +0.0304 +0.0023 +0.0009 -0.0001 -0.0134 -0.0115 +0.0137
```

- Flat imprints a systematic positive step at almost every edge. That is the staircase: the
  model is too bright at the top of each band and too faint at the bottom, for a falling
  spectrum. Linear brings most edges down to the within-band level.
- **The remaining linear steps** (edges 1, 6, 8, 12–14) sit next to RFI-heavy bands at about
  909, 1177, 1284 and 1551–1605 MHz. Through those bands, the tent fit interpolates between
  poorly constrained knots.

### Residuals after self-cal (16 ch/image)

| | core rms | src rms | off rms |
|---|---|---|---|
| flat | 4.928e-4 | 3.426e-4 | 1.686e-4 |
| linear | 4.926e-4 | 3.299e-4 | 1.676e-4 |

By sub-band position inside the model band, linear / flat (0 and 3 = edges):

| pos | core | src | off |
|---|---|---|---|
| 0 | 1.012 | **0.876** | 0.986 |
| 1 | 1.006 | 0.978 | 1.000 |
| 2 | 0.998 | 1.001 | 1.002 |
| 3 | 0.987 | **0.969** | 0.982 |

- Per model band, core linear/flat ranges over 0.980–1.017, with no trend.
- MFS core rms(lin − flat) = 4.1e-6, against rms(flat) = 1.85e-4 (2%).
- **Interpretation:**
  - The residual hardly cares about the prediction, because self-cal is flexible enough (G per
    16 ch) to absorb the flat model's error.
  - Where that error goes is what differs. With flat it ends up in G and therefore in
    CORRECTED_DATA (the staircase of Experiment 2). With linear it largely doesn't.
  - The src improvement at the band edges is where you'd expect a staircase to bite.
  - Since CORRECTED_DATA feeds the next imaging round, linear is the more faithful choice even
    though the residual metric barely moves.

## Experiment 4: Experiment 3 on the full MS (24 scans)

`degrid_tests/fullms/`, run 2026-11-01/02 at pfb-imaging 51e7905. Two arms:

- **Flat:** the existing `MODEL_DATA` (Oct 30 flat degrid).
- **Linear:** `MODEL_LIN`, degridded with `--spectral linear` and 64-ch chunks.

Both arms ran QuartiCal with `qcselfcal_amp.yaml`, changing only the recipe, the columns, the
gain dirs and **`flags: false`**. Both therefore start from and leave the same FLAG. Each arm's
`CORRECTED_RESIDUAL_*` was imaged at 16 ch/image with c4data's settings.

Wall time per step:

| step | time |
|---|---|
| degrid | 4.7 h |
| QuartiCal, each arm | 4.8 h |
| imager, each arm | 9.3 h (pass 1 5.8 h, pass 2 2.3 h, scratch removal 19 min) |

### Gains

| prediction | edge step, two scans | edge step, full MS | within bands, full MS | antenna/scan mean within bands |
|---|---|---|---|---|
| flat | 0.0167 | **0.0164** | 0.0045 | 0.0023 |
| linear | 0.0060 | **0.0063** | 0.0039 | 0.0012 |

Antenna/scan-mean step at each of the 14 model-band edges:

```
flat   -0.0257 +0.0119 +0.0180 +0.0137 +0.0169 +0.0364 +0.0104 +0.0179 +0.0098 +0.0121 +0.0089 +0.0029 +0.0044 +0.0204
linear -0.0245 +0.0057 +0.0027 -0.0033 +0.0003 +0.0154 +0.0041 +0.0147 +0.0024 +0.0013 -0.0006 -0.0051 -0.0158 +0.0045
```

The remaining linear steps are at the same RFI-adjacent edges (1, 6, 8, 13) as on the subset.
See landmanbester/pfb-model-spec#42.

### Residuals

| | core rms | src rms | off rms |
|---|---|---|---|
| flat | 2.059e-4 | 1.422e-4 | 3.981e-5 |
| linear | 2.070e-4 | 1.324e-4 | 3.966e-5 |

Linear / flat by sub-band position inside the model band (0 and 3 = edges):

| pos | core | src | off |
|---|---|---|---|
| 0 | 1.014 | **0.838** | 0.992 |
| 1 | 1.008 | 0.970 | 1.000 |
| 2 | 1.007 | 0.999 | 1.004 |
| 3 | 0.993 | **0.938** | 0.986 |

- Per model band, core linear/flat ranges over 0.981–1.033. The 1.033 is band 1, next to the
  edge-1 RFI band.
- MFS core rms(lin − flat) = 1.3e-6, against rms(flat) = 1.2e-4 (1%).
- **Same picture as the subset, with a stronger src result.** The source region is 16% and 6%
  better at the low and high band edges, and 7% better overall. The core and off-source
  residuals are unchanged to within ~1%.

## Conclusions and follow-ups

- **Use `--spectral linear` with the default 64-ch chunks for self-cal predictions.** This is
  evidence for #387 (making linear the default).
- **#355 must be fixed before testing whether degrid can beat deconv's residual.** The fix
  sits in worktree branch `worktree-agent-a515330d5b911f705`, unmerged. Note that existing
  single-field `.dt` files, including c4data, carry the mirrored BEAM, so re-imaging is needed.
- **pfb-model-spec question:** should the spectral fit be weighted or smoothed, or should
  RFI-flagged bands be masked, so the tent doesn't kink through poorly constrained bands?
- **Ordinary experiment debt:**
  - D59 still says "Not yet measured on real data" and should be updated.
  - Setup overhead is about 30 min per degrid run and 21 min per imager run, regardless of
    `--scan-names`/`--freq-range` subsetting. That deserves a profile.
- **Cleanup on com08 (done 2026-11-01):**
  - Deleted every `degrid_tests` image tree, FITS and npz file; scripts, logs, summaries,
    plots and the freqacc2 gain tables are kept.
  - Dropped the experiment columns from both MSs.
  - Dropped `CORRECTED_{DATA,RESIDUAL}{3,4}` and `MODEL_DATA_BEAM` from the main MS.
    Experiments 1–2 can no longer be rerun as they stand.

## Caveats

- **Few scans:** Experiments 1–2 use 4 scans and 4 model bands; Experiment 3 uses 2 scans.
  Parallactic-angle coverage is limited, but none of the conclusions depend on it.
- **One calibration recipe:** Experiment 3 tests a single recipe (`qcselfcal_amp`). A less
  flexible G (wider freq interval) would push more of the model error into the residual, and
  the residual comparison would then favour linear more clearly.
- **Experiment 2's linear runs are not a fair test.** They are kept because they explain the
  staircase mechanism.
