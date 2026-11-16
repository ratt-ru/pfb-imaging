# Spectral-index ripples: one diagnostic run

**Status:** one run, from August 2026, kept for its numbers. Not conclusive: the script's own
self-check failed (see below). The run's raw output (an `.npz` and two summary PNGs) was not
kept. Re-run `scripts/check_spi_ripples.py` to regenerate it.

**Data:** `output/mosaic/withbeam_newfout_gcx17fix_bkp_I.dt`. That is timeid 0 of a UHF
mosaic: 7 bands at 595.8–984.1 MHz (a frequency ratio of 1.65), 7840² pixels, 2.308″ cells.
The common restoring beam was (5.085, 4.668) px at PA 155.9°, which is band 0's native beam.
The script analysed a 512² cutout at the image centre.

Background: the residual term of the restored image was the leading suspect for the ripples in a
per-pixel spectral-index fit (#312; wiki D32/D33).

## What it measured

| diagnostic | result | reading |
|---|---|---|
| 1. sidelobes after homogenisation to the common beam | core rms difference 0.10–0.20 against wing rms difference ~0.01 | homogenisation leaves the wings mismatched only at the 1e-2 level |
| 2. dirty-beam sidelobes scale as 1/ν | wing correlation with band 0: −0.2–0.0 raw, 0.66–0.90 after rescaling by ν_b/ν_0 | confirmed: a static sidelobe pattern becomes a spectral signal |
| 4. clean-beam volume / dirty main-lobe volume | 1.03 at band 0, falling to 0.71 at band 6 (39% spread) | residual (Jy/dirty beam) enters the restored sum with a band-dependent scale |
| 0. consistency ratio E_b = FFT(PSF_b)·T_b/Ĝ, residual-power weighted | ⟨E_b⟩ 0.77 → 0.70, rms(E_b − 1) 0.26–0.33 | the trend implies a spurious α of −0.20, uniform over the residual |
| 3. α with and without the residual term | α[full] − α[model]: median 0.0015, std 0.032; α[model] std 0.355 | **the residual term is ~10× smaller than the sky's own α spread** |
| 5. FFT padding (pfrac 1 against 0.2) | std 0.0000 | padding does not contribute |
| 6. geometry of α fine structure | peaks at 38 px (7.5 restoring beams, 88″); anisotropy 4.15 at PA 5° | anisotropic at a fixed PA, so it looks like PSF striping, not primary-beam rings |

The fit masked at 2.5 × rms (3.96e-4 Jy/beam), which kept 98.1% of the pixels. The median
residual flux fraction over the mask was 0.005.

## Caveat: the self-check failed

The script rebuilds the restored cutout term by term. The rebuild matched the stored
`CRESIDUAL` (8.8e-7 relative) and `IMAGE` (2.3e-9). But for exactly Gaussian PSFs, E came out
1.66e-3 away from 1, so the script printed "SELF-CHECK FAILED … measuring a convention error,
not the data". The diagnostics that use E (diagnostic 0, and the spurious α of −0.20) are not
trustworthy until that convention error is found.

## Reading

If diagnostic 3 holds, the residual term is not what makes the ripples. It changes α by about
0.03 rms, against 0.35 for the deconvolved sky. Diagnostic 6 points at the PSF, through
striping at a fixed position angle. Next steps:

1. Fix the E self-check.
2. Repeat on a second field, to tell PSF striping from something specific to this mosaic.
