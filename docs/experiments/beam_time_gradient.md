# Time-averaged beam in the deconv gradient (#401, step 0)

**Status:** first results, from a coarse run on com08, November 2026.
- The gradient numbers below are at a transferred, unconverged model. The rerun at a
  deconvolved model is pending, and it is the number of record.
- The frequency effect is measured on the beams only. No frequency Δg has been computed.
- The averaged Hessian (#401 step 3) is not measured here.
- `--super-resolution-factor 1.0` sits on the edge of #182: points at the edge of the uv grid
  smear off it, and pfb only matches wsclean from about 1.1. Treat absolute numbers from this
  run as provisional. Later runs use 1.2.

**Data:** ESO137-006, MeerKAT+ L-band, `CORRECTED_DATA_LIN`. The imager ran at commit 1836fbc,
coarser than the production command:
- 4 bands of 256 channels;
- `--chan-average 4`;
- `--super-resolution-factor 1.0`, giving 5488² at 1.19″;
- `--baseline-groups`, `--beam-model L`, `--robustness 0`, `--weight-grouping per-band`;
- `--transfer-model-from output/c4data_beampart_16b_I_main.mds`, a 16-band model rendered onto
  these 4 bands;
- `--keep-scratch`.

The scratch holds 24 scans × 3 baseline groups per band, each piece with its own beam. Wall time
was 3350 s: setup 750, pass 1 1832, pass 2 690.

Script: `scripts/profiling/beam_time_gradient.py`.
- `gradient` mode re-grids every scan with pass 2's robust weights. It computes the
  beam-attenuated residual twice: once with each scan's own beam (`g_exact`) and once with the
  partition's `wsum_nat`-mean beam, which is what deconv uses today (`g_avg`).
- `beams` mode compares beams on a coarse 128² field-centred grid, with no gridding.

## Checks

Summing the per-scan products reproduces what pass 2 stored: `DIRTY` to ≤ 5e-7 relative, and
`BRESIDUAL` (= `g_avg`) to ≤ 8e-6, the gridder's `epsilon`. So `g_avg` is deconv's gradient.

## Gradient, at the transferred model

All values are wsum-normalised, with `Δg = g_exact − g_avg`. `rms` is the MAD rms of `g_avg`
inside the main lobe (band beam > 0.5 of its peak). `g_avg` is beam-attenuated, so an rms over
the whole image understates the noise.

| band | ν (MHz) | ‖Δg‖/‖g‖ | max\|Δg\| / rms | where (r, B̄) | Δg/g there | max\|Δg\| / rms, where \|g\| < 5 rms |
|---|---|---|---|---|---|---|
| 0 | 1003 | 1.5% | 4.2 | 0.83°, 0.35 | −2.8% | 3.8 |
| 1 | 1175 | 1.1% | 2.2 | 0.67°, 0.37 | −3.8% | 2.0 |
| 2 | 1404 | 2.2% | 4.6 | 0.37°, 0.65 | +20% | 4.3 |
| 3 | 1599 | 1.6% | 3.4 | 0.25°, 0.78 | −17% | 2.9 |

- `Δg` is affine in the model. Where the model fits the data, its data and model terms cancel
  to first order, so this table is an upper bound on the bias at convergence. The model here is
  a 16-band model rendered onto 4 bands, and it leaves `max|g_avg|` at thousands × rms. The rerun
  at a deconvolved model settles it.
- Even so, `max|Δg|` is 2–4 × rms in the quiet parts of the residual, not only at bright
  residuals. The worst pixels are in the main lobe or on its shoulder, at 0.25–0.83°.

The step-0 decision rule is max |Δg| ≥ rms. It holds at this model.

## Beams

Maximum |ΔB|, measured directly on 53.5 MHz bands (the production 64 channels) about each
band's frequency, over the main lobe and shoulder (B̄ > 0.1):

| effect | MM | MPM | MPMP |
|---|---|---|---|
| time: per-scan beam vs mean | 0.019–0.023 | 0.012–0.032 | 0.005–0.053 |
| frequency: band edge to edge | 0.021–0.047 | 0.025–0.046 | 0.032–0.045 |
| frequency: flat vs chord at mid band | 0.002–0.006 | 0.001–0.003 | 0.002–0.004 |

- A flat beam errs by about half the edge-to-edge change at a band's edge channels:
  ~0.01–0.023. That is **comparable to the time effect**.
- A beam linear in frequency within the band leaves 0.001–0.006, so it removes most of the
  frequency error, as degrid's `--spectral linear` does (D59).
- In the sidelobes (B̄ < 0.1) both effects are smaller: time up to 0.022, frequency up to 0.033
  edge to edge.

## The time variation is low-rank

SVD of the 24 per-scan beams minus their mean, decimated 8× on the image grid:

| band, group | energy, rank 1 / 2 / 3 / 4 | max \|error\|, rank 0 / 2 / 4 / 6 |
|---|---|---|
| 0, MM | 0.657 / 0.992 / 0.997 / 0.9995 | 0.022 / 0.0047 / 0.0011 / 0.0005 |
| 2, MM | 0.643 / 0.984 / 0.995 / 0.998 | 0.025 / 0.0058 / 0.0019 / 0.0007 |
| 0, MPMP | 0.646 / 0.869 / 0.958 / 0.984 | 0.012 / 0.0070 / 0.0030 / 0.0004 |
| 2, MPMP | 0.702 / 0.885 / 0.970 / 0.993 | 0.014 / 0.0058 / 0.0015 / 0.0007 |

Two components hold 98–99% of the MM variation, which is the quadrupolar asymmetry rotating with
parallactic angle. Four bring the worst error to ~0.001–0.003.

This suggests an alternative to one gridding term per time block. Write
`B(t, ν) ≈ B̄ + Σ_k a_k(t, ν) φ_k(l, m)`, so the forward model is
`G(B̄x) + Σ_k G(a_k ∘ (φ_k x))`, with per-visibility coefficients. That costs K + 1 degrid-and-grid
pairs per partition, whatever the number of time blocks. A beam linear in frequency has the same
form, with per-channel coefficients.

**Parked.** The SVD covered only this image, r ≲ 0.9°, so it never saw a beam null. A null that
rotates with parallactic angle is unlikely to stay low-rank at wider fields. Other open points:
- the gain only appears above about K + 1 time blocks;
- per-integration coefficients, and frequency folded into the same basis, are untested;
- it does not give a separate beam per time block, which per-time beam-orientation checks
  want.

## A safe beam cell

The #365 plan is for pass 1 to store the beam on a small field-centred grid, and for pass 2 to
place it on the image grid bilinearly, once per partition. `scripts/profiling/beam_cell_study.py` measured the bilinear error against the beam
evaluated at dx/16 directly, where dx = 0.0625° is the meerkat-beams L cube's spacing. It
covered:
- each group, Stokes I;
- 1000 and 1711 MHz;
- one scan and the full track;
- out to ±1.5°.

Inside the inscribed disk the error quarters with every halving of the cell, so it is pure
bilinear curvature. The worst error is in the main lobe:

| cell h | h / (λ_min/D) at 1711 MHz | main lobe | shoulder | sidelobes |
|---|---|---|---|---|
| dx | 0.084–0.093 | 6.8–8.9e-3 | 1.8–2.5e-3 | 1.7–2.5e-3 |
| dx/2 | 0.042–0.047 | 1.7–2.2e-3 | 4.5–6.3e-4 | 4.2–6.1e-4 |
| dx/4 | 0.021–0.023 | 4.3–5.6e-4 | 1.1–1.6e-4 | 1.0–1.5e-4 |
| dx/8 | 0.011–0.012 | 1.1–1.4e-4 | 2.9–3.9e-5 | 2.6–3.9e-5 |

The main-lobe maximum is **(h / (λ_min/D))²**, with D the largest dish in the group:
- MM, D = 13.5 m: 0.96–1.00 times that;
- MPMP, D = 15 m: 1.02–1.03 times;
- MPM, D = 15 m: 0.92 times. With D = 13.5 m it is 1.12–1.13 times, so the cross group's beam
  is narrower than MM's.

A tolerance `tol` therefore gives **h = √tol · λ_min/D_max**. Taking D_max as the array's
largest dish is conservative for every group.

The corners of the ±1.5° square reach 2.1°. Rotated with parallactic angle, they leave the cube
(±2°), and there the error does not converge with the cell at all (up to 9e-3). That is a
coverage limit, not a sampling one.

Scope: Stokes I, the L cube, r ≤ 1.5°. The study subsampled one fine evaluation, so it still
has to be confirmed that evaluating the wizard directly at the coarse cell gives the same
values.

## Cost

At 5488² with 20 threads, each scan × group term took ~27 s for five gridder calls: one grid,
plus two degrid-and-grid pairs. That is 1960–2220 s per band for 72 terms. A time-resolved
gradient pays roughly one degrid-and-grid pair, ~10 s here, per time block per baseline group
per band per major cycle. Today it pays one per baseline group.

## Raw output

On com08, `/data7/lbester/MKPLUS/ESO137MKP/s0/`:
- `results/band####_{gradient,beams,beams_53.5MHz}.json`;
- FITS of `g_avg`, `dg`, `dg_data` and `dg_model` per band;
- the kept `s0_coarse_I.scratch` and `.dt`.
