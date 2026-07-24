# Look-ahead — deferred work for the MSv4 DataTree imager

Register of known-but-deferred work, recorded so it is not lost after the
`imager` branch merges. Items are roughly ordered by when they should be picked
up. Nothing here blocks the current merge; each is a follow-up.

Context: the `pfb imager` two-pass DataTree pipeline (`<out>_<PRODUCT>.dt` + `.scratch`)
landed on the `imager` branch. See `.claude/rules/architecture.md §8` and
`docs/wiki/imager-pipeline.md`.

---

## 1. Test data carries no signal — DONE (0.1.0, #277)

The shared test MS used to have an all-zero `DATA` column, which made the old
imager↔init+grid equivalence test vacuous. Fixed by `tests/conftest.py::sky_truth`
(seeded predicted-sky injection, ~10% flags, one fully flagged channel,
consistent `FLAG_ROW`, module-scoped) feeding ground-truth recovery tests
(`test_imager.py`, `test_imager_pol.py`, `test_deconv.py::test_deconv_groundtruth`)
in place of the deleted equivalence test. See `docs/wiki/design-decisions.md` D2/D20.

## 2. Multi-correlation `wsum == 0` guards (before polarized imaging)

For `product="I"` a band node only exists if it has weight, so `wsum > 0`
always and these divides are safe. For **polarized products (Q/U/V)** a *present*
band can contain a fully-flagged correlation (`wsum[c] == 0`), which would write
`inf`/`NaN`. Guard each divide (e.g. `np.divide(num, wsum, where=wsum>0,
out=zeros)`; empty correlation → zero, **not** a hard `raise`):

- `core/imager.py:163` — `PSFPARSN = fitcleanbeam(psf_sum / wsum_sum)`
- `core/imager.py:635` — MFS beam fit `psf_mfs[tid] / wsum_mfs[tid]`
- `operators/hessian.py` — `HessianTree.dot` divides by `self.wsum`
- `utils/fits.py` — `dt2fits` MFS and cube normalisation by `wsum`/`wsums`

This needs a **fully-flagged-correlation test** to be meaningful — still open;
`sky_truth` (§1) covers a fully flagged channel but not a fully flagged
correlation within a present band. (Copilot review #5/#6/#7/#9/#10.)

## 3. `robustness > 2` → natural short-circuit (consistency cleanup)

`grid_partition`/`_grid_image` apply `counts_to_weights` for any non-`None`
`robustness`, but the docstring/CLI advertise `robustness > 2` as natural
weighting. Normalise `robustness > 2` to `None` so pass-2 logic and band
metadata match the documented semantics, and guard the `counts is None` +
`robustness` set case in `grid_partition` (currently crashes on `counts.copy()`).
Low priority — Briggs is already ≈ natural at `R > 2`. (Copilot review #3/#4;
note Copilot's "nearly-uniform" wording is backwards — high robustness tends to
natural.)

## 4. Band indexing across paths (cosmetic)

When a band is dropped, the imager keeps the original band id (gap, e.g. nodes
`0,1,3`) while the legacy `init+grid` reindexes contiguously (`0,1,2`). MFS
products are unaffected (sum over nodes). If a per-band comparison or alignment
is ever needed, match on `freq_out`, not on the band index.

## 5. Transient-injection end-to-end test (designed, unimplemented)

The `hci` `inject_transients` feature (`utils/transients.py`, wired through
`core/hci.py`/`stokes2im.py`) has no end-to-end test. A reviewed design existed and
was removed with the other specs when `docs/superpowers/` became ephemeral scratch
(recoverable from git history if ever needed); everything durable from it is
summarised here. Key ingredients to preserve: isolate the
transient with `data_column="DATA-DATA"` (zero base data); place it on an exact pixel
centre by inverting the SIN projection so flux reads without PSF ambiguity; predict
per-bin flux from `generate_transient_spectra` profiles averaged over contiguous
time/chan blocks; assert against the raw `cube`, never `cube_mean` (the RMS-flag
gotcha in `docs/wiki/design-decisions.md`); an unexplained recovered/injected scale
factor is reported, not absorbed into a tolerance.

## 6. Larger follow-ups (from the design review)

- Make `HessianTree` a Ray actor so per-`(band,time)` partitions can be loaded
  once on a node and reused across minor-cycle iterations.
- Baseline-group partitioning / per-antenna-pair Mueller beams for MeerKAT+
  (the `baseline_group` partition key is in place but only ever `"all"` today).
