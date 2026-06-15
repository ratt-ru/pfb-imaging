# Look-ahead — deferred work for the MSv4 DataTree imager

Register of known-but-deferred work, recorded so it is not lost after the
`imager` branch merges. Items are roughly ordered by when they should be picked
up. Nothing here blocks the current merge; each is a follow-up.

Context: the `pfb imager` two-pass DataTree pipeline (`<out>_<P>.dt` + `.scratch`)
landed on the `imager` branch. See `.claude/rules/architecture.md §8` and
`docs/superpowers/specs/2026-06-04-imager-datatree-design.md`.

---

## 1. Test data carries no signal (highest priority)

The shared test MS downloaded from Google Drive has an **all-zero `DATA`
column**. Locally-cached older copies still have visibilities, which is why the
suite passes locally but the imager↔init+grid equivalence test was vacuous on
CI (both images zero). It is currently divide-by-zero-safe but only checks that
the two paths *agree* and run end to end — see the `FIXME` in
`tests/test_imager.py::test_imager_matches_init_grid_single_field`.

**To make it meaningful everywhere:**
- Populate `DATA` with a deterministic model (predict point sources with ducc0
  `dirty2vis`, as `test_kclean`/`test_sara` already do), ideally once per session
  in `conftest.py`.
- As of arcae 0.5.2 (ratt-ru/arcae#211, #212) casacore (daskms) and arcae coexist in one
  process, so the populate step can write the MS via daskms/casacore directly in the
  `test_imager.py` process — no subprocess isolation needed.
- While doing this, **fully flag one band** as a realistic baseline (RFI). This
  is already handled gracefully end to end (the band is dropped — no node, no
  Ray task, no NaN; verified for edge and interior bands), so it mainly buys
  coverage. Note it drops the suite-wide band count (e.g. 4 → 3); audit the
  MSv2 tests that may assume a fixed band count before flipping it on globally.

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

This needs a **fully-flagged-correlation test** to be meaningful, which ties
into the data-population work in §1. (Copilot review #5/#6/#7/#9/#10.)

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

## 5. Larger follow-ups (from the design spec)

- Wire the (not-yet-operational) `deconv`/`sara`/`kclean` consumers to read the
  `.dt` tree instead of the legacy `.dds`.
- Make `HessianTree` a Ray actor so per-`(band,time)` partitions can be loaded
  once on a node and reused across minor-cycle iterations.
- Baseline-group partitioning / per-antenna-pair Mueller beams for MeerKAT+
  (the `baseline_group` partition key is in place but only ever `"all"` today).
