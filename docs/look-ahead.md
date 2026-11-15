# Look-ahead — deferred work for the MSv4 DataTree imager

Register of known-but-deferred work, recorded so it is not lost after the
`imager` branch merges. Items are roughly ordered by when they should be picked
up. Nothing here blocks the current merge; each is a follow-up.

Context: the `pfb imager` two-pass DataTree pipeline (`<out>_<PRODUCT>.dt` + `.scratch`).
See `docs/wiki/imager-pipeline.md`.

---

## 1. Multi-correlation `wsum == 0` guards (before polarized imaging)

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
`tests/conftest.py::sky_truth` covers a fully flagged channel but not a fully flagged
correlation within a present band. (Copilot review #5/#6/#7/#9/#10.)

## 2. `robustness > 2` → natural short-circuit (consistency cleanup)

`grid_partition`/`_grid_image` apply `counts_to_weights` for any non-`None`
`robustness`, but the docstring/CLI advertise `robustness > 2` as natural
weighting. Normalise `robustness > 2` to `None` so pass-2 logic and band
metadata match the documented semantics, and guard the `counts is None` +
`robustness` set case in `grid_partition` (currently crashes on `counts.copy()`).
Low priority — Briggs is already ≈ natural at `R > 2`. (Copilot review #3/#4;
note Copilot's "nearly-uniform" wording is backwards — high robustness tends to
natural.)

## 3. Band indexing across paths (cosmetic)

When a band is dropped, the imager keeps the original band id (gap, e.g. nodes
`0,1,3`) while the legacy `init+grid` reindexes contiguously (`0,1,2`). MFS
products are unaffected (sum over nodes). If a per-band comparison or alignment
is ever needed, match on `freq_out`, not on the band index.

## 4. Transient-injection end-to-end test (done)

Implemented as `tests/test_hci.py::test_hci_injected_transient_has_the_analytic_dynamic_spectrum`
(zeroed base data via `DATA-DATA`, a source on an exact pixel centre, per-bin flux predicted
from `generate_transient_spectra`, asserted against the raw `cube`), with the position tests
beside it. Kept here only so the numbering of the sections below does not move.

## 5. Larger follow-ups (from the design review)

- Make `HessianTree` a Ray actor so per-`(band,time)` partitions can be loaded
  once on a node and reused across minor-cycle iterations.
- Baseline-group partitioning / per-antenna-pair Mueller beams for MeerKAT+
  (the `baseline_group` partition key is in place but only ever `"all"` today).
