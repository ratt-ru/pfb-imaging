---
okf_version: "0.1"
---

# pfb-imaging LLM wiki

Deep internal knowledge that is expensive or impossible to re-derive from source.
Primary reader: an LLM agent working in this repo; humans are a secondary audience.
Pages follow the Open Knowledge Format v0.1 (YAML frontmatter; the custom
`last_verified_commit` field pins the commit each page's claims were checked against).

**Maintenance rule:** any change that invalidates a page updates the page, its
`timestamp` and its `last_verified_commit` in the same session/PR.

**Specs and plans are ephemeral.** `docs/superpowers/` (specs/plans written during
brainstorming/planning) is gitignored working scratch: before finishing a branch, fold
any durable knowledge into this wiki and delete/abandon the spec and plan files. Wiki
pages cite code, tests, PRs, commits and issues as sources — never spec/plan paths.

| Page | What it covers | When to read |
|------|----------------|--------------|
| [deconv-primer.md](deconv-primer.md) | The PFB major cycle, SARA prior, primal-dual/forward-backward math mapped to code; the load-bearing constants (`nu = nbasis`, total-wsum normalisation, λ schedule, reweighting semantics); Protocol seams and Ray topology; legacy-oracle traps. | Before touching `deconv/`, `opt/`, `prox/` or `core/deconv.py`, or when a solve converges slowly/diverges. |
| [design-decisions.md](design-decisions.md) | Context/Decision/Rationale/Consequences ledger for the architectural choices, plus Known debt and Recurring gotchas. | When asking "why is it built this way", before "fixing" something that looks wrong, or before re-litigating a past decision. |
| [memory-and-ray.md](memory-and-ray.md) | The three memory-retention layers on the Ray + MSv4 path, the rss/peak telemetry that separates them, and the deconv band workers' scheduling/pinning rules. | Before touching imager pass 1/2 or the band workers, and when debugging memory footprint or Ray scheduling stalls. |
| [imager-pipeline.md](imager-pipeline.md) | Why the imager writes a DataTree; the two-pass data flow; `.dt` layout; counts/`weight_grouping` and `concat_row` semantics; the HessianTree/`residual_from_partitions` split; known risks. | Before touching `core/imager.py`, `operators/gridder.py` pass-2 code or the `.dt` schema, or when reasoning about weighting granularity. |
| [image-and-beam-orientation.md](image-and-beam-orientation.md) | The measured axis conventions of the wgridder image, hci cube/FITS, BeamWizard maps and `reproject_interp`; post-mortem of the transpose+flip beam hack; the validated corrected reprojection. | Before touching `beam_for_band`, `utils/beam.py`, hci coordinate/header code, or anything that transposes/flips an image or beam array. |
