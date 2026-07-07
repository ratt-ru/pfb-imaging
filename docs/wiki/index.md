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

| Page | What it covers | When to read |
|------|----------------|--------------|
| [deconv-primer.md](deconv-primer.md) | The PFB major cycle, SARA prior, primal-dual/forward-backward math mapped to code; the load-bearing constants (`nu = nbasis`, total-wsum normalisation, λ schedule, reweighting semantics); Protocol seams and Ray topology; legacy-oracle traps. | Before touching `deconv/`, `opt/`, `prox/` or `core/deconv.py`, or when a solve converges slowly/diverges. |
| [design-decisions.md](design-decisions.md) | Context/Decision/Rationale/Consequences ledger for the architectural choices, plus Known debt and Recurring gotchas. | When asking "why is it built this way", before "fixing" something that looks wrong, or before re-litigating a past decision. |
| [memory-and-ray.md](memory-and-ray.md) | The three memory-retention layers on the Ray + MSv4 path, the rss/peak telemetry that separates them, and the deconv band workers' scheduling/pinning rules. | Before touching imager pass 1/2 or the band workers, and when debugging memory footprint or Ray scheduling stalls. |
| [imager-pipeline.md](imager-pipeline.md) | *Stub.* Two-pass MSv4 imager, `.dt` tree layout, weighting groupings. | Currently covered by `.claude/rules/architecture.md` §8; content migrates here when next reworked. |
