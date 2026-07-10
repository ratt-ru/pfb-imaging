---
type: Subsystem Notes
title: MSv4 DataTree imager pipeline (stub)
description: Placeholder for the two-pass imager, .dt tree layout and weighting-grouping knowledge currently held in .claude/rules/architecture.md §8.
tags: [imager, msv4, datatree, weighting, stub]
timestamp: 2026-07-07T20:10:49Z
last_verified_commit: f6c8a80
---

# MSv4 DataTree imager pipeline (stub)

**Scope when filled:** the two Ray-distributed passes (`stokes_vis` fine pieces +
`.scratch` cache; `_grid_image` partition gridding and band-node sums), the
`<output>_<PRODUCT>.dt` tree layout (`band{b}_time{t}` nodes, `part{p}` children,
which variables live where), the counts reduction and `weight_grouping` semantics,
and the imager ↔ legacy `init`+`grid` equivalence testing strategy.

Until migrated, this content lives in `.claude/rules/architecture.md` §8 and
`docs/superpowers/specs/2026-06-04-imager-datatree-design.md`. Memory discipline for
both passes: [memory-and-ray.md](memory-and-ray.md).
