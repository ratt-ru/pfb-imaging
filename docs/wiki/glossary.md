---
type: Glossary
title: Domain glossary
description: The canonical names for the imaging and deconvolution concepts, with the synonyms to avoid.
tags: [glossary, imager, deconv, datatree, parallelism]
timestamp: 2026-11-27T21:00:00Z
last_verified_commit: 076879d
---

# Domain glossary

The words this project uses for its own concepts. Use them in code, issues, tests and docs,
and avoid the listed synonyms. Each definition says what a thing is; how it is stored lives in
[imager-pipeline.md](imager-pipeline.md).

**Time block** and **Hessian term** are planned under #401 and are not in the code yet. Until
then a partition has no time slot, and the Hessian has one term per partition.

## The `.dt` and its terms

**Output image**:
One band and one time of the deconvolved sky: the domain that the Hessian and the model are
summed and solved over. Its time axis is separate from the time block.
_Avoid_: band node (that is the storage), image group

**Partition**:
One term of the gradient: the visibilities, weights and beam of one
`(ms, field, spw, baseline group, time block)` slice of an output image's data.
_Avoid_: piece (a piece is the finer pass-1 unit a partition is reduced from), chunk

**Time block**:
The time slot in a partition's identity, one or more scans. It splits an output image's data
for the gradient and does not create an output image.
_Avoid_: time chunk, timeid (that is the output image's time index)

**Hessian term**:
One term of the approximate Hessian: the image-space PSF and beam of the partitions that share
a field, spw and baseline group, summed over their time blocks.
_Avoid_: partition (a Hessian term is coarser), Hessian partition

## Parallelism

**Worker**:
One node of the cluster a run executes on. Reserved for multi-node runs (#399); it is never
a process, a thread or a concurrent task.
_Avoid_: worker for a Ray worker process or a task slot

**Core budget**:
The logical CPU cores a run may use across its cluster (`--ncpu`), shared out afresh by each
pass of the imager.
_Avoid_: nworkers, nthreads

**Memory budget**:
The memory a run may use across its cluster (`--mem-limit`), shared out afresh by each pass
of the imager against each task's estimated peak.
_Avoid_: mem limit per worker

**Concurrent tasks**:
How many tasks of one pass run at once: as many as both the core budget and the memory
budget allow.
_Avoid_: workers, nworkers
