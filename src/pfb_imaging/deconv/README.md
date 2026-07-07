# Deconvolution module

The general deconvolution interface composes four pieces behind the
`DeconvSolver` Protocol (see `docs/superpowers/specs/2026-07-06-gendeconv-protocols-design.md`
and issue #185):

    PFBSolver(hess, forward_alg, backward_alg, prox)

- `hess` satisfies `operators.LinearOperator` (e.g. `HessTreeRay`, Ray band-actors)
- `forward_alg` satisfies `opt.ForwardSolver` (e.g. `PCG`)
- `backward_alg` satisfies `opt.BackwardSolver` (`PrimalDual`, `ForwardBackward`)
- `prox` satisfies `deconv.Regulariser` (`prox.L21`, `prox.L1`; owns its weights)

No inheritance anywhere: the interfaces are `typing.Protocol` classes, satisfied
structurally. Conformance is enforced at the seams — `PFBSolver.__init__`
validates all four pieces, the backward solvers validate the regulariser (and
its `psi`) in `setup()`, and the regularisers validate `psi` — via
`operators.require_protocol`, which raises a `TypeError` naming the missing
methods/attributes. To contribute an algorithm:

1. Write a regulariser: a plain class with `psi`, `nu` and
   `prox(v, vout, lam, sigma=1.0)` = `prox_{(lam/sigma) g}(v/sigma)` in-place.
   Optional fast paths (`dual_update`) and reweighting hooks
   (`init_reweighting`/`update_weights`/`reweight_active`) are sniffed, not required.
2. Register a factory in `deconv/presets.py` mapping CLI options to a `PFBSolver`.
3. Algorithms that do not decompose as forward/prox implement the
   `DeconvSolver` Protocol (`first/forward/backward/last`) directly instead.

The legacy per-algorithm drivers (`core/sara.py`, `core/kclean.py`, reading the
`.dds`) remain as validation oracles; `pfb deconv` reads the imager `.dt`.
