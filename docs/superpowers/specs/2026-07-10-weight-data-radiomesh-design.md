# weight_data on radiomesh generated expressions — design

**Date:** 2026-07-10
**Issues:** ratt-ru/pfb-imaging#273 (numba cache weirdness in `hci`), ratt-ru/pfb-imaging#183
(replace sympy in `weight_data` with radiomesh auto-generated functions)
**Branch:** `issue273`

## Problem

`weight_data` (`utils/weighting.py`) builds its per-stokes vis/weight functions by running
sympy and `lambdify` inside `@overload` resolution, then njit-ing the lambdified functions and
capturing them in the `_impl` closure. This has two consequences, verified by local repro:

1. **The numba disk cache can never hit.** Numba's cache key is
   `(signature, cpu_target, (code_hash, closure_hash))`, where `closure_hash` hashes the
   *pickled closure cells*. Pickling an njit `Dispatcher` embeds a per-process UUID, so the
   closure hash differs on every overload resolution. Every compile is a miss followed by a
   fresh ~280 kB `.nbc` write; nothing is ever loaded. The cache dir (hardcoded `/tmp/numba`,
   see #270) grows without bound. In the #273 run this eventually hit
   `OSError: [Errno 28] No space left on device` at task 425 and killed the pipeline.
2. **Every fresh Ray worker pays ~6 s** of sympy + LLVM compilation. The nested `hci` pattern
   (`batch_stokes_image` blocking in `ray.get` on `stokes_image` subtasks) spawns/reaps worker
   processes throughout a run, so this cost recurs for the whole run.

A secondary churn source: the numba signature includes each array's readonly flag and layout,
which vary per task (Ray zero-copy args arrive readonly; `astype`/rephasing copies are
writable; the jones `np.swapaxes` view is non-contiguous). Each new combination forces a full
recompile (+ cache write).

**Key verified facts** (repro scripts referenced in #273):

- Pickled `Dispatcher` bytes differ across processes even for a static module-level `@njit`
  function → any overload impl closing over a dispatcher is uncacheable.
- Plain module-level functions (raw or `register_jitable`) pickle by reference →
  byte-identical across processes → closures over them are cache-stable.
- An overload impl referencing only module globals / plain-function cells cache-hits across
  processes.

## Goal

`weight_data` compiles from pre-generated, statically defined expressions; its full compile
chain is disk-cacheable so only the first worker per node/CPU-arch pays compilation; signature
churn is collapsed. Behaviour (numerics, output dtypes, error surface) is unchanged.

## Non-goals

- No change to the `hci` Ray dispatch pattern (waits for #237).
- No fix for the shared `/tmp/numba` cache-dir collision problem (#270) beyond removing the
  unbounded growth.
- No full-jones minvar (stays `NotImplementedError`, as today).
- `utils/correlations.py` has an unrelated function also named `weight_data`; out of scope.

## Design

### Part 1 — radiomesh (separate repo, lands first)

Extend `radiomesh/scripts/gen_expr.py` and regenerate `radiomesh/generated/_stokes_expr.py`:

- **Diag-jones variants.** Substitute `jp01 = jp10 = jq01 = jq10 = 0` into the coherency and
  weight matrices *before* `simplify`, and emit
  `{POL}_VIS_DIAGJONES_{S}(v00, v01, v10, v11, jp00, jp11, jq00, jq11)` and
  `{POL}_WEIGHT_DIAGJONES_{S}(w00, w01, w10, w11, jp00, jp11, jq00, jq11)` for
  `POL ∈ {LINEAR, CIRCULAR}`, `S ∈ {I, Q, U, V}`. These reproduce the compact expressions the
  sympy path produces today for pfb's hot diag path (jones `ndim == 5`).
- **Minvar weights** (diag-only). From each diag-substituted weight element `w`, emit
  `4 * Min(*expand(w).args)` as `{POL}_WEIGHT_MINVAR_DIAGJONES_{S}`, mirroring
  `stokes_funcs`' minvar mode. Minvar cannot be derived from the full-jones expressions: the
  `Min` over expansion terms would select the structurally-zero cross terms. The generator maps
  sympy's `Min(...)` rendering to Python's builtin `min` (numba-supported for scalars), the
  same way it already maps `conjugate → conj`.
- **Registry.** `CONVERT_FNS` gains the new keys — data type `"WEIGHT_MINVAR"` and jones mode
  `"DIAGJONES"` — purely additively. Existing keys and the `data_conv_fn` intrinsic are
  untouched.
- **Test** (radiomesh): numeric check that DIAGJONES == JONES evaluated with zero
  off-diagonals, and that minvar matches a sympy-computed oracle.

pfb-imaging pins the resulting commit: `radiomesh @ git+https://github.com/ratt-ru/radiomesh@<sha>`
in the `[full]` extra, swapped to a version pin once radiomesh releases.

### Part 2 — pfb-imaging (branch `issue273`, three commits, one mechanism each)

**Commit 1 — expression swap (fixes the uncacheable closure).**

- `utils/stokes.py`: delete the sympy `stokes_funcs`; add a selector with the same
  signature-in-spirit (`(data, jones, product, pol, nc, wgt_mode)` literals in, functions out)
  that keeps today's validation logic verbatim (product availability vs `nc`/`pol`, `remprod`
  check, mode check) and returns **tuples of plain radiomesh functions** — one vis fn and one
  wgt fn per requested stokes — selected from `radiomesh.generated._stokes_expr.CONVERT_FNS`
  (`DIAGJONES` family for jones `ndim == 5`, `JONES` family for `ndim == 6`,
  `WEIGHT_MINVAR_DIAGJONES` for minvar).
- `utils/weighting.py`: `nb_weight_data_impl` calls the selector and returns an `_impl`
  specialised on `ns = len(product)` — four small statically-written variants using constant
  tuple indexing (`vis_fns[0](...)`, `vis_fns[1](...)`, …); no `literal_unroll`. The nc == 2
  conventions (`w01 = w10 = 1.0`, `v01 = v10 = 0j`) and the diag/full jones scalar extraction
  (`jp00 = gp[chan, 0]`, … / `gp[chan, 0, 0]`, …) move inline into `_impl`.
- **Cache-safety rule (load-bearing):** closure cells of `_impl` (and anything it calls) may
  contain only plain module-level functions and ints — never njit dispatchers. This is what
  makes the cache key stable.
- Output dtypes preserved exactly (the current `_impl.returns` coercion to
  complex128/float64); the oracle test pins this.
- `pyproject.toml`: add the radiomesh git pin to `[full]`; drop `sympy` from runtime deps if
  `stokes_funcs` was its last runtime consumer.
- Call sites (`stokes2im.py`, `stokes2vis.py`, `stokes2vis_msv4.py`) unchanged.

**Commit 2 — end-to-end cacheability + regression test.**

- Flip the outer `weight_data` to `cache=True` (the overload already inherits `cache=True`
  from `JIT_OPTIONS`). Fresh Ray workers then warm-start from disk instead of recompiling.
- Regression test for #273: run a small `weight_data` call in subprocess 1 (cold cache dir via
  `NUMBA_CACHE_DIR`), then in subprocess 2 assert (via `NUMBA_DEBUG_CACHE` output and `.nbc`
  file count) that the compile is served by a cache **load** and no new `.nbc` is written.

**Commit 3 — collapse signature churn.**

- In `stokes_image` (`utils/stokes2im.py`), canonicalise the eight array args immediately
  before the `weight_data` call: `np.ascontiguousarray` (copies only genuinely non-contiguous
  inputs, e.g. the jones `swapaxes` view) and readonly views (`x.view()` +
  `writeable = False`; always legal in that direction, zero-copy). `weight_data` never mutates
  its inputs. Result: one signature per dtype configuration per process instead of one per
  readonly/layout permutation.

### Testing

- **`tests/test_stokes_expr.py`** (new): the old sympy derivation survives as an in-test
  oracle; numeric equivalence of radiomesh-backed `weight_data` against it across
  pol × product × nc × wgt_mode with random jones, flags, and weights; asserts output dtypes.
- **Cache regression test** (commit 2, above).
- Existing `tests/test_polproducts.py` (gains on/off) and the full suite as end-to-end guards.

### Error handling

Unchanged surface: same `ValueError`s for invalid product/pol/nc/mode combinations,
`NotImplementedError` for minvar + full jones. radiomesh is imported at module scope of
`utils/stokes.py` — not on any CLI-lightweight path.

### Risks / notes

- The exact output-dtype behaviour of the current `_impl.returns` coercion must be replicated;
  the oracle test is the guard.
- The readonly-view normalisation (commit 3) changes the arrays' flags as seen by *later* code
  in `stokes_image` only if the same references are reused — the normalised views are local
  to the `weight_data` call.
- Legacy MSv2 users of `weight_data` (`stokes2vis.py`) get commits 1–2 for free; their
  signature churn is milder (Dask path) and is not normalised in this PR.
