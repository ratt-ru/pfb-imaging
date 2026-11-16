# Profiling scripts

Run everything from the repo root, in an environment with the scientific stack
(`uv sync --extra all --group dev`):

```bash
uv run python scripts/profiling/<script>.py --help
```

Two kinds of script live here: **reports** over the captures a real `pfb` run
leaves behind, and **kernel benchmarks** that time one piece in isolation on
synthetic data.

## Which one do I want?

| question | script |
|---|---|
| Where did a `pfb imager` / `deconv` / `degrid` / `hci` run spend its time? | `profile_report.py` |
| What allocated the memory in that run? | `memray_report.py` |
| Is the GP frequency prior worth its cost on this `.dt`? | `profile_freq_correlated_hessian.py` |
| Did a solver loop get slower, independent of its operator? | `profile_pcg.py`, `profile_primal_dual.py`, `profile_power_method.py` |
| Did the wavelet dictionary get slower? | `profile_wavelets.py`, then down the stack below |

## Reports over a pipeline run

Start here for anything that is not a kernel. Set the environment variables on
the `pfb` command, then point the report at what they wrote.

- **`profile_report.py`** — driver phases, the Ray task timeline and per-task
  cProfile captures (`PFB_RAY_TIMELINE`, `PFB_CPROFILE_DIR`). Its docstring is
  the how-to.
- **`memray_report.py`** — allocation sites per Ray task from
  `PFB_MEMRAY_DIR` captures. Its docstring says how to read the rows that are
  mappings and not resident memory.

For `pfb deconv`, read the always-on log lines first: the primal-dual section
clock and the band workers' `Ray overhead` per method.
`docs/wiki/memory-and-ray.md` ("Where the time goes") says how.

## Benchmarks on a real `.dt`

- **`profile_freq_correlated_hessian.py`** — the deconv preconditioner with and
  without the GP frequency prior (#307): three forward solves on one worker
  pool that separate the lost band-parallel fast path, the driver-side coupling
  arithmetic and the extra CG iterations. Needs a `.dt` with at least two
  bands. Read the driver's CPU column, not only its wall time; the docstring
  says why.

## Solver loops on a toy problem

Each wraps its operator in a clock, so the report splits operator time from the
solver's own bookkeeping, and prints tracemalloc's peak (numpy allocations
only). All take `--shape`, `--maxit`, `--warmup` and `--nthreads`.

- **`profile_pcg.py`** — `pcg_numba` (the in-worker CG) against `cg_inplace`
  (the driver's cube-level solve, two cubes fewer, D58). Same iterates, so the
  script checks they agree.
- **`profile_primal_dual.py`** — the legacy `primal_dual_numba` oracle against
  `PrimalDual` + `L21`, over a real in-process wavelet dictionary. The two take
  different steps for more than one basis (D57); `--bases self` makes them
  agree exactly.
- **`profile_power_method.py`** — `power_method` (numpy) against
  `power_method_numba`.

## Wavelets, bottom-up

Run the lower ones to isolate a regression before trusting the operator-level
number.

1. **`profile_dwt_convolutions.py`** — the scalar 1D convolution kernels:
   generic loop without fastmath against the unrolled fastmath form, per
   Daubechies filter length. `--simd-only` inspects the generated ASM and LLVM
   IR for packed SSE/AVX and FMA instructions.
2. **`profile_polyphase.py`** — axis-0 polyphase convolutions against plain
   row-wise ones. The no-transpose DWT depends on the strided access being
   competitive.
3. **`profile_dwt2d.py`** — the full 2D DWT, transposing (`dwt2d`) against
   no-transpose (`dwt2d_nocopyt`): forward, inverse, round trip, and that the
   two agree.
4. **`profile_wavelets.py`** — the `Psi` operator across bands: `Psi`,
   `PsiNocopyt` and the Ray-actor `PsiNocopytRay`, with `dot`/`hdot` timings
   and per-actor RSS/USS. This is the number deconv feels, twice per iteration.

### What the SIMD check does not cover

`--simd-only` looks at the 1D kernels only. The polyphase kernels (strided
access, which may vectorise differently), the `prange` wrappers and the
composed `dwt2d_nocopyt`/`Psi` operators are not inspected, and nothing in CI
would catch a numba or LLVM upgrade that silently dropped vectorisation.

## Writing a new one

`_common.py` holds the shared pieces. Call `setup_threads()` before the first
parallel numba kernel: importing `pfb_imaging` requests the TBB threading layer
but does not load `libtbb.so`, and a script that skips this dies with "No
threading layer could be loaded". Import `_common` before `ray`, since it also
turns off Ray's `uv run` runtime-env hook, which otherwise relaunches workers
without the `[full]` extra.
