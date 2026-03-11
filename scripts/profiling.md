# Profiling scripts

The profiling scripts are layered bottom-up: low-level kernels first,
then composite operators built from them. Run the lower-level scripts
to isolate regressions before looking at the full operator benchmarks.

## Scripts (bottom-up)

### `profile_dwt_convolutions.py`

Profiles the scalar convolution kernels (`downsampling_convolution`,
`upsampling_convolution_valid_sf`, etc.) in isolation.

**What it measures:**
- Timing: OLD (generic loop, no fastmath) vs NEW (unrolled + fastmath)
  for each Daubechies filter length (db1–db5).
- SIMD analysis (`--simd-only`): inspects Numba-generated ASM and LLVM IR
  for vectorization evidence (SSE/AVX packed ops, FMA instructions,
  YMM/ZMM register usage).

**Why:** These kernels are the innermost loops of the wavelet transform.
Small improvements here multiply across levels, bases, and bands.

### `profile_polyphase.py`

Profiles axis-0 polyphase convolutions vs plain row-wise (axis-1)
convolutions.

**What it measures:**
- Timing ratio (polyphase / row-wise) for downsample and upsample pairs
  across filter lengths.

**Why:** The nocopyt DWT replaces axis-1 convolution + transpose with
direct axis-0 polyphase convolution. This script verifies that the
polyphase approach is competitive despite strided memory access, which
justifies eliminating the transpose entirely.

### `profile_dwt2d.py`

Profiles the full 2D DWT: old (copyt) vs new (nocopyt) implementations.

**What it measures:**
- Forward (dwt2d), inverse (idwt2d), and round-trip timing.
- Correctness: nocopyt coefficients == old coefficients transposed.
- Round-trip reconstruction error.

**Why:** Validates that the nocopyt layout (x-first coefficients, no
transpose buffers) is faster end-to-end for a single 2D image, before
scaling up to multi-band operators.

### `profile_wavelets.py`

Profiles the top-level Psi operator variants across multiple bands.

**What it measures:**
- Separate `dot` (analysis) and `hdot` (synthesis) timing.
- Round-trip reconstruction error.
- Memory: main process RSS delta, system memory delta, and (for Ray)
  per-actor RSS/USS breakdown.
- Threading diagnostics for Ray actors (layer, thread count, PID).

**Variants compared:**
- `Psi` — original copyt implementation with thread pool.
- `PsiNocopyt` — polyphase nocopyt with thread pool.
- `PsiNocopytRay` — nocopyt with Ray actor processes (true parallelism,
  no GIL contention).

**Why:** This is the benchmark that matters for deconvolution performance.
The Psi operator is called twice per iteration (dot + hdot), so its
round-trip time directly determines iteration cost.

## SIMD status

SIMD analysis is currently limited to `profile_dwt_convolutions.py`,
which inspects the ASM/LLVM IR of the 1D convolution kernels only. It
checks for:

- SSE packed double ops (`addpd`, `mulpd`, etc.)
- AVX double ops (v-prefix: `vaddpd`, `vmulpd`, etc.)
- AVX-512 ops (`vaddpd` with zmm registers)
- FMA instructions (`vfmadd`, `vfmsub`, `vfnmadd`)
- Register width usage (xmm = 128-bit, ymm = 256-bit, zmm = 512-bit)
- LLVM IR vector types (`<N x double>`) and FMA intrinsics

### What is NOT checked

- **Polyphase kernels**: The axis-0 polyphase convolutions
  (`conv_downsample_axis0_polyphase_pair`, etc.) are not inspected for
  SIMD. These have a different memory access pattern (strided axis-0)
  that may inhibit vectorization differently than the contiguous axis-1
  kernels.

- **Parallel loop overhead**: The `prange` wrapper functions are not
  inspected. Numba's parallel layer adds scheduling overhead that the
  SIMD analysis doesn't capture.

- **Higher-level operators**: The `dwt2d_nocopyt` and `Psi` operators
  compose multiple kernels but are not individually inspected for
  vectorization quality.

- **Auto-vectorization regressions**: There is no CI gate that would
  catch a Numba or LLVM upgrade silently dropping vectorization.

### Recommendations for future work

1. **Extend SIMD inspection to polyphase kernels.** The `inspect_simd`
   function in `profile_dwt_convolutions.py` is generic — extract it
   into a shared utility and call it from `profile_polyphase.py` on the
   `conv_downsample_axis0_polyphase_pair` and
   `conv_upsample_axis0_polyphase_pair` functions.

2. **Add a lightweight SIMD regression check.** A fast test (no timing,
   just compile + inspect ASM) that asserts key kernels emit at least N
   FMA or packed SIMD instructions. Run it in CI to catch vectorization
   regressions from Numba/LLVM upgrades.

3. **Profile with `perf stat` for hardware counters.** The current SIMD
   analysis is static (ASM inspection). Runtime counters like
   `fp_arith_inst_retired.256b_packed_double` would confirm that
   vectorized code paths are actually exercised at the expected rate.

4. **Investigate polyphase vectorization.** The strided axis-0 access
   pattern may prevent the compiler from auto-vectorizing the inner
   loop. If so, consider manual gather/scatter intrinsics or
   restructuring the polyphase kernel to process multiple columns in
   contiguous chunks.

5. **Benchmark with AVX-512 explicitly.** If the target hardware
   supports AVX-512, set `NUMBA_CPU_FEATURES=+avx512f,+avx512dq` and
   compare. Wider vectors may help the polyphase kernels where there are
   enough independent columns to fill 512-bit registers.
