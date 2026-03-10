# DWT Implementation Notes

## Architecture

The 2D DWT is implemented as hand-written Numba JIT kernels in `wavelets.py`,
wrapped by two operator classes in `operators/psi.py`:

- **`PsiBand`** (`@jitclass`) — parallelizes within each wavelet level via
  `prange` over image rows. Shared scratch buffers force sequential iteration
  over bases. Multi-band parallelism via `ThreadPoolExecutor` in the `Psi` class.
- **`PsiBasis`** (Python class + standalone `@njit(parallel=True)` functions) —
  parallelizes over the **basis axis** with a single `prange`. Per-basis
  contiguous scratch buffers eliminate data races. Bands processed sequentially,
  reusing buffers.

### Data flow (analysis direction, single level)

```
image (nx, ny)
  -> row-wise downsampling_convolution (parallel over rows via prange)
  -> cbuff (nax, nay)  [lo | hi interleaved along columns]
  -> copyt transpose -> cbufft (nay, nax)
  -> col-wise downsampling_convolution (parallel over cols via prange)
  -> coeffs (nay, nax)  [2x2 quadrant layout: LL | LH / HL | HH]
  -> approx = copyt(coeffs[:midy, :midx]) into pre-allocated buffer
     -> next level input
```

Synthesis (`idwt2d_level`) reverses this with `upsampling_convolution_valid_sf`.

### Key design choices
- Coefficients stored **transposed** relative to input (performance).
- Explicit tiled `copyt()` for cache-friendly transpose.
- Pre-allocated buffers (`alpha`, `cbuff`, `cbufft`, `approx`, `image`) avoid
  allocations in hot loops.
- `fastmath=True` on convolution kernels enables FMA and SIMD vectorization.
- Array-based bookkeeping (`ix`, `iy`, `sx`, `sy` as int64 arrays) replaces
  the old `numba.typed.Dict` approach for lower overhead.

### Serial variants

`wavelets.py` provides `_seq` suffix functions (`copyt_seq`, `dwt2d_level_seq`,
`idwt2d_level_seq`, `dwt2d_seq`, `idwt2d_seq`) that are identical to the
parallel versions but use plain `range` instead of `prange`. These exist for
use inside an outer `prange` loop (e.g., `PsiBasis`) to avoid nested parallel
region overhead.

---

## Parallelization Strategies

### PsiBand (row-parallel)

```
for each basis (sequential):
    for each level (sequential):
        prange over rows -> downsampling_convolution
        copyt (prange)
        prange over cols -> downsampling_convolution
        copyt (prange)
```

- ~54 fork/join barriers per `dot` call (6 bases × 3 levels × 3 prange
  regions per level).
- Shared scratch buffers (`cbuff`, `cbufft`, `approx`) prevent basis-level
  parallelism.
- Thread pool dispatches bands to separate `PsiBand` instances (each with its
  own buffers).

### PsiBasis (basis-parallel)

```
prange over bases (single fork/join):
    for each level (sequential):
        range over rows -> downsampling_convolution (serial)
        copyt_seq (serial)
        range over cols -> downsampling_convolution (serial)
        copyt_seq (serial)
reduce: sum per-basis images into output
```

- **2 fork/join barriers** per dot+hdot call (one for dot, one for hdot).
- Per-basis contiguous buffers: `cbuff[nbasis, nxmax, nymax]`, etc.
- Bands processed sequentially, reusing all per-basis buffers.

### Memory trade-off

For 6 bases at 2048², 3-level decomposition:

| | PsiBand (nband=1) | PsiBasis (nband=1) | PsiBand (nband=8) | PsiBasis (nband=8) |
|---|---|---|---|---|
| Scratch buffers | ~77 MB (1 set) | ~550 MB (6 sets) | ~616 MB (8 sets) | ~550 MB (6 sets, shared) |

PsiBasis uses more memory for nband=1 but becomes more efficient for nband > ~7
because it reuses per-basis buffers across bands instead of duplicating per-band
instances.

---

## Completed Optimizations

### 1. `fastmath=True` on convolution kernels

Enabled on `downsampling_convolution` and `upsampling_convolution_valid_sf`.
LLVM now emits FMA instructions and AVX 256-bit packed double operations.

Measured via ASM inspection:
- **OLD** (no fastmath): 0 FMA, 0 packed ops, scalar only (addsd/mulsd)
- **NEW** (fastmath): 15 FMA instructions, 22 packed AVX double ops, 538 YMM
  register references, 115 LLVM vector operations

Round-trip reconstruction error remains < 1e-10 (decimal=10 in tests).

### 2. Pre-allocated approx buffer

Replaced `coeffs[0:midy, 0:midx].T.copy()` (allocation per level per basis)
with `copyt` into a pre-allocated buffer. Eliminates all allocations inside the
multi-level DWT loop.

At 2048² with db5, 3 levels:
- Old: 3 allocations totalling ~11 KB per `dwt2d` call
- New: 1 buffer of ~33 KB reused across levels (amortized to zero)

### 3. Array-based bookkeeping

Replaced `numba.typed.Dict` with plain int64 arrays for `ix`, `iy`, `sx`, `sy`,
`spx`, `spy`. Filter banks stored in `numba.typed.List` indexed by integer
wavelet ID. Eliminates dictionary lookup overhead inside JIT code.

### 4. Persistent thread pool

`ThreadPoolExecutor` created once in `Psi.__init__` and reused across calls.
Skipped entirely for nband=1 (direct call, no pool overhead).

### 5. `NUMBA_NUM_THREADS` ceiling fix

`numba.set_num_threads(n)` can only reduce below the process-wide
`NUMBA_NUM_THREADS` ceiling, never increase. The profiling script now parses
`--psi-nthreads` before importing numba and sets `NUMBA_NUM_THREADS`
accordingly.

### 6. PsiBasis operator

Basis-parallel Psi operator using a single `prange` over the basis axis with
per-basis contiguous scratch buffers. Eliminates the many fork/join barriers
of the row-parallel approach.

---

## Performance Results

### PsiBand vs PsiBasis (8 threads, 6 bases, nband=1)

| Image Size | PsiBand | PsiBasis | Speedup |
|------------|---------|----------|---------|
| 512² | 17.6 ms | 12.1 ms | **1.46x** |
| 1024² | 58.8 ms | 48.3 ms | **1.22x** |
| 2048² | 217.7 ms | 180.3 ms | **1.21x** |
| 4096² | 827.6 ms | 803.3 ms | **1.03x** |

PsiBasis advantage is largest at smaller image sizes where fork/join overhead
dominates compute. At 4096² the per-level compute is large enough to amortize
the fork/join cost, so the gap shrinks.

At 1 thread both strategies are identical (~609 ms at 2048²), confirming the
improvement is purely from better parallelization, not algorithmic changes.

### Scaling with threads (2048², 6 bases, nband=1)

| Threads | PsiBand | PsiBasis |
|---------|---------|----------|
| 1 | 609 ms | 610 ms |
| 8 | 218 ms | 180 ms |
| 16 | 201 ms | 193 ms |

Diminishing returns beyond 8 threads because:
- PsiBand: deeper levels have fewer rows than threads
- PsiBasis: only 6 bases, so only 6 of 16 threads have work

---

## Test Coverage

### test_wavelets.py
- Forward-inverse round-trip matching PyWavelets for db1, db4, db5 at shapes
  (128,256), (512,128) and levels 1-3.
- Coefficient packing matches PyWavelets layout.

### test_wavelets_optimized.py (97 tests)
- **Approx buffer round-trip**: db1/db4/db5 × 3 shapes × 3 levels
- **Psi adjoint**: `<Ψx, α> = <x, Ψ†α>` for multiple sizes and levels
- **Psi multi-band**: nband=2,4 matches independent single-band
- **Self basis round-trip**: identity basis through dot/hdot
- **Minimum size inputs**: smallest valid images for db1/db2
- **Non-square inputs**: (16,256) and (256,16)
- **Fastmath accuracy vs PyWavelets**: round-trip matches pywt reference
- **Serial DWT round-trip**: `dwt2d_seq`/`idwt2d_seq` for all wavelets/sizes
- **Serial vs parallel consistency**: `dwt2d_seq` produces identical coefficients
  to `dwt2d`
- **PsiBasis round-trip**: single wavelet basis, self-only, mixed bases
- **PsiBasis adjoint**: inner product test for multiple sizes
- **PsiBasis matches PsiBand**: forward and adjoint outputs identical to decimal=12

---

## Remaining Optimization Opportunities

### Medium impact

1. **Scope buffer zeroing in `idwt2d_level`**
   Currently zeros full `cbufft` and `image` at every level. Should zero only
   the sub-region actually used: `cbufft[0:nay, 0:nx] = 0.0`.

2. **Adaptive strategy selection**
   Auto-select PsiBand vs PsiBasis based on image size and thread count.
   PsiBasis wins at smaller sizes; PsiBand wins when rows >> threads at large
   sizes. A heuristic crossover around nx*ny ~ 8M could switch automatically.

### Lower impact / exploratory

3. **Specialize convolution for common filter lengths**
   Generate versions for filter_size = 2 (Haar/db1), 4, 6, 8, 10 using
   `numba.literally()` or dispatch. Enables full loop unrolling. Most impactful
   for short filters (db1-db3).

4. **Pre-reverse filters for forward access**
   Reverse `dec_lo`, `dec_hi` at construction so the hot center loop becomes
   a forward dot product. Forward sequential access is friendlier to SIMD
   prefetch.

5. **Skip copyt for small matrices**
   At coarse decomposition levels the matrices are small (e.g., 32×32). The
   `prange` scheduling overhead in `copyt` may exceed the cost of operating
   on non-contiguous memory directly. A size threshold could skip the transpose.

6. **Lifting scheme**
   Replace the filter-bank convolution with the lifting scheme. For db1 (Haar)
   this is trivially `(a+b)/2, (a-b)/2` with no multiply. For longer
   Daubechies wavelets, lifting reduces the operation count by ~50%.

7. **DUCC0 / SciPy for convolution**
   For longer filters, `scipy.signal.fftconvolve` or DUCC's FFT for the
   convolution step might outperform the direct form, especially for
   filter lengths > 10.
