# ruff: noqa
"""
Profile downsampling and upsampling convolutions in isolation.

Compares:
  - OLD: generic loop (no unrolling, no fastmath)
  - NEW: unrolled for filter sizes 2,4,6,8,10 + fastmath

Usage:
    python scripts/profile_dwt_convolutions.py [--size 4096] [--warmup 5] [--repeats 50]
"""

import argparse
import gc
from time import time

import numba
import numpy as np

# ── OLD convolutions: generic loop, no fastmath, no unrolling, inlined ──────────


@numba.njit(nogil=True, cache=False, inline="never", fastmath=False)
def old_downsampling_convolution(input, output, filter, step):
    i = step - 1
    o = 0
    input_size = input.shape[0]
    filter_size = filter.shape[0]

    while i < filter_size and i < input_size:
        fsum = input.dtype.type(0)
        j = 0
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    while i < input_size:
        fsum = input.dtype.type(0)
        j = 0
        while j < filter_size:
            fsum += input[i - j] * filter[j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    while i < filter_size:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    while i < input_size + filter_size - 1:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j < filter_size:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1


@numba.njit(nogil=True, cache=False, inline="never", fastmath=False)
def old_upsampling_convolution_valid_sf(input, filter, output):
    input_size = input.shape[0]
    filter_size = filter.shape[0]
    output_size = output.shape[0]
    o = 0
    i = (filter_size // 2) - 1
    stopping_criteria = filter_size // 2
    while i < input_size and o < output_size:
        sum_even = input.dtype.type(0)
        sum_odd = input.dtype.type(0)
        j = 0
        j2 = 0
        while j < stopping_criteria:
            input_element = input[i - j]
            sum_even += filter[j2] * input_element
            sum_odd += filter[j2 + 1] * input_element
            j += 1
            j2 += 2
        output[o] += sum_even
        output[o + 1] += sum_odd
        i += 1
        o += 2


# ── NEW convolutions: unrolled + fastmath (copy from wavelets.py) ──────


@numba.njit(nogil=True, cache=False, inline="never", fastmath=True)
def new_downsampling_convolution(input, output, filter, step):
    i = step - 1
    o = 0
    input_size = input.shape[0]
    filter_size = filter.shape[0]

    # left boundary overhang
    while i < filter_size and i < input_size:
        fsum = input.dtype.type(0)
        j = 0
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    # center — unrolled for common filter sizes
    if filter_size == 2:
        f0, f1 = filter[0], filter[1]
        while i < input_size:
            output[o] = f0 * input[i] + f1 * input[i - 1]
            i += step
            o += 1
    elif filter_size == 4:
        f0, f1, f2, f3 = filter[0], filter[1], filter[2], filter[3]
        while i < input_size:
            output[o] = f0 * input[i] + f1 * input[i - 1] + f2 * input[i - 2] + f3 * input[i - 3]
            i += step
            o += 1
    elif filter_size == 6:
        f0, f1, f2, f3, f4, f5 = filter[0], filter[1], filter[2], filter[3], filter[4], filter[5]
        while i < input_size:
            output[o] = (
                f0 * input[i]
                + f1 * input[i - 1]
                + f2 * input[i - 2]
                + f3 * input[i - 3]
                + f4 * input[i - 4]
                + f5 * input[i - 5]
            )
            i += step
            o += 1
    elif filter_size == 8:
        f0, f1, f2, f3 = filter[0], filter[1], filter[2], filter[3]
        f4, f5, f6, f7 = filter[4], filter[5], filter[6], filter[7]
        while i < input_size:
            output[o] = (
                f0 * input[i]
                + f1 * input[i - 1]
                + f2 * input[i - 2]
                + f3 * input[i - 3]
                + f4 * input[i - 4]
                + f5 * input[i - 5]
                + f6 * input[i - 6]
                + f7 * input[i - 7]
            )
            i += step
            o += 1
    elif filter_size == 10:
        f0, f1, f2, f3, f4 = filter[0], filter[1], filter[2], filter[3], filter[4]
        f5, f6, f7, f8, f9 = filter[5], filter[6], filter[7], filter[8], filter[9]
        while i < input_size:
            output[o] = (
                f0 * input[i]
                + f1 * input[i - 1]
                + f2 * input[i - 2]
                + f3 * input[i - 3]
                + f4 * input[i - 4]
                + f5 * input[i - 5]
                + f6 * input[i - 6]
                + f7 * input[i - 7]
                + f8 * input[i - 8]
                + f9 * input[i - 9]
            )
            i += step
            o += 1
    else:
        while i < input_size:
            fsum = input.dtype.type(0)
            j = 0
            while j < filter_size:
                fsum += input[i - j] * filter[j]
                j += 1
            output[o] = fsum
            i += step
            o += 1

    # center (filter wider than input)
    while i < filter_size:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    # right boundary overhang
    while i < input_size + filter_size - 1:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j < filter_size:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1


@numba.njit(nogil=True, cache=False, inline="never", fastmath=True)
def new_upsampling_convolution_valid_sf(input, filter, output):
    input_size = input.shape[0]
    filter_size = filter.shape[0]
    output_size = output.shape[0]
    o = 0
    i = (filter_size // 2) - 1

    if filter_size == 2:
        f0, f1 = filter[0], filter[1]
        while i < input_size and o < output_size:
            x0 = input[i]
            output[o] += f0 * x0
            output[o + 1] += f1 * x0
            i += 1
            o += 2
    elif filter_size == 4:
        f0, f1, f2, f3 = filter[0], filter[1], filter[2], filter[3]
        while i < input_size and o < output_size:
            x0 = input[i]
            x1 = input[i - 1]
            output[o] += f0 * x0 + f2 * x1
            output[o + 1] += f1 * x0 + f3 * x1
            i += 1
            o += 2
    elif filter_size == 6:
        f0, f1, f2, f3, f4, f5 = filter[0], filter[1], filter[2], filter[3], filter[4], filter[5]
        while i < input_size and o < output_size:
            x0 = input[i]
            x1 = input[i - 1]
            x2 = input[i - 2]
            output[o] += f0 * x0 + f2 * x1 + f4 * x2
            output[o + 1] += f1 * x0 + f3 * x1 + f5 * x2
            i += 1
            o += 2
    elif filter_size == 8:
        f0, f1, f2, f3 = filter[0], filter[1], filter[2], filter[3]
        f4, f5, f6, f7 = filter[4], filter[5], filter[6], filter[7]
        while i < input_size and o < output_size:
            x0 = input[i]
            x1 = input[i - 1]
            x2 = input[i - 2]
            x3 = input[i - 3]
            output[o] += f0 * x0 + f2 * x1 + f4 * x2 + f6 * x3
            output[o + 1] += f1 * x0 + f3 * x1 + f5 * x2 + f7 * x3
            i += 1
            o += 2
    elif filter_size == 10:
        f0, f1, f2, f3, f4 = filter[0], filter[1], filter[2], filter[3], filter[4]
        f5, f6, f7, f8, f9 = filter[5], filter[6], filter[7], filter[8], filter[9]
        while i < input_size and o < output_size:
            x0 = input[i]
            x1 = input[i - 1]
            x2 = input[i - 2]
            x3 = input[i - 3]
            x4 = input[i - 4]
            output[o] += f0 * x0 + f2 * x1 + f4 * x2 + f6 * x3 + f8 * x4
            output[o + 1] += f1 * x0 + f3 * x1 + f5 * x2 + f7 * x3 + f9 * x4
            i += 1
            o += 2
    else:
        stopping_criteria = filter_size // 2
        while i < input_size and o < output_size:
            sum_even = input.dtype.type(0)
            sum_odd = input.dtype.type(0)
            j = 0
            j2 = 0
            while j < stopping_criteria:
                input_element = input[i - j]
                sum_even += filter[j2] * input_element
                sum_odd += filter[j2 + 1] * input_element
                j += 1
                j2 += 2
            output[o] += sum_even
            output[o + 1] += sum_odd
            i += 1
            o += 2


# ── Row-loop wrappers (simulate how DWT calls the convolutions) ────────
# These apply the convolution to every row of a 2D array, which is the
# actual hot loop pattern in dwt2d_level / idwt2d_level.


@numba.njit(nogil=True, cache=False)
def apply_downsample_rows(image, output, filt, step, func):
    """Apply downsampling convolution to every row of image."""
    nrows = image.shape[0]
    for i in range(nrows):
        func(image[i, :], output[i, :], filt, step)


@numba.njit(nogil=True, cache=False)
def apply_upsample_rows(input_2d, filt, output_2d, func):
    """Apply upsampling convolution to every row of input."""
    nrows = input_2d.shape[0]
    for i in range(nrows):
        func(input_2d[i, :], filt, output_2d[i, :])


# ── Profiling ──────────────────────────────────────────────────────────


def coeff_size(nsignal, nfilter):
    return (nsignal + nfilter - 1) // 2


def profile_downsampling(size, wavelet, warmup, repeats):
    """Profile downsampling convolution over all rows of a 2D array."""
    import pywt

    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.ascontiguousarray(wvlt.filter_bank[0])
    dec_hi = np.ascontiguousarray(wvlt.filter_bank[1])
    filter_length = len(dec_lo)

    rng = np.random.default_rng(42)
    image = rng.standard_normal((size, size))

    ncols_out = coeff_size(size, filter_length)
    out_old = np.empty((size, ncols_out))
    out_new = np.empty((size, ncols_out))

    # warmup — compile both variants
    for _ in range(warmup):
        apply_downsample_rows(image, out_old, dec_lo, 2, old_downsampling_convolution)
        apply_downsample_rows(image, out_new, dec_lo, 2, new_downsampling_convolution)

    # verify correctness
    apply_downsample_rows(image, out_old, dec_lo, 2, old_downsampling_convolution)
    apply_downsample_rows(image, out_new, dec_lo, 2, new_downsampling_convolution)
    max_diff = np.max(np.abs(out_old - out_new))

    # time old
    gc.collect()
    times_old = []
    for _ in range(repeats):
        t0 = time()
        apply_downsample_rows(image, out_old, dec_lo, 2, old_downsampling_convolution)
        apply_downsample_rows(image, out_old, dec_hi, 2, old_downsampling_convolution)
        times_old.append(time() - t0)

    # time new
    gc.collect()
    times_new = []
    for _ in range(repeats):
        t0 = time()
        apply_downsample_rows(image, out_new, dec_lo, 2, new_downsampling_convolution)
        apply_downsample_rows(image, out_new, dec_hi, 2, new_downsampling_convolution)
        times_new.append(time() - t0)

    return np.array(times_old), np.array(times_new), max_diff


def profile_upsampling(size, wavelet, warmup, repeats):
    """Profile upsampling convolution over all rows of a 2D array."""
    import pywt

    wvlt = pywt.Wavelet(wavelet)
    rec_lo = np.ascontiguousarray(wvlt.filter_bank[2])
    rec_hi = np.ascontiguousarray(wvlt.filter_bank[3])
    filter_length = len(rec_lo)

    ncols_in = coeff_size(size, filter_length)
    rng = np.random.default_rng(42)
    input_2d = rng.standard_normal((size, ncols_in))

    # output size: 2*ncols_in - filter_length + 2
    ncols_out = 2 * ncols_in - filter_length + 2
    out_old = np.zeros((size, ncols_out))
    out_new = np.zeros((size, ncols_out))

    # warmup
    for _ in range(warmup):
        out_old[:] = 0.0
        apply_upsample_rows(input_2d, rec_lo, out_old, old_upsampling_convolution_valid_sf)
        out_new[:] = 0.0
        apply_upsample_rows(input_2d, rec_lo, out_new, new_upsampling_convolution_valid_sf)

    # verify correctness
    out_old[:] = 0.0
    apply_upsample_rows(input_2d, rec_lo, out_old, old_upsampling_convolution_valid_sf)
    apply_upsample_rows(input_2d, rec_hi, out_old, old_upsampling_convolution_valid_sf)
    out_new[:] = 0.0
    apply_upsample_rows(input_2d, rec_lo, out_new, new_upsampling_convolution_valid_sf)
    apply_upsample_rows(input_2d, rec_hi, out_new, new_upsampling_convolution_valid_sf)
    max_diff = np.max(np.abs(out_old - out_new))

    # time old
    gc.collect()
    times_old = []
    for _ in range(repeats):
        out_old[:] = 0.0
        t0 = time()
        apply_upsample_rows(input_2d, rec_lo, out_old, old_upsampling_convolution_valid_sf)
        apply_upsample_rows(input_2d, rec_hi, out_old, old_upsampling_convolution_valid_sf)
        times_old.append(time() - t0)

    # time new
    gc.collect()
    times_new = []
    for _ in range(repeats):
        out_new[:] = 0.0
        t0 = time()
        apply_upsample_rows(input_2d, rec_lo, out_new, new_upsampling_convolution_valid_sf)
        apply_upsample_rows(input_2d, rec_hi, out_new, new_upsampling_convolution_valid_sf)
        times_new.append(time() - t0)

    return np.array(times_old), np.array(times_new), max_diff


def print_result(label, times_old, times_new, max_diff):
    med_old = np.median(times_old) * 1000
    med_new = np.median(times_new) * 1000
    min_old = np.min(times_old) * 1000
    min_new = np.min(times_new) * 1000
    speedup = np.median(times_old) / np.median(times_new)

    print(f"  {label:40s}  {med_old:7.2f}ms  {med_new:7.2f}ms  {speedup:5.2f}x  {max_diff:.1e}")


def inspect_simd(func, sig, label):
    """Inspect a numba dispatcher's ASM for SIMD instructions.

    Returns a dict mapping instruction class to count.
    """
    import re

    # x86 SIMD instruction prefixes/patterns (SSE, AVX, AVX-512)
    simd_patterns = {
        "SSE (scalar double)": re.compile(r"\b(addsd|mulsd|subsd|divsd|sqrtsd|maxsd|minsd|fmasd)\b"),
        "SSE (packed double)": re.compile(r"\b(addpd|mulpd|subpd|divpd|sqrtpd|maxpd|minpd)\b"),
        "SSE (packed single)": re.compile(r"\b(addps|mulps|subps|divps|sqrtps|maxps|minps)\b"),
        "AVX (v-prefix double)": re.compile(
            r"\bv(addpd|mulpd|subpd|divpd|sqrtpd|maxpd|minpd|fmadd[0-9]*pd|fmsub[0-9]*pd|fnmadd[0-9]*pd)\b"
        ),
        "AVX (v-prefix single)": re.compile(
            r"\bv(addps|mulps|subps|divps|sqrtps|maxps|minps|fmadd[0-9]*ps|fmsub[0-9]*ps|fnmadd[0-9]*ps)\b"
        ),
        "AVX scalar double": re.compile(
            r"\bv(addsd|mulsd|subsd|divsd|sqrtsd|fmadd[0-9]*sd|fmsub[0-9]*sd|fnmadd[0-9]*sd)\b"
        ),
        "AVX-512 (zmm)": re.compile(r"\bzmm\d+\b"),
        "FMA": re.compile(r"\bv?fmadd[0-9]*(sd|ss|pd|ps)\b"),
        "FMA (fmsub)": re.compile(r"\bv?fmsub[0-9]*(sd|ss|pd|ps)\b"),
        "FMA (fnmadd)": re.compile(r"\bv?fnmadd[0-9]*(sd|ss|pd|ps)\b"),
    }

    # LLVM IR vectorization hints
    llvm_vec_patterns = {
        "LLVM vector ops": re.compile(r"<\d+ x double>"),
        "LLVM FMA intrinsic": re.compile(r"@llvm\.fma\."),
        "LLVM fmuladd": re.compile(r"@llvm\.fmuladd\."),
        "LLVM vector shuffle": re.compile(r"shufflevector"),
    }

    results = {}

    # get ASM
    try:
        func.compile(sig)
        asm_dict = func.inspect_asm()
        asm_text = "\n".join(asm_dict.values())

        for name, pat in simd_patterns.items():
            count = len(pat.findall(asm_text))
            if count > 0:
                results[name] = count

        # count register usage to gauge vector width
        ymm_count = len(re.findall(r"\bymm\d+\b", asm_text))
        xmm_count = len(re.findall(r"\bxmm\d+\b", asm_text))
        zmm_count = len(re.findall(r"\bzmm\d+\b", asm_text))
        results["xmm register refs"] = xmm_count
        results["ymm register refs"] = ymm_count
        results["zmm register refs"] = zmm_count

    except Exception as e:
        results["ASM error"] = str(e)

    # get LLVM IR
    llvm_results = {}
    try:
        llvm_dict = func.inspect_llvm()
        llvm_text = "\n".join(llvm_dict.values())

        for name, pat in llvm_vec_patterns.items():
            count = len(pat.findall(llvm_text))
            if count > 0:
                llvm_results[name] = count

    except Exception as e:
        llvm_results["LLVM error"] = str(e)

    return results, llvm_results


def print_simd_report():
    """Compile all key convolution functions and report SIMD usage."""
    f64_1d = numba.types.Array(numba.float64, 1, "C")

    # trigger compilation with real data
    n = 64
    arr1d = np.zeros(n)
    out1d = np.zeros(n)
    filt = np.array([0.5, 0.5])

    new_downsampling_convolution(arr1d, out1d[:32], filt, 2)
    new_upsampling_convolution_valid_sf(arr1d[:32], filt, out1d)
    old_downsampling_convolution(arr1d, out1d[:32], filt, 2)
    old_upsampling_convolution_valid_sf(arr1d[:32], filt, out1d)

    # also compile with longer filters to see if vectorization kicks in
    filt8 = np.zeros(8)  # db4 filter length
    filt10 = np.zeros(10)  # db5 filter length
    new_downsampling_convolution(arr1d, out1d[:32], filt8, 2)
    old_downsampling_convolution(arr1d, out1d[:32], filt8, 2)
    new_downsampling_convolution(arr1d, out1d[:32], filt10, 2)
    old_downsampling_convolution(arr1d, out1d[:32], filt10, 2)

    functions_to_inspect = [
        (
            "NEW downsampling_convolution (fastmath)",
            new_downsampling_convolution,
            (f64_1d, f64_1d, f64_1d, numba.int64),
        ),
        (
            "OLD downsampling_convolution (no fastmath)",
            old_downsampling_convolution,
            (f64_1d, f64_1d, f64_1d, numba.int64),
        ),
        (
            "NEW upsampling_convolution_valid_sf (fastmath)",
            new_upsampling_convolution_valid_sf,
            (f64_1d, f64_1d, f64_1d),
        ),
        (
            "OLD upsampling_convolution_valid_sf (no fastmath)",
            old_upsampling_convolution_valid_sf,
            (f64_1d, f64_1d, f64_1d),
        ),
    ]

    from llvmlite import binding as llvm

    print(f"\n{'=' * 72}")
    print("  SIMD / Vectorization Analysis")
    print(f"{'=' * 72}")
    cpu_name = llvm.get_host_cpu_name()
    print(f"  CPU target: {cpu_name}")

    try:
        features = llvm.get_host_cpu_features()
        simd_features = sorted(
            k for k, v in features.items() if v and any(s in k for s in ["sse", "avx", "fma", "mmx"])
        )
        print(f"  SIMD features: {', '.join(simd_features)}")
    except Exception:
        pass

    print(f"  Numba version: {numba.__version__}")
    print()

    for label, func, sig in functions_to_inspect:
        asm_results, llvm_results = inspect_simd(func, sig, label)

        print(f"  {label}")
        print(f"  {'─' * 60}")

        if not asm_results and not llvm_results:
            print("    No SIMD instructions detected")
        else:
            if asm_results:
                print("    ASM:")
                for k, v in sorted(asm_results.items(), key=lambda x: -x[1]):
                    if v > 0:
                        print(f"      {k:35s}  {v:5d}")
            if llvm_results:
                print("    LLVM IR:")
                for k, v in sorted(llvm_results.items(), key=lambda x: -x[1]):
                    if v > 0:
                        print(f"      {k:35s}  {v:5d}")
        print()

    # Summary comparison
    print(f"  {'─' * 60}")
    print("  Key takeaways:")

    new_asm_results, _ = inspect_simd(
        new_downsampling_convolution,
        (f64_1d, f64_1d, f64_1d, numba.int64),
        "new_ds",
    )
    old_asm_results, _ = inspect_simd(
        old_downsampling_convolution,
        (f64_1d, f64_1d, f64_1d, numba.int64),
        "old_ds",
    )

    new_fma = (
        new_asm_results.get("FMA", 0) + new_asm_results.get("FMA (fmsub)", 0) + new_asm_results.get("FMA (fnmadd)", 0)
    )
    old_fma = (
        old_asm_results.get("FMA", 0) + old_asm_results.get("FMA (fmsub)", 0) + old_asm_results.get("FMA (fnmadd)", 0)
    )
    new_ymm = new_asm_results.get("ymm register refs", 0)
    old_ymm = old_asm_results.get("ymm register refs", 0)
    new_packed = new_asm_results.get("SSE (packed double)", 0) + new_asm_results.get("AVX (v-prefix double)", 0)
    old_packed = old_asm_results.get("SSE (packed double)", 0) + old_asm_results.get("AVX (v-prefix double)", 0)

    print(
        f"    FMA instructions:  OLD={old_fma}  NEW={new_fma}  {'✓ fastmath enables FMA' if new_fma > old_fma else '✗ no FMA difference'}"
    )
    print(
        f"    Packed SIMD ops:   OLD={old_packed}  NEW={new_packed}  {'✓ vectorized' if new_packed > 0 else '✗ scalar only'}"
    )
    print(f"    YMM (256-bit) use: OLD={old_ymm}  NEW={new_ymm}  {'✓ AVX width' if new_ymm > old_ymm else '—'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Profile DWT convolution kernels")
    parser.add_argument("--size", type=int, default=2048, help="Square image dimension")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=30, help="Timed iterations")
    parser.add_argument("--simd-only", action="store_true", help="Only run SIMD analysis, skip timing")
    args = parser.parse_args()

    size = args.size
    warmup = args.warmup
    repeats = args.repeats

    wavelets = ["db1", "db2", "db3", "db4", "db5"]

    print(f"Image: ({size}, {size}),  warmup={warmup},  repeats={repeats}")
    print(f"Numba {numba.__version__},  threads={numba.config.NUMBA_NUM_THREADS}")
    print()

    # ── SIMD analysis ──────────────────────────────────────────────────
    print_simd_report()

    if args.simd_only:
        return

    # ── Downsampling convolution ───────────────────────────────────────
    print(f"{'=' * 80}")
    print("  Downsampling convolution (lo + hi filters, all rows)")
    print(f"{'=' * 80}")
    print(f"  {'wavelet (filter_len)':40s}  {'OLD':>7s}  {'NEW':>7s}  {'speed':>5s}  {'diff':>7s}")
    print(f"  {'─' * 74}")

    for wavelet in wavelets:
        import pywt

        flen = len(pywt.Wavelet(wavelet).filter_bank[0])
        times_old, times_new, diff = profile_downsampling(size, wavelet, warmup, repeats)
        print_result(f"{wavelet} (filter_len={flen})", times_old, times_new, diff)

    # ── Upsampling convolution ─────────────────────────────────────────
    print()
    print(f"{'=' * 80}")
    print("  Upsampling convolution (lo + hi filters, all rows)")
    print(f"{'=' * 80}")
    print(f"  {'wavelet (filter_len)':40s}  {'OLD':>7s}  {'NEW':>7s}  {'speed':>5s}  {'diff':>7s}")
    print(f"  {'─' * 74}")

    for wavelet in wavelets:
        flen = len(pywt.Wavelet(wavelet).filter_bank[0])
        times_old, times_new, diff = profile_upsampling(size, wavelet, warmup, repeats)
        print_result(f"{wavelet} (filter_len={flen})", times_old, times_new, diff)

    print()


if __name__ == "__main__":
    main()
