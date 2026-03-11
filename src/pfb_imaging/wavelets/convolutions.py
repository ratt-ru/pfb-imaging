import numba as nb
from numba import prange


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def downsampling_convolution(input, output, filter, step):
    i = step - 1
    o = 0
    input_size = input.shape[0]
    filter_size = filter.shape[0]

    # left boundary overhang (few iterations, not worth unrolling)
    while i < filter_size and i < input_size:
        fsum = input.dtype.type(0)
        j = 0
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    # center — unrolled for common filter sizes (db1-db5)
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
        f0, f1, f2, f3, f4, f5 = (
            filter[0],
            filter[1],
            filter[2],
            filter[3],
            filter[4],
            filter[5],
        )
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
        # generic fallback
        while i < input_size:
            fsum = input.dtype.type(0)
            j = 0
            while j < filter_size:
                fsum += input[i - j] * filter[j]
                j += 1
            output[o] = fsum
            i += step
            o += 1

    # center (filter wider than input — rare, small data)
    while i < filter_size:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1

    # right boundary overhang (few iterations)
    while i < input_size + filter_size - 1:
        fsum = input.dtype.type(0)
        j = i - input_size + 1
        while j < filter_size:
            fsum += filter[j] * input[i - j]
            j += 1
        output[o] = fsum
        i += step
        o += 1


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def upsampling_convolution_valid_sf(input, filter, output):
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
        f0, f1, f2, f3, f4, f5 = (
            filter[0],
            filter[1],
            filter[2],
            filter[3],
            filter[4],
            filter[5],
        )
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
        # generic fallback
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


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def upsampling_convolution_valid_sf_set(input, filter, output):
    """Like upsampling_convolution_valid_sf but writes (=) instead of accumulating (+=).

    Used as the first filter call in idwt2d_level, eliminating the need to zero
    the output buffer before the convolution pair.
    """
    input_size = input.shape[0]
    filter_size = filter.shape[0]
    output_size = output.shape[0]

    o = 0
    i = (filter_size // 2) - 1

    if filter_size == 2:
        f0, f1 = filter[0], filter[1]
        while i < input_size and o < output_size:
            x0 = input[i]
            output[o] = f0 * x0
            output[o + 1] = f1 * x0
            i += 1
            o += 2
    elif filter_size == 4:
        f0, f1, f2, f3 = filter[0], filter[1], filter[2], filter[3]
        while i < input_size and o < output_size:
            x0 = input[i]
            x1 = input[i - 1]
            output[o] = f0 * x0 + f2 * x1
            output[o + 1] = f1 * x0 + f3 * x1
            i += 1
            o += 2
    elif filter_size == 6:
        f0, f1, f2, f3, f4, f5 = (
            filter[0],
            filter[1],
            filter[2],
            filter[3],
            filter[4],
            filter[5],
        )
        while i < input_size and o < output_size:
            x0 = input[i]
            x1 = input[i - 1]
            x2 = input[i - 2]
            output[o] = f0 * x0 + f2 * x1 + f4 * x2
            output[o + 1] = f1 * x0 + f3 * x1 + f5 * x2
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
            output[o] = f0 * x0 + f2 * x1 + f4 * x2 + f6 * x3
            output[o + 1] = f1 * x0 + f3 * x1 + f5 * x2 + f7 * x3
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
            output[o] = f0 * x0 + f2 * x1 + f4 * x2 + f6 * x3 + f8 * x4
            output[o + 1] = f1 * x0 + f3 * x1 + f5 * x2 + f7 * x3 + f9 * x4
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
            output[o] = sum_even
            output[o + 1] = sum_odd
            i += 1
            o += 2


# Convention: pywt 'zero' boundary mode throughout.
#
# ANALYSIS:
#   y[i] = sum_k h[k] * x[2i + 1 - k],   x[n] = 0 outside [0, N)
#   Output length: N_out = (N - 1) // 2 + K // 2
#
#   This is equivalent to np.convolve(x, h)[1::2][:N_out], i.e. the full linear
#   convolution sampled at odd positions then truncated.
#
#   Polyphase split (k = 2m even, k = 2m+1 odd):
#     even taps dec_lo = h[0::2]: Y0[i] = sum_m dec_lo[m] * X[2i + 1 - 2m, :]   (ODD  input rows)
#     odd  taps dec_hi = h[1::2]: Y1[i] = sum_m dec_hi[m] * X[2i     - 2m, :]   (EVEN input rows)
#
# SYNTHESIS:
#   y[n] = sum_k g[k] * x_up[n - k + K - 2]  where x_up[2i] = X[i,:], x_up[2i+1] = 0
#   (offset K-2, confirmed for all standard orthogonal/biorthogonal wavelets)
#
#   Polyphase split (even/odd output rows):
#     Even outputs: Y[2i,  :] = sum_m g0[m] * X[i - m + K//2 - 1, :]   (even taps g0)
#     Odd  outputs: Y[2i+1,:] = sum_m g1[m] * X[i - m + K//2 - 1, :]   (odd  taps g1)
#
#   Both output phases use the SAME gather index (i - m + K//2 - 1), weighted by
#   different filter taps. This is a direct consequence of the K-2 offset being even.


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def conv_downsample_axis0_polyphase(x, h, y):
    """
    In-place analysis: convolve x along axis 0 with h, downsample by 2.
    pywt 'zero' boundary convention. No transpose or allocation required.

    Parameters
    ----------
    x : (n, n_cols) float64 C-contiguous array
    h : (k,)   filter kernel  (e.g. wavelet dec_lo or dec_hi)
    y : (n_out, n_cols) pre-allocated output array, n_out = (n-1)//2 + k//2
        Will be overwritten.
    """
    n = x.shape[0]
    n_cols = x.shape[1]
    k = h.shape[0]
    k0 = (k + 1) // 2
    k1 = k // 2
    n_out = y.shape[0]
    i_lo = k0 - 1
    i_hi = (n - 2) // 2

    # --- Boundary rows (serial, few iterations) ---
    for i in range(min(i_lo, n_out)):
        for j in range(n_cols):
            acc = 0.0
            for m in range(k0):
                r = 2 * i + 1 - 2 * m
                if 0 <= r < n:
                    acc += h[2 * m] * x[r, j]
            for m in range(k1):
                r = 2 * i - 2 * m
                if 0 <= r < n:
                    acc += h[2 * m + 1] * x[r, j]
            y[i, j] = acc
    for i in range(max(i_hi + 1, i_lo), n_out):
        for j in range(n_cols):
            acc = 0.0
            for m in range(k0):
                r = 2 * i + 1 - 2 * m
                if 0 <= r < n:
                    acc += h[2 * m] * x[r, j]
            for m in range(k1):
                r = 2 * i - 2 * m
                if 0 <= r < n:
                    acc += h[2 * m + 1] * x[r, j]
            y[i, j] = acc

    # --- Interior (parallel, unrolled for common filter sizes) ---
    int_lo = max(i_lo, 0)
    int_hi = min(i_hi + 1, n_out)
    if k == 2:
        h0, h1 = h[0], h[1]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                y[i, j] = h0 * x[2 * i + 1, j] + h1 * x[2 * i, j]
    elif k == 4:
        h0, h1, h2, h3 = h[0], h[1], h[2], h[3]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                y[i, j] = h0 * x[2 * i + 1, j] + h2 * x[2 * i - 1, j] + h1 * x[2 * i, j] + h3 * x[2 * i - 2, j]
    elif k == 6:
        h0, h1, h2, h3, h4, h5 = h[0], h[1], h[2], h[3], h[4], h[5]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                y[i, j] = (
                    h0 * x[2 * i + 1, j]
                    + h2 * x[2 * i - 1, j]
                    + h4 * x[2 * i - 3, j]
                    + h1 * x[2 * i, j]
                    + h3 * x[2 * i - 2, j]
                    + h5 * x[2 * i - 4, j]
                )
    elif k == 8:
        h0, h1, h2, h3 = h[0], h[1], h[2], h[3]
        h4, h5, h6, h7 = h[4], h[5], h[6], h[7]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                y[i, j] = (
                    h0 * x[2 * i + 1, j]
                    + h2 * x[2 * i - 1, j]
                    + h4 * x[2 * i - 3, j]
                    + h6 * x[2 * i - 5, j]
                    + h1 * x[2 * i, j]
                    + h3 * x[2 * i - 2, j]
                    + h5 * x[2 * i - 4, j]
                    + h7 * x[2 * i - 6, j]
                )
    elif k == 10:
        h0, h1, h2, h3, h4 = h[0], h[1], h[2], h[3], h[4]
        h5, h6, h7, h8, h9 = h[5], h[6], h[7], h[8], h[9]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                y[i, j] = (
                    h0 * x[2 * i + 1, j]
                    + h2 * x[2 * i - 1, j]
                    + h4 * x[2 * i - 3, j]
                    + h6 * x[2 * i - 5, j]
                    + h8 * x[2 * i - 7, j]
                    + h1 * x[2 * i, j]
                    + h3 * x[2 * i - 2, j]
                    + h5 * x[2 * i - 4, j]
                    + h7 * x[2 * i - 6, j]
                    + h9 * x[2 * i - 8, j]
                )
    else:
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                acc = 0.0
                for m in range(k0):
                    acc += h[2 * m] * x[2 * i + 1 - 2 * m, j]
                for m in range(k1):
                    acc += h[2 * m + 1] * x[2 * i - 2 * m, j]
                y[i, j] = acc


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def conv_upsample_axis0_polyphase(x, h, y):
    """
    In-place synthesis: upsample x by 2 along axis 0, convolve with h.
    pywt 'zero' boundary convention. No upsampled array is allocated.

    Parameters
    ----------
    x : (n, n_cols)        float64 C-contiguous subband array
    h : (k,)          synthesis filter  (e.g. wavelet rec_lo or rec_hi)
    y : (out_size, n_cols) pre-allocated output array.  Will be overwritten.
    """
    n = x.shape[0]
    n_cols = x.shape[1]
    k = h.shape[0]
    k0 = (k + 1) // 2
    k1 = k // 2
    half = k // 2
    out_size = y.shape[0]
    n_half = out_size // 2
    i_lo = k0 - half
    i_hi = n - half

    # --- Boundary rows (serial, few iterations) ---
    for i in range(min(max(i_lo, 0), n_half)):
        for j in range(n_cols):
            acc_e = 0.0
            acc_o = 0.0
            for m in range(k0):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_e += h[2 * m] * x[idx, j]
            for m in range(k1):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_o += h[2 * m + 1] * x[idx, j]
            y[2 * i, j] = acc_e
            y[2 * i + 1, j] = acc_o
    for i in range(max(min(i_hi, n_half), max(i_lo, 0)), n_half):
        for j in range(n_cols):
            acc_e = 0.0
            acc_o = 0.0
            for m in range(k0):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_e += h[2 * m] * x[idx, j]
            for m in range(k1):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_o += h[2 * m + 1] * x[idx, j]
            y[2 * i, j] = acc_e
            y[2 * i + 1, j] = acc_o

    # --- Interior (parallel, unrolled for common filter sizes) ---
    int_lo = max(i_lo, 0)
    int_hi = min(i_hi, n_half)
    if k == 2:
        h0, h1 = h[0], h[1]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v = x[i, j]
                y[2 * i, j] = h0 * v
                y[2 * i + 1, j] = h1 * v
    elif k == 4:
        h0, h1, h2, h3 = h[0], h[1], h[2], h[3]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 1, j]
                v1 = x[i, j]
                y[2 * i, j] = h0 * v0 + h2 * v1
                y[2 * i + 1, j] = h1 * v0 + h3 * v1
    elif k == 6:
        h0, h1, h2, h3, h4, h5 = h[0], h[1], h[2], h[3], h[4], h[5]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 2, j]
                v1 = x[i + 1, j]
                v2 = x[i, j]
                y[2 * i, j] = h0 * v0 + h2 * v1 + h4 * v2
                y[2 * i + 1, j] = h1 * v0 + h3 * v1 + h5 * v2
    elif k == 8:
        h0, h1, h2, h3 = h[0], h[1], h[2], h[3]
        h4, h5, h6, h7 = h[4], h[5], h[6], h[7]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 3, j]
                v1 = x[i + 2, j]
                v2 = x[i + 1, j]
                v3 = x[i, j]
                y[2 * i, j] = h0 * v0 + h2 * v1 + h4 * v2 + h6 * v3
                y[2 * i + 1, j] = h1 * v0 + h3 * v1 + h5 * v2 + h7 * v3
    elif k == 10:
        h0, h1, h2, h3, h4 = h[0], h[1], h[2], h[3], h[4]
        h5, h6, h7, h8, h9 = h[5], h[6], h[7], h[8], h[9]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 4, j]
                v1 = x[i + 3, j]
                v2 = x[i + 2, j]
                v3 = x[i + 1, j]
                v4 = x[i, j]
                y[2 * i, j] = h0 * v0 + h2 * v1 + h4 * v2 + h6 * v3 + h8 * v4
                y[2 * i + 1, j] = h1 * v0 + h3 * v1 + h5 * v2 + h7 * v3 + h9 * v4
    else:
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                acc_e = 0.0
                acc_o = 0.0
                for m in range(k0):
                    acc_e += h[2 * m] * x[i - m + half - 1, j]
                for m in range(k1):
                    acc_o += h[2 * m + 1] * x[i - m + half - 1, j]
                y[2 * i, j] = acc_e
                y[2 * i + 1, j] = acc_o

    # --- Trailing odd row (if out_size is odd) ---
    if out_size % 2 == 1:
        i = n_half
        for j in range(n_cols):
            acc = 0.0
            for m in range(k0):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc += h[2 * m] * x[idx, j]
            y[2 * i, j] = acc


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def conv_upsample_axis0_polyphase_add(x, h, y):
    """
    Like conv_upsample_axis0_polyphase but accumulates (+=) into y
    instead of overwriting.
    """
    n = x.shape[0]
    n_cols = x.shape[1]
    k = h.shape[0]
    k0 = (k + 1) // 2
    k1 = k // 2
    half = k // 2
    out_size = y.shape[0]
    n_half = out_size // 2
    i_lo = k0 - half
    i_hi = n - half

    # --- Boundary rows (serial, few iterations) ---
    for i in range(min(max(i_lo, 0), n_half)):
        for j in range(n_cols):
            acc_e = 0.0
            acc_o = 0.0
            for m in range(k0):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_e += h[2 * m] * x[idx, j]
            for m in range(k1):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_o += h[2 * m + 1] * x[idx, j]
            y[2 * i, j] += acc_e
            y[2 * i + 1, j] += acc_o
    for i in range(max(min(i_hi, n_half), max(i_lo, 0)), n_half):
        for j in range(n_cols):
            acc_e = 0.0
            acc_o = 0.0
            for m in range(k0):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_e += h[2 * m] * x[idx, j]
            for m in range(k1):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc_o += h[2 * m + 1] * x[idx, j]
            y[2 * i, j] += acc_e
            y[2 * i + 1, j] += acc_o

    # --- Interior (parallel, unrolled for common filter sizes) ---
    int_lo = max(i_lo, 0)
    int_hi = min(i_hi, n_half)
    if k == 2:
        h0, h1 = h[0], h[1]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v = x[i, j]
                y[2 * i, j] += h0 * v
                y[2 * i + 1, j] += h1 * v
    elif k == 4:
        h0, h1, h2, h3 = h[0], h[1], h[2], h[3]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 1, j]
                v1 = x[i, j]
                y[2 * i, j] += h0 * v0 + h2 * v1
                y[2 * i + 1, j] += h1 * v0 + h3 * v1
    elif k == 6:
        h0, h1, h2, h3, h4, h5 = h[0], h[1], h[2], h[3], h[4], h[5]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 2, j]
                v1 = x[i + 1, j]
                v2 = x[i, j]
                y[2 * i, j] += h0 * v0 + h2 * v1 + h4 * v2
                y[2 * i + 1, j] += h1 * v0 + h3 * v1 + h5 * v2
    elif k == 8:
        h0, h1, h2, h3 = h[0], h[1], h[2], h[3]
        h4, h5, h6, h7 = h[4], h[5], h[6], h[7]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 3, j]
                v1 = x[i + 2, j]
                v2 = x[i + 1, j]
                v3 = x[i, j]
                y[2 * i, j] += h0 * v0 + h2 * v1 + h4 * v2 + h6 * v3
                y[2 * i + 1, j] += h1 * v0 + h3 * v1 + h5 * v2 + h7 * v3
    elif k == 10:
        h0, h1, h2, h3, h4 = h[0], h[1], h[2], h[3], h[4]
        h5, h6, h7, h8, h9 = h[5], h[6], h[7], h[8], h[9]
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                v0 = x[i + 4, j]
                v1 = x[i + 3, j]
                v2 = x[i + 2, j]
                v3 = x[i + 1, j]
                v4 = x[i, j]
                y[2 * i, j] += h0 * v0 + h2 * v1 + h4 * v2 + h6 * v3 + h8 * v4
                y[2 * i + 1, j] += h1 * v0 + h3 * v1 + h5 * v2 + h7 * v3 + h9 * v4
    else:
        for i in prange(int_lo, int_hi):
            for j in range(n_cols):
                acc_e = 0.0
                acc_o = 0.0
                for m in range(k0):
                    acc_e += h[2 * m] * x[i - m + half - 1, j]
                for m in range(k1):
                    acc_o += h[2 * m + 1] * x[i - m + half - 1, j]
                y[2 * i, j] += acc_e
                y[2 * i + 1, j] += acc_o

    # --- Trailing odd row (if out_size is odd) ---
    if out_size % 2 == 1:
        i = n_half
        for j in range(n_cols):
            acc = 0.0
            for m in range(k0):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc += h[2 * m] * x[idx, j]
            y[2 * i, j] += acc


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def conv_downsample_axis0_polyphase_pair(x, dec_lo, dec_hi, y0, y1):
    """
    In-place analysis for a filter pair (lo+hi) in a single prange.
    Both filters share the same gather indices — only the weights differ.

    Parameters
    ----------
    x       : (n, n_cols)     input array, C-contiguous
    dec_lo, dec_hi  : (k,)       filter pair — must have same length
    y0, y1  : (n_out, n_cols) pre-allocated outputs for dec_lo and dec_hi respectively
              n_out = (n-1)//2 + k//2
    """
    n = x.shape[0]
    n_cols = x.shape[1]
    k = dec_lo.shape[0]
    k0 = (k + 1) // 2
    k1 = k // 2
    n_out = y0.shape[0]
    i_lo = k0 - 1
    i_hi = (n - 2) // 2

    for i in prange(n_out):
        if i_lo <= i <= i_hi:
            # --- Interior: unrolled for common filter sizes ---
            if k == 2:
                l0, l1 = dec_lo[0], dec_lo[1]
                h0, h1 = dec_hi[0], dec_hi[1]
                for j in range(n_cols):
                    xe = x[2 * i + 1, j]
                    xo = x[2 * i, j]
                    y0[i, j] = l0 * xe + l1 * xo
                    y1[i, j] = h0 * xe + h1 * xo
            elif k == 4:
                l0, l1, l2, l3 = dec_lo[0], dec_lo[1], dec_lo[2], dec_lo[3]
                h0, h1, h2, h3 = dec_hi[0], dec_hi[1], dec_hi[2], dec_hi[3]
                for j in range(n_cols):
                    xe0 = x[2 * i + 1, j]
                    xe1 = x[2 * i - 1, j]
                    xo0 = x[2 * i, j]
                    xo1 = x[2 * i - 2, j]
                    y0[i, j] = l0 * xe0 + l2 * xe1 + l1 * xo0 + l3 * xo1
                    y1[i, j] = h0 * xe0 + h2 * xe1 + h1 * xo0 + h3 * xo1
            elif k == 6:
                l0, l1, l2, l3, l4, l5 = (dec_lo[0], dec_lo[1], dec_lo[2], dec_lo[3], dec_lo[4], dec_lo[5])
                h0, h1, h2, h3, h4, h5 = (dec_hi[0], dec_hi[1], dec_hi[2], dec_hi[3], dec_hi[4], dec_hi[5])
                for j in range(n_cols):
                    xe0 = x[2 * i + 1, j]
                    xe1 = x[2 * i - 1, j]
                    xe2 = x[2 * i - 3, j]
                    xo0 = x[2 * i, j]
                    xo1 = x[2 * i - 2, j]
                    xo2 = x[2 * i - 4, j]
                    y0[i, j] = l0 * xe0 + l2 * xe1 + l4 * xe2 + l1 * xo0 + l3 * xo1 + l5 * xo2
                    y1[i, j] = h0 * xe0 + h2 * xe1 + h4 * xe2 + h1 * xo0 + h3 * xo1 + h5 * xo2
            elif k == 8:
                l0, l1, l2, l3 = dec_lo[0], dec_lo[1], dec_lo[2], dec_lo[3]
                l4, l5, l6, l7 = dec_lo[4], dec_lo[5], dec_lo[6], dec_lo[7]
                h0, h1, h2, h3 = dec_hi[0], dec_hi[1], dec_hi[2], dec_hi[3]
                h4, h5, h6, h7 = dec_hi[4], dec_hi[5], dec_hi[6], dec_hi[7]
                for j in range(n_cols):
                    xe0 = x[2 * i + 1, j]
                    xe1 = x[2 * i - 1, j]
                    xe2 = x[2 * i - 3, j]
                    xe3 = x[2 * i - 5, j]
                    xo0 = x[2 * i, j]
                    xo1 = x[2 * i - 2, j]
                    xo2 = x[2 * i - 4, j]
                    xo3 = x[2 * i - 6, j]
                    y0[i, j] = l0 * xe0 + l2 * xe1 + l4 * xe2 + l6 * xe3 + l1 * xo0 + l3 * xo1 + l5 * xo2 + l7 * xo3
                    y1[i, j] = h0 * xe0 + h2 * xe1 + h4 * xe2 + h6 * xe3 + h1 * xo0 + h3 * xo1 + h5 * xo2 + h7 * xo3
            elif k == 10:
                l0, l1, l2, l3, l4 = (dec_lo[0], dec_lo[1], dec_lo[2], dec_lo[3], dec_lo[4])
                l5, l6, l7, l8, l9 = (dec_lo[5], dec_lo[6], dec_lo[7], dec_lo[8], dec_lo[9])
                h0, h1, h2, h3, h4 = (dec_hi[0], dec_hi[1], dec_hi[2], dec_hi[3], dec_hi[4])
                h5, h6, h7, h8, h9 = (dec_hi[5], dec_hi[6], dec_hi[7], dec_hi[8], dec_hi[9])
                for j in range(n_cols):
                    xe0 = x[2 * i + 1, j]
                    xe1 = x[2 * i - 1, j]
                    xe2 = x[2 * i - 3, j]
                    xe3 = x[2 * i - 5, j]
                    xe4 = x[2 * i - 7, j]
                    xo0 = x[2 * i, j]
                    xo1 = x[2 * i - 2, j]
                    xo2 = x[2 * i - 4, j]
                    xo3 = x[2 * i - 6, j]
                    xo4 = x[2 * i - 8, j]
                    y0[i, j] = (
                        l0 * xe0
                        + l2 * xe1
                        + l4 * xe2
                        + l6 * xe3
                        + l8 * xe4
                        + l1 * xo0
                        + l3 * xo1
                        + l5 * xo2
                        + l7 * xo3
                        + l9 * xo4
                    )
                    y1[i, j] = (
                        h0 * xe0
                        + h2 * xe1
                        + h4 * xe2
                        + h6 * xe3
                        + h8 * xe4
                        + h1 * xo0
                        + h3 * xo1
                        + h5 * xo2
                        + h7 * xo3
                        + h9 * xo4
                    )
            else:
                for j in range(n_cols):
                    acc0 = 0.0
                    acc1 = 0.0
                    for m in range(k0):
                        xval = x[2 * i + 1 - 2 * m, j]
                        acc0 += dec_lo[2 * m] * xval
                        acc1 += dec_hi[2 * m] * xval
                    for m in range(k1):
                        xval = x[2 * i - 2 * m, j]
                        acc0 += dec_lo[2 * m + 1] * xval
                        acc1 += dec_hi[2 * m + 1] * xval
                    y0[i, j] = acc0
                    y1[i, j] = acc1
        else:
            # --- Boundary: generic with bounds checks ---
            for j in range(n_cols):
                acc0 = 0.0
                acc1 = 0.0
                for m in range(k0):
                    r = 2 * i + 1 - 2 * m
                    if 0 <= r < n:
                        xval = x[r, j]
                        acc0 += dec_lo[2 * m] * xval
                        acc1 += dec_hi[2 * m] * xval
                for m in range(k1):
                    r = 2 * i - 2 * m
                    if 0 <= r < n:
                        xval = x[r, j]
                        acc0 += dec_lo[2 * m + 1] * xval
                        acc1 += dec_hi[2 * m + 1] * xval
                y0[i, j] = acc0
                y1[i, j] = acc1


@nb.njit(nogil=True, cache=True, inline="always", fastmath=True)
def conv_upsample_axis0_polyphase_pair(x0, x1, rec_lo, rec_hi, y):
    """
    In-place synthesis for a filter pair in a single prange.
    y = upsample(x0, rec_lo) + upsample(x1, rec_hi)
    Accumulates both contributions into y in one pass.

    Parameters
    ----------
    x0, x1  : (n, n_cols)      lo and hi subband arrays
    rec_lo, rec_hi  : (k,)        rec_lo and rec_hi — must have same length
    y       : (out_size, n_cols)    pre-allocated output, will be overwritten
    """
    n = x0.shape[0]
    n_cols = x0.shape[1]
    k = rec_lo.shape[0]
    k0 = (k + 1) // 2
    k1 = k // 2
    half = k // 2
    out_size = y.shape[0]
    n_half = out_size // 2
    i_lo = k0 - half
    i_hi = n - half

    for i in prange(n_half):
        if i_lo <= i < i_hi:
            # --- Interior: unrolled for common filter sizes ---
            if k == 2:
                rl0, rl1 = rec_lo[0], rec_lo[1]
                rh0, rh1 = rec_hi[0], rec_hi[1]
                for j in range(n_cols):
                    a = x0[i, j]
                    b = x1[i, j]
                    y[2 * i, j] = rl0 * a + rh0 * b
                    y[2 * i + 1, j] = rl1 * a + rh1 * b
            elif k == 4:
                rl0, rl1, rl2, rl3 = rec_lo[0], rec_lo[1], rec_lo[2], rec_lo[3]
                rh0, rh1, rh2, rh3 = rec_hi[0], rec_hi[1], rec_hi[2], rec_hi[3]
                for j in range(n_cols):
                    a0 = x0[i + 1, j]
                    a1 = x0[i, j]
                    b0 = x1[i + 1, j]
                    b1 = x1[i, j]
                    y[2 * i, j] = rl0 * a0 + rl2 * a1 + rh0 * b0 + rh2 * b1
                    y[2 * i + 1, j] = rl1 * a0 + rl3 * a1 + rh1 * b0 + rh3 * b1
            elif k == 6:
                rl0, rl1, rl2, rl3, rl4, rl5 = (rec_lo[0], rec_lo[1], rec_lo[2], rec_lo[3], rec_lo[4], rec_lo[5])
                rh0, rh1, rh2, rh3, rh4, rh5 = (rec_hi[0], rec_hi[1], rec_hi[2], rec_hi[3], rec_hi[4], rec_hi[5])
                for j in range(n_cols):
                    a0 = x0[i + 2, j]
                    a1 = x0[i + 1, j]
                    a2 = x0[i, j]
                    b0 = x1[i + 2, j]
                    b1 = x1[i + 1, j]
                    b2 = x1[i, j]
                    y[2 * i, j] = rl0 * a0 + rl2 * a1 + rl4 * a2 + rh0 * b0 + rh2 * b1 + rh4 * b2
                    y[2 * i + 1, j] = rl1 * a0 + rl3 * a1 + rl5 * a2 + rh1 * b0 + rh3 * b1 + rh5 * b2
            elif k == 8:
                rl0, rl1, rl2, rl3 = rec_lo[0], rec_lo[1], rec_lo[2], rec_lo[3]
                rl4, rl5, rl6, rl7 = rec_lo[4], rec_lo[5], rec_lo[6], rec_lo[7]
                rh0, rh1, rh2, rh3 = rec_hi[0], rec_hi[1], rec_hi[2], rec_hi[3]
                rh4, rh5, rh6, rh7 = rec_hi[4], rec_hi[5], rec_hi[6], rec_hi[7]
                for j in range(n_cols):
                    a0 = x0[i + 3, j]
                    a1 = x0[i + 2, j]
                    a2 = x0[i + 1, j]
                    a3 = x0[i, j]
                    b0 = x1[i + 3, j]
                    b1 = x1[i + 2, j]
                    b2 = x1[i + 1, j]
                    b3 = x1[i, j]
                    y[2 * i, j] = rl0 * a0 + rl2 * a1 + rl4 * a2 + rl6 * a3 + rh0 * b0 + rh2 * b1 + rh4 * b2 + rh6 * b3
                    y[2 * i + 1, j] = (
                        rl1 * a0 + rl3 * a1 + rl5 * a2 + rl7 * a3 + rh1 * b0 + rh3 * b1 + rh5 * b2 + rh7 * b3
                    )
            elif k == 10:
                rl0, rl1, rl2, rl3, rl4 = (rec_lo[0], rec_lo[1], rec_lo[2], rec_lo[3], rec_lo[4])
                rl5, rl6, rl7, rl8, rl9 = (rec_lo[5], rec_lo[6], rec_lo[7], rec_lo[8], rec_lo[9])
                rh0, rh1, rh2, rh3, rh4 = (rec_hi[0], rec_hi[1], rec_hi[2], rec_hi[3], rec_hi[4])
                rh5, rh6, rh7, rh8, rh9 = (rec_hi[5], rec_hi[6], rec_hi[7], rec_hi[8], rec_hi[9])
                for j in range(n_cols):
                    a0 = x0[i + 4, j]
                    a1 = x0[i + 3, j]
                    a2 = x0[i + 2, j]
                    a3 = x0[i + 1, j]
                    a4 = x0[i, j]
                    b0 = x1[i + 4, j]
                    b1 = x1[i + 3, j]
                    b2 = x1[i + 2, j]
                    b3 = x1[i + 1, j]
                    b4 = x1[i, j]
                    y[2 * i, j] = (
                        rl0 * a0
                        + rl2 * a1
                        + rl4 * a2
                        + rl6 * a3
                        + rl8 * a4
                        + rh0 * b0
                        + rh2 * b1
                        + rh4 * b2
                        + rh6 * b3
                        + rh8 * b4
                    )
                    y[2 * i + 1, j] = (
                        rl1 * a0
                        + rl3 * a1
                        + rl5 * a2
                        + rl7 * a3
                        + rl9 * a4
                        + rh1 * b0
                        + rh3 * b1
                        + rh5 * b2
                        + rh7 * b3
                        + rh9 * b4
                    )
            else:
                for j in range(n_cols):
                    acc_e = 0.0
                    acc_o = 0.0
                    for m in range(k0):
                        x0val = x0[i - m + half - 1, j]
                        x1val = x1[i - m + half - 1, j]
                        acc_e += rec_lo[2 * m] * x0val + rec_hi[2 * m] * x1val
                    for m in range(k1):
                        x0val = x0[i - m + half - 1, j]
                        x1val = x1[i - m + half - 1, j]
                        acc_o += rec_lo[2 * m + 1] * x0val + rec_hi[2 * m + 1] * x1val
                    y[2 * i, j] = acc_e
                    y[2 * i + 1, j] = acc_o
        else:
            # --- Boundary: generic with bounds checks ---
            for j in range(n_cols):
                acc_e = 0.0
                acc_o = 0.0
                for m in range(k0):
                    idx = i - m + half - 1
                    if 0 <= idx < n:
                        x0val = x0[idx, j]
                        x1val = x1[idx, j]
                        acc_e += rec_lo[2 * m] * x0val + rec_hi[2 * m] * x1val
                for m in range(k1):
                    idx = i - m + half - 1
                    if 0 <= idx < n:
                        x0val = x0[idx, j]
                        x1val = x1[idx, j]
                        acc_o += rec_lo[2 * m + 1] * x0val + rec_hi[2 * m + 1] * x1val
                y[2 * i, j] = acc_e
                y[2 * i + 1, j] = acc_o

    # --- Trailing odd row (if out_size is odd) ---
    if out_size % 2 == 1:
        i = n_half
        for j in range(n_cols):
            acc = 0.0
            for m in range(k0):
                idx = i - m + half - 1
                if 0 <= idx < n:
                    acc += rec_lo[2 * m] * x0[idx, j] + rec_hi[2 * m] * x1[idx, j]
            y[2 * i, j] = acc
