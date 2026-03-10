import numba


# from the answer given here
# https://stackoverflow.com/questions/67431966/how-to-avoid-huge-overhead-of-single-threaded-numpys-transpose
@numba.njit(nogil=True, cache=True, parallel=True, inline="always")
def copyt(mat, out):
    block_size, tile_size = 256, 32  # To be tuned
    n, m = mat.shape
    for tmp in numba.prange((m + block_size - 1) // block_size):
        i = tmp * block_size
        for j in range(0, n, block_size):
            timin, timax = i, min(i + block_size, m)
            tjmin, tjmax = j, min(j + block_size, n)
            for ti in range(timin, timax, tile_size):
                for tj in range(tjmin, tjmax, tile_size):
                    out[ti : ti + tile_size, tj : tj + tile_size] = mat[tj : tj + tile_size, ti : ti + tile_size].T


@numba.njit(nogil=True, cache=True)
def coeff_size(nsignal, nfilter):
    return (nsignal + nfilter - 1) // 2


@numba.njit(nogil=True, cache=True)
def signal_size(ncoeff, nfilter):
    return 2 * ncoeff - nfilter + 2


@numba.njit(nogil=True, cache=True, inline="always", fastmath=True)
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


@numba.njit(nogil=True, cache=True, inline="always", fastmath=True)
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


@numba.njit(nogil=True, cache=True, parallel=True)
def dwt2d_level(image, coeffs, cbuff, cbufft, dec_lo, dec_hi, approx):
    """
    Map image to coeffs for a single level

    image   - (nx, ny) signal
    coeffs  - (nay, nax) output coeffs
    cbuff   - (nax, nay) coeff buffer
    cbufft  - (nay, nax) coeff buffer
    dec_lo  - (filter_size) length low pass filter for decomposition
    dec_hi  - (filter_size) length high pass filter for decomposition
    approx  - pre-allocated buffer for approx coeffs (avoids allocation)

    The dimension of the output is given by

    nax = (nx + filter_size - 1)//2
    nay = (ny + filter_size - 1)//2

    The output is transposed with respect to the input for
    performance reasons.

    This function is not meant to be called directly.
    """
    nx, ny = image.shape
    nay, nax = coeffs.shape

    midy = nay // 2
    for i in numba.prange(nx):
        # approx
        downsampling_convolution(image[i, :], cbuff[i, 0:midy], dec_lo, 2)
        # detail
        downsampling_convolution(image[i, :], cbuff[i, midy:], dec_hi, 2)

    # prefer over repeatedly convolving the non-contiguous array
    copyt(cbuff, cbufft)

    midx = nax // 2
    for i in numba.prange(nay):
        # approx
        downsampling_convolution(cbufft[i, 0:nx], coeffs[i, 0:midx], dec_lo, 2)
        # detail
        downsampling_convolution(cbufft[i, 0:nx], coeffs[i, midx:], dec_hi, 2)

    # approx coeffs are in the top left corner of out
    # transpose into pre-allocated buffer to avoid allocation
    copyt(coeffs[0:midy, 0:midx], approx[0:midx, 0:midy])


@numba.njit(nogil=True, cache=True)
def dwt2d(image, coeffs, cbuff_flat, cbufft_flat, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx):
    """
    Multi-level 2D image to coeffs transform

    image      - (nx, ny) signal
    coeffs     - (nay, nax) output coeffs
    cbuff_flat - 1D scratch buffer, reshaped per level for contiguous access
    cbufft_flat- 1D scratch buffer, reshaped per level for contiguous access
    ix,iy      - (nlevel, 2) int64 arrays with start/stop indices for packing coeffs
    sx,sy      - (nlevel,) int64 arrays with coeff sizes at each level
    dec_lo     - (filter_size) length low pass filter for decomposition
    dec_hi     - (filter_size) length high pass filter for decomposition
    nlevel     - the number of decomposition levels
    approx     - (nxmax, nymax) pre-allocated buffer for approx coeffs

    """
    approx_in = image
    for i in range(nlevel):
        nax = 2 * sx[i]
        nay = 2 * sy[i]
        highy = iy[i, 1]
        lowy = highy - nay
        highx = ix[i, 1]
        lowx = highx - nax
        # reshape flat buffers to contiguous 2D for this level
        cbuff = cbuff_flat[0 : nax * nay].reshape(nax, nay)
        cbufft = cbufft_flat[0 : nay * nax].reshape(nay, nax)
        dwt2d_level(
            approx_in,
            coeffs[lowy:highy, lowx:highx],
            cbuff,
            cbufft,
            dec_lo,
            dec_hi,
            approx,
        )
        approx_in = approx[0 : sx[i], 0 : sy[i]]


@numba.njit(nogil=True, cache=True, parallel=True)
def idwt2d_level(coeffs, image, cbuff, cbufft, rec_lo, rec_hi):
    """
    Map coeffs to image for a single level

    coeffs  - (nay, nax) coeffs
    image   - (nx, ny) output signal
    cbuff   - (nax, nay) coeff buffer
    cbufft  - (nay, nax) coeff buffer
    rec_lo  - (filter_size) length low pass filter for reconstruction
    rec_hi  - (filter_size) length high pass filter for reconstruction

    The dimension of the output signal is give by

    nx = 2*nax - filter_size + 2
    ny = 2*nay - filter_size + 2

    The output is transposed with respect to the input for
    performance reasons.

    This function is not meant to be called directly.

    """
    nay, nax = coeffs.shape
    nx, ny = image.shape

    # zero since accumulated into
    cbufft[...] = 0.0
    image[...] = 0.0

    midx = nax // 2
    for i in numba.prange(nay):
        upsampling_convolution_valid_sf(coeffs[i, 0:midx], rec_lo, cbufft[i, :])
        upsampling_convolution_valid_sf(coeffs[i, midx:], rec_hi, cbufft[i, :])

    copyt(cbufft, cbuff)

    midy = nay // 2
    for i in numba.prange(nx):
        upsampling_convolution_valid_sf(cbuff[i, 0:midy], rec_lo, image[i, :])
        upsampling_convolution_valid_sf(cbuff[i, midy:], rec_hi, image[i, :])


@numba.njit(nogil=True, cache=True)
def idwt2d(coeffs, image, alpha, cbuff_flat, cbufft_flat, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel):
    """
    Multi-level 2D coeffs to image transform

    coeffs      - (nay, nax) output coeffs
    image       - (nx, ny) signal
    alpha       - (nay, nax) coeff buffer
    cbuff_flat  - 1D scratch buffer, reshaped per level for contiguous access
    cbufft_flat - 1D scratch buffer, reshaped per level for contiguous access
    ix,iy       - (nlevel, 2) int64 arrays with start/stop indices for packing coeffs
    sx,sy       - (nlevel,) int64 arrays with coeff sizes at each level
    spx,spy     - (nlevel,) int64 arrays with signal sizes at each level
    rec_lo      - (filter_size) length low pass filter for reconstruction
    rec_hi      - (filter_size) length high pass filter for reconstruction
    nlevel      - the number of decomposition levels

    """
    nx, ny = image.shape
    alpha[...] = coeffs  # to avoid overwriting coeffs
    for i in range(nlevel - 1, -1, -1):
        nax = sx[i]
        nay = sy[i]

        highx = ix[i, 1]
        lowx = highx - 2 * nax
        highy = iy[i, 1]
        lowy = highy - 2 * nay

        nxo = spx[i]
        nyo = spy[i]

        if i < nlevel - 1:
            copyt(image[0:nax, 0:nay], alpha[lowy : lowy + nay, lowx : lowx + nax])

        # reshape flat buffers to contiguous 2D for this level
        cbuff = cbuff_flat[0 : 2 * nax * 2 * nay].reshape(2 * nax, 2 * nay)
        cbufft = cbufft_flat[0 : 2 * nay * 2 * nax].reshape(2 * nay, 2 * nax)
        idwt2d_level(
            alpha[lowy:highy, lowx:highx],
            image[0:nxo, 0:nyo],
            cbuff,
            cbufft,
            rec_lo,
            rec_hi,
        )


# ── Serial (non-parallel) versions for use inside outer prange ────────
# These are identical to the parallel versions above but use plain range
# instead of prange, avoiding nested parallel region overhead.


@numba.njit(nogil=True, cache=True, inline="always")
def copyt_seq(mat, out):
    """Cache-friendly serial transpose (no prange)."""
    block_size, tile_size = 256, 32
    n, m = mat.shape
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            timin, timax = i, min(i + block_size, m)
            tjmin, tjmax = j, min(j + block_size, n)
            for ti in range(timin, timax, tile_size):
                for tj in range(tjmin, tjmax, tile_size):
                    out[ti : ti + tile_size, tj : tj + tile_size] = mat[tj : tj + tile_size, ti : ti + tile_size].T


@numba.njit(nogil=True, cache=True)
def dwt2d_level_seq(image, coeffs, cbuff, cbufft, dec_lo, dec_hi, approx):
    """Single-level forward DWT without prange (serial)."""
    nx, ny = image.shape
    nay, nax = coeffs.shape

    midy = nay // 2
    for i in range(nx):
        downsampling_convolution(image[i, :], cbuff[i, 0:midy], dec_lo, 2)
        downsampling_convolution(image[i, :], cbuff[i, midy:], dec_hi, 2)

    copyt_seq(cbuff, cbufft)

    midx = nax // 2
    for i in range(nay):
        downsampling_convolution(cbufft[i, 0:nx], coeffs[i, 0:midx], dec_lo, 2)
        downsampling_convolution(cbufft[i, 0:nx], coeffs[i, midx:], dec_hi, 2)

    copyt_seq(coeffs[0:midy, 0:midx], approx[0:midx, 0:midy])


@numba.njit(nogil=True, cache=True)
def dwt2d_seq(image, coeffs, cbuff_flat, cbufft_flat, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx):
    """Multi-level forward DWT without prange (serial)."""
    approx_in = image
    for i in range(nlevel):
        nax = 2 * sx[i]
        nay = 2 * sy[i]
        highy = iy[i, 1]
        lowy = highy - nay
        highx = ix[i, 1]
        lowx = highx - nax
        # reshape flat buffers to contiguous 2D for this level
        cbuff = cbuff_flat[0 : nax * nay].reshape(nax, nay)
        cbufft = cbufft_flat[0 : nay * nax].reshape(nay, nax)
        dwt2d_level_seq(
            approx_in,
            coeffs[lowy:highy, lowx:highx],
            cbuff,
            cbufft,
            dec_lo,
            dec_hi,
            approx,
        )
        approx_in = approx[0 : sx[i], 0 : sy[i]]


@numba.njit(nogil=True, cache=True)
def idwt2d_level_seq(coeffs, image, cbuff, cbufft, rec_lo, rec_hi):
    """Single-level inverse DWT without prange (serial)."""
    nay, nax = coeffs.shape
    nx, ny = image.shape

    cbufft[...] = 0.0
    image[...] = 0.0

    midx = nax // 2
    for i in range(nay):
        upsampling_convolution_valid_sf(coeffs[i, 0:midx], rec_lo, cbufft[i, :])
        upsampling_convolution_valid_sf(coeffs[i, midx:], rec_hi, cbufft[i, :])

    copyt_seq(cbufft, cbuff)

    midy = nay // 2
    for i in range(nx):
        upsampling_convolution_valid_sf(cbuff[i, 0:midy], rec_lo, image[i, :])
        upsampling_convolution_valid_sf(cbuff[i, midy:], rec_hi, image[i, :])


@numba.njit(nogil=True, cache=True)
def idwt2d_seq(coeffs, image, alpha, cbuff_flat, cbufft_flat, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel):
    """Multi-level inverse DWT without prange (serial)."""
    nx, ny = image.shape
    alpha[...] = coeffs
    for i in range(nlevel - 1, -1, -1):
        nax = sx[i]
        nay = sy[i]
        highx = ix[i, 1]
        lowx = highx - 2 * nax
        highy = iy[i, 1]
        lowy = highy - 2 * nay
        nxo = spx[i]
        nyo = spy[i]
        if i < nlevel - 1:
            copyt_seq(image[0:nax, 0:nay], alpha[lowy : lowy + nay, lowx : lowx + nax])
        # reshape flat buffers to contiguous 2D for this level
        cbuff = cbuff_flat[0 : 2 * nax * 2 * nay].reshape(2 * nax, 2 * nay)
        cbufft = cbufft_flat[0 : 2 * nay * 2 * nax].reshape(2 * nay, 2 * nax)
        idwt2d_level_seq(
            alpha[lowy:highy, lowx:highx],
            image[0:nxo, 0:nyo],
            cbuff,
            cbufft,
            rec_lo,
            rec_hi,
        )
