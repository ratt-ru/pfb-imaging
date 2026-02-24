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


@numba.njit(nogil=True, cache=True, inline="always", fastmath=False, error_model="numpy")
def downsampling_convolution(input, output, filter, step):
    i = step - 1
    o = 0
    input_size = input.shape[0]
    filter_size = filter.shape[0]

    # left boundary overhang
    while i < filter_size and i < input_size:
        fsum = input.dtype.type(0)  # init to zero with correct type
        j = 0

        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1

        output[o] = fsum

        i += step
        o += 1

    # center (if input equal or wider than filter: input_size >= filter_size)
    while i < input_size:
        fsum = input.dtype.type(0)
        j = 0

        while j < filter_size:
            fsum += input[i - j] * filter[j]
            j += 1

        output[o] = fsum

        i += step
        o += 1

    # center (if filter is wider than input: filter_size > input_size)
    while i < filter_size:
        fsum = input.dtype.type(0)
        j = 0
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
        j = 0
        j = i - input_size + 1

        while j < filter_size:
            fsum += filter[j] * input[i - j]
            j += 1

        output[o] = fsum

        i += step
        o += 1


@numba.njit(nogil=True, cache=True, inline="always", fastmath=False, error_model="numpy")
def upsampling_convolution_valid_sf(input, filter, output):
    input_size = input.shape[0]
    filter_size = filter.shape[0]
    output_size = output.shape[0]

    # if filter_size % 2:
    #     raise ValueError("even filter required for upsampling")

    # if input_size < (filter_size // 2):
    #     raise ValueError("input.shape[0] < (filter.shape[0] // 2)")

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


@numba.njit(nogil=True, cache=True, parallel=True)
def dwt2d_level(image, coeffs, cbuff, cbufft, dec_lo, dec_hi):
    """
    Map image to coeffs for a single level

    image   - (nx, ny) signal
    coeffs  - (nay, nax) output coeffs
    cbuff   - (nax, nay) coeff buffer
    cbufft  - (nay, nax) coeff buffer
    dec_lo  - (filter_size) length low pass filter for decomposition
    dec_hi  - (filter_size) length high pass filter for decomposition

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
    # copy and transpose to ensure correct order and
    # that input is not overwritten at the next level
    return coeffs[0:midy, 0:midx].T.copy()


@numba.njit(nogil=True, cache=True, parallel=True)
def dwt2d(image, coeffs, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel):
    """
    Multi-level 2D image to coeffs transform

    image   - (nx, ny) signal
    coeffs  - (nay, nax) output coeffs
    cbuff   - (nax, nay) coeff buffer
    cbufft  - (nay, nax) coeff buffer
    ix,iy   - (nlevel) dicts containing start and stop indices for packing coeffs
    sx,sy   - (nlevel) tuples containing coeff sizes at each level
    spx,spy - (nlevel) tuples containing signal sizes at each level
    dec_lo  - (filter_size) length low pass filter for decomposition
    dec_hi  - (filter_size) length high pass filter for decomposition
    nlevel  - the number of decomposition levels

    The dimension of the output is given by

    nax = (nx + filter_size - 1)//2
    nay = (ny + filter_size - 1)//2

    The output is transposed with respect to the input for
    performance reasons.

    """
    nx, ny = image.shape
    approx = image
    for i in range(nlevel):
        _, highx = ix[i]
        lowx = highx - 2 * sx[i]
        _, highy = iy[i]
        lowy = highy - 2 * sy[i]
        # detail coeffs go directly into output
        # returned approx coeffs are safe to overwrite since copied
        approx = dwt2d_level(
            approx,
            coeffs[lowy:highy, lowx:highx],
            cbuff[lowx:highx, lowy:highy],
            cbufft[lowy:highy, lowx:highx],
            dec_lo,
            dec_hi,
        )


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


@numba.njit(nogil=True, cache=True, parallel=True)
def idwt2d(coeffs, image, alpha, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel):
    """
    Multi-level 2D coeffs to image transform

    coeffs  - (nay, nax) output coeffs
    image   - (nx, ny) signal
    alpha   - (nay, nax) coeff buffer
    cbuff   - (nax, nay) coeff buffer
    cbufft  - (nay, nax) coeff buffer
    ix,iy   - (nlevel) dicts containing start and stop indices for packing coeffs
    sx,sy   - (nlevel) tuples containing coeff sizes at each level
    spx,spy - (nlevel) tuples containing signal sizes at each level
    dec_lo  - (filter_size) length low pass filter for decomposition
    dec_hi  - (filter_size) length high pass filter for decomposition
    nlevel  - the number of decomposition levels

    The dimension of the output is given by

    nx = 2*nax - filter_size + 2
    ny = 2*nay - filter_size + 2

    The output is transposed with respect to the input for
    performance reasons.

    """
    nx, ny = image.shape
    alpha[...] = coeffs  # to avoid overwriting coeffs
    for i in range(nlevel - 1, -1, -1):
        # coeff size at current level
        nax = sx[i]
        nay = sy[i]

        # slice indices for alpha
        _, highx = ix[i]
        lowx = highx - 2 * nax
        _, highy = iy[i]
        lowy = highy - 2 * nay

        # signal size at current level
        nxo = spx[i]
        nyo = spy[i]

        # previous output is the top left corner of the next input
        # note the difference in indexing (nax,nay instead of nxo,nyo)
        # that stems from the redundant coefficient
        if i < nlevel - 1:
            copyt(image[0:nax, 0:nay], alpha[lowy : lowy + nay, lowx : lowx + nax])

        # need a buffer the size of the coeffs at next level
        idwt2d_level(
            alpha[lowy:highy, lowx:highx],
            image[0:nxo, 0:nyo],
            cbuff[0 : 2 * nax, 0 : 2 * nay],
            cbufft[0 : 2 * nay, 0 : 2 * nax],
            rec_lo,
            rec_hi,
        )
