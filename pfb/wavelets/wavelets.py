import numba

@numba.njit(nogil=True, cache=True)
def coeff_size(nsignal, nfilter):
    return (nsignal + nfilter - 1)//2


@numba.njit(nogil=True, cache=True)
def signal_size(ncoeff, nfilter):
    return 2*ncoeff - nfilter + 2


@numba.njit(nogil=True, cache=True, inline='always')
def downsampling_convolution(input, output, filter, step):

    i = step - 1
    o = 0
    N = input.shape[0]
    F = filter.shape[0]


    # left boundary overhang
    while i < F and i < N:
        fsum = input.dtype.type(0)  # init to zero with correct type
        j = 0

        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1

        output[o] = fsum

        i += step
        o += 1

    # center (if input equal or wider than filter: N >= F)
    while i < N:
        fsum = input.dtype.type(0)
        j = 0

        while j < F:
            fsum += input[i - j] * filter[j]
            j += 1

        output[o] = fsum

        i += step
        o += 1

    # center (if filter is wider than input: F > N)
    while i < F:
        fsum = input.dtype.type(0)
        j = 0
        j = i - N + 1

        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1

        output[o] = fsum

        i += step
        o += 1

    # right boundary overhang
    while i < N + F - 1:
        fsum = input.dtype.type(0)
        j = 0
        j = i - N + 1

        while j < F:
            fsum += filter[j] * input[i - j]
            j += 1

        output[o] = fsum

        i += step
        o += 1


@numba.njit(nogil=True, cache=True, inline='always')
def upsampling_convolution_valid_sf(input, filter, output):

    N = input.shape[0]
    F = filter.shape[0]
    O = output.shape[0]

    o = 0
    Fo2 = F//2
    i = Fo2 - 1

    while i < N and o < O:
        sum_even = input.dtype.type(0)
        sum_odd = input.dtype.type(0)

        for j in range(Fo2):
            j2 = 2*j
            inimj = input[i-j]
            sum_odd += filter[j2 + 1] * inimj
            sum_even += filter[j2] * inimj

        output[o] += sum_even
        output[o + 1] += sum_odd

        i += 1
        o += 2


@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def dwt2d_level(image, coeffs, cbuff, cbuffT, dec_lo, dec_hi):
    """
    Map image to coeffs for a single level

    image   - (nx, ny) signal
    coeffs  - (nay, nax) output coeffs
    cbuff   - (nax, nay) coeff buffer
    cbuffT  - (nay, nax) coeff buffer
    dec_lo  - (F) length low pass filter for decomposition
    dec_hi  - (F) length high pass filter for decomposition

    The dimension of the output is given by

    nax = (nx + F - 1)//2
    nay = (ny + F - 1)//2

    The output is transposed with respect to the input for
    performance reasons.

    This function is not meant to be called directly.
    """
    nx, ny = image.shape
    # # Ensure even signal
    # if nx%2 or ny%2:
    #     raise ValueError('Signal must be even')
    nax, nay = coeffs.shape

    midy = nay//2
    for i in numba.prange(nx):
        # approx
        downsampling_convolution(image[i, :], cbuff[i, 0:midy], dec_lo, 2)
        # detail
        downsampling_convolution(image[i, :], cbuff[i, midy:], dec_hi, 2)

    # prefer over repeatedly convolving the non-contiguous array
    for j in numba.prange(nax):
        cbuffT[:, j] = cbuff[j]

    midx = nax//2
    for i in numba.prange(nay):
        # approx
        downsampling_convolution(cbuffT[i, 0:nx], coeffs[i, 0:midx], dec_lo, 2)
        # detail
        downsampling_convolution(cbuffT[i, 0:nx], coeffs[i, midx:], dec_hi, 2)

    # approx coeffs are in the top left corner of out
    # copy and transpose to ensure correct order and
    # that input is not overwritten at the next level
    return coeffs[0:midy, 0:midx].T.copy()


@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def dwt2d(image, coeffs, cbuff, cbuffT, ix, iy,
          sx, sy, dec_lo, dec_hi, nlevel):
    '''
    Multi-level 2D image to coeffs transform

    image   - (nx, ny) signal
    coeffs  - (nay, nax) output coeffs
    cbuff   - (nax, nay) coeff buffer
    cbuffT  - (nay, nax) coeff buffer
    ix,iy   - (nlevel) dicts containing start and stop indices for packing coeffs
    sx,sy   - (nlevel) tuples containing coeff sizes at each level
    spx,spy - (nlevel) tuples containing signal sizes at each level
    dec_lo  - (F) length low pass filter for decomposition
    dec_hi  - (F) length high pass filter for decomposition
    nlevel  - the number of decomposition levels

    The dimension of the output is given by

    nax = (nx + F - 1)//2
    nay = (ny + F - 1)//2

    The output is transposed with respect to the input for
    performance reasons.

    '''
    nx, ny = image.shape
    approx = image
    for i in range(nlevel):
        _, highx = ix[i]
        lowx = highx - 2*sx[i]
        _, highy = iy[i]
        lowy = highy - 2*sy[i]
        # detail coeffs go directly into output
        # returned approx coeffs are safe to overwrite since copied
        approx = dwt2d_level(approx,
                             coeffs[lowy:highy, lowx:highx],
                             cbuff[lowx:highx, lowy:highy],
                             cbuffT[lowy:highy, lowx:highx],
                             dec_lo, dec_hi)


@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def idwt2d_level(coeffs, image, cbuff, cbuffT, rec_lo, rec_hi):
    '''
    Map coeffs to image for a single level

    coeffs  - (nay, nax) coeffs
    image   - (nx, ny) output signal
    cbuff   - (nax, nay) coeff buffer
    cbuffT  - (nay, nax) coeff buffer
    rec_lo  - (F) length low pass filter for reconstruction
    rec_hi  - (F) length high pass filter for reconstruction

    The dimension of the output signal is give by

    nx = 2*nax - F + 2
    ny = 2*nay - F + 2

    The output is transposed with respect to the input for
    performance reasons.

    This function is not meant to be called directly.

    '''
    nay, nax = coeffs.shape
    nx, ny = image.shape

    # zero since accumulated into
    cbuff[...] = 0.0
    cbuffT[...] = 0.0
    image[...] = 0.0

    midx = nax//2
    for i in numba.prange(nay):
        upsampling_convolution_valid_sf(coeffs[i, 0:midx], rec_lo, cbuffT[i, :])
        upsampling_convolution_valid_sf(coeffs[i, midx:], rec_hi, cbuffT[i, :])


    # import ipdb; ipdb.set_trace()
    for j in numba.prange(nay):
        cbuff[:, j] = cbuffT[j]

    midy = nay//2
    for i in numba.prange(nx):
        upsampling_convolution_valid_sf(cbuff[i, 0:midy], rec_lo, image[i, :])
        upsampling_convolution_valid_sf(cbuff[i, midy:], rec_hi, image[i, :])


@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def idwt2d(coeffs, image, alpha, cbuff, cbuffT, ix, iy,
           sx, sy, spx, spy, rec_lo, rec_hi, nlevel):
    '''
    Multi-level 2D coeffs to image transform

    coeffs  - (nay, nax) output coeffs
    image   - (nx, ny) signal
    alpha   - (nay, nax) coeff buffer
    cbuff   - (nax, nay) coeff buffer
    cbuffT  - (nay, nax) coeff buffer
    ix,iy   - (nlevel) dicts containing start and stop indices for packing coeffs
    sx,sy   - (nlevel) tuples containing coeff sizes at each level
    spx,spy - (nlevel) tuples containing signal sizes at each level
    dec_lo  - (F) length low pass filter for decomposition
    dec_hi  - (F) length high pass filter for decomposition
    nlevel  - the number of decomposition levels

    The dimension of the output is given by

    nx = 2*nax - F + 2
    ny = 2*nay - F + 2

    The output is transposed with respect to the input for
    performance reasons.

    '''
    nx, ny = image.shape
    alpha[...] = coeffs  # to avoid overwriting coeffs
    for i in range(nlevel-1, -1, -1):
        # coeff size at current level
        nax = sx[i]
        nay = sy[i]

        # slice indices for alpha
        _, highx = ix[i]
        lowx = highx - 2*nax
        _, highy = iy[i]
        lowy = highy - 2*nay

        # signal size at current level
        nxo = spx[i]
        nyo = spy[i]

        # previous output is the top left corner of the next input
        # note the difference in indexing (nax,nay instead of nxo,nyo)
        # that stems from the redundant coefficient
        if i < nlevel - 1:
            alpha[lowy:lowy+nay, lowx:lowx+nax] = image[0:nax, 0:nay].T

        # need a buffer the size of the coeffs at next level
        idwt2d_level(alpha[lowx:highx, lowy:highy],
                     image[0:nyo, 0:nxo],
                     cbuff[0:2*nax, 0:2*nay],
                     cbuffT[0:2*nay, 0:2*nax],
                     rec_lo, rec_hi)
