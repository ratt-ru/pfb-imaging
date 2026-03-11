import numba

from pfb_imaging.wavelets.convolutions import (
    conv_downsample_axis0_polyphase_pair,
    conv_upsample_axis0_polyphase_pair,
    downsampling_convolution,
    upsampling_convolution_valid_sf,
    upsampling_convolution_valid_sf_set,
)


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
def dwt2d(image, coeffs, cbuff, cbufft, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx):
    """
    Multi-level 2D image to coeffs transform

    image   - (nx, ny) signal
    coeffs  - (nay, nax) output coeffs
    cbuff   - (nxmax, nymax) scratch buffer
    cbufft  - (nymax, nxmax) scratch buffer
    ix,iy   - (nlevel, 2) int64 arrays with start/stop indices for packing coeffs
    sx,sy   - (nlevel,) int64 arrays with coeff sizes at each level
    dec_lo  - (filter_size) length low pass filter for decomposition
    dec_hi  - (filter_size) length high pass filter for decomposition
    nlevel  - the number of decomposition levels
    approx  - (nxmax, nymax) pre-allocated buffer for approx coeffs

    """
    approx_in = image
    for i in range(nlevel):
        nax = 2 * sx[i]
        nay = 2 * sy[i]
        highy = iy[i, 1]
        lowy = highy - nay
        highx = ix[i, 1]
        lowx = highx - nax
        dwt2d_level(
            approx_in,
            coeffs[lowy:highy, lowx:highx],
            cbuff[0:nax, 0:nay],
            cbufft[0:nay, 0:nax],
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

    midx = nax // 2
    for i in numba.prange(nay):
        # _set writes (=) instead of accumulating, so no zeroing needed
        upsampling_convolution_valid_sf_set(coeffs[i, 0:midx], rec_lo, cbufft[i, :])
        upsampling_convolution_valid_sf(coeffs[i, midx:], rec_hi, cbufft[i, :])

    copyt(cbufft, cbuff)

    midy = nay // 2
    for i in numba.prange(nx):
        upsampling_convolution_valid_sf_set(cbuff[i, 0:midy], rec_lo, image[i, :])
        upsampling_convolution_valid_sf(cbuff[i, midy:], rec_hi, image[i, :])


@numba.njit(nogil=True, cache=True)
def idwt2d(coeffs, image, alpha, cbuff, cbufft, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel):
    """
    Multi-level 2D coeffs to image transform

    coeffs  - (nay, nax) output coeffs
    image   - (nx, ny) signal
    alpha   - (nay, nax) coeff buffer
    cbuff   - (nxmax, nymax) scratch buffer
    cbufft  - (nymax, nxmax) scratch buffer
    ix,iy   - (nlevel, 2) int64 arrays with start/stop indices for packing coeffs
    sx,sy   - (nlevel,) int64 arrays with coeff sizes at each level
    spx,spy - (nlevel,) int64 arrays with signal sizes at each level
    rec_lo  - (filter_size) length low pass filter for reconstruction
    rec_hi  - (filter_size) length high pass filter for reconstruction
    nlevel  - the number of decomposition levels

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

        idwt2d_level(
            alpha[lowy:highy, lowx:highx],
            image[0:nxo, 0:nyo],
            cbuff[0 : 2 * nax, 0 : 2 * nay],
            cbufft[0 : 2 * nay, 0 : 2 * nax],
            rec_lo,
            rec_hi,
        )


# ── No-copyt versions using polyphase axis-0 convolutions ─────────────
#
# These eliminate ALL copyt transpositions by performing axis-0
# convolutions directly via the polyphase decomposition from
# pfb_imaging.wavelets.convolutions.


@numba.njit(nogil=True, cache=True, parallel=True)
def dwt2d_level_nocopyt(image, coeffs, cbuff, dec_lo, dec_hi):
    """Single-level forward 2D DWT without transpose operations.

    Parameters
    ----------
    image  : (nx, ny) input signal
    coeffs : (2*sx, 2*sy) output, x-first layout
    cbuff  : (nx, 2*sy) scratch buffer
    dec_lo : (K,) low-pass decomposition filter
    dec_hi : (K,) high-pass decomposition filter
    """
    nx = image.shape[0]
    nax, nay = coeffs.shape
    sx = nax // 2
    sy = nay // 2

    # Phase 1: downsample-convolve along axis 1 (contiguous rows)
    for i in numba.prange(nx):
        downsampling_convolution(image[i, :], cbuff[i, 0:sy], dec_lo, 2)
        downsampling_convolution(image[i, :], cbuff[i, sy:nay], dec_hi, 2)

    # Phase 2: downsample-convolve along axis 0 via polyphase
    # Y0 = lo_x rows (LL|LH), Y1 = hi_x rows (HL|HH)
    conv_downsample_axis0_polyphase_pair(cbuff, dec_lo, dec_hi, coeffs[0:sx, :], coeffs[sx:nax, :])


@numba.njit(nogil=True, cache=True)
def dwt2d_nocopyt(image, coeffs, cbuff, ix, iy, sx, sy, dec_lo, dec_hi, nlevel):
    """Multi-level forward 2D DWT without transpose operations.

    Parameters
    ----------
    image  : (nx, ny) input signal
    coeffs : (ntotx, ntoty) output coefficients, x-first layout
    cbuff  : (nxmax, 2*symax) scratch buffer
    ix, iy : (nlevel, 2) int64 start/stop indices per level
    sx, sy : (nlevel,) int64 coeff sizes per level
    dec_lo : (K,) low-pass decomposition filter
    dec_hi : (K,) high-pass decomposition filter
    nlevel : number of decomposition levels
    """
    approx_in = image
    for i in range(nlevel):
        nax = 2 * sx[i]
        nay = 2 * sy[i]
        highx = ix[i, 1]
        lowx = highx - nax
        highy = iy[i, 1]
        lowy = highy - nay
        nx_in = approx_in.shape[0]
        dwt2d_level_nocopyt(
            approx_in,
            coeffs[lowx:highx, lowy:highy],
            cbuff[0:nx_in, 0:nay],
            dec_lo,
            dec_hi,
        )
        # LL subband is at top-left of the block — use directly as
        # input for the next level (view, no copy or transpose)
        approx_in = coeffs[lowx : lowx + sx[i], lowy : lowy + sy[i]]


@numba.njit(nogil=True, cache=True, parallel=True)
def idwt2d_level_nocopyt(coeffs, image, cbuff, rec_lo, rec_hi):
    """Single-level inverse 2D DWT without transpose operations.

    Parameters
    ----------
    coeffs : (2*sx, 2*sy) input coefficients, x-first layout
    image  : (nx, ny) output signal
    cbuff  : (nx, 2*sy) scratch buffer
    rec_lo : (K,) low-pass reconstruction filter
    rec_hi : (K,) high-pass reconstruction filter
    """
    nax, nay = coeffs.shape
    nx, ny = image.shape
    sx = nax // 2
    sy = nay // 2

    # Phase 1: upsample+convolve along axis 0 via polyphase
    # X0 = lo_x rows (LL|LH), X1 = hi_x rows (HL|HH)
    conv_upsample_axis0_polyphase_pair(coeffs[0:sx, :], coeffs[sx:nax, :], rec_lo, rec_hi, cbuff)

    # Phase 2: upsample+convolve along axis 1 (contiguous rows)
    for i in numba.prange(nx):
        upsampling_convolution_valid_sf_set(cbuff[i, 0:sy], rec_lo, image[i, :])
        upsampling_convolution_valid_sf(cbuff[i, sy:nay], rec_hi, image[i, :])


@numba.njit(nogil=True, cache=True)
def idwt2d_nocopyt(coeffs, image, alpha, cbuff, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel):
    """Multi-level inverse 2D DWT without transpose operations.

    Parameters
    ----------
    coeffs : (ntotx, ntoty) input coefficients, x-first layout
    image  : (nx, ny) output signal
    alpha  : (ntotx, ntoty) working copy of coeffs (avoids overwriting)
    cbuff  : (nxmax, 2*symax) scratch buffer
    ix, iy : (nlevel, 2) int64 start/stop indices per level
    sx, sy : (nlevel,) int64 coeff sizes per level
    spx, spy : (nlevel,) int64 signal sizes per level
    rec_lo : (K,) low-pass reconstruction filter
    rec_hi : (K,) high-pass reconstruction filter
    nlevel : number of decomposition levels
    """
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
            # Copy inner reconstruction into LL slot (plain copy, no transpose)
            alpha[lowx : lowx + nax, lowy : lowy + nay] = image[0:nax, 0:nay]

        idwt2d_level_nocopyt(
            alpha[lowx : lowx + 2 * nax, lowy : lowy + 2 * nay],
            image[0:nxo, 0:nyo],
            cbuff[0:nxo, 0 : 2 * nay],
            rec_lo,
            rec_hi,
        )
