import pywt
import numpy as np
import numba
from numba import prange
from numba.typed import List

PARALLEL = True

@numba.njit(nogil=True, cache=True)
def coeff_size(nsignal, nfilter):
    return (nsignal + nfilter - 1)//2


@numba.njit(nogil=True, cache=True)
def coeff_size_at_level(nsignal, nfilter, level):
    ncoeff = nsignal

    for _ in range(level):
        ncoeff = (ncoeff + nfilter - 1)//2

    return ncoeff

@numba.njit(nogil=True, cache=True)
def get_buffer_blocks(image_shape, filter_size, level):

    result = List()

    dim0, dim1 = image_shape
    result.append(dim0 * dim1)

    for _ in range(level):
        dim0 = coeff_size(dim0, filter_size)
        dim1 = coeff_size(dim1, filter_size)

        result.pop()
        result.extend([dim0 * dim1] * 4)

    return result


@numba.njit(nogil=True, cache=True)
def quadrants_from_dwt_buffer(buffer, image_shape, filter_size, level):

    image_size0, image_size1 = image_shape

    blocks = get_buffer_blocks(image_shape, filter_size, level)

    approx_size0 = coeff_size_at_level(image_size0, filter_size, level)
    approx_size1 = coeff_size_at_level(image_size1, filter_size, level)
    approx_shape = (approx_size0, approx_size1)

    q11 = buffer[sum(blocks[:-4]): sum(blocks[:-3])].reshape(approx_shape)
    q10 = buffer[sum(blocks[:-3]): sum(blocks[:-2])].reshape(approx_shape)
    q01 = buffer[sum(blocks[:-2]): sum(blocks[:-1])].reshape(approx_shape)
    q00 = buffer[sum(blocks[:-1]): sum(blocks)].reshape(approx_shape)

    return (q11, q10, q01, q00)


@numba.njit(nogil=True, cache=True)
def get_idwt_buffer_blocks(image_shape, filter_size, level):

    result = List()

    dim0, dim1 = image_shape
    result.append(dim0 * dim1)

    for _ in range(level):
        dim0 = coeff_size(dim0, filter_size)
        dim1 = coeff_size(dim1, filter_size)

        result.pop()
        result.extend([dim0 * dim1] * 4)

    result[-1] = get_signal_shape_at_level(image_shape, filter_size, level)

    return result


@numba.njit(nogil=True, cache=True)
def quadrants_from_idwt_buffer(buffer, image_shape, filter_size, level):

    image_size0, image_size1 = image_shape

    blocks = get_idwt_buffer_blocks(image_shape, filter_size, level)

    signal_shape = signal_size_at_level(image_shape, filter_size, level)
    block_shape = get_block_shape_at_level(image.shape, rec_lo.size, i + 1)

    q11 = buffer[sum(blocks[:-4]): sum(blocks[:-3])].reshape(block_shape)
    q10 = buffer[sum(blocks[:-3]): sum(blocks[:-2])].reshape(block_shape)
    q01 = buffer[sum(blocks[:-2]): sum(blocks[:-1])].reshape(block_shape)
    q00 = buffer[sum(blocks[:-1]): sum(blocks)].reshape(signal_shape)

    return (q11, q10, q01, q00)


@numba.njit(nogil=True, cache=True)
def get_block_shape_at_level(image_shape, filter_size, level):

    dim0, dim1 = image_shape

    for _ in range(level):
        dim0 = coeff_size(dim0, filter_size)
        dim1 = coeff_size(dim1, filter_size)

    return (dim0, dim1)


@numba.njit(nogil=True, cache=True)
def get_signal_shape_at_level(image_shape, filter_size, level):

    dim0, dim1 = image_shape

    for _ in range(level):
        dim0 = coeff_size(dim0, filter_size)
        dim1 = coeff_size(dim1, filter_size)

    dim0 = signal_size(dim0, filter_size)
    dim1 = signal_size(dim1, filter_size)

    return (dim0, dim1)


@numba.njit(nogil=True, cache=True)
def get_buffer_size(image_shape, filter_size, level):
    return sum(get_buffer_blocks(image_shape, filter_size, level))

@numba.njit(nogil=True, cache=True)
def signal_size(ncoeff, nfilter):
    return 2*ncoeff - nfilter + 2

@numba.njit(nogil=True, cache=True)
def signal_size_at_level(ncoeff, nfilter, lvel):
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



@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def dwt2d_level(image, quadrants, dec_lo, dec_hi):
    """
    This function is not meant to be called directly
    """
    nx, ny = image.shape

    x_dim = coeff_size(nx, dec_lo.size)
    y_dim = coeff_size(ny, dec_lo.size)

    approx_x = np.zeros((y_dim, nx))
    detail_x = np.zeros((y_dim, nx))

    quadrant00 = quadrants[-1]
    quadrant01 = quadrants[-2]
    quadrant10 = quadrants[-3]
    quadrant11 = quadrants[-4]

    for i in prange(nx):
        # approx - low pass
        downsampling_convolution(image[i, :], approx_x[:, i], dec_lo, 2)
        # detail - high pass
        downsampling_convolution(image[i, :], detail_x[:, i], dec_hi, 2)

    for i in prange(y_dim):
        # high pass - high pass
        downsampling_convolution(detail_x[i, :], quadrant11[:, i], dec_hi, 2)
        # high pass - low pass
        downsampling_convolution(detail_x[i, :], quadrant01[:, i], dec_lo, 2)
        # low pass - high pass
        downsampling_convolution(approx_x[i, :], quadrant10[:, i], dec_hi, 2)
        # low pass - low pass
        downsampling_convolution(approx_x[i, :], quadrant00[:, i], dec_lo, 2)


@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def dwt2d(image, buffer, dec_lo, dec_hi, nlevel):
    '''
    image       - (nx, ny) signal
    buffer      - coeff buffer,the size of which is detemined by the level
                  and filter size
    dec_lo      - low pass decomposition filter
    dec_hi      - high pass decomposition filter
    '''

    image_shape = image.shape
    filter_size = dec_lo.size

    # Possibly unecessary?
    buffer[:image.size] = image.ravel()
    image = buffer[:image.size].reshape(image_shape)

    for level in range(1, nlevel + 1):

        quadrants = quadrants_from_dwt_buffer(
            buffer, image_shape, filter_size, level
        )

        dwt2d_level(image, quadrants, dec_lo, dec_hi)

        image = quadrants[-1]



@numba.njit(nogil=True, cache=True, inline='always')
def upsampling_convolution_valid_sf(input, filter, output):

    N = input.shape[0]
    F = filter.shape[0]
    O = output.shape[0]

    # if F % 2:
    #     raise ValueError("even filter required for upsampling")

    # if N < (F // 2):
    #     raise ValueError("input.shape[0] < (filter.shape[0] // 2)")

    o = 0
    i = (F // 2) - 1
    stopping_criteria = F // 2

    while i < N and o < O:
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


@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def idwt2d_level(recon, quadrants, rec_lo, rec_hi):
    '''
    outT is a buffer the size of out.T
    tmp is a buffer the size of alpha
    tmpT is a buffer the size of alpha.T
    '''

    dec11, dec10, dec01, dec00 = quadrants

    quad_x_dim, quad_y_dim = dec11.shape  # Coefficient quadrant array shape

    rec_x_dim = signal_size(quad_x_dim, rec_lo.size)
    rec_y_dim = signal_size(quad_y_dim, rec_lo.size)

    approx_x = np.zeros((rec_x_dim, quad_y_dim))
    detail_x = np.zeros((rec_x_dim, quad_y_dim))

    for i in prange(quad_y_dim): # Upsampling convolution along column.
        upsampling_convolution_valid_sf(dec00[:, i], rec_lo, approx_x[:, i])
        upsampling_convolution_valid_sf(dec10[:, i], rec_hi, approx_x[:, i])
        upsampling_convolution_valid_sf(dec01[:, i], rec_lo, detail_x[:, i])
        upsampling_convolution_valid_sf(dec11[:, i], rec_hi, detail_x[:, i])

    recon[:] = 0  # Clear the buffer as we need to accumulate!

    for i in prange(rec_x_dim): # Upsampling convolution along column.
        upsampling_convolution_valid_sf(approx_x[i, :], rec_lo, recon[i, :])
        upsampling_convolution_valid_sf(detail_x[i, :], rec_hi, recon[i, :])


@numba.njit(nogil=True, cache=True, parallel=PARALLEL)
def idwt2d(buffer, image, rec_lo, rec_hi, nlevel):
    '''

    coeffs           - (nax, nay) coeffs
    image             - (nx, ny) output array
    imageT            - (ny, nx) signal buffer to avoid non-contiguous access
    tmpa            - (nax, nay) coeff buffer that avoids overwriting coeffs
    tmp             - (nay, nax) coeff buffer that avoids copy and transpose
    tmpT            - (nay, nax) coeff buffer that avoids copy and transpose
    ix              - dict keyed on level containing x start and stop indices i.e. ix[level] = (ix, fx)
    iy              - dict keyed on level containing y start and stop indices i.e. iy[level] = (iy, fy)
    '''

    quadrants = quadrants_from_dwt_buffer(
        buffer, image.shape, rec_lo.size, nlevel
    )

    blocks = get_buffer_blocks(image.shape, rec_lo.size, nlevel)
    block_dim = get_block_shape_at_level(image.shape, rec_lo.size, nlevel)
    recon_size = block_dim[0] * block_dim[1]
    recon_shape = (block_dim[0], block_dim[1])

    for i in range(nlevel - 1, -1, -1):

        block_dim = get_block_shape_at_level(image.shape, rec_lo.size, i + 1)

        q11 = buffer[sum(blocks[:-4]): sum(blocks[:-3])].reshape(block_dim)
        q10 = buffer[sum(blocks[:-3]): sum(blocks[:-2])].reshape(block_dim)
        q01 = buffer[sum(blocks[:-2]): sum(blocks[:-1])].reshape(block_dim)
        q00 = buffer[sum(blocks[:-1]): sum(blocks)].reshape(recon_shape)

        quadrants = (q11, q10, q01, q00)

        if i == 0:
            recon = image
        else:
            recon_dim0 = signal_size(block_dim[0], rec_lo.size)
            recon_dim1 = signal_size(block_dim[1], rec_lo.size)
            recon_size = recon_dim0 * recon_dim1
            recon_shape = (recon_dim0, recon_dim1)

            blocks = blocks[:-4]
            blocks.append(recon_size)

            recon = buffer[sum(blocks[:-1]):sum(blocks)].reshape(recon_shape)

        idwt2d_level(recon, quadrants, rec_lo, rec_hi)
