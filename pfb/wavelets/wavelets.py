from collections import namedtuple
import warnings

import numba
import numba.core.types as nbtypes
from numba.core.cgutils import is_nonelike
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import register_jitable, overload
from numba.typed import Dict, List
import numpy as np

from pfb.wavelets.coefficients import coefficients
from pfb.wavelets.common import NUMBA_SEQUENCE_TYPES
from pfb.wavelets.convolution import downsampling_convolution
from pfb.wavelets.convolution import upsampling_convolution_valid_sf
from pfb.wavelets.modes import Modes, promote_mode
from pfb.wavelets.intrinsics import slice_axis, force_type_contiguity, not_optional


BaseWavelet = namedtuple("BaseWavelet", (
    "support_width",
    "symmetry",
    "orthogonal",
    "biorthogonal",
    "compact_support",
    "family_name",
    "short_name"))


DiscreteWavelet = namedtuple("DiscreteWavelet",
                             BaseWavelet._fields + (
                                 "dec_hi",
                                 "dec_lo",
                                 "rec_hi",
                                 "rec_lo",
                                 "vanishing_moments_psi",
                                 "vanishing_moments_phi"))


@register_jitable
def str_to_int(s):
    final_index = len(s) - 1
    value = 0

    for i, v in enumerate(s):
        digit = (ord(v) - 48)

        if digit < 0 or digit > 9:
            raise ValueError("Invalid integer string")

        value += digit * (10 ** (final_index - i))

    return value


@register_jitable
def dwt_max_level(data_length, filter_length):
    if filter_length < 2:
        raise ValueError("Invalid wavelet filter length")

    if filter_length <= 1 or data_length < (filter_length - 1):
        return 0

    return int(np.floor(np.log2(data_length / (filter_length - 1))))


@register_jitable
def dwt_buffer_length(input_length, filter_length, mode):
    if input_length < 1 or filter_length < 1:
        return 0

    if mode is Modes.periodisation:
        return (input_length // 2) + (1 if input_length % 2 else 0)
    else:
        return (input_length + filter_length - 1) // 2


@register_jitable
def idwt_buffer_length(coeffs_length, filter_length, mode):
    if mode is Modes.periodisation:
        return 2 * coeffs_length
    else:
        return 2 * coeffs_length - filter_length + 2


@register_jitable
def dwt_coeff_length(data_length, filter_length, mode):
    if data_length < 1:
        raise ValueError("data_length < 1")

    if filter_length < 1:
        raise ValueError("filter_length < 1")

    return dwt_buffer_length(data_length, filter_length, mode)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def discrete_wavelet(wavelet):
    if not isinstance(wavelet, nbtypes.types.UnicodeType):
        raise TypeError("wavelet must be a string")

    def impl(wavelet):
        if wavelet.startswith("db"):
            offset = 2
            order = str_to_int(wavelet[offset:])

            coeffs = coefficients("db", order - 1)
            index = np.arange(len(coeffs))
            rec_lo = coeffs
            dec_lo = np.flipud(rec_lo)
            rec_hi = np.where(index % 2, -1, 1) * dec_lo
            dec_hi = np.where(index[::-1] % 2, -1, 1) * rec_lo

            return DiscreteWavelet(support_width=2 * order - 1,
                                   symmetry="asymmetric",
                                   orthogonal=1,
                                   biorthogonal=1,
                                   compact_support=1,
                                   family_name="Daubechies",
                                   short_name="db",
                                   rec_lo=rec_lo,
                                   dec_lo=dec_lo,
                                   rec_hi=rec_hi,
                                   dec_hi=dec_hi,
                                   vanishing_moments_psi=order,
                                   vanishing_moments_phi=0)

        else:
            raise ValueError("Unknown wavelet")

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def promote_wavelets(wavelets, naxis):
    if not isinstance(naxis, nbtypes.Integer):
        raise TypeError("naxis must be an integer")

    if isinstance(wavelets, nbtypes.misc.UnicodeType):
        def impl(wavelets, naxis):
            return List([wavelets] * naxis)

    elif (isinstance(wavelets, NUMBA_SEQUENCE_TYPES) and
            isinstance(wavelets.dtype, nbtypes.misc.UnicodeType)):

        def impl(wavelets, naxis):
            if len(wavelets) != naxis:
                raise ValueError("len(wavelets) != len(axis)")

            return List(wavelets)
    else:
        raise TypeError("wavelet must be a string, "
                        "a list of strings "
                        "or a tuple of strings. "
                        "Got %s." % wavelets)

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def promote_axis(axis, ndim):
    if not isinstance(ndim, nbtypes.Integer):
        raise TypeError("ndim must be an integer")

    if isinstance(axis, nbtypes.Integer):
        def impl(axis, ndim):
            axis = axis + ndim if axis < 0 else axis
            return List([axis])

    elif (isinstance(axis, NUMBA_SEQUENCE_TYPES) and
            isinstance(axis.dtype, nbtypes.Integer)):
        def impl(axis, ndim):
            if len(axis) > ndim:
                raise ValueError("len(axis) > data.ndim")

            for a in axis:
                if a >= ndim:
                    raise ValueError("axis[i] >= data.ndim")

            return List([a + ndim if a < 0 else a for a in axis])

    else:
        raise TypeError("axis must be an integer, "
                        "a list of integers "
                        "or a tuple of integers. "
                        "Got %s." % axis)

    return impl


@numba.generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def dwt_axis(data, wavelet, mode, axis):
    def impl(data, wavelet, mode, axis):
        coeff_len = dwt_coeff_length(
            data.shape[axis], len(
                wavelet.dec_hi), mode)
        out_shape = tuple_setitem(data.shape, axis, coeff_len)

        if axis < 0 or axis >= data.ndim:
            raise ValueError("0 <= axis < data.ndim failed")

        ca = np.empty(out_shape, dtype=data.dtype)
        cd = np.empty(out_shape, dtype=data.dtype)

        # Iterate over all points except along the slicing axis
        for idx in np.ndindex(*tuple_setitem(data.shape, axis, 1)):
            initial_in_row = slice_axis(data, idx, axis, None)
            initial_ca_row = slice_axis(ca, idx, axis, None)
            initial_cd_row = slice_axis(cd, idx, axis, None)

            # The numba array type returned by slice_axis assumes
            # non-contiguity in the general case.
            # However, the slice may actually be contiguous in layout
            # If so, cast the array type to obtain type contiguity
            # else, copy the slice to obtain contiguity in
            # both type and layout
            if initial_in_row.flags.c_contiguous:
                in_row = force_type_contiguity(initial_in_row)
            else:
                in_row = initial_in_row.copy()

            if initial_ca_row.flags.c_contiguous:
                ca_row = force_type_contiguity(initial_ca_row)
            else:
                ca_row = initial_ca_row.copy()

            if initial_cd_row.flags.c_contiguous:
                cd_row = force_type_contiguity(initial_cd_row)
            else:
                cd_row = initial_cd_row.copy()

            # Compute the approximation and detail coefficients
            downsampling_convolution(in_row, ca_row, wavelet.dec_lo, mode, 2)
            downsampling_convolution(in_row, cd_row, wavelet.dec_hi, mode, 2)

            # If necessary, copy back into the output
            if not initial_ca_row.flags.c_contiguous:
                initial_ca_row[:] = ca_row[:]

            if not initial_cd_row.flags.c_contiguous:
                initial_cd_row[:] = cd_row[:]

        return ca, cd

    return impl


@numba.generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def idwt_axis(approx_coeffs, detail_coeffs,
              wavelet, mode, axis):

    have_approx = not is_nonelike(approx_coeffs)
    have_detail = not is_nonelike(detail_coeffs)

    if not have_approx and not have_detail:
        raise ValueError("Either approximation or detail "
                         "coefficients must be present")

    dtypes = [approx_coeffs.dtype if have_approx else None,
              detail_coeffs.dtype if have_detail else None]

    out_dtype = np.result_type(*(np.dtype(dt.name) for dt in dtypes if dt))

    if have_approx and have_detail:
        if approx_coeffs.ndim != detail_coeffs.ndim:
            raise ValueError("approx_coeffs.ndim != detail_coeffs.ndim")

    def impl(approx_coeffs, detail_coeffs,
             wavelet, mode, axis):

        if have_approx and have_detail:
            coeff_shape = approx_coeffs.shape
            it = enumerate(zip(approx_coeffs.shape, detail_coeffs.shape))

            # NOTE(sjperkins)
            # Clip the coefficient dimensions to the smallest dimensions
            # pywt clips in waverecn and fails in idwt and idwt_axis
            # on heterogenous coefficient shapes.
            # The actual clipping is performed in slice_axis
            for i, (asize, dsize) in it:
                size = asize if asize < dsize else dsize
                coeff_shape = tuple_setitem(coeff_shape, i, size)

        elif have_approx:
            coeff_shape = approx_coeffs.shape
        elif have_detail:
            coeff_shape = detail_coeffs.shape
        else:
            raise ValueError("Either approximation or detail must be present")

        if not (0 <= axis < len(coeff_shape)):
            raise ValueError(("0 <= axis < coeff.ndim does not hold"))

        idwt_len = idwt_buffer_length(coeff_shape[axis],
                                      wavelet.rec_lo.shape[0],
                                      mode)
        out_shape = tuple_setitem(coeff_shape, axis, idwt_len)
        output = np.empty(out_shape, dtype=out_dtype)

        # Iterate over all points except along the slicing axis
        for idx in np.ndindex(*tuple_setitem(output.shape, axis, 1)):
            initial_out_row = slice_axis(output, idx, axis, None)

            # Zero if we have a contiguous slice, else allocate
            if initial_out_row.flags.c_contiguous:
                out_row = force_type_contiguity(initial_out_row)
                out_row[:] = 0
            else:
                out_row = np.zeros_like(initial_out_row)

            # Apply approximation coefficients if they exist
            if approx_coeffs is not None:
                initial_ca_row = slice_axis(approx_coeffs, idx,
                                            axis, coeff_shape[axis])

                if initial_ca_row.flags.c_contiguous:
                    ca_row = force_type_contiguity(initial_ca_row)
                else:
                    ca_row = initial_ca_row.copy()

                upsampling_convolution_valid_sf(ca_row, wavelet.rec_lo,
                                                out_row, mode)

            # Apply detail coefficients if they exist
            if detail_coeffs is not None:
                initial_cd_row = slice_axis(detail_coeffs, idx,
                                            axis, coeff_shape[axis])

                if initial_cd_row.flags.c_contiguous:
                    cd_row = force_type_contiguity(initial_cd_row)
                else:
                    cd_row = initial_cd_row.copy()

                upsampling_convolution_valid_sf(cd_row, wavelet.rec_hi,
                                                out_row, mode)

            # Copy back output row if the output space was non-contiguous
            if not initial_out_row.flags.c_contiguous:
                initial_out_row[:] = out_row

        return output

    return impl


@numba.generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def dwt(data, wavelet, mode="symmetric", axis=None):

    if isinstance(data, nbtypes.misc.Optional):
        if not isinstance(data.type, nbtypes.npytypes.Array):
            raise TypeError(f"data must be ndarray. Got {data.type}")
    elif not isinstance(data, nbtypes.npytypes.Array):
        raise TypeError(f"data must be an ndarray. Got {data}")

    have_axis = not is_nonelike(axis)

    def impl(data, wavelet, mode="symmetric", axis=None):
        if not have_axis:
            axis = List(range(data.ndim))

        paxis = promote_axis(axis, data.ndim)
        naxis = len(paxis)
        pmode = promote_mode(mode, naxis)
        pwavelets = [discrete_wavelet(w) for w
                     in promote_wavelets(wavelet, naxis)]

        coeffs = List([("", data)])

        for a, (ax, m, wv) in enumerate(zip(paxis, pmode, pwavelets)):
            new_coeffs = List()

            for subband, x in coeffs:
                ca, cd = dwt_axis(x, wv, m, ax)
                new_coeffs.append((subband + "a", ca))
                new_coeffs.append((subband + "d", cd))

            coeffs = new_coeffs

        dict_coeffs = Dict()

        for name, coeff in coeffs:
            dict_coeffs[name] = coeff

        return dict_coeffs

    return impl


@register_jitable
def coeff_product(args, repeat=1):
    # Adapted from
    # https://docs.python.org/3/library/itertools.html#itertools.product
    pools = List()

    for i in range(repeat):
        pools.append(args)

    result = List()

    for i in range(len(args) - 1):
        result.append('')

    for pool in pools:
        result = List([x + y for x in result for y in pool])

    return result


@numba.generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def idwt(coeffs, wavelet, mode='symmetric', axis=None):

    have_axis = not is_nonelike(axis)

    def impl(coeffs, wavelet, mode='symmetric', axis=None):
        ndim_transform = max([len(key) for key in coeffs.keys()])
        coeff_shapes = [v.shape for v in coeffs.values()]

        if not have_axis:
            axis = List(range(ndim_transform))
            ndim = ndim_transform
        else:
            ndim = len(coeff_shapes[0])

        paxis = promote_axis(axis, ndim)
        naxis = len(paxis)
        pmode = promote_mode(mode, naxis)
        pwavelets = List([discrete_wavelet(w) for w
                          in promote_wavelets(wavelet, naxis)])

        it = list(enumerate(zip(paxis, pwavelets, pmode)))

        for key_length, (ax, wv, m) in it[::-1]:
            new_coeffs = {}
            new_keys = coeff_product('ad', key_length)

            for key in new_keys:
                L = coeffs[key + 'a']
                H = coeffs[key + 'd']
                new_coeffs[key] = idwt_axis(L, H, wv, m, ax)

            coeffs = new_coeffs

        return coeffs['']

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def promote_level(sizes, dec_lens, level=None):
    have_level = not is_nonelike(level)

    if isinstance(sizes, nbtypes.Integer):
        int_sizes = True
    elif (isinstance(sizes, NUMBA_SEQUENCE_TYPES) and
            isinstance(sizes.dtype, nbtypes.Integer)):
        int_sizes = False
    else:
        raise TypeError("sizes must be an integer or "
                        "sequence of integers")

    if isinstance(dec_lens, nbtypes.Integer):
        int_dec_len = True
    elif (isinstance(dec_lens, NUMBA_SEQUENCE_TYPES) and
            isinstance(dec_lens.dtype, nbtypes.Integer)):
        int_dec_len = False
    else:
        raise TypeError("dec_len must be an integer or "
                        "sequence of integers")

    def impl(sizes, dec_lens, level=None):
        if int_sizes:
            sizes = List([sizes])

        if int_dec_len:
            dec_lens = List([dec_lens])

        max_level = min([dwt_max_level(s, d) for s, d in zip(sizes, dec_lens)])

        if not have_level:
            level = max_level
        elif level < 0:
            raise ValueError("Negative levels are invalid. Minimum level is 0")
        elif level > max_level:
            # with numba.objmode():
            #     warnings.warn("Level value is too high. "
            #                   "All coefficients will experience "
            #                   "boundary effects")
            pass

        return level

    return impl


@numba.generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def wavedecn(data, wavelet, mode='symmetric', level=None, axis=None):
    have_axis = not is_nonelike(axis)

    def impl(data, wavelet, mode='symmetric', level=None, axis=None):
        if not have_axis:
            axis = List(range(data.ndim))

        paxis = promote_axis(axis, data.ndim)
        naxis = len(paxis)
        # pmodes = promote_mode(mode, naxis)
        pwavelets = [discrete_wavelet(w) for w
                     in promote_wavelets(wavelet, naxis)]
        dec_lens = [w.dec_hi.shape[0] for w in pwavelets]
        sizes = [data.shape[ax] for ax in paxis]
        plevel = promote_level(sizes, dec_lens, level)

        coeffs_list = List()

        a = data

        for i in range(plevel):
            coeffs = dwt(a, wavelet, mode, paxis)
            a = not_optional(coeffs.pop('a' * naxis))
            coeffs_list.append(coeffs)

        coeffs_list.append({'aa': a})

        coeffs_list.reverse()

        return coeffs_list

    return impl


@numba.generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def waverecn(coeffs, wavelet, mode='symmetric', axis=None):
    # ca = coeffs[0]['aa']
    # if not isinstance(ca, nbtypes.npytypes.Array):
    #     raise TypeError("ca must be an ndarray")

    have_axis = not is_nonelike(axis)
    # ndim_slices = (slice(None),) * ca.ndim

    def impl(coeffs, wavelet, mode='symmetric', axis=None):
        ca = coeffs[0]['aa']
        if len(coeffs) == 1:
            return ca

        coeff_ndims = [ca.ndim]
        coeff_shapes = [ca.shape]

        for c in coeffs[1:]:
            coeff_ndims.extend([v.ndim for v in c.values()])
            coeff_shapes.extend([v.shape for v in c.values()])

        unique_coeff_ndims = np.unique(np.array(coeff_ndims))

        if len(unique_coeff_ndims) == 1:
            ndim = unique_coeff_ndims[0]
        else:
            raise ValueError("Coefficient dimensions don't match")

        if not have_axis:
            axis = List(range(ndim))

        paxes = promote_axis(axis, ndim)
        naxis = len(paxes)

        for idx, c in enumerate(coeffs[1:]):
            c[not_optional('a' * naxis)] = ca
            ca = idwt(c, wavelet, mode, axis)

        return ca

    return impl


@numba.njit(nogil=True, fastmath=True, cache=True)
def ravel_coeffs(coeffs):
    a_coeffs = coeffs[0]['aa']

    ndim = a_coeffs.ndim

    # initialize with the approximation coefficients.
    a_size = a_coeffs.size
    arr_size = a_size
    for c in coeffs[1:]:
        for k, v in c.items():
            arr_size += v.size
    coeff_arr = np.empty((arr_size, ), dtype=a_coeffs.dtype)

    a_slice = slice(a_size)
    coeff_arr[a_slice] = a_coeffs.ravel()

    # initialize list of coefficient slices
    coeff_slices = List()
    coeff_shapes = List()
    tmp1 = Dict()
    tmp1['aa'] = a_slice
    coeff_slices.append(tmp1)
    tmp2 = Dict()
    tmp2['aa'] = a_coeffs.shape
    coeff_shapes.append(tmp2)
    coeff_shapes[-1]['aa'] = a_coeffs.shape

    if len(coeffs) == 1:
        return coeff_arr, coeff_slices, coeff_shapes

    # loop over the detail cofficients, embedding them in coeff_arr
    offset = a_size
    for coeff_dict in coeffs[1:]:
        # new dictionaries for detail coefficient slices and shapes
        tmp1 = Dict()
        tmp2 = Dict()

        # sort to make sure key order is consistent across Python versions
        keys = sorted(coeff_dict.keys())
        for i, key in enumerate(keys):
            d = coeff_dict[key]
            sl = slice(offset, offset + d.size)
            offset += d.size
            coeff_arr[sl] = d.ravel()
            tmp1[key] = sl
            tmp2[key] = d.shape

        coeff_slices.append(tmp1)
        coeff_shapes.append(tmp2)
    return coeff_arr, coeff_slices, coeff_shapes


@numba.njit(nogil=True, fastmath=True, cache=True)
def unravel_coeffs(arr, coeff_slices, coeff_shapes, output_format='wavedecn'):
    arr = np.asarray(arr)
    coeffs = List()
    tmp = Dict()
    tmp['aa'] = arr[coeff_slices[0]['aa']].reshape(coeff_shapes[0]['aa'])
    coeffs.append(tmp)

    # difference coefficients at each level
    for n in range(1, len(coeff_slices)):
        slice_dict = coeff_slices[n]
        shape_dict = coeff_shapes[n]
        d = Dict()
        for k, v in coeff_slices[n].items():
            d[k] = arr[v].reshape(shape_dict[k])
        coeffs.append(d)
    return coeffs


@numba.njit(nogil=True, fastmath=True, cache=True)
def wavelet_setup(x, bases, nlevels):
    # set up dictionary info
    iys = Dict()
    sys = Dict()
    nmax = x[0].ravel().size
    ntot = List()
    ntot.append(nmax)
    for base in bases:
        if base == 'self':
            continue
        alpha = wavedecn(x[0], base, mode='zero', level=nlevels)
        y, iy, sy = ravel_coeffs(alpha)
        iys[base] = iy
        sys[base] = sy
        ntot.append(y.size)
        nmax = np.maximum(nmax, y.size)

    return iys, sys, ntot, nmax
