from collections import namedtuple

import numba
import numba.core.types as nbtypes
from numba.core.cgutils import is_nonelike
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import register_jitable, overload
from numba.typed import Dict, List
import numpy as np

from pfb.wavelets.coefficients import coefficients
from pfb.wavelets.convolution import downsampling_convolution
from pfb.wavelets.convolution import upsampling_convolution_valid_sf
from pfb.wavelets.modes import Modes, promote_mode
from pfb.wavelets.intrinsics import slice_axis, force_type_contiguity


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

            return DiscreteWavelet(support_width=2*order - 1,
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

    ltypes = (nbtypes.containers.List,
              nbtypes.containers.ListType,
              nbtypes.containers.UniTuple)

    if isinstance(wavelets, nbtypes.misc.UnicodeType):
        def impl(wavelets, naxis):
            return numba.typed.List([wavelets] * naxis)

    elif (isinstance(wavelets, ltypes) and
            isinstance(wavelets.dtype, nbtypes.misc.UnicodeType)):

        def impl(wavelets, naxis):
            if len(wavelets) != naxis:
                raise ValueError("len(wavelets) != len(axis)")

            return numba.typed.List(wavelets)
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

    ltypes = (nbtypes.containers.List,
              nbtypes.containers.ListType,
              nbtypes.containers.UniTuple)

    if isinstance(axis, nbtypes.Integer):
        def impl(axis, ndim):
            return numba.typed.List([axis])

    elif (isinstance(axis, ltypes) and
            isinstance(axis.dtype, nbtypes.Integer)):
        def impl(axis, ndim):
            if len(axis) > ndim:
                raise ValueError("len(axis) > data.ndim")

            for a in axis:
                if a >= ndim:
                    raise ValueError("axis[i] >= data.ndim")

            return numba.typed.List(axis)
    else:
        raise TypeError("axis must be an integer, "
                        "a list of integers "
                        "or a tuple of integers. "
                        "Got %s." % axis)

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def dwt_axis(data, wavelet, mode, axis):
    def impl(data, wavelet, mode, axis):
        coeff_len = dwt_coeff_length(data.shape[axis], len(wavelet.dec_hi), mode)
        out_shape = tuple_setitem(data.shape, axis, coeff_len)

        if axis < 0 or axis >= data.ndim:
            raise ValueError("0 <= axis < data.ndim failed")

        ca = np.empty(out_shape, dtype=data.dtype)
        cd = np.empty(out_shape, dtype=data.dtype)

        # Iterate over all points except along the slicing axis
        for idx in np.ndindex(*tuple_setitem(data.shape, axis, 1)):
            initial_in_row = slice_axis(data, idx, axis)
            initial_ca_row = slice_axis(ca, idx, axis)
            initial_cd_row = slice_axis(cd, idx, axis)

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


@numba.generated_jit(nopython=True, nogil=True)
def idwt_axis(approx_coeffs, detail_coeffs,
              wavelet, mode, axis):

    have_approx = not is_nonelike(approx_coeffs)
    have_detail = not is_nonelike(detail_coeffs)

    def impl(approx_coeffs, detail_coeffs,
            wavelet, mode, axis):

        if have_approx:
            shape = approx_coeffs.shape
            dtype = approx_coeffs.dtype
        elif have_detail:
            shape = detail_coeffs.shape
            dtype = detail_coeffs.dtype
        else:
            raise ValueError("Either approximation or detail must be present")

        if (have_approx and have_detail and
            approx_coeffs.shape != detail_coeffs.shape):

                raise ValueError("approx_coeffs.shape != detail_coeffs.shape")

        if not (0 <= axis < len(shape)):
            raise ValueError(("0 <= axis < coeff.ndim does not hold"))

        idwt_len = idwt_buffer_length(shape[axis], wavelet.rec_lo.shape[0], mode)
        shape = tuple_setitem(shape, axis, idwt_len)
        output = np.empty(shape, dtype=dtype)

        # Iterate over all points except along the slicing axis
        for idx in np.ndindex(*tuple_setitem(output.shape, axis, 1)):
            initial_out_row = slice_axis(output, idx, axis)

            # Zero if we have a contiguous slice, else allocate
            if initial_out_row.flags.c_contiguous:
                out_row = force_type_contiguity(initial_out_row)
                out_row[:] = 0
            else:
                out_row = np.zeros_like(initial_out_row)

            # Apply approximation coefficients if they exist
            if approx_coeffs is not None:
                initial_ca_row = slice_axis(approx_coeffs, idx, axis)

                if initial_ca_row.flags.c_contiguous:
                    ca_row = force_type_contiguity(initial_ca_row)
                else:
                    ca_row = initial_ca_row.copy()

                upsampling_convolution_valid_sf(ca_row, wavelet.rec_lo,
                                                out_row, mode)

            # Apply detail coefficients if they exist
            if detail_coeffs is not None:
                initial_cd_row = slice_axis(detail_coeffs, idx, axis)

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



@numba.generated_jit(nopython=True, nogil=True, cache=True)
def dwt(data, wavelet, mode="symmetric", axis=None):

    if not isinstance(data, nbtypes.npytypes.Array):
        raise TypeError("data must be an ndarray")

    have_axis = not is_nonelike(axis)
    is_complex = isinstance(data.dtype, nbtypes.Complex)

    def impl(data, wavelet, mode="symmetric", axis=None):
        if not have_axis:
            axis = numba.typed.List(range(data.ndim))

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
    # Adapted from https://docs.python.org/3/library/itertools.html#itertools.product
    pools = [args] * repeat
    result = [''] * (len(args) - 1)

    for pool in pools:
        result = [x + y for x in result for y in pool]

    return result

@numba.generated_jit(nopython=True, nogil=True)
def idwt(coeffs, wavelet, mode='symmetric', axis=None):

    have_axis = not is_nonelike(axis)

    def impl(coeffs, wavelet, mode='symmetric', axis=None):
        ndim_transform = max([len(key) for key in coeffs.keys()])
        coeff_shapes = [v.shape for v in coeffs.values()]

        for cs in coeff_shapes[1:]:
            if cs != coeff_shapes[0]:
                raise ValueError("Mismatch in coefficient shapes")

        if not have_axis:
            axis = numba.typed.List(range(ndim_transform))
            ndim = ndim_transform
        else:
            ndim = len(coeff_shapes[0])

        paxis = promote_axis(axis, ndim)
        naxis = len(paxis)
        pmode = promote_mode(mode, naxis)
        pwavelets = [discrete_wavelet(w) for w
                     in promote_wavelets(wavelet, naxis)]

        it = list(enumerate(zip(paxis, pwavelets, pmode)))

        for key_length, (ax, wv, m) in it[::-1]:
            new_coeffs = {}
            new_keys = coeff_product('ad', key_length)

            for key in new_keys:
                L = coeffs[key + 'a']
                H = coeffs[key + 'd']
                # print(key, "low", None if L is None else L.shape)
                # print(key, "high", None if H is None else H.shape)

                new_coeffs[key] = idwt_axis(L, H, wv, m, ax)

            coeffs = new_coeffs

        return coeffs['']

    return impl
