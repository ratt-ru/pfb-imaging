from collections import namedtuple

import numba
import numba.core.types as nbtypes
import numpy as np

from pfb.wavelets.coefficients import coefficients


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

_VALID_MODES = ["zeropad", "symmetric", "constant_edge",
                "smooth", "periodic", "periodisation",
                "reflect", "asymmetric", "antireflect"]


@numba.njit(nogil=True, cache=True)
def str_to_int(s):
    final_index = len(s) - 1
    value = 0

    for i, v in enumerate(s):
        digit = (ord(v) - 48)

        if digit < 0 or digit > 9:
            raise ValueError("Invalid integer string")

        value += digit * (10 ** (final_index - i))

    return value


@numba.njit(nogil=True, cache=True)
def dwt_buffer_length(input_length, filter_length, mode):
    if input_length < 1 or filter_length < 1:
        return 0

    if mode == "periodisation":
        return (input_length // 2) + (1 if input_length % 2 else 0)
    else:
        return (input_length + filter_length - 1) // 2


@numba.njit(nogil=True, cache=True)
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
            ncoeffs = len(coeffs)
            rec_lo = coeffs
            dec_lo = np.flipud(rec_lo)
            rec_hi = np.where(np.arange(ncoeffs) % 2, -1, 1) * dec_lo
            dec_hi = np.where(rec_lo % 2, -1, 1) * rec_lo

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

    if isinstance(wavelets, nbtypes.misc.UnicodeType):
        def impl(wavelets, naxis):
            return numba.typed.List([wavelets] * naxis)

    elif ((isinstance(wavelets, nbtypes.containers.List) or
          isinstance(wavelets, nbtypes.containers.UniTuple)) and
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

    if isinstance(axis, nbtypes.Integer):
        def impl(axis, ndim):
            return numba.typed.List([axis])

    elif ((isinstance(axis, nbtypes.containers.List) or
          isinstance(axis, nbtypes.containers.UniTuple)) and
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
def promote_mode(mode, naxis):
    if not isinstance(naxis, nbtypes.Integer):
        raise TypeError("naxis must be an integer")

    if isinstance(mode, nbtypes.misc.UnicodeType):
        def impl(mode, naxis):
            return numba.typed.List([mode for _ in range(naxis)])

    elif ((isinstance(mode, nbtypes.containers.List) or
          isinstance(mode, nbtypes.containers.UniTuple)) and
            isinstance(mode.dtype, nbtypes.UnicodeType)):
        def impl(mode, naxis):
            if len(mode) != naxis:
                raise ValueError("len(mode) != len(axis)")

            return numba.typed.List(mode)
    else:
        raise TypeError("mode must be a string, "
                        "a list of strings "
                        "or a tuple of strings. "
                        "Got %s." % mode)

    return impl



@numba.generated_jit(nopython=True, nogil=True, cache=True)
def downsampling_convolution(input, N, filter,
                             output, step, mode):
    def impl(input, N, filter, F, output,
             step, mode):
        i = step - 1
        o = 0

        if mode == "smooth" and N < 2:
            mode = "constant_edge"

        while i < F and i < N:
            sum = input.dtype.type(0)

            for j in range(i):
                sum += filter[j] * input[i-j]

            i += step
            o += 1

    return impl

from numba.cpython.unsafe.tuple import tuple_setitem

@numba.generated_jit(nopython=True, nogil=True, cache=True)
def dwt_axis(data, wavelet, mode, axis):
    def impl(data, wavelet, mode, axis):
        out_shape = data.shape

        for i, s in enumerate(data.shape):
            if i == axis:
                d = dwt_coeff_length(data.shape[i], len(wavelet.dec_hi), mode)
            else:
                d = s

            out_shape = tuple_setitem(out_shape, i, d)

        ca = np.empty(out_shape, dtype=data.dtype)
        cd = np.empty(out_shape, dtype=data.dtype)
        N = 1

        loop_axes = data.shape[:-1]

        for i, s in enumerate(data.shape):
            pass

        # for i, s in enumerate(data.shape):
        #     if i != axis:
        #         N *= s

        # for i in range(N):
        #     reduced_idx = i
        #     input_offset = 0
        #     output_offset = 0

        #     for j, (shape, stride) in enumerate(zip(data.shape, data.strides)):
        #         j_inv = data.ndim - 1 - j

        #         if j_inv != axis:
        #             axis_idx = reduced_idx % data.shape[j_inv]
        #             reduced_idx /= data.shape[j_inv]
        #             input_offset += (axis_idx * data.strides[j_inv])
        #             output_offset += (axis_idx * data.strides[j_inv])

            # downsampling_convolution(flat_data, N,
            #                          wavelet.dec_lo,
            #                          ca, mode)



        return ca, cd

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def dwt(data, wavelet, mode="symmetric", axis=0):

    if not isinstance(data, nbtypes.npytypes.Array):
        raise TypeError("data must be an ndarray")

    is_complex = isinstance(data.dtype, nbtypes.Complex)

    def impl(data, wavelet, mode="symmetric", axis=0):
        paxis = promote_axis(axis, data.ndim)
        naxis = len(paxis)
        pmode = promote_mode(mode, naxis)
        pwavelets = [discrete_wavelet(w) for w
                     in promote_wavelets(wavelet, naxis)]

        for a, (ax, m, wv) in enumerate(zip(paxis, pmode, pwavelets)):
            print(a, m, ax)
            ca, cd = dwt_axis(data, wv, m, ax)

        # if is_complex:
        #     print("complex", paxis, pmode)
        # else:
        #     print("real", paxis, pmode)


    return impl

@numba.generated_jit(nopython=True, nogil=True, cache=True)
def idwt():
    def impl():
        pass

    return impl
