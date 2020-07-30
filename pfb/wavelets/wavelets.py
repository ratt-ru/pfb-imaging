from collections import namedtuple
from enum import Enum

import numba
import numba.core.types as nbtypes
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import register_jitable, overload
import numpy as np

from pfb.wavelets.coefficients import coefficients
from pfb.wavelets.utils import slice_axis, force_type_contiguity


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


class Modes(Enum):
    zeropad = 0
    symmetric = 1
    constant_edge = 2
    smooth = 3
    periodic = 4
    periodisation = 4
    reflect = 5
    asymmetric = 6
    antireflect = 7


def mode_str_to_enum(mode_str):
    pass


@overload(mode_str_to_enum)
def mode_str_to_enum_impl(mode_str):
    if isinstance(mode_str, nbtypes.UnicodeType):
        # Modes.zeropad.name doesn't work in the jitted code
        # so expand it all.
        zeropad_name = Modes.zeropad.name
        symmetric_name = Modes.symmetric.name
        constant_edge_name = Modes.constant_edge.name
        smooth_name = Modes.smooth.name
        periodic_name = Modes.periodic.name
        periodisation_name = Modes.periodisation.name
        reflect_name = Modes.reflect.name
        asymmetric_name = Modes.asymmetric.name
        antireflect_name = Modes.antireflect.name

        def impl(mode_str):
            mode_str = mode_str.lower()

            if mode_str == zeropad_name:
                return Modes.zeropad
            elif mode_str == symmetric_name:
                return Modes.symmetric
            elif mode_str == constant_edge_name:
                return Modes.constant_edge
            elif mode_str == smooth_name:
                return Modes.smooth
            elif mode_str == periodic_name:
                return Modes.periodic
            elif mode_str == periodisation_name:
                return Modes.periodisation
            elif mode_str == reflect_name:
                return Modes.reflect
            elif mode_str == asymmetric_name:
                return Modes.asymmetric
            elif mode_str == antireflect_name:
                return Modes.antireflect
            else:
                raise ValueError("Unknown mode string")

        return impl


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

    if mode == "periodisation":
        return (input_length // 2) + (1 if input_length % 2 else 0)
    else:
        return (input_length + filter_length - 1) // 2


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
            return numba.typed.List([mode_str_to_enum(mode) for _ in range(naxis)])

    elif ((isinstance(mode, nbtypes.containers.List) or
          isinstance(mode, nbtypes.containers.UniTuple)) and
            isinstance(mode.dtype, nbtypes.UnicodeType)):
        def impl(mode, naxis):
            if len(mode) != naxis:
                raise ValueError("len(mode) != len(axis)")

            return numba.typed.List([mode_str_to_enum(m) for m in mode])
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


@numba.generated_jit(nopython=True, nogil=True)
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

        # Iterate over all points except along the slicing axis
        for idx in np.ndindex(*tuple_setitem(data.shape, axis, 1)):
            initial_in_row = slice_axis(data, idx, axis)
            initial_out_row = slice_axis(ca, idx, axis)

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

            if initial_out_row.flags.c_contiguous:
                out_row = force_type_contiguity(initial_out_row)
            else:
                out_row = initial_out_row.copy()

        return ca, cd

    return impl


@numba.generated_jit(nopython=True, nogil=True)
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
            ca, cd = dwt_axis(data, wv, m, ax)

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def idwt():
    def impl():
        pass

    return impl
