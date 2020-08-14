from enum import Enum

import numba
from numba.extending import overload
import numba.core.types as nbtypes

from pfb.wavelets.common import NUMBA_SEQUENCE_TYPES

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


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def promote_mode(mode, naxis):
    if not isinstance(naxis, nbtypes.Integer):
        raise TypeError("naxis must be an integer")

    if isinstance(mode, nbtypes.misc.UnicodeType):
        def impl(mode, naxis):
            return numba.typed.List([mode_str_to_enum(mode) for _ in range(naxis)])

    elif (isinstance(mode, NUMBA_SEQUENCE_TYPES) and
            isinstance(mode.dtype, nbtypes.misc.UnicodeType)):

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
