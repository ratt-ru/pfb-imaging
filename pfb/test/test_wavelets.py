import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pfb.wavelets.wavelets import (dwt, idwt,
                                   str_to_int,
                                   promote_axis,
                                   promote_mode,
                                   discrete_wavelet)



def test_str_to_int():
    assert str_to_int("111") == 111
    assert str_to_int("23") == 23
    assert str_to_int("3") == 3


def test_promote_mode():
    assert ["s"] == list(promote_mode("s", 1))
    assert ["s", "s", "s"] == list(promote_mode("s", 3))

    assert ["s"] == list(promote_mode(["s"], 1))
    assert ["s"] == list(promote_mode(("s",), 1))

    with pytest.raises(ValueError):
        assert ["s"] == list(promote_mode(["s"], 2))

    assert ["s", "t"] == list(promote_mode(["s", "t"], 2))
    assert ["s", "t"] == list(promote_mode(("s", "t"), 2))

    with pytest.raises(ValueError):
        assert ["s", "t"] == list(promote_mode(["s", "t"], 3))

    with pytest.raises(ValueError):
        assert ["s", "t"] == list(promote_mode(["s", "t"], 1))


def test_promote_axis():
    assert [0] == list(promote_axis(0, 1))
    assert [0] == list(promote_axis([0], 1))
    assert [0] == list(promote_axis((0,), 1))

    with pytest.raises(ValueError):
        assert [0, 1] == list(promote_axis((0, 1), 1))

    assert [0, 1] == list(promote_axis((0, 1), 2))
    assert [0, 1] == list(promote_axis([0, 1], 2))

    assert [0, 1] == list(promote_axis((0, 1), 3))


@pytest.mark.parametrize("wavelet", ["db1", "db4", "db5"])
def test_discrete_wavelet(wavelet):
    pfb_wave = discrete_wavelet(wavelet)

    pywt = pytest.importorskip("pywt")
    py_wave = pywt.Wavelet(wavelet)

    # assert py_wave.support_width == pfb_wave.support_width
    assert py_wave.orthogonal == pfb_wave.orthogonal
    assert py_wave.biorthogonal == pfb_wave.biorthogonal
    #assert py_wave.compact_support == pfb_wave.compact_support
    assert py_wave.family_name == pfb_wave.family_name
    assert py_wave.short_family_name == pfb_wave.short_name
    assert py_wave.vanishing_moments_phi == pfb_wave.vanishing_moments_phi
    assert py_wave.vanishing_moments_psi == pfb_wave.vanishing_moments_psi

    assert_array_almost_equal(py_wave.rec_lo, pfb_wave.rec_lo)
    assert_array_almost_equal(py_wave.dec_lo, pfb_wave.dec_lo)
    assert_array_almost_equal(py_wave.rec_hi, pfb_wave.rec_hi)
    assert_array_almost_equal(py_wave.rec_lo, pfb_wave.rec_lo)


def test_dwt():
    data = np.random.random((111, 126))
    dwt(data, "db1", "symmetric", 0)
    dwt(data, ("db1", "db2"), ("symmetric", "symmetric"), (0, 1))



from numba.core import types, cgutils
from numba.extending import intrinsic


@intrinsic
def sliced_indexing_tuple(typingctx, input_tuple, axis):
    if not isinstance(input_tuple, types.UniTuple):
        raise types.TypingError("tuple_type must be an homogenous tuple %s" % input_tuple)

    if not input_tuple.count > 0:
        raise types.TypingError("tuple_type must be > 0 elements")

    return_type = types.UniTuple(types.slice2_type, input_tuple.count)
    sig = return_type(input_tuple, axis)

    def codegen(context, builder, signature, args):
        input_type = signature.args[0]
        axis_type = signature.args[1]
        return_type = signature.return_type

        # Allocate an empty tuple
        llvm_tuple_type = context.get_value_type(return_type)
        tup = cgutils.get_null_value(llvm_tuple_type)

        def slicer(tup, i, axis):
            idx = tup[i]
            #print(tup, i, axis)
            return slice(0, idx) if i == axis else slice(idx, idx+1)

        for i in range(return_type.count):
            # Get a constant index value
            tup_idx = context.get_constant(types.intp, i)
            # Arguments passed into slicer
            slicer_args = [args[0], tup_idx, args[1]]
            slicer_sig = return_type.dtype(input_type, types.intp, axis_type)
            # Call slicer
            data = context.compile_internal(builder, slicer,
                                            slicer_sig,
                                            slicer_args)

            # Insert result of slicer into the tuple
            tup = builder.insert_value(tup, data, i)

        # Return the tuple
        return tup

    return sig, codegen


def test_slicing():
    from numba.cpython.unsafe.tuple import tuple_setitem
    import numba

    @numba.njit(nogil=True, cache=True)
    def fn(a, axis=1):

        shape = a.shape
        for ax, d in enumerate(a.shape):
            shape = tuple_setitem(shape, ax, 1 if ax == axis else d)

        for i in np.ndindex(shape):
            i = tuple_setitem(i, axis, a.shape[axis])
            idx = sliced_indexing_tuple(i, axis)
            row = a[idx]

            if not row.flags.c_contiguous:
                row = row.copy()


    A = np.random.random((8, 9, 10))
    B = fn(A, axis=1)