import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import numba
from numba.core import types, cgutils
from numba.extending import intrinsic
from numba.np.arrayobj import make_view, fix_integer_index
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


@intrinsic
def slice_axis(typingctx, array, index, axis):
    return_type = array.copy(ndim=1)

    if not isinstance(array, types.Array):
        raise TypeError("array is not an Array")

    if (not isinstance(index, types.UniTuple) or
        not isinstance(index.dtype, types.Integer)):
        raise TypeError("index is not a Homogenous Tuple of Integers")

    if len(index) != array.ndim:
        raise TypeError("array.ndim != len(index")

    if not isinstance(axis, types.Integer):
        raise TypeError("axis is not an Integer")

    sig = return_type(array, index, axis)

    def codegen(context, builder, signature, args):
        array_type, idx_type, axis_type = signature.args
        array, idx, axis = args
        array = context.make_array(array_type)(context, builder, array)

        zero = context.get_constant(types.intp, 0)
        llvm_intp_t = context.get_value_type(types.intp)
        ndim = array_type.ndim

        view_shape = cgutils.alloca_once(builder, llvm_intp_t)
        view_stride = cgutils.alloca_once(builder, llvm_intp_t)

        # Final array indexes. We only know the slicing index at runtime
        # so we need to recreate idx but with zero at the slicing axis
        indices = cgutils.alloca_once(builder, llvm_intp_t, size=array_type.ndim)

        for ax in range(array_type.ndim):
            llvm_ax = context.get_constant(types.intp, ax)
            predicate = builder.icmp_unsigned("!=", llvm_ax, axis)

            with builder.if_else(predicate) as (not_equal, equal):
                with not_equal:
                    # If this is not the slicing axis,
                    # use the appropriate tuple index
                    value = builder.extract_value(idx, ax)
                    builder.store(value, builder.gep(indices, [llvm_ax]))

                with equal:
                    # If this is the slicing axis,
                    # store zero as the index.
                    # Also record the stride and shape
                    builder.store(zero, builder.gep(indices, [llvm_ax]))
                    size = builder.extract_value(array.shape, ax)
                    stride = builder.extract_value(array.strides, ax)
                    builder.store(size, view_shape)
                    builder.store(stride, view_stride)


        # Build a python list from indices
        tmp_indices = []

        for i in range(ndim):
            i = context.get_constant(types.intp, i)
            tmp_indices.append(builder.load(builder.gep(indices, [i])))

        # Get the data pointer obtained from indexing the array
        dataptr = cgutils.get_item_pointer(context, builder,
                                           array_type, array,
                                           tmp_indices,
                                           wraparound=True,
                                           boundscheck=True)

        # Set up the shape and stride. There'll only be one
        # dimension, corresponding to the axis along which we slice
        view_shapes = [builder.load(view_shape)]
        view_strides = [builder.load(view_stride)]

        # Make a view with the data pointer, shapes and strides
        retary = make_view(context, builder,
                           array_type, array, return_type,
                           dataptr, view_shapes, view_strides)
        return retary._getvalue()

    return sig, codegen

def test_slicing():
    @numba.njit
    def fn(a, index, axis=1):
        return slice_axis(a, index, axis)

    A = np.random.random((8, 9, 10))

    for axis in range(A.ndim):
        tup_idx = tuple(np.random.randint(1, d) for d in A.shape)
        slice_idx = tuple(slice(None) if a == axis else i for a, i in enumerate(tup_idx))

        B = fn(A, tup_idx, axis)
        assert_array_equal(B, A[slice_idx])


if __name__ == "__main__":
    test_slicing()