import numba
from numba.cpython.unsafe.tuple import tuple_setitem
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from pfb.wavelets.wavelets import (dwt, dwt_axis,
                                   idwt, idwt_axis,
                                   str_to_int,
                                   promote_axis,
                                   discrete_wavelet)
from pfb.wavelets.modes import (Modes,
                                promote_mode,
                                mode_str_to_enum)

from pfb.wavelets.intrinsics import slice_axis


def test_str_to_int():
    assert str_to_int("111") == 111
    assert str_to_int("23") == 23
    assert str_to_int("3") == 3


def test_promote_mode():
    assert [Modes.symmetric] == list(promote_mode("symmetric", 1))
    assert [Modes.symmetric]*3 == list(promote_mode("symmetric", 3))

    assert [Modes.symmetric] == list(promote_mode(["symmetric"], 1))
    assert [Modes.symmetric] == list(promote_mode(("symmetric",), 1))

    with pytest.raises(ValueError):
        assert [Modes.symmetric] == list(promote_mode(["symmetric"], 2))

    list_inputs = ["symmetric", "reflect"]
    tuple_inputs = tuple(list_inputs)
    result_enums = [Modes.symmetric, Modes.reflect]

    assert result_enums == list(promote_mode(list_inputs, 2))
    assert result_enums == list(promote_mode(tuple_inputs, 2))

    with pytest.raises(ValueError):
        assert result_enums == list(promote_mode(list_inputs, 3))

    with pytest.raises(ValueError):
        assert result_enums == list(promote_mode(list_inputs, 1))


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
    assert_array_almost_equal(py_wave.dec_hi, pfb_wave.dec_hi)


@pytest.mark.parametrize("wavelet", ["db1", "db4", "db5"])
@pytest.mark.parametrize("data_shape", [(13,), (12, 7)])
def test_dwt_idwt_axis(wavelet, data_shape):
    pywt = pytest.importorskip("pywt")
    data = np.random.random(size=data_shape)
    pywt_dwt_axis = pywt._dwt.dwt_axis
    pywt_idwt_axis = pywt._dwt.idwt_axis

    pywt_wavelet = pywt.Wavelet(wavelet)
    pywt_mode = pywt.Modes.symmetric

    wavelet = discrete_wavelet(wavelet)

    for axis in reversed(range(len(data_shape))):
        # Deconstruct
        ca, cd = dwt_axis(data, wavelet, Modes.symmetric, axis)
        pywt_ca, pywt_cd = pywt_dwt_axis(data, pywt_wavelet, pywt_mode, axis)
        assert_array_almost_equal(ca, pywt_ca)
        assert_array_almost_equal(cd, pywt_cd)

        # Reconstruct with both approximation and detail
        pywt_out = pywt_idwt_axis(ca, cd, pywt_wavelet, pywt_mode, axis)
        output = idwt_axis(ca, cd, wavelet, Modes.symmetric, axis)
        assert_array_almost_equal(output, pywt_out)

        # Reconstruct with approximation only
        pywt_out = pywt_idwt_axis(ca, None, pywt_wavelet, pywt_mode, axis)
        output = idwt_axis(ca, None, wavelet, Modes.symmetric, axis)
        assert_array_almost_equal(output, pywt_out)

        # Reconstruct with detail only
        pywt_out = pywt_idwt_axis(None, cd, pywt_wavelet, pywt_mode, axis)
        output = idwt_axis(None, cd, wavelet, Modes.symmetric, axis)
        assert_array_almost_equal(output, pywt_out)


def test_dwt():
    pywt = pytest.importorskip("pywt")
    data = np.random.random((5, 8, 11))
    res = dwt(data, "db1", "symmetric", 0)
    res = dwt(data, ("db1", "db2"), ("symmetric", "symmetric"), (0, 1))

    pywt_res = pywt.dwtn(data, ("db1", "db2"), ("symmetric", "symmetric"), (0, 1))

    for k, v in res.items():
        vv = pywt_res[k]
        assert_array_almost_equal(v, vv)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_slice_axis(ndim):
    @numba.njit
    def fn(a, index, axis=1):
        return slice_axis(a, index, axis)

    A = np.random.random(np.random.randint(4, 10, size=ndim))
    assert A.ndim == ndim

    for axis in range(A.ndim):
        # Randomly choose indexes within the array
        tup_idx = tuple(np.random.randint(0, d) for d in A.shape)
        # Replace index with slice along desired axis
        slice_idx = tuple(slice(None) if a == axis else i for a, i in enumerate(tup_idx))

        As = A[slice_idx]
        B = fn(A, tup_idx, axis)

        assert_array_equal(As, B)

        if ndim == 1:
            assert B.flags.c_contiguous == As.flags.c_contiguous
            assert B.flags.f_contiguous == As.flags.f_contiguous
            assert B.flags.aligned == As.flags.aligned
            assert B.flags.writeable == As.flags.writeable
            assert B.flags.writebackifcopy == As.flags.writebackifcopy
            assert B.flags.updateifcopy == As.flags.updateifcopy

            # TODO(sjperkins)
            # Why is owndata True in the
            # case of the numba intrinsic, but
            # not in the case of numpy?
            assert B.flags.owndata != As.flags.owndata
        else:
            assert B.flags == As.flags

        # Check that modifying the numba slice
        # modifies the numpy slice
        B[:] = np.arange(B.shape[0])
        assert_array_equal(As, B)



def test_internal_slice_axis():
    @numba.njit
    def fn(A):
        for axis in range(A.ndim):
            for i in np.ndindex(*tuple_setitem(A.shape, axis, 1)):
                S = slice_axis(A, i, axis)

                if S.flags.c_contiguous != (S.itemsize == S.strides[0]):
                    raise ValueError("contiguity flag doesn't match layout")

    fn(np.random.random((8, 9, 10)))
