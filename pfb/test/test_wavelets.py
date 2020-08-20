from itertools import product

import numba
from numba.cpython.unsafe.tuple import tuple_setitem
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from pfb.wavelets.wavelets import (dwt, dwt_axis,
                                   idwt, idwt_axis,
                                   dwt_max_level,
                                   str_to_int,
                                   coeff_product,
                                   promote_axis,
                                   promote_level,
                                   discrete_wavelet,
                                   wavedecn, waverecn)
from pfb.wavelets.modes import (Modes,
                                promote_mode,
                                mode_str_to_enum)

from pfb.wavelets.intrinsics import slice_axis


convert_mode = numba.njit(lambda s: mode_str_to_enum(s))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("extent", [-1, 0, 1])
def test_slice_axis(ndim, extent):
    @numba.njit
    def fn(a, index, axis=1, extent=None):
        return slice_axis(a, index, axis, extent)

    A = np.random.random(np.random.randint(4, 10, size=ndim))
    assert A.ndim == ndim

    for axis in range(A.ndim):
        # Randomly choose indexes within the array
        tup_idx = tuple(np.random.randint(0, d) for d in A.shape)
        # Replace index with slice along desired axis
        ax_size = A.shape[axis]

        ext = None if extent == 0 else ax_size + extent
        slice_idx = tuple(slice(ext) if a == axis
                          else i for a, i in enumerate(tup_idx))

        As = A[slice_idx]
        B = fn(A, tup_idx, axis, ext)

        assert_array_equal(As, B)

        if ndim == 1:
            assert B.flags.c_contiguous is As.flags.c_contiguous
            assert B.flags.f_contiguous is As.flags.f_contiguous
            assert B.flags.aligned is As.flags.aligned
            assert B.flags.writeable is As.flags.writeable
            assert B.flags.writebackifcopy is As.flags.writebackifcopy
            assert B.flags.updateifcopy is As.flags.updateifcopy

            # TODO(sjperkins)
            # Why is owndata True in the
            # case of the numba intrinsic, but
            # not in the case of numpy?
            assert B.flags.owndata is not (extent or As.flags.owndata)
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
                S = slice_axis(A, i, axis, None)

                if S.flags.c_contiguous != (S.itemsize == S.strides[0]):
                    raise ValueError("contiguity flag doesn't match layout")

    fn(np.random.random((8, 9, 10)))


@pytest.mark.parametrize("repeat", range(5))
def test_coeff_product(repeat):
    res = coeff_product('ad', repeat=repeat)
    coeffs = [''.join(c) for c in product('ad', repeat=repeat)]
    assert list(res) == coeffs


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


@pytest.mark.parametrize("data", [500, 100, 12])
@pytest.mark.parametrize("filter", [500, 100, 24])
def test_dwt_max_level(data, filter):
    pywt = pytest.importorskip("pywt")
    assert pywt.dwt_max_level(data, filter) == dwt_max_level(data, filter)


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
@pytest.mark.parametrize("mode", ["symmetric", "zero"])
def test_dwt_idwt_axis(wavelet, mode, data_shape):
    pywt = pytest.importorskip("pywt")
    data = np.random.random(size=data_shape)
    enum_mode = convert_mode(mode)

    pywt_dwt_axis = pywt._dwt.dwt_axis
    pywt_idwt_axis = pywt._dwt.idwt_axis

    pywt_wavelet = pywt.Wavelet(wavelet)
    pywt_mode = pywt.Modes.from_object(mode)

    wavelet = discrete_wavelet(wavelet)

    for axis in reversed(range(len(data_shape))):
        # Deconstruct
        ca, cd = dwt_axis(data, wavelet, enum_mode, axis)
        pywt_ca, pywt_cd = pywt_dwt_axis(data, pywt_wavelet, pywt_mode, axis)
        assert_array_almost_equal(ca, pywt_ca)
        assert_array_almost_equal(cd, pywt_cd)

        # Reconstruct with both approximation and detail
        pywt_out = pywt_idwt_axis(ca, cd, pywt_wavelet, pywt_mode, axis)
        output = idwt_axis(ca, cd, wavelet, enum_mode, axis)
        assert_array_almost_equal(output, pywt_out)

        # Reconstruct with approximation only
        pywt_out = pywt_idwt_axis(ca, None, pywt_wavelet, pywt_mode, axis)
        output = idwt_axis(ca, None, wavelet, enum_mode, axis)
        assert_array_almost_equal(output, pywt_out)

        # Reconstruct with detail only
        pywt_out = pywt_idwt_axis(None, cd, pywt_wavelet, pywt_mode, axis)
        output = idwt_axis(None, cd, wavelet, enum_mode, axis)
        assert_array_almost_equal(output, pywt_out)


def test_dwt_idwt():
    pywt = pytest.importorskip("pywt")
    data = np.random.random((5, 8, 11))

    res = dwt(data, "db1", "symmetric")
    pywt_res = pywt.dwtn(data, "db1", "symmetric")
    for k, v in res.items():
        assert_array_almost_equal(v, pywt_res[k])

    res = dwt(data, "db1", "symmetric", 1)
    pywt_res = pywt.dwtn(data, "db1", "symmetric", (1,))
    for k, v in res.items():
        assert_array_almost_equal(v, pywt_res[k])

    res = dwt(data, ("db1", "db2"), ("symmetric", "symmetric"), (0, 1))
    pywt_res = pywt.dwtn(data, ("db1", "db2"), ("symmetric", "symmetric"), (0, 1))
    for k, v in res.items():
        assert_array_almost_equal(v, pywt_res[k])

    output = idwt(res, ("db1", "db2"), ("symmetric", "symmetric"), (0, 1))
    pywt_out = pywt.idwtn(pywt_res, ("db1", "db2"), ("symmetric", "symmetric"), (0, 1))
    assert_array_almost_equal(output, pywt_out)

xf = pytest.mark.xfail(reason="level too high")

@pytest.mark.parametrize("data_shape", [(50, 24, 63)])
@pytest.mark.parametrize("level", list(range(5)) + [
    pytest.param(i, marks=xf) for i in range(6, 10)])
def test_wavedecn_waverecn(data_shape, level):
    pywt = pytest.importorskip("pywt")
    data = np.random.random(data_shape)

    out = pywt.wavedecn(data, "db1", "symmetric")
    a, coeffs = wavedecn(data, "db1", "symmetric")

    assert_array_almost_equal(a, out[0])

    for d1, d2 in zip(out[1:], coeffs):
        assert list(d1.keys()) == list(d2.keys())

        for k, v in d1.items():
            assert_array_almost_equal(v, d2[k])

    pywt_rec = pywt.waverecn(out, "db1", "symmetric")
    rec = waverecn(a, coeffs, "db1", "symmetric")
    assert_array_almost_equal(pywt_rec, rec)

    out = pywt.wavedecn(data, "db1", "symmetric", axes=(1, 2))
    a, coeffs = wavedecn(data, "db1", "symmetric", axis=(1, 2))

    assert_array_almost_equal(a, out[0])

    for d1, d2 in zip(out[1:], coeffs):
        assert list(d1.keys()) == list(d2.keys())

        for k, v in d1.items():
            assert_array_almost_equal(v, d2[k])

    pywt_rec = pywt.waverecn(out, "db1", "symmetric", axes=(1, 2))
    rec = waverecn(a, coeffs, "db1", "symmetric", axis=(1, 2))
    assert_array_almost_equal(pywt_rec, rec)

    # Test various levels of decomposition
    out = pywt.wavedecn(data, "db1", "symmetric", level=level, axes=(1, 2))
    a, coeffs = wavedecn(data, "db1", "symmetric", level=level, axis=(1, 2))

    assert_array_almost_equal(a, out[0])

    for d1, d2 in zip(out[1:], coeffs):
        assert list(d1.keys()) == list(d2.keys())

        for k, v in d1.items():
            assert_array_almost_equal(v, d2[k])

    pywt_rec = pywt.waverecn(out, "db1", "symmetric", axes=(1, 2))
    rec = waverecn(a, coeffs, "db1", "symmetric", axis=(1, 2))
    assert_array_almost_equal(pywt_rec, rec)

