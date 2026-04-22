from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from numpy.testing import assert_allclose

from pfb_imaging.core.hci import (
    _FITS_AXIS_KEYWORDS,
    _STOKES_FITS_INDEX,
    _make_fits_header,
)
from pfb_imaging.core.hci import (
    hci as hci_core,
)

pmp = pytest.mark.parametrize


def _make_base_fits_header():
    """Build a 5-axis FITS header mirroring the layout stored by hci.

    Axis order matches make_dummy_dataset: (X, Y, TIME/INTEGRATION, FREQ, STOKES).
    """
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 5
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    hdr["NAXIS3"] = 5
    hdr["NAXIS4"] = 3
    hdr["NAXIS5"] = 1
    hdr["EXTEND"] = True
    hdr["BSCALE"] = 1.0
    hdr["BZERO"] = 0.0
    hdr["BUNIT"] = "Jy/beam"
    hdr["EQUINOX"] = 2000.0
    hdr["BTYPE"] = "Intensity"
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CTYPE3"] = "INTEGRATION"
    hdr["CTYPE4"] = "FREQ"
    hdr["CTYPE5"] = "STOKES"
    hdr["CRVAL1"] = 10.0
    hdr["CRVAL2"] = -30.0
    hdr["CRVAL3"] = 0.0
    hdr["CRVAL4"] = 1.0e9
    hdr["CRVAL5"] = 1
    hdr["CRPIX1"] = 33
    hdr["CRPIX2"] = 33
    hdr["CRPIX3"] = 1
    hdr["CRPIX4"] = 1
    hdr["CRPIX5"] = 1
    hdr["CDELT1"] = -1e-5
    hdr["CDELT2"] = 1e-5
    hdr["CDELT3"] = 60.0
    hdr["CDELT4"] = 1e6
    hdr["CDELT5"] = 1
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CUNIT3"] = "s"
    hdr["CUNIT4"] = "Hz"
    hdr["CUNIT5"] = ""
    hdr["WCSAXES"] = 5
    return dict(hdr)


def test_stokes_fits_index_codes():
    """The module-level Stokes -> FITS code mapping must match the FITS standard."""
    assert _STOKES_FITS_INDEX == {"I": 1, "Q": 2, "U": 3, "V": 4}


def test_fits_axis_keywords_complete():
    """All axis-numbered FITS keywords we care about are present in the rewrite set."""
    for kw in ("CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT", "NAXIS"):
        assert kw in _FITS_AXIS_KEYWORDS


def test_make_fits_header_drop_time_axis():
    """Dropping the TIME axis (FITS axis 3) yields a 4-axis (X, Y, FREQ, STOKES) header.

    The FREQ axis must be renumbered to axis 3 and STOKES to axis 4, with
    mandatory keywords appearing first as required by StreamingHDU.
    """
    base = _make_base_fits_header()
    stokes = ["I"]
    # numpy shape (STOKES, FREQ, Y, X) -> FITS axes (X=1, Y=2, FREQ=3, STOKES=4)
    shape = (1, 3, 64, 64)
    hdr = _make_fits_header(base, shape, stokes, drop_axes={3})

    assert hdr["NAXIS"] == 4
    assert hdr["NAXIS1"] == 64
    assert hdr["NAXIS2"] == 64
    assert hdr["NAXIS3"] == 3
    assert hdr["NAXIS4"] == 1
    assert hdr["CTYPE1"] == "RA---SIN"
    assert hdr["CTYPE2"] == "DEC--SIN"
    assert hdr["CTYPE3"] == "FREQ"
    assert hdr["CTYPE4"] == "STOKES"
    assert hdr["CRVAL3"] == 1.0e9
    assert hdr["CDELT3"] == 1e6
    # Stokes axis values come from the FITS code table, not the base header
    assert hdr["CRVAL4"] == 1
    assert hdr["CDELT4"] == 1
    assert hdr["CUNIT4"] == ""
    # mandatory keywords appear first in key order (StreamingHDU requirement)
    keys = list(hdr.keys())
    expected_prefix = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4"]
    assert keys[: len(expected_prefix)] == expected_prefix
    assert hdr["WCSAXES"] == 4
    # the old axis-5 (STOKES) keywords must not leak through
    for kw in ("CTYPE5", "CRVAL5", "CRPIX5", "CDELT5", "CUNIT5", "NAXIS5"):
        assert kw not in hdr


def test_make_fits_header_drop_freq_axis():
    """Dropping the FREQ axis (FITS axis 4) yields a 4-axis (X, Y, TIME, STOKES) header."""
    base = _make_base_fits_header()
    # numpy shape (STOKES, TIME, Y, X)
    shape = (1, 5, 64, 64)
    hdr = _make_fits_header(base, shape, ["I"], drop_axes={4})

    assert hdr["NAXIS"] == 4
    assert hdr["NAXIS3"] == 5
    assert hdr["CTYPE3"] == "INTEGRATION"
    assert hdr["CRVAL3"] == 0.0
    assert hdr["CDELT3"] == 60.0
    assert hdr["CTYPE4"] == "STOKES"


@pmp(
    "stokes,expected_crval,expected_cdelt",
    [
        (["I"], 1, 1),
        (["Q", "U"], 2, 1),
        (["I", "Q", "U", "V"], 1, 1),
    ],
)
def test_make_fits_header_stokes_axis(stokes, expected_crval, expected_cdelt):
    """The STOKES axis CRVAL/CDELT are derived from the Stokes code table."""
    base = _make_base_fits_header()
    n_stokes = len(stokes)
    shape = (n_stokes, 3, 64, 64)
    hdr = _make_fits_header(base, shape, stokes, drop_axes={3})

    assert hdr["NAXIS4"] == n_stokes
    assert hdr["CTYPE4"] == "STOKES"
    assert hdr["CRVAL4"] == expected_crval
    assert hdr["CDELT4"] == expected_cdelt


def test_make_fits_header_does_not_mutate_input():
    """_make_fits_header must not mutate the caller's dict."""
    base = _make_base_fits_header()
    snapshot = dict(base)
    _make_fits_header(base, (1, 3, 64, 64), ["I"], drop_axes={3})
    assert base == snapshot


def test_hci_channels_per_bin_invariance_no_beam(ms_name, tmp_path):
    """channels_per_bin must not change hci output when no beam model is supplied.

    With channels_per_image=-1 (a single image per spw) and no beam correction,
    the per-bin sub-division is only relevant for beam application and
    deconvolution. The final cube is a weight-normalised sum over bins of
    linear wgridder outputs, which telescopes back to the single-bin result.
    We therefore expect cube, cube_mean and weight to be numerically
    equivalent across channels_per_bin=-1 and channels_per_bin=4.
    """
    out_full = str(tmp_path / "hci_full.zarr")
    out_bin4 = str(tmp_path / "hci_bin4.zarr")

    common_kwargs = dict(
        product="I",
        channels_per_image=-1,
        integrations_per_image=10,
        images_per_chunk=6,
        max_simul_chunks=1,
        field_of_view=0.5,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        beam_model=None,
        robustness=0.0,
        epsilon=1e-7,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    hci_core([ms_name], out_full, channels_per_bin=-1, **common_kwargs)
    hci_core([ms_name], out_bin4, channels_per_bin=4, **common_kwargs)

    ds_full = xr.open_zarr(out_full)
    ds_bin4 = xr.open_zarr(out_bin4)

    # output geometry must match so we can compare element-wise
    assert ds_full.cube.shape == ds_bin4.cube.shape
    assert ds_full.cube_mean.shape == ds_bin4.cube_mean.shape
    assert ds_full.weight.shape == ds_bin4.weight.shape

    cube_full = ds_full.cube.values
    cube_bin4 = ds_bin4.cube.values
    mean_full = ds_full.cube_mean.values
    mean_bin4 = ds_bin4.cube_mean.values
    weight_full = ds_full.weight.values
    weight_bin4 = ds_bin4.weight.values

    # sanity: the run actually produced data
    assert np.any(weight_full > 0)
    assert np.any(weight_bin4 > 0)

    # weights telescope exactly: sum_b max(PSF_b) at the central pixel equals
    # max(PSF_full) since each band's PSF max sits at the same pixel.
    assert_allclose(weight_bin4, weight_full, rtol=1e-6, atol=1e-6)

    cube_scale = float(max(np.abs(cube_full).max(), np.abs(cube_bin4).max(), 1.0))
    mean_scale = float(max(np.abs(mean_full).max(), np.abs(mean_bin4).max(), 1.0))
    assert_allclose(cube_bin4, cube_full, rtol=1e-5, atol=1e-5 * cube_scale)
    assert_allclose(mean_bin4, mean_full, rtol=1e-5, atol=1e-5 * mean_scale)


def test_hci_produces_expected_output_structure(ms_name, tmp_path):
    """hci writes a zarr dataset with the documented variables and coords."""
    out = str(tmp_path / "hci_structure.zarr")
    hci_core(
        [ms_name],
        out,
        product="I",
        channels_per_image=-1,
        integrations_per_image=20,
        images_per_chunk=3,
        max_simul_chunks=1,
        field_of_view=0.5,
        super_resolution_factor=2.0,
        nworkers=1,
        nthreads=1,
        robustness=0.0,
        epsilon=1e-7,
        overwrite=True,
        keep_ray_alive=True,
        log_directory=str(tmp_path / "logs"),
    )

    ds = xr.open_zarr(out)
    for name in ("cube", "cube_mean", "rms", "weight", "nonzero", "flag", "channel_width"):
        assert name in ds, f"{name} missing from hci output"
    assert list(ds.coords["STOKES"].values) == ["I"]
    assert ds.cube.dims == ("STOKES", "FREQ", "TIME", "Y", "X")
    assert ds.cube_mean.dims == ("STOKES", "FREQ", "Y", "X")
    # single spw + channels_per_image=-1 -> single frequency bin in the cube
    assert ds.sizes["FREQ"] == 1
    # 60 integrations / integrations_per_image=20 -> 3 time bins
    assert ds.sizes["TIME"] == 3
    # channel_width should be populated with the per-band bandwidth
    assert np.all(ds.channel_width.values > 0)


def test_hci_rejects_existing_output_without_overwrite(ms_name, tmp_path):
    """hci must refuse to clobber an existing output unless overwrite=True."""
    out = str(tmp_path / "hci_exists.zarr")
    Path(out).mkdir()  # pretend a previous run left its output in place
    # write a marker file so we can tell if the directory was removed
    (Path(out) / "marker").write_text("do not delete")

    with pytest.raises(RuntimeError, match="exists"):
        hci_core(
            [ms_name],
            out,
            product="I",
            channels_per_image=-1,
            integrations_per_image=60,
            images_per_chunk=1,
            max_simul_chunks=1,
            field_of_view=0.5,
            super_resolution_factor=2.0,
            nworkers=1,
            nthreads=1,
            overwrite=False,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )

    assert (Path(out) / "marker").exists()
