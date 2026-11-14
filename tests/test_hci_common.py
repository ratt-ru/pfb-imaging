"""hci_common: FITS headers, beams, scaffold layout, and an import path free of dask-ms and pyrap."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from meerkat_beams.utils import BeamWizard
from numpy.testing import assert_allclose

from pfb_imaging.utils.hci_common import (
    FITS_AXIS_KEYWORDS,
    STOKES_FITS_INDEX,
    beam_for_band,
    beam_gain_for_source,
    build_scaffold,
    default_phase_dir,
    hci_beam_band,
    make_fits_header,
)
from pfb_imaging.utils.misc import radec_barycentre


@pytest.mark.slow  # a fresh interpreter importing the scientific stack: ~14-18 s (D45)
def test_hci_imports_without_daskms_or_pyrap():
    code = (
        "import sys\n"
        "sys.modules['daskms'] = None\n"
        "sys.modules['pyrap'] = None\n"
        "sys.modules['casacore'] = None\n"
        "import pfb_imaging.core.hci\n"
        "import pfb_imaging.cli.hci\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_build_scaffold_layout():
    out_times = np.array([5.0e9, 5.0e9 + 8.0, 5.0e9 + 16.0])
    out_freqs = np.array([1.2e9])
    ds, attrs = build_scaffold(
        out_times,
        out_freqs,
        np.array([10.0]),
        np.array([-30.0]),
        product="I",
        beam=True,
        psf_out=True,
        nx=64,
        ny=64,
        nx_psf=128,
        ny_psf=128,
        cell_deg=1e-3,
        spatial_chunk=128,
        images_per_chunk=2,
        obs_label=None,
    )
    assert ds.cube.dims == ("STOKES", "FREQ", "TIME", "Y", "X")
    assert ds.cube.shape == (1, 1, 3, 64, 64)
    # spatial chunk clipped to the smallest image dimension
    assert ds.cube.data.chunksize == (1, 1, 2, 64, 64)
    assert ds.rms.data.chunksize == (1, 1, 2)
    assert "beam_weight" in ds and "psf" in ds and "psf2" in ds
    assert attrs["fits_dims"][2] == ("TIME", "TIME")
    assert default_phase_dir(np.zeros(2), np.zeros(2)) is None


def test_build_scaffold_does_not_let_a_small_psf_cap_the_cube_chunks():
    """A 128-pixel PSF once clipped a 3840-pixel cube's spatial chunk to 128 (#358)."""
    ds, _ = build_scaffold(
        np.array([5.0e9, 5.0e9 + 8.0]),
        np.array([1.2e9]),
        np.array([10.0]),
        np.array([-30.0]),
        product="I",
        beam=True,
        psf_out=True,
        nx=640,
        ny=640,
        nx_psf=128,
        ny_psf=128,
        cell_deg=1e-3,
        spatial_chunk=320,
        images_per_chunk=2,
        obs_label=None,
    )
    assert ds.cube.data.chunksize == ds.beam_weight.data.chunksize == (1, 1, 2, 320, 320)
    assert ds.cube_mean.data.chunksize == (1, 1, 320, 320)
    assert ds.psf.data.chunksize == (1, 1, 2, 128, 128)
    assert ds.psf2.data.chunksize == (1, 128, 128)


def test_default_phase_dir_with_fields_sharing_a_coordinate():
    """#320: fields sharing an RA or a Dec must not desynchronise the dedup."""
    # three distinct pointings on one Dec strip, the first one seen twice (one
    # per scan) -- unique-ing RA and Dec separately gives 3 RAs and 1 Dec
    ra = np.deg2rad([10.0, 10.0, 11.0, 12.0])
    dec = np.deg2rad([-30.0, -30.0, -30.0, -30.0])

    phase_dir = default_phase_dir(ra, dec)

    ra_str, dec_str = phase_dir.split(",")
    coord = SkyCoord(ra_str, dec_str, frame="fk5", unit=(units.hourangle, units.deg))
    expected = np.rad2deg(radec_barycentre(np.deg2rad([[10.0, -30.0], [11.0, -30.0], [12.0, -30.0]])))
    assert_allclose([coord.ra.deg, coord.dec.deg], expected, atol=1e-6)


def test_default_phase_dir_single_pointing_is_none():
    """One pointing over many scans keeps the MS phase centre."""
    ra = np.deg2rad([10.0, 10.0, 10.0])
    dec = np.deg2rad([-30.0, -30.0, -30.0])
    assert default_phase_dir(ra, dec) is None


@pytest.mark.parametrize("given, band", [(None, None), ("L", "L"), ("l", "L"), ("s0", "S0"), ("U", "U")])
def test_hci_beam_band_takes_a_meerkat_band_name(given, band):
    """--beam-model names the band, as imager and degrid take it (D25)."""
    assert hci_beam_band(given) == band


@pytest.mark.parametrize("given", ["meerkat-beams", "katbeam", "UHF", ""])
def test_hci_beam_band_refuses_anything_else(given):
    with pytest.raises(ValueError, match="meerkat-beams band"):
        hci_beam_band(given)


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
    assert STOKES_FITS_INDEX == {"I": 1, "Q": 2, "U": 3, "V": 4}


def test_fits_axis_keywords_complete():
    """All axis-numbered FITS keywords we care about are present in the rewrite set."""
    for kw in ("CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT", "NAXIS"):
        assert kw in FITS_AXIS_KEYWORDS


def test_make_fits_header_drop_time_axis():
    """Dropping the TIME axis (FITS axis 3) yields a 4-axis (X, Y, FREQ, STOKES) header.

    The FREQ axis must be renumbered to axis 3 and STOKES to axis 4, with
    mandatory keywords appearing first as required by StreamingHDU.
    """
    base = _make_base_fits_header()
    stokes = ["I"]
    # numpy shape (STOKES, FREQ, Y, X) -> FITS axes (X=1, Y=2, FREQ=3, STOKES=4)
    shape = (1, 3, 64, 64)
    hdr = make_fits_header(base, shape, stokes, drop_axes={3})

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
    hdr = make_fits_header(base, shape, ["I"], drop_axes={4})

    assert hdr["NAXIS"] == 4
    assert hdr["NAXIS3"] == 5
    assert hdr["CTYPE3"] == "INTEGRATION"
    assert hdr["CRVAL3"] == 0.0
    assert hdr["CDELT3"] == 60.0
    assert hdr["CTYPE4"] == "STOKES"


@pytest.mark.parametrize(
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
    hdr = make_fits_header(base, shape, stokes, drop_axes={3})

    assert hdr["NAXIS4"] == n_stokes
    assert hdr["CTYPE4"] == "STOKES"
    assert hdr["CRVAL4"] == expected_crval
    assert hdr["CDELT4"] == expected_cdelt


def test_make_fits_header_does_not_mutate_input():
    """make_fits_header must not mutate the caller's dict."""
    base = _make_base_fits_header()
    snapshot = dict(base)
    make_fits_header(base, (1, 3, 64, 64), ["I"], drop_axes={3})
    assert base == snapshot


def _mock_beam_wizard(**attrs):
    """A BeamWizard stand-in that passes the isinstance checks in hci_common."""
    mock = MagicMock(spec=BeamWizard)
    for name, value in attrs.items():
        setattr(mock, name, value)
    return mock


def test_transient_beam_gain_grid():
    """The beam applied to an injected transient must be evaluated at the source's
    true sky position, on the chunk's own (row, chan) grid.

    get_time_variable_beamgain takes an astropy Time and returns (nchan, ntime),
    while the dynamic spectrum it multiplies is (nrow, nchan) on the MS grid.
    """
    utime = np.array([5.0e9, 5.0e9 + 8.0, 5.0e9 + 16.0])  # MJD seconds
    nbl = 3
    time = np.repeat(utime, nbl)
    freq = np.array([1.0e9, 1.1e9])
    ra_deg, dec_deg = 45.0, -30.0

    gain = 1.0 + np.arange(freq.size * utime.size, dtype=float).reshape(freq.size, utime.size)
    bw = _mock_beam_wizard()
    bw.get_time_variable_beamgain.return_value = gain

    out = beam_gain_for_source(bw, ra_deg, dec_deg, "I", utime, time, freq)

    # (nrow, nchan), each row carrying the gain of its own integration
    assert out.shape == (time.size, freq.size)
    assert_allclose(out, np.repeat(gain.T, nbl, axis=0))

    (coord,), kwargs = bw.get_time_variable_beamgain.call_args
    assert_allclose(coord.ra.deg, ra_deg)
    assert_allclose(coord.dec.deg, dec_deg)
    assert isinstance(kwargs["times"], Time)
    assert_allclose(kwargs["times"].mjd, utime / (24 * 3600))
    assert_allclose(kwargs["freq"], freq)


@pytest.mark.parametrize("product", ("I", "QI", "IQUV"))
def test_transient_beam_gain_stokes(product):
    """Transients are injected into the first (IQUV-ordered) Stokes plane only,
    so the gain must be the beam of that product, whatever order product is given in."""
    utime = np.array([5.0e9])
    freq = np.array([1.0e9])
    bw = _mock_beam_wizard()
    bw.get_time_variable_beamgain.return_value = np.ones((1, 1))

    beam_gain_for_source(bw, 45.0, -30.0, product, utime, utime.copy(), freq)

    kwargs = bw.get_time_variable_beamgain.call_args.kwargs
    assert kwargs["i"] == "I"
    assert kwargs["j"] == "I"


def test_beam_for_band_stokes_order():
    """beam_for_band must fill the Stokes axis in I, Q, U, V order to match the
    Stokes axis of the data (stokes_expr_funcs always emits I, Q, U, V) and the
    sorted corr coordinate written to the output dataset."""
    product = "IQUV"
    nx = ny = 32
    cell_deg = 1e-2
    l_beam = (np.arange(64) - 32) * cell_deg
    m_beam = (np.arange(64) - 32) * cell_deg
    # a distinct constant per Stokes plane so a permutation is detectable
    levels = {"I": 1.0, "Q": 2.0, "U": 3.0, "V": 4.0}

    bw = _mock_beam_wizard(bds=SimpleNamespace(X=SimpleNamespace(values=l_beam), Y=SimpleNamespace(values=m_beam)))
    bw.get_rotation_averaged_beam.side_effect = lambda **kw: (
        np.full((m_beam.size, l_beam.size), levels[kw["i"]]),
        None,
    )

    radec = np.array([0.5, -0.5])
    pbeam = beam_for_band(
        bw,
        np.array([5.0e9]),
        product,
        np.float64,
        np.array([1.0e9]),
        radec,
        radec,
        cell_deg,
        nx,
        ny,
    )

    assert [kw["i"] for _, kw in bw.get_rotation_averaged_beam.call_args_list] == ["I", "Q", "U", "V"]
    for index, prod in enumerate("IQUV"):
        assert_allclose(pbeam[index], levels[prod], rtol=1e-6)


@pytest.mark.parametrize("grid_dtype", [np.float32, np.float64])
def test_beam_for_band_reuses_the_reprojection_and_matches_reproject_interp(grid_dtype):
    """The coordinate map is built once per field and changes nothing (#358).

    The reference is the pre-cache body: reproject_interp per Stokes, zeroed outside the footprint.
    """
    from astropy.wcs import WCS
    from reproject import reproject_interp

    from pfb_imaging.utils import hci_common

    hci_common._beam_reprojection.cache_clear()
    nx, ny, cell_deg = 40, 36, 1.5e-2
    l_beam = ((np.arange(64) - 32) * 1e-2).astype(grid_dtype)
    m_beam = ((np.arange(48) - 24) * 1.2e-2).astype(grid_dtype)
    rng = np.random.default_rng(0)
    bw = _mock_beam_wizard(bds=SimpleNamespace(X=SimpleNamespace(values=l_beam), Y=SimpleNamespace(values=m_beam)))
    bw.get_rotation_averaged_beam.side_effect = lambda **kw: (rng.random((m_beam.size, l_beam.size)), None)
    radec, radec_new = np.array([0.5, -0.5]), np.array([0.502, -0.499])

    for _ in range(3):
        beam_for_band(bw, np.array([5.0e9]), "I", np.float64, np.array([1e9]), radec, radec_new, cell_deg, nx, ny)
    info = hci_common._beam_reprojection.cache_info()
    assert (info.misses, info.hits) == (1, 2)

    # the same beam through the uncached reprojection
    plane = rng.random((m_beam.size, l_beam.size))
    bw.get_rotation_averaged_beam.side_effect = lambda **kw: (plane, None)
    got = beam_for_band(bw, np.array([5.0e9]), "I", np.float64, np.array([1e9]), radec, radec_new, cell_deg, nx, ny)
    dl, dm = l_beam[1] - l_beam[0], m_beam[1] - m_beam[0]
    wcs_ref = WCS(naxis=2)
    wcs_ref.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_ref.wcs.cdelt = np.array((dl, dm))
    wcs_ref.wcs.cunit = ["deg", "deg"]
    wcs_ref.wcs.crval = np.rad2deg(radec)
    wcs_ref.wcs.crpix = [1.0 - l_beam[0] / dl, 1.0 - m_beam[0] / dm]
    wcs_target = WCS(naxis=2)
    wcs_target.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_target.wcs.cdelt = np.array((-cell_deg, cell_deg))
    wcs_target.wcs.cunit = ["deg", "deg"]
    wcs_target.wcs.crval = np.rad2deg(radec_new)
    wcs_target.wcs.crpix = [1 + nx // 2, 1 + ny // 2]
    want, footprint = reproject_interp((plane, wcs_ref), wcs_target, shape_out=(ny, nx))
    want[~footprint.astype(bool)] = 0.0
    assert (got[0] > 0).any() and (got[0] == 0).any()  # the footprint edge is in the image
    np.testing.assert_array_equal(got[0], want)
