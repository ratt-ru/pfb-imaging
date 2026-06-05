"""Tree-aware FITS export from the imager .dt DataTree (casacore-free)."""

import numpy as np
import xarray as xr
from astropy.io import fits as afits

from pfb_imaging.utils.fits import dt2fits


def _band_node(timeid, freq_out, val, wsum=10.0, nx=8):
    return xr.Dataset(
        {
            "DIRTY": (("corr", "x", "y"), np.full((1, nx, nx), float(val))),
            "WSUM": (("corr",), np.array([float(wsum)])),
        },
        coords={"corr": ["I"]},
        attrs={"timeid": timeid, "freq_out": freq_out, "time_out": 5.0e9, "ra": 0.1, "dec": -0.2, "cell_rad": 1.0e-6},
    )


def test_dt2fits_mfs(tmp_path):
    store = str(tmp_path / "out.dt")
    _band_node(0, 1.0e9, 1.0).to_zarr(store, group="band0000_time0000", mode="w")
    _band_node(0, 1.1e9, 3.0).to_zarr(store, group="band0001_time0000", mode="a")

    outname = str(tmp_path / "img")
    dt2fits(store, "DIRTY", outname, norm_wsum=True, do_mfs=True, do_cube=False)

    data = np.squeeze(afits.getdata(outname + "_dirty_time0_mfs.fits"))
    assert data.shape == (8, 8)
    # norm_wsum MFS = sum(cube) / sum(wsum) per pixel = (1 + 3) / (10 + 10)
    np.testing.assert_allclose(data.flat[0], (1.0 + 3.0) / 20.0, rtol=1e-5)


def test_dt2fits_cube_has_band_axis(tmp_path):
    # 3 bands: set_wcs's reference-channel index (nchan//2+1) needs nchan>2
    store = str(tmp_path / "out2.dt")
    _band_node(0, 1.0e9, 1.0).to_zarr(store, group="band0000_time0000", mode="w")
    _band_node(0, 1.1e9, 3.0).to_zarr(store, group="band0001_time0000", mode="a")
    _band_node(0, 1.2e9, 5.0).to_zarr(store, group="band0002_time0000", mode="a")

    outname = str(tmp_path / "img2")
    dt2fits(store, "DIRTY", outname, norm_wsum=True, do_mfs=False, do_cube=True)

    data = afits.getdata(outname + "_dirty_time0.fits")
    # FITS axes (numpy, reversed): (STOKES, FREQ=band, DEC, RA)
    assert data.shape == (1, 3, 8, 8)
    # norm_wsum cube: per-band value / per-band wsum
    np.testing.assert_allclose(data[0, 0].flat[0], 1.0 / 10.0, rtol=1e-5)
    np.testing.assert_allclose(data[0, 1].flat[0], 3.0 / 10.0, rtol=1e-5)
    np.testing.assert_allclose(data[0, 2].flat[0], 5.0 / 10.0, rtol=1e-5)


def test_dt2fits_no_column_is_noop(tmp_path):
    store = str(tmp_path / "out3.dt")
    _band_node(0, 1.0e9, 1.0).to_zarr(store, group="band0000_time0000", mode="w")
    # MODEL is not present on the band node -> returns column, writes nothing
    assert dt2fits(store, "MODEL", str(tmp_path / "img3"), do_mfs=True, do_cube=False) == "MODEL"
