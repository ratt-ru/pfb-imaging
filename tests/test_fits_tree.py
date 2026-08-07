"""Tree-aware FITS export from the imager .dt DataTree (casacore-free)."""

import numpy as np
import xarray as xr
from astropy.io import fits as afits

from pfb_imaging.utils.fits import dt2fits


def _band_node(timeid, freq_out, val, wsum=10.0, nx=8, bandid=0):
    return xr.Dataset(
        {
            "DIRTY": (("corr", "y", "x"), np.full((1, nx, nx), float(val))),
            "WSUM": (("corr",), np.array([float(wsum)])),
        },
        coords={"corr": ["I"]},
        attrs={
            "timeid": timeid,
            "bandid": bandid,
            "freq_out": freq_out,
            "time_out": 5.0e9,
            "ra": 0.1,
            "dec": -0.2,
            "cell_rad": 1.0e-6,
        },
    )


def test_dt2fits_mfs(tmp_path):
    store = str(tmp_path / "out.dt")
    _band_node(0, 1.0e9, 1.0, bandid=0).to_zarr(store, group="band0000_time0000", mode="w")
    _band_node(0, 1.1e9, 3.0, bandid=1).to_zarr(store, group="band0001_time0000", mode="a")

    outname = str(tmp_path / "img")
    dt2fits(store, "DIRTY", outname, norm_wsum=True, do_mfs=True, do_cube=False)

    data = np.squeeze(afits.getdata(outname + "_dirty_time0_mfs.fits"))
    assert data.shape == (8, 8)
    # norm_wsum MFS = sum(cube) / sum(wsum) per pixel = (1 + 3) / (10 + 10)
    np.testing.assert_allclose(data.flat[0], (1.0 + 3.0) / 20.0, rtol=1e-5)


def test_dt2fits_cube_has_band_axis(tmp_path):
    # 3 bands: set_wcs's reference-channel index (nchan//2+1) needs nchan>2
    store = str(tmp_path / "out2.dt")
    _band_node(0, 1.0e9, 1.0, bandid=0).to_zarr(store, group="band0000_time0000", mode="w")
    _band_node(0, 1.1e9, 3.0, bandid=1).to_zarr(store, group="band0001_time0000", mode="a")
    _band_node(0, 1.2e9, 5.0, bandid=2).to_zarr(store, group="band0002_time0000", mode="a")

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
    _band_node(0, 1.0e9, 1.0, bandid=0).to_zarr(store, group="band0000_time0000", mode="w")
    # MODEL is not present on the band node -> returns column, writes nothing
    assert dt2fits(store, "MODEL", str(tmp_path / "img3"), do_mfs=True, do_cube=False) == "MODEL"


def test_dt2fits_orders_planes_by_bandid(tmp_path):
    """Effective frequencies are data-dependent and can invert under severe
    asymmetric flagging; plane order must follow bandid, which is monotonic in
    frequency by construction (issue #296).
    """
    store = str(tmp_path / "out.dt")
    # band 0 deliberately labelled at a HIGHER frequency than band 1
    _band_node(0, 1.4e9, 1.0, bandid=0).to_zarr(store, group="band0000_time0000", mode="w")
    _band_node(0, 1.1e9, 3.0, bandid=1).to_zarr(store, group="band0001_time0000", mode="a")

    outname = str(tmp_path / "img")
    dt2fits(store, "DIRTY", outname, norm_wsum=True, do_mfs=False, do_cube=True)

    data = np.squeeze(afits.getdata(outname + "_dirty_time0.fits"))
    assert data.shape == (2, 8, 8)
    # plane 0 is bandid 0 (value 1.0/10), not the lower-frequency bandid 1
    np.testing.assert_allclose(data[0].flat[0], 0.1, rtol=1e-5)
    np.testing.assert_allclose(data[1].flat[0], 0.3, rtol=1e-5)


def _band_node_with_psfpars(timeid, freq_out, val, psfparsn, psfparsf, wsum=10.0, nx=8, bandid=0):
    ds = _band_node(timeid, freq_out, val, wsum=wsum, nx=nx, bandid=bandid)
    return ds.assign(
        {
            "PSFPARSN": (("corr", "bpar"), np.asarray(psfparsn, dtype=float)[None]),
            "PSFPARSF": (("corr", "bpar"), np.asarray(psfparsf, dtype=float)[None]),
        }
    ).assign_coords({"bpar": ["BMAJ", "BMIN", "BPA"]})


def test_dt2fits_drop_bands_excludes_from_cube_and_mfs(tmp_path):
    """A dropped band leaves the cube entirely and contributes to no MFS
    reduction -- image sum, wsum or freq. Zero planes are deliberately NOT
    used; the resulting non-uniform frequency axis is issue #302's problem.
    """
    store = str(tmp_path / "drop.dt")
    _band_node(0, 1.0e9, 1.0, bandid=0).to_zarr(store, group="band0000_time0000", mode="w")
    _band_node(0, 1.1e9, 3.0, bandid=1).to_zarr(store, group="band0001_time0000", mode="a")
    _band_node(0, 1.2e9, 5.0, bandid=2).to_zarr(store, group="band0002_time0000", mode="a")

    outname = str(tmp_path / "img")
    dt2fits(store, "DIRTY", outname, norm_wsum=True, do_mfs=True, do_cube=True, drop_bands=[1])

    cube = afits.getdata(outname + "_dirty_time0.fits")
    assert cube.shape == (1, 2, 8, 8)
    np.testing.assert_allclose(cube[0, 0].flat[0], 1.0 / 10.0, rtol=1e-5)
    np.testing.assert_allclose(cube[0, 1].flat[0], 5.0 / 10.0, rtol=1e-5)

    mfs = np.squeeze(afits.getdata(outname + "_dirty_time0_mfs.fits"))
    np.testing.assert_allclose(mfs.flat[0], (1.0 + 5.0) / 20.0, rtol=1e-5)

    hdr = afits.getheader(outname + "_dirty_time0_mfs.fits")
    assert hdr["DROPBAND"] == "1"
    assert hdr["WSUM"] == 20.0


def test_dt2fits_psfpars_var_selects_the_named_variable(tmp_path):
    """The cube BEAMS table and BMAJ{i} cards come from psfpars_var, so
    restore can publish the final resolution (PSFPARSF) rather than the
    native one (PSFPARSN).
    """
    store = str(tmp_path / "pp.dt")
    _band_node_with_psfpars(0, 1.0e9, 1.0, [2.0, 1.0, 0.0], [6.0, 5.0, 0.0], bandid=0).to_zarr(
        store, group="band0000_time0000", mode="w"
    )
    _band_node_with_psfpars(0, 1.1e9, 3.0, [3.0, 1.5, 0.0], [6.0, 5.0, 0.0], bandid=1).to_zarr(
        store, group="band0001_time0000", mode="a"
    )

    outname = str(tmp_path / "pp")
    dt2fits(store, "DIRTY", outname, do_mfs=False, do_cube=True, psfpars_var="PSFPARSF")

    hdr = afits.getheader(outname + "_dirty_time0.fits")
    cell_deg = np.rad2deg(1.0e-6)
    np.testing.assert_allclose(hdr["BMAJ1"], 6.0 * cell_deg, rtol=1e-5)
    np.testing.assert_allclose(hdr["BMAJ2"], 6.0 * cell_deg, rtol=1e-5)


def test_dt2fits_psfpars_var_defaults_to_native(tmp_path):
    """Default behaviour is unchanged: PSFPARSN drives the beam cards."""
    store = str(tmp_path / "ppn.dt")
    _band_node_with_psfpars(0, 1.0e9, 1.0, [2.0, 1.0, 0.0], [6.0, 5.0, 0.0], bandid=0).to_zarr(
        store, group="band0000_time0000", mode="w"
    )
    _band_node_with_psfpars(0, 1.1e9, 3.0, [3.0, 1.5, 0.0], [6.0, 5.0, 0.0], bandid=1).to_zarr(
        store, group="band0001_time0000", mode="a"
    )

    outname = str(tmp_path / "ppn")
    dt2fits(store, "DIRTY", outname, do_mfs=False, do_cube=True)

    hdr = afits.getheader(outname + "_dirty_time0.fits")
    cell_deg = np.rad2deg(1.0e-6)
    np.testing.assert_allclose(hdr["BMAJ1"], 2.0 * cell_deg, rtol=1e-5)
    np.testing.assert_allclose(hdr["BMAJ2"], 3.0 * cell_deg, rtol=1e-5)
