"""imager --transfer-model-from (#309): upfront refusals and the pass-2 residual."""

from pathlib import Path

import numpy as np
import pytest

from pfb_imaging.core.imager import imager as imager_core


def _ms_radec(ms_name):
    import arcae

    with arcae.table(f"{ms_name}::FIELD") as tab:
        return tuple(float(v) for v in np.asarray(tab.getcol("PHASE_DIR")).squeeze())


def _write_mds(path, *, radec, cell_rad, stokes=("I",), nx=32, ny=32, freqs=(1.0e9, 1.1e9)):
    from pfb_model_spec.utils.io import build_mds_dataset

    freqs = np.asarray(freqs)
    ns = len(stokes)
    coeffs = np.ones((ns, 1, 1))  # one flat-spectrum component per plane
    ds = build_mds_dataset(
        coeffs,
        np.array([ny // 2]),
        np.array([nx // 2]),
        "t0",
        ["t0"],
        "t",
        "f",
        np.array([1.62393461e9]),
        freqs,
        cell_rad,
        nx,
        ny,
        0.0,
        0.0,
        False,
        True,
        False,
        radec,
        list(stokes),
        "test",
        freq_bounds=np.stack([freqs - 5e7, freqs + 5e7], axis=1),
    )
    ds.to_zarr(str(path), mode="w")
    return str(path)


def _no_ray(monkeypatch):
    import pfb_imaging.core.imager as imager_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)


def test_transfer_refuses_a_missing_product_before_ray(ms_name, tmp_path, monkeypatch):
    _no_ray(monkeypatch)
    mds = _write_mds(tmp_path / "q.mds", radec=_ms_radec(ms_name), cell_rad=1e-5, stokes=("Q",))
    with pytest.raises(ValueError, match="missing"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "out"),
            product="I",
            transfer_model_from=mds,
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )


def test_transfer_refuses_a_missing_path_before_ray(ms_name, tmp_path, monkeypatch):
    _no_ray(monkeypatch)
    with pytest.raises(ValueError, match="No .mds"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "out"),
            transfer_model_from=str(tmp_path / "nope.mds"),
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )


def test_check_transfer_geometry_refusals_and_warnings(tmp_path):
    from pfb_model_spec.utils.spec import open_mds

    from pfb_imaging.core.imager import check_transfer_geometry

    radec = (1.0, -0.5)
    ds = open_mds(_write_mds(tmp_path / "m.mds", radec=radec, cell_rad=1e-5))
    centres = np.array([1.0e9, 1.1e9])

    assert check_transfer_geometry(ds, grid_radec=radec, cell_rad=1e-5, band_centres=centres) == []
    assert check_transfer_geometry(ds, grid_radec=radec, cell_rad=5e-6, band_centres=centres) == []  # finer ok
    # RA either side of zero is a tiny separation, not ~2 pi
    ds0 = open_mds(_write_mds(tmp_path / "z.mds", radec=(1e-12, -0.5), cell_rad=1e-5))
    assert check_transfer_geometry(ds0, grid_radec=(2 * np.pi - 1e-12, -0.5), cell_rad=1e-5, band_centres=centres) == []

    with pytest.raises(ValueError, match="[Tt]angent"):
        check_transfer_geometry(ds, grid_radec=(1.0, -0.4), cell_rad=1e-5, band_centres=centres)
    with pytest.raises(ValueError, match="pfb-model-spec#23"):
        check_transfer_geometry(ds, grid_radec=radec, cell_rad=2e-5, band_centres=centres)
    (msg,) = check_transfer_geometry(ds, grid_radec=radec, cell_rad=1e-5, band_centres=np.array([1.0e9, 1.3e9]))
    assert "extrapolat" in msg


def test_transfer_refuses_a_coarser_grid_before_pass_1(ms_name, tmp_path):
    """Geometry is only known after the MS metadata read; still nothing is gridded."""
    mds = _write_mds(tmp_path / "fine.mds", radec=_ms_radec(ms_name), cell_rad=1e-9)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="pfb-model-spec#23"):
        imager_core(
            [Path(ms_name)],
            str(out),
            channels_per_image=2,
            product="I",
            field_of_view=1.0,
            transfer_model_from=mds,
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )
    assert not list(Path(f"{out}_I.scratch").glob("band*"))


def test_transfer_path_is_resolved_to_absolute(ms_name, tmp_path, monkeypatch):
    _no_ray(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _write_mds(tmp_path / "rel.mds", radec=_ms_radec(ms_name), cell_rad=1e-5, stokes=("Q",))
    # relative path that only resolves from tmp_path; the product refusal proves it was opened
    with pytest.raises(ValueError, match="missing"):
        imager_core(
            [Path(ms_name)],
            "out",
            product="I",
            transfer_model_from="rel.mds",
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )
