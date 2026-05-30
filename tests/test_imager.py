"""Smoke test for the MSv4 imager core function."""

from pathlib import Path

from pfb_imaging.core.imager import imager as imager_core


def test_imager_runs(ms_name):
    """imager() runs through with ipi=15 and cpi=2 on the test MS."""
    test_dir = Path(ms_name).resolve().parent
    outname = str(test_dir / "test_imager")

    imager_core(
        [Path(ms_name)],
        outname,
        integrations_per_image=15,
        channels_per_image=2,
        product="I",
        overwrite=True,
        keep_ray_alive=True,
    )

    xds_dir = test_dir / "test_imager_I.xds"
    assert xds_dir.is_dir()
    assert any(xds_dir.glob("*.zarr")), "imager produced no zarr datasets"
