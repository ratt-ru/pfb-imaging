"""Smoke test for the MSv4 DataTree imager (pass 1 -> .scratch tree)."""

from pathlib import Path

import numpy as np
import xarray as xr

from pfb_imaging.core.imager import imager as imager_core


def test_imager_pass1_writes_scratch_tree(ms_name):
    """imager() pass 1 writes a .scratch DataTree of fine averaged Stokes pieces."""
    test_dir = Path(ms_name).resolve().parent
    outname = str(test_dir / "test_imager")

    imager_core(
        [Path(ms_name)],
        outname,
        integrations_per_image=15,
        channels_per_image=2,
        product="I",
        field_of_view=1.0,
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = xr.open_datatree(outname + "_I.scratch", engine="zarr", chunks=None)
    # output-image groups: band{b:04d}_time{t:04d}
    image_names = [n for n in dt.children if n.startswith("band")]
    assert image_names, "no output-image groups written"

    # each image group has at least one partition piece with the expected vars
    image = dt[sorted(image_names)[0]]
    pieces = [name for name in image.children]
    assert pieces, "image group has no partition pieces"
    piece = image[pieces[0]]
    for v in ("VIS", "WEIGHT", "MASK", "UVW", "FREQ", "BEAM", "COUNTS"):
        assert v in piece.ds, f"piece missing {v}"
    assert piece.ds.attrs["baseline_group"] == "all"
    ncorr = piece.ds.corr.size
    assert piece.ds.COUNTS.shape[0] == ncorr
    assert np.isfinite(piece.ds.VIS.values).all()
