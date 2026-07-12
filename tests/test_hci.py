"""In-process smoke test for the hci sub-command (issue #273 consumer coverage).

Runs the Ray-distributed batch_stokes_image/stokes_image path on the small
test MS for both weighting modes and sanity-checks the output zarr cube.
minvar is the mode that exercised the weight_data recompile bug in
production.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pfb_imaging.core.hci import hci as hci_core

pmp = pytest.mark.parametrize


@pmp("wgt_mode", ("l2", "minvar"))
def test_hci_writes_cube(wgt_mode, ms_name, tmp_path):
    outname = str(tmp_path / f"test_hci_{wgt_mode}.zarr")

    hci_core(
        [Path(ms_name)],
        outname,
        product="I",
        data_column="DATA",
        integrations_per_image=8,
        images_per_chunk=4,
        max_simul_chunks=2,
        field_of_view=1.0,
        precision="single",
        epsilon=1e-4,  # ducc0 single-precision kernels need epsilon >~ 1e-5
        wgt_mode=wgt_mode,
        overwrite=True,
        keep_ray_alive=True,
    )

    ds = xr.open_zarr(outname, chunks=None)
    cube = ds.cube.values  # (STOKES, FREQ, TIME, Y, X)
    wsum = ds.weight.values  # (STOKES, FREQ, TIME)

    assert (wsum > 0).any(), "no data was imaged"
    imaged = cube[wsum > 0]
    assert np.isfinite(imaged).all()
    assert np.abs(imaged).max() > 0
