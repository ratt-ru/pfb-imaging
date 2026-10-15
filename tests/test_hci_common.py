"""hci_common: scaffold layout, and the hci-msv4 import path stays free of dask-ms and pyrap."""

import subprocess
import sys

import numpy as np
import pytest

from pfb_imaging.utils.hci_common import build_scaffold, default_phase_dir


@pytest.mark.slow  # a fresh interpreter importing the scientific stack: ~14-18 s (D45)
def test_hci_msv4_imports_without_daskms_or_pyrap():
    code = (
        "import sys\n"
        "sys.modules['daskms'] = None\n"
        "sys.modules['pyrap'] = None\n"
        "sys.modules['casacore'] = None\n"
        "import pfb_imaging.core.hci_msv4\n"
        "import pfb_imaging.cli.hci_msv4\n"
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
