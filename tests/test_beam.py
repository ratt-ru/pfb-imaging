import numpy as np
import pytest
from katbeam import JimBeam

from pfb_imaging.utils.beam import eval_beam, interp_beam

pmp = pytest.mark.parametrize


@pmp("beam_model", ("kbl", "kbuhf", None))
@pmp("nx", (24, 128))
@pmp("ny", (64, 92))
def test_beam(beam_model, nx, ny):
    freq = 1e9
    cell_deg = 1e-2
    beam, l_coord, m_coord = interp_beam(freq, nx, ny, cell_deg, beam_model)
    ll, mm = np.meshgrid(l_coord, m_coord, indexing="ij")
    bvals = eval_beam(beam, l_coord, m_coord, ll, mm)

    if beam_model is None:
        assert (bvals == 1).all()
    elif beam_model == "kbl":
        jb = JimBeam("MKAT-AA-L-JIM-2020").I
        assert np.allclose(bvals, jb(ll, mm, freq / 1e6), atol=1e-10)
    elif beam_model == "kbuhf":
        jb = JimBeam("MKAT-AA-UHF-JIM-2020").I
        assert np.allclose(bvals, jb(ll, mm, freq / 1e6), atol=1e-10)
