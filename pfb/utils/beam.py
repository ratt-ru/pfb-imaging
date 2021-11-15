import numpy as np
from katbeam import JimBeam
import dask.array as da

def _katbeam_impl(freq, nx, ny, cell_deg, beam_dtype):
    nband = freq.size
    refpix = 1 + nx//2
    l_coord = -np.arange(1 - refpix, 1 + npix - refpix) * cell_deg
    refpix = ny//2
    m_coord = np.arange(1 - refpix, 1 + npix - refpix) * cell_deg
    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
    beam = JimBeam('MKAT-AA-L-JIM-2020')
    pbeam = np.zeros((nband, nx, ny), dtype=beam_dtype)
    for i in range(nband):  # freq in MHz
        pbeam[i] = beam.I(xx, yy, freq[i]/1e6).astype(beam_dtype)
    return pbeam

def katbeam(freq, nx, ny, cell_deg, beam_dtype=np.float32):
    return da.blockwise(_katbeam_impl, ("nband, nx, ny"),
                        freq, ("nband",),
                        nx, None,
                        ny, None,
                        cell_deg, None,
                        dtype, None,
                        btype=None,
                        new_axes={"nx": nx, "ny": ny},
                        dtype=beam_dtype)
