import numpy as np
from katbeam import JimBeam
import dask.array as da

def _katbeam_impl(freq, nx, ny, cell_deg, beam_dtype):
    nband = np.size(freq)
    refpix = 1 + nx//2
    l_coord = -np.arange(1 - refpix, 1 + nx - refpix) * cell_deg
    refpix = ny//2
    m_coord = np.arange(1 - refpix, 1 + ny - refpix) * cell_deg
    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
    beam = JimBeam('MKAT-AA-L-JIM-2020')
    pbeam = np.zeros((nband, nx, ny), dtype=beam_dtype)
    if nband > 1:
        for i in range(nband):  # freq in MHz
            pbeam[i] = beam.I(xx, yy, freq[i]/1e6).astype(beam_dtype)
    else:
        pbeam[0] = beam.I(xx, yy, freq/1e6).astype(beam_dtype)
    return pbeam

def katbeam(freq, nx, ny, cell_deg, beam_dtype=np.float32):
    return da.blockwise(_katbeam_impl, ("nband, nx, ny"),
                        freq, ("nband",),
                        nx, None,
                        ny, None,
                        cell_deg, None,
                        beam_dtype, None,
                        new_axes={"nx": nx, "ny": ny},
                        dtype=beam_dtype)

def beam2obj(bvals, l, m):
    return da.blockwise(_beam2obj, ('x',),
                        bvals, ('l','m'),
                        l, ('l',),
                        m, ('m',),
                        dtype=object,
                        new_axes={'x': 1})

def _beam2obj(bvals, l, m):
    return _beam2obj_impl(bvals[0][0], l[0], m[0])

from scipy.interpolate import RectBivariateSpline
def _beam2obj_impl(bvals, l, m):
    bo = RectBivariateSpline(l, m, bvals)
    return np.array([bo], dtype=object)
