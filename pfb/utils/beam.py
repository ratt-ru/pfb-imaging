import numpy as np
from functools import partial
from katbeam import JimBeam
import dask.array as da

def Id_beam(l, m):
    return np.ones(l.shape, dtype=float)

def _interp_beam_impl(freq, nx, ny, cell_deg, btype):
    '''
    A function that returns an object array containing a function
    returning beam values given (l,m) coordinates at a single frequency.
    Frequency mapped to imaging band extenally. Result is meant to be
    passed into eval_beam below.
    '''
    l = (-(nx//2) + np.arange(nx)) * cell_deg
    m = (-(ny//2) + np.arange(ny)) * cell_deg
    ll, mm = np.meshgrid(l, m, indexing='ij')

    if btype is None:
        beam = Id_beam
    else:
        btype = btype.lower()
        btype = btype.replace('-', '_')
        if btype in ["kbl", "kb_l", "katbeam_l"]:
            # katbeam L band
            beam = partial(JimBeam('MKAT-AA-L-JIM-2020').I, freqMHz=freq)
        elif btype in ["kbuhf", "kb_uhf", "katbeam_uhf"]:
            # katbeam L band
            beam = partial(JimBeam('MKAT-AA-UHF-JIM-2020').I, freqMHz=freq)
        else:
            raise ValueError(f"Unknown beam model {btype}")
    return np.array([beam], dtype=object)


def interp_beam(freq, nx, ny, cell_deg, btype):
    '''
    Blockwise wrapper that returns an object array containing a function
    returning beam values given (l,m) coordinates at a single frequency.
    Frequency mapped to imaging band extenally. Result is meant to be
    passed into eval_beam below.
    '''
    return da.blockwise(_interp_beam_impl, '1',
                        freq, None,
                        nx, None,
                        ny, None,
                        cell_deg, None,
                        btype, None,
                        new_axes={'1': 1},
                        dtype=object)

def _eval_beam_impl(beam_object_array, l, m):
    return beam_object_array[0](l, m)

def _eval_beam(beam_object_array, l, m):
    return _eval_beam_impl(beam_object_array[0], l, m)

def eval_beam(beam_object_array, l, m):
    if l.ndim == 2:
        lout = ('nx', 'ny')
        mout = ('nx', 'ny')
    else:
        lout = ('nx',)
        mout = ('ny',)
    return da.blockwise(_eval_beam, ('nx', 'ny'),
                        beam_object_array, ('1',),
                        l, lout,
                        m, mout,
                        dtype=float)
