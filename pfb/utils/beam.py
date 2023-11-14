import numpy as np
from functools import partial
from katbeam import JimBeam
import dask.array as da
from numba.core.errors import NumbaDeprecationWarning
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    from africanus.rime.fast_beam_cubes import beam_cube_dde
    from africanus.rime import parallactic_angles


def _interp_beam_impl(freq, nx, ny, cell_deg, btype,
                      utime=None, ant_pos=None, phase_dir=None):
    '''
    A function that returns an object array containing a function
    returning beam values given (l,m) coordinates at a single frequency.
    Frequency mapped to imaging band extenally. Result is meant to be
    passed into eval_beam below.
    '''
    if isinstance(freq, np.ndarray):
        assert freq.size == 1
        freq = freq[0]
    if btype is None:
        beam = np.ones((nx, ny), dtype=float)
    elif btype.endswith('.npz'):
        # these are expected to be in the format given here
        # https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/index.html
        dct = np.load(btype)
        beam = dct['abeam']
        l = np.deg2rad(dct['ldeg'])
        m = np.deg2rad(dct['mdeg'])
        ll, mm = np.meshgrid(l, m, indexing='ij')
        lm = np.vstack((ll.flatten(), mm.flatten())).T
        beam_extents = np.array([[l.min(), l.max()], [m.min(), m.max()]])
        bfreqs = dct['freq']
        beam_amp = (beam[0, :, :, :] * beam[0, :, :, :].conj() +
                    beam[-1, :, :, :] * beam[-1, :, :, :].conj())/2.0
        beam_amp = np.transpose(beam_amp, (1,2,0))[:, :, :, None, None].real
    else:
        btype = btype.lower()
        btype = btype.replace('-', '_')
        l = (-(nx//2) + np.arange(nx)) * cell_deg
        m = (-(ny//2) + np.arange(ny)) * cell_deg
        ll, mm = np.meshgrid(l, m, indexing='ij')
        if btype in ["kbl", "kb_l", "katbeam_l"]:
            # katbeam L band
            beam_amp = JimBeam('MKAT-AA-L-JIM-2020').I(ll.flatten(),
                                                       mm.flatten(),
                                                       freqMHz=freq/1e6)
        elif btype in ["kbuhf", "kb_uhf", "katbeam_uhf"]:
            # katbeam L band
            beam_amp = JimBeam('MKAT-AA-UHF-JIM-2020').I(ll.flatten(),
                                                       mm.flatten(),
                                                       freqMHz=freq/1e6)
        else:
            raise ValueError(f"Unknown beam model {btype}")
        beam_amp = beam_amp[:, :, None, None, None]

    parangles = parallactic_angles(utime, ant_pos, phase_dir, backend='')
    # mean over antanna nant -> 1
    parangles = np.mean(parangles, axis=1, keepdims=True)
    nant = 1
    # beam_cube_dde requirements
    nband = 1
    ntimes = utime.size
    ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
    point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
    beam_image = beam_cube_dde(np.ascontiguousarray(beam_amp),
                               beam_extents, bfreqs,
                               lm, parangles, point_errs,
                               ant_scale, np.array((freq,))).squeeze()
    return beam_image


def interp_beam(freq, nx, ny, cell_deg, btype,
                utime=None, ant_pos=None, phase_dir=None):
    '''
    Blockwise wrapper that returns an object array containing a function
    returning beam values given (l,m) coordinates at a single frequency.
    Frequency mapped to imaging band extenally. Result is meant to be
    passed into eval_beam below.
    '''
    if btype.endwith('.npz'):
        dct = np.load(btype)
        nx = dct['ldeg'].size
        ny = dct['mdeg'].size

    return da.blockwise(_interp_beam_impl, 'xy',
                        freq, None,
                        nx, None,
                        ny, None,
                        cell_deg, None,
                        btype, None,
                        utime, None,
                        ant_pos, None,
                        phase_dir, None,
                        new_axes={'x': nx, 'y': ny},
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


def get_beam_meta(msname):
    from pyrap.tables import table
    ms = table(msname)
    time = ms.getcol('TIME')
    utime = np.unique(time)
    field = table(f'{msname}::FIELD')
    phase_dir = field.getcol('PHASE_DIR').squeeze()
    ant = table(f'{msname}::ANTENNA')
    ant_pos = ant.getcol('POSITION')

    return utime, phase_dir, ant_pos
