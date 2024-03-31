import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from functools import partial
from katbeam import JimBeam
import dask.array as da
from numba.core.errors import NumbaDeprecationWarning
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    from africanus.rime.fast_beam_cubes import beam_cube_dde
    from africanus.rime import parallactic_angles
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('BEAM')

def _interp_beam_impl(freq, nx, ny, cell_deg, btype,
                      utime=None, ant_pos=None, phase_dir=None):
    '''
    A function that returns an object array containing a function
    returning beam values given (l,m) coordinates at a single frequency.
    Frequency mapped to imaging band extenally. Result is meant to be
    passed into eval_beam below.
    '''
    if isinstance(freq, np.ndarray):
        assert freq.size == 1, "Only single frequency interpolation currently supported"
        freq = freq[0]
    if btype is None:
        return np.ones((nx, ny), dtype=float)
    elif btype.endswith('.npz'):
        # these are expected to be in the format given here
        # https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/index.html
        dct = np.load(btype)
        beam = dct['abeam']
        l = np.deg2rad(dct['ldeg'])
        m = np.deg2rad(dct['mdeg'])
        ll, mm = np.meshgrid(l, m, indexing='ij')
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
        beam_amp = beam_amp.reshape(nx, ny)[:, :, None, None, None]
        bfreqs = np.array((freq,))

    if utime is None:
        return beam_amp.squeeze()

    parangles = parallactic_angles(utime, ant_pos, phase_dir, backend='astropy')
    # mean over antanna nant -> 1
    parangles = np.mean(parangles, axis=1, keepdims=True)
    nant = 1
    # beam_cube_dde requirements
    nband = 1
    ntimes = utime.size
    ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
    point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
    beam_extents = np.array([[l.min(), l.max()], [m.min(), m.max()]])
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    beam_image = beam_cube_dde(np.ascontiguousarray(beam_amp),
                            beam_extents, bfreqs,
                            lm, parangles, point_errs,
                            ant_scale, np.array((freq,))).squeeze()
    import ipdb; ipdb.set_trace()
    return beam_image.squeeze()


def interp_beam(freq, nx, ny, cell_deg, btype,
                utime=None, ant_pos=None, phase_dir=None):
    '''
    Blockwise wrapper that returns an object array containing a function
    returning beam values given (l,m) coordinates at a single frequency.
    Frequency mapped to imaging band extenally. Result is meant to be
    passed into eval_beam below.
    '''
    if btype is not None and btype.endswith('.npz'):
        dct = np.load(btype)
        l = dct['ldeg']
        m = dct['mdeg']
        nx = l.size
        ny = m.size
        cell_deg = l[1] - l[0]
        assert cell_deg == m[1] - m[0], 'Beam coords must be on a square grid'
    else:
        l = (-(nx//2) + np.arange(nx)) * cell_deg
        m = (-(ny//2) + np.arange(ny)) * cell_deg

    beam_image = da.blockwise(_interp_beam_impl, 'xy',
                        freq, None,
                        nx, None,
                        ny, None,
                        cell_deg, None,
                        btype, None,
                        utime, None,
                        ant_pos, None,
                        phase_dir, None,
                        new_axes={'x': nx, 'y': ny},
                        dtype=float)
    # l = da.from_array(l, chunks=-1)
    # m = da.from_array(m, chunks=-1)
    return beam_image, l, m


def _eval_beam(beam_image, l_in, m_in, l_out, m_out):
    if l_out.ndim == 2:
        ll = l_out
        mm = m_out
    elif l_out.ndim == 1:
        ll, mm = np.meshgrid(l_out, m_out, indexing='ij')
    else:
        msg = 'Only 1 or 2D coordinates supported for beam evaluation'
        raise ValueError(msg)

    try:
        beamo = RGI((l_in, m_in), beam_image,
                    bounds_error=True, method='linear')
        return beamo((ll, mm))
    except Exception as e:
        print(e)
        print(f"{e} raised in beam evaluation. "
              "Consider setting init.max_field_of_view > grid.field_of_view")
        beamo = RGI((l_in, m_in), beam_image,
                    bounds_error=False, method='linear', fill_value=None)
        return beamo((ll, mm))


def eval_beam(beam_image, l_in, m_in, l_out, m_out):
    nxo = l_out.shape[0]
    nyo = m_out.shape[-1]
    return da.blockwise(_eval_beam, 'xy',
                        beam_image, 'xy',
                        l_in, None,
                        m_in, None,
                        l_out, None,
                        m_out, None,
                        adjust_chunks={'x': nxo, 'y': nyo},
                        dtype=float,
                        meta=np.empty((nxo, nyo), dtype=float))
