import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from katbeam import JimBeam
from africanus.rime.fast_beam_cubes import beam_cube_dde
from africanus.rime import parallactic_angles


def interp_beam(freq, nx, ny, cell_deg, btype,
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
        l = (-(nx//2) + np.arange(nx)) * cell_deg
        m = (-(ny//2) + np.arange(ny)) * cell_deg
        return np.ones((nx, ny), dtype=float), l, m
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
        return beam_amp.squeeze(), l, m

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

    return beam_image.squeeze(), l, m


def eval_beam(beam_image, l_in, m_in, l_out, m_out):
    if l_out.ndim == 2:
        ll = l_out
        mm = m_out
    elif l_out.ndim == 1:
        ll, mm = np.meshgrid(l_out, m_out, indexing='ij')
    else:
        msg = 'Only 1 or 2D coordinates supported for beam evaluation'
        raise ValueError(msg)

    if (beam_image == 1.0).all():
        return np.ones_like(ll)
    else:  # this gets expensive
        beamo = RGI((l_in, m_in), beam_image,
                    bounds_error=False, method='linear', fill_value=1.0)
        return beamo((ll, mm))
